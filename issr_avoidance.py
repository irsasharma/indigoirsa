"""
issr_avoidance.py
=================
Contrail-mitigation simulation library. Import this; do not run directly.
Use run_issr.py to configure and execute.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from dataclasses import dataclass, field
from typing import Literal

from pyproj import Transformer
from pycontrails import Flight, MetDataset
from pycontrails.core.fuel import JetA
from pycontrails.models.emissions import Emissions
from pycontrails.models.ps_model import PSFlight
from pycontrails.physics import units
from pycontrails.models.cocip import Cocip
from pycontrails.models.humidity_scaling import ExponentialBoostHumidityScaling
from pycontrails.physics import geo


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def lonlat_to_m(lon, lat, ref_lon, ref_lat):
    """Convert lon/lat [deg] to local Cartesian metres via Transverse Mercator."""
    transformer = Transformer.from_crs(
        "epsg:4326",
        f"+proj=tmerc +lat_0={ref_lat} +lon_0={ref_lon} +k=1 +x_0=0 +y_0=0",
        always_xy=True,
    )
    x_m, y_m = transformer.transform(
        np.asarray(lon, dtype=float), np.asarray(lat, dtype=float)
    )
    return x_m, y_m


def q_sat_ice(T, p_Pa):
    """Saturation specific humidity over ice [kg/kg] via Magnus formula."""
    eps = 0.6220
    e_sat = 611.2 * np.exp(22.46 * (T - 273.16) / (T - 0.55))
    return eps * e_sat / (p_Pa - e_sat)





# ---------------------------------------------------------------------------
# Parameter dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SimParams:
    """Simulation domain and temporal parameters."""
    t_start: pd.Timestamp = field(default_factory=lambda: pd.Timestamp("2025-01-20 13:00"))
    dt_fl: pd.Timedelta = field(default_factory=lambda: pd.Timedelta(minutes=2))
    duration_fl: pd.Timedelta = field(default_factory=lambda: pd.Timedelta(hours=1))
    dt_met: pd.Timedelta = field(default_factory=lambda: pd.Timedelta(hours=1))
    duration_met: pd.Timedelta = field(default_factory=lambda: pd.Timedelta(hours=12))
    lon_bounds: tuple = (0.0, 8.0)
    lat_bounds: tuple = (0.0, 2.0)
    alt_bounds: tuple = (10_000.0, 15_000.0)
    hres: float = 0.1    # horizontal resolution [deg]
    vres: float = 100.0  # vertical resolution [m]


@dataclass
class FlParams:
    """Flight parameters."""
    ac_type: str = "A320"
    target_alt: float = None      # cruise altitude [m]; None -> domain midpoint
    alt_delta: float = 1000.0     # altitude shift for S1/S2 [m]
    fl_lon_bounds: tuple = (0.0, 4.0)
    rocd: float = 5.0             # rate of climb/descent [m/s]
    avoid_frac: float = 1.0       # fraction of ISSR Gaussian to avoid [0-1]
    cruise_mach: float = 0.78     # cruise Mach number passed to PSFlight


@dataclass
class MetParams:
    """Uniform background atmospheric state."""
    air_temperature: float = 220.0
    eastward_wind: float = 0.0
    northward_wind: float = 0.0
    lagrangian_tendency_of_air_pressure: float = 0.0


@dataclass
class ISSRParams:
    """Gaussian ISSR parameters."""
    centroid: tuple = (2.0, 1.0, 12_000.0)  # (lon deg, lat deg, alt m)
    sigma_parallel: float = 150_000.0        # along-track half-width [m]
    sigma_perp: float = 50_000.0             # across-track half-width [m]
    sigma_z: float = 500.0                   # vertical half-width [m]
    rhi_bg: float = 0.75    # background RHi
    rhi_peak: float = 1.20  # peak RHi at centroid

# ---------------------------------------------------------------------------
# Main simulation class
# ---------------------------------------------------------------------------

class ContrailSimulator:
    """
    Contrail-mitigation simulation over a synthetic Gaussian ISSR.

    Instantiate with parameter dataclasses, then call run_all_strategies()
    or run_strategy() directly.
    """

    def __init__(self, sim_params=None, fl_params=None, met_params=None, issr_params=None):
        self.sim_params = sim_params or SimParams()
        self.fl_params = fl_params or FlParams()
        self.met_params = met_params or MetParams()
        self.issr_params = issr_params or ISSRParams()
        self.build_grid()

    def build_grid(self):
        sp = self.sim_params
        h, v, eps = sp.hres, sp.vres, 1e-9

        self.lons = np.arange(sp.lon_bounds[0] + h / 2, sp.lon_bounds[1] + eps, h)
        self.lats = np.arange(sp.lat_bounds[0] + h / 2, sp.lat_bounds[1] + eps, h)
        self.alts = np.arange(sp.alt_bounds[0] + v / 2, sp.alt_bounds[1] + eps, v)

        self.met_lons = np.arange(sp.lon_bounds[0] - h / 2, sp.lon_bounds[1] + h + eps, h)
        self.met_lats = np.arange(sp.lat_bounds[0] - h / 2, sp.lat_bounds[1] + h + eps, h)
        self.met_alts = np.arange(sp.alt_bounds[0] - v / 2, sp.alt_bounds[1] + v + eps, v)
        self.met_levels = units.m_to_pl(self.met_alts)

        ref_lon, ref_lat = sp.lon_bounds[0], sp.lat_bounds[0]
        self.met_lons_m, _ = lonlat_to_m(
            self.met_lons, np.full_like(self.met_lons, ref_lat), ref_lon, ref_lat
        )
        _, self.met_lats_m = lonlat_to_m(
            np.full_like(self.met_lats, ref_lon), self.met_lats, ref_lon, ref_lat
        )

        self.times_fl = pd.date_range(
            start=sp.t_start, end=sp.t_start + sp.duration_fl, freq=sp.dt_fl
        )
        self.times_met = pd.date_range(
            start=sp.t_start, end=sp.t_start + sp.duration_met, freq=sp.dt_met
        )

    def _rhi_field(self):
        """Gaussian RHi field. Returns shape (n_lon, n_lat, n_level)."""
        ip = self.issr_params
        ref_lon, ref_lat = self.sim_params.lon_bounds[0], self.sim_params.lat_bounds[0]
        cx, cy = lonlat_to_m(
            np.array([ip.centroid[0]]), np.array([ip.centroid[1]]), ref_lon, ref_lat
        )
        cx, cy, cz = float(cx[0]), float(cy[0]), ip.centroid[2]

        dx = self.met_lons_m[:, None, None] - cx
        dy = self.met_lats_m[None, :, None] - cy
        dz = self.met_alts[None, None, :] - cz

        M = np.exp(
            -(dx ** 2) / (2 * ip.sigma_parallel ** 2)
            - (dy ** 2) / (2 * ip.sigma_perp ** 2)
            - (dz ** 2) / (2 * ip.sigma_z ** 2)
        )
        return ip.rhi_bg + (ip.rhi_peak - ip.rhi_bg) * M

    def _q_field(self):
        """
        RHi -> specific humidity, broadcast over time.
        
        Returns shape (n_lon, n_lat, n_level, n_time).
        """
        T = self.met_params.air_temperature
        qs = q_sat_ice(T, self.met_levels * 100.0)
        rhi = self._rhi_field()
        q_3d = np.clip(rhi * qs[None, None, :], 1e-9, None)
        n_t = len(self.times_met)
        return np.broadcast_to(q_3d[..., None], q_3d.shape + (n_t,)).copy()

    def _calc_sdr(self, lons, lats, times):
        times_np = np.asarray(times, dtype="datetime64[ns]")

        lon_grid, lat_grid, time_grid = np.meshgrid(
            lons,
            lats,
            times_np,
            indexing="ij",
        )

        sdr = geo.solar_direct_radiation(
            lon_grid,
            lat_grid,
            time_grid,
        )

        albedo = 0.30
        olr = 240.0  # W m-2
        dt_s = self.sim_params.dt_met.total_seconds()

        sw = (1.0 - albedo) * sdr * dt_s
        lw = np.full_like(sw, -olr * dt_s)

        return sw, lw


    def gen_met(self):
        """
        Build synthetic meteorology and radiation datasets.

        Pressure-level met uses the Gaussian ISSR humidity field over a
        uniform background atmosphere from MetParams.  Radiation is set to
        a typical nighttime scenario (no solar, standard OLR).
        No ERA5 download or cache required.
        """
        mp = self.met_params

        lons   = self.met_lons.astype("float32")
        lats   = self.met_lats.astype("float32")
        levels = self.met_levels.astype("float32")   # hPa
        times  = self.times_met

        n_lon, n_lat, n_lev, n_t = len(lons), len(lats), len(levels), len(times)
        shape4 = (n_lon, n_lat, n_lev, n_t)
        ones   = np.ones(shape4, dtype="float32")
        dims4  = ["longitude", "latitude", "level", "time"]
        coords4 = dict(longitude=lons, latitude=lats, level=levels, time=times)

        ds = xr.Dataset(
            {
                "air_temperature": (
                    dims4, ones * np.float32(mp.air_temperature)),
                "specific_humidity": (
                    dims4, self._q_field().astype("float32")),
                "eastward_wind": (
                    dims4, ones * np.float32(mp.eastward_wind)),
                "northward_wind": (
                    dims4, ones * np.float32(mp.northward_wind)),
                "lagrangian_tendency_of_air_pressure": (
                    dims4, ones * np.float32(mp.lagrangian_tendency_of_air_pressure)),
                "tau_cirrus": (
                    dims4, np.zeros(shape4, dtype="float32")),
            },
            coords=coords4,
        )
        met = MetDataset(ds)

        sw, lw = self._calc_sdr(lons, lats, times)

        # Radiation (single-level) — nighttime: solar = 0, standard OLR
        shape3  = (n_lon, n_lat, n_t)
        dims3   = ["longitude", "latitude", "time"]
        coords3 = dict(longitude=lons, latitude=lats, time=times)

        ds_rad = xr.Dataset(
            {
                "top_net_solar_radiation": xr.DataArray(
                    sw.astype("float32"),
                    dims=dims3,
                    attrs={"units": "J m**-2"},
                ),
                "top_net_thermal_radiation": xr.DataArray(
                    lw.astype("float32"),
                    dims=dims3,
                    attrs={"units": "J m**-2"},
                ),
            },
            coords=coords3,
            attrs={"provider": "ECMWF", "dataset": "ERA5", "product": "reanalysis"},
        ).expand_dims({"level": [-1]})
        rad = MetDataset(ds_rad)

        return met, rad

    def gen_flight(self, alt=None, alt_target=None, direction=0, avoid_frac=None):
        """
        Generate a flight trajectory with optional ISSR avoidance.

        All strategies share the same baseline longitude/latitude path.
        For avoidance (direction != 0), the aircraft climbs or descends at
        *rocd* [m/s] starting at a diversion point set by *avoid_frac*,
        levels off at *alt_target*, then returns symmetrically after the ISSR.

        Parameters
        ----------
        alt : float, optional
            Baseline cruise altitude [m].
        alt_target : float, optional
            Target altitude during avoidance [m]. Ignored when direction=0.
        direction : int
            +1 = climb, -1 = descend, 0 = straight (no deviation).
        avoid_frac : float, optional
            Fraction of the Gaussian to avoid [0-1].
            0 = no diversion; 1 = avoid entire sigma region.
            Defaults to fl_params.avoid_frac.
        """
        sim_params, fl_params, issr_params = self.sim_params, self.fl_params, self.issr_params

        if alt is None:
            alt = fl_params.target_alt or 0.5 * (sim_params.alt_bounds[0] + sim_params.alt_bounds[1])
        if avoid_frac is None:
            avoid_frac = fl_params.avoid_frac

        n_wp = len(self.times_fl)
        lat_mid = 0.5 * (sim_params.lat_bounds[0] + sim_params.lat_bounds[1])
        lons = np.linspace(fl_params.fl_lon_bounds[0], fl_params.fl_lon_bounds[1], n_wp)
        alts = np.full(n_wp, float(alt))

        # Avoidance logic: climb or descend to alt_target, then return to baseline.
        if direction != 0 and alt_target is not None and avoid_frac > 0:
            cx_lon = issr_params.centroid[0]
            m_per_deg = 111_319.0 * np.cos(np.deg2rad(lat_mid))

            # Distance from centroid at which diversion begins.
            # avoid_frac=1 -> divert starting 1-sigma out; avoid_frac=0 -> no margin
            x_div_m = issr_params.sigma_parallel * avoid_frac
            x_div_deg = x_div_m / m_per_deg

            # Ground speed from waypoint spacing and time step
            dt_s = (self.times_fl[1] - self.times_fl[0]).total_seconds()
            dlon = (fl_params.fl_lon_bounds[1] - fl_params.fl_lon_bounds[0]) / (n_wp - 1)
            v_gnd = dlon * m_per_deg / dt_s  # m/s

            # Climb/descent distance needed to reach alt_target
            dx_trans_deg = abs(alt_target - alt) / fl_params.rocd * v_gnd / m_per_deg

            # Four transition longitudes
            lon_start  = cx_lon - x_div_deg - dx_trans_deg  # begin climb/descend
            lon_level  = cx_lon - x_div_deg                 # reached alt_target
            lon_resume = cx_lon + x_div_deg                 # begin return
            lon_end    = cx_lon + x_div_deg + dx_trans_deg  # back to baseline

            for i, lon in enumerate(lons):
                if lon <= lon_start:
                    alts[i] = alt
                elif lon < lon_level:
                    frac = (lon - lon_start) / (lon_level - lon_start)
                    alts[i] = alt + frac * (alt_target - alt)
                elif lon <= lon_resume:
                    alts[i] = alt_target
                elif lon < lon_end:
                    frac = (lon - lon_resume) / (lon_end - lon_resume)
                    alts[i] = alt_target + frac * (alt - alt_target)
                else:
                    alts[i] = alt

        df = pd.DataFrame({
            "longitude": lons,
            "latitude": np.full(n_wp, lat_mid),
            "altitude": alts,
            "time": self.times_fl,
            "mach_number": np.full(n_wp, fl_params.cruise_mach),
        })
        return Flight(
            data=df,
            attrs={"aircraft_type": fl_params.ac_type, "flight_id": "synthetic_fl_001"},
            fuel=JetA(),
        )

    def run_cocip(self, fl, met=None, rad=None, **cocip_kwargs):
        """Run CoCiP on a Flight object, returning the updated Flight and contrail DataFrame."""
        if met is None or rad is None:
            met, rad = self.gen_met()

        humidity_scaling = ExponentialBoostHumidityScaling()

        fl = PSFlight(met=met).eval(fl)
        fl = Emissions(met=met, humidity_scaling=humidity_scaling).eval(fl)

        # max_age must not push contrail time past the end of the met data.
        # The last flight waypoint is at t_start + duration_fl, so the
        # furthest any contrail can run is duration_met - duration_fl.
        max_age = self.sim_params.duration_met - self.sim_params.duration_fl

        cocip = Cocip(
            met=met, rad=rad,
            dt_integration=np.timedelta64(10, "m"),
            max_age=max_age,
            humidity_scaling=humidity_scaling,
            **cocip_kwargs,
        )
        fl_out = cocip.eval(fl)
        contrail = cocip.contrail
        return fl_out, contrail
    
    def run_flights(self, flight_cases: pd.DataFrame, **cocip_kwargs):
        """
        Run every row in flight_cases as one independent flight case.

        Expected columns:
            case
            alt
            alt_target
            avoidance
        """
        required = {"case", "alt", "alt_target", "avoidance"}
        missing = required - set(flight_cases.columns)

        if missing:
            raise ValueError(f"flight_cases is missing required columns: {missing}")
        
        allowed = {"none", "descend", "climb"}
        bad = set(flight_cases["avoidance"]) - allowed

        if bad:
            raise ValueError(f"Invalid avoidance values: {bad}. Use one of {allowed}.")

        met, rad = self.gen_met()
        results = {}

        direction_map = {
            "none": 0,
            "descend": -1,
            "climb": 1,
        }

        for _, row in flight_cases.iterrows():
            case_name = row["case"]

            direction = direction_map[row["avoidance"]]

            alt_target = row["alt_target"]
            if pd.isna(alt_target):
                alt_target = None

            fl = self.gen_flight(
                alt=float(row["alt"]),
                alt_target=alt_target,
                direction=direction,
            )

            

            fl_out, contrail = self.run_cocip(
                fl,
                met=met,
                rad=rad,
                **cocip_kwargs,
            )

            print(fl_out.dataframe.columns)

            results[case_name] = {
                "inputs": row.to_dict(),
                "flight": fl_out,
                "contrail": contrail,
            }

        return results

    def summarise_results(self, results: dict) -> pd.DataFrame:
        rows = []

        for case_name, result in results.items():
            inputs = result["inputs"]
            fl = result["flight"]
            con = result["contrail"]

            fl_df = fl.dataframe

            if con is None or con.empty:
                total_ef = 0.0
                mean_lw_rf = np.nan
                contrail_points = 0
            else:
                total_ef = float(con["ef"].sum()) if "ef" in con.columns else 0.0
                mean_lw_rf = float(con["rf_lw"].mean()) if "rf_lw" in con.columns else np.nan
                contrail_points = len(con)

            mean_fuel_flow = (
                float(fl_df["fuel_flow"].mean())
                if "fuel_flow" in fl_df.columns
                else np.nan
            )

            max_fuel_flow = float(fl_df["fuel_flow"].max()) if "fuel_flow" in fl_df.columns else np.nan
            min_altitude = float(fl_df["altitude"].min())
            max_altitude = float(fl_df["altitude"].max())

            rows.append({
                "case": case_name,
                "altitude_m": inputs["alt"],
                "target_altitude_m": inputs["alt_target"],
                "avoidance": inputs["avoidance"],
                "total_ef_J": total_ef,
                "mean_lw_rf_W_m2": mean_lw_rf,
                "contrail_points": contrail_points,
                "mean_fuel_flow_kg_s": mean_fuel_flow,
                "max_fuel_flow_kg_s": max_fuel_flow,
                "min_altitude_m": min_altitude,
                "max_altitude_m": max_altitude,
            })

        return pd.DataFrame(rows).set_index("case")

    def save_results(self, results, summary, output_dir="results"):
        import os

        os.makedirs(output_dir, exist_ok=True)

        summary.to_csv(f"{output_dir}/summary.csv")

        for case_name, result in results.items():
            flight_df = result["flight"].dataframe
            flight_df.to_csv(f"{output_dir}/flight_{case_name}.csv", index=False)

            contrail = result["contrail"]
            if contrail is not None and not contrail.empty:
                contrail.to_csv(f"{output_dir}/contrail_{case_name}.csv", index=False)

        # Save ISSR field for reproducibility
        rhi = self._rhi_field()
        np.save(f"{output_dir}/rhi_field.npy", rhi)

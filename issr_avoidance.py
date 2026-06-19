"""
issr_avoidance.py
=================
Synthetic contrail-mitigation simulation framework.

Compares altitude-based contrail avoidance (S1/S2) with SAF-blend strategies
(S3) against a no-avoidance baseline (S0), using the pycontrails CoCiP model
over a synthetic Gaussian ISSR.

Workflow
--------
    sim = ISSRAvoidance()              # default parameters
    sim.preprocess()                   # build met, generate flight, PS + emissions
    results = sim.run_strategy("S0")   # baseline CoCiP simulation
    results = sim.run_strategy("S1")   # altitude-avoidance simulation
    all_res = sim.run_all_strategies() # all four strategies in one call
    sim.plot_rhi()                     # visualise the ISSR humidity field

Avoidance strategies
--------------------
    S0  Baseline  – fly through the ISSR at cruise altitude.
    S1  Descend   – fly below the ISSR by ``fl_params.fl_alt_delta`` metres.
    S2  Climb     – fly above the ISSR by ``fl_params.fl_alt_delta`` metres.
    S3  Combined  – descend (as S1) with a 100 % SAF blend.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from typing import Literal
from pyproj import Transformer
from dataclasses import dataclass, field

from pycontrails import Flight, MetDataset
from pycontrails.core.models import Model
from pycontrails.core.fuel import SAFBlend
from pycontrails.models.emissions import Emissions
from pycontrails.models.ps_model import PSFlight
from pycontrails.physics import units
from pycontrails.models.cocip import Cocip


# ---------------------------------------------------------------------------
# Module-level utilities
# ---------------------------------------------------------------------------

def lonlat_to_m(
    lon: np.ndarray,
    lat: np.ndarray,
    ref_lon: float,
    ref_lat: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert geographic coordinates to local Cartesian metres.

    Uses a Transverse Mercator projection centred on (ref_lon, ref_lat).

    Parameters
    ----------
    lon, lat : array-like
        Geographic coordinates [degrees].
    ref_lon, ref_lat : float
        Projection origin [degrees].

    Returns
    -------
    x_m, y_m : np.ndarray
        Easting and northing [m].
    """
    transformer = Transformer.from_crs(
        "epsg:4326",
        (
            f"+proj=tmerc +lat_0={ref_lat} +lon_0={ref_lon}"
            " +k=1 +x_0=0 +y_0=0"
        ),
        always_xy=True,
    )
    x_m, y_m = transformer.transform(np.asarray(lon, dtype=float),
                                     np.asarray(lat, dtype=float))
    return x_m, y_m


def q_sat_ice(T: np.ndarray, p_Pa: np.ndarray) -> np.ndarray:
    """
    Saturation specific humidity over ice [kg kg⁻¹].

    Uses the Magnus formula (Alduchov & Eskridge 1996).

    Parameters
    ----------
    T : array-like
        Temperature [K].
    p_Pa : array-like
        Pressure [Pa].

    Returns
    -------
    np.ndarray
        Saturation specific humidity [kg kg⁻¹].
    """
    eps = 0.6220  # ratio of molar masses H₂O / dry air
    e_sat = 611.2 * np.exp(22.46 * (T - 273.16) / (T - 0.55))  # [Pa]
    return eps * e_sat / (p_Pa - e_sat)

# ---------------------------------------------------------------------------
# Parameter dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SimParams:
    """Simulation grid and temporal parameters."""

    # -- Temporal domain --
    t_fl: tuple = field(
        default_factory=lambda: (
            pd.to_datetime("2025-01-20 13:00:00"),
            pd.Timedelta(minutes=2),
            pd.Timedelta(hours=1),
        )
    )
    """(start, dt, duration) for flight waypoint time-stamps."""

    t_con: tuple = field(
        default_factory=lambda: (
            pd.to_datetime("2025-01-20 13:00:00"),
            pd.Timedelta(hours=1),
            pd.Timedelta(hours=12),
        )
    )
    """(start, dt, duration) for the contrail / met time axis."""

    # -- Spatial domain --
    lon_bounds: tuple[float, float] = (0.0, 8.0)        # [°] — wide enough for ~10 m/s wind × 12 h advection (~3.9°) plus flight path
    lat_bounds: tuple[float, float] = (0.0, 2.0)        # [°]
    alt_bounds: tuple[float, float] = (10000.0, 14000.0)  # [m] — wide enough to cover avoidance strategies (±1500 m from 12 km ISSR)
    hres_sim: float = 0.25    # horizontal resolution [°]
    vres_sim: float = 500.0   # vertical resolution [m]


@dataclass
class FlParams:
    """Flight and fuel parameters."""

    ac_type: str = "A320"
    pct_blend: float = 0.0          # SAF blend percentage [0–100]
    fl_speed: float = 230.0         # cruise true airspeed [m s⁻¹] (informational)
    fl_target_alt: float | None = None
    """Cruise altitude [m]. ``None`` → midpoint of ``alt_bounds``."""
    fl_alt_delta: float = 1500.0    # altitude shift for avoidance strategies [m]
    fl_lon_bounds: tuple[float, float] = (0.0, 4.0)
    """Flight longitude range [°]. Narrower than the met domain so advected
    contrails stay inside the wider met domain throughout the simulation."""


@dataclass
class ContrailParams:
    """Initial contrail plume parameters (passed to CoCiP where applicable)."""

    depth: float = 50.0    # initial plume depth [m]
    width: float = 50.0    # initial plume width [m]
    verbose_outputs: bool = False


@dataclass
class MetParams:
    """Uniform background atmospheric state."""

    air_temperature: float = 220.0                    # [K]
    eastward_wind: float = 0.0                       # [m s⁻¹]
    northward_wind: float = 0.0                       # [m s⁻¹]
    lagrangian_tendency_of_air_pressure: float = 0.0  # [Pa s⁻¹]


@dataclass
class ISSRParams:
    """
    Gaussian ISSR representation.

    The ISSR is modelled as a smooth perturbation of specific humidity:

        q = q_bg + Δq · M(lon, lat, alt)

    where M is a trivariate Gaussian centred on ``issr_centroid`` with
    independent half-widths ``sigma_parallel`` (along-track / longitude),
    ``sigma_perp`` (across-track / latitude), and ``sigma_z`` (vertical).
    Background RHi = ``rhi_bg``; peak RHi at centroid = ``rhi_peak``.
    """

    issr_centroid: tuple[float, float, float] = (2.0, 1.0, 12000.0)
    """(lon [°], lat [°], alt [m]) of the ISSR peak. Placed in western half so the contrail advects through it."""

    sigma_parallel: float = 150_000.0   # along-track (longitude) half-width [m]
    sigma_perp: float = 50_000.0        # across-track (latitude) half-width [m]
    sigma_z: float = 500.0              # vertical half-width [m]

    rhi_bg: float = 0.75    # background RHi outside the ISSR [–]
    rhi_peak: float = 1.20  # peak RHi at ISSR centroid [–]


class ISSRAvoidance(Model):
    """
    Synthetic contrail-mitigation simulation framework.

    Generates a straight east-west flight through (or around) a synthetic
    Gaussian ISSR and evaluates contrail climate impact using CoCiP across
    four avoidance strategies.

    Parameters
    ----------
    sim_params : SimParams, optional
    fl_params : FlParams, optional
    contrail_params : ContrailParams, optional
    met_params : MetParams, optional
    issr_params : ISSRParams, optional

    Examples
    --------
    >>> sim = ISSRAvoidance()
    >>> sim.preprocess()
    >>> results = sim.run_all_strategies()
    >>> df = sim.compare_strategies(results)
    """

    name = "ISSRAvoidance"
    long_name = "ISSR Contrail Avoidance Simulation Framework"

    def __init__(
        self,
        sim_params: SimParams | None = None,
        fl_params: FlParams | None = None,
        contrail_params: ContrailParams | None = None,
        met_params: MetParams | None = None,
        issr_params: ISSRParams | None = None,
    ) -> None:
        super().__init__()

        self.sim_params = sim_params or SimParams()
        self.fl_params = fl_params or FlParams()
        self.contrail_params = contrail_params or ContrailParams()
        self.met_params = met_params or MetParams()
        self.issr_params = issr_params or ISSRParams()

        self._build_grid(self.sim_params)

        sp = self.sim_params
        self.times_fl = pd.date_range(
            start=sp.t_fl[0],
            end=sp.t_fl[0] + sp.t_fl[2],
            freq=sp.t_fl[1],
        )
        self.times_con = pd.date_range(
            start=sp.t_con[0],
            end=sp.t_con[0] + sp.t_con[2],
            freq=sp.t_con[1],
        )

        # Populated by preprocess()
        self.fl: Flight | None = None
        self.met: MetDataset | None = None
        self.rad: MetDataset | None = None

    # ------------------------------------------------------------------
    # Grid construction
    # ------------------------------------------------------------------

    def _build_grid(self, sp: SimParams) -> None:
        """Construct domain and padded met grid arrays from *sp*."""
        h = sp.hres_sim
        v = sp.vres_sim
        eps = 1e-9  # ensures np.arange includes the upper endpoint

        # -- Core domain grid (cell centres) --
        self.lons = np.arange(sp.lon_bounds[0] + h / 2, sp.lon_bounds[1] + eps, h)
        self.lats = np.arange(sp.lat_bounds[0] + h / 2, sp.lat_bounds[1] + eps, h)
        self.alts = np.arange(sp.alt_bounds[0] + v / 2, sp.alt_bounds[1] + eps, v)
        self.levels = units.m_to_pl(self.alts)

        # -- Padded met grid: one extra cell on each side for interpolation --
        self.met_lons = np.arange(sp.lon_bounds[0] - h / 2,
                                  sp.lon_bounds[1] + h + eps, h)
        self.met_lats = np.arange(sp.lat_bounds[0] - h / 2,
                                  sp.lat_bounds[1] + h + eps, h)
        self.met_alts = np.arange(sp.alt_bounds[0] - v / 2,
                                  sp.alt_bounds[1] + v + eps, v)
        self.met_levels = units.m_to_pl(self.met_alts)  # hPa, descending order

        # -- Local Cartesian axes [m] for Gaussian ISSR computation --
        ref_lon = float(sp.lon_bounds[0])
        ref_lat = float(sp.lat_bounds[0])
        self.met_lons_m, _ = lonlat_to_m(
            self.met_lons, np.full_like(self.met_lons, ref_lat), ref_lon, ref_lat
        )
        _, self.met_lats_m = lonlat_to_m(
            np.full_like(self.met_lats, ref_lon), self.met_lats, ref_lon, ref_lat
        )

    # ------------------------------------------------------------------
    # ISSR field generation
    # ------------------------------------------------------------------

    def _gen_issr_field(self) -> np.ndarray:
        """
        Build a 4-D specific-humidity array embedding a Gaussian ISSR.

        Background RHi = ``issr_params.rhi_bg``; peak RHi at the centroid
        = ``issr_params.rhi_peak``. The field is uniform in time.

        Returns
        -------
        np.ndarray
            Shape ``(n_lon, n_lat, n_level, n_time)`` [kg kg⁻¹].
        """
        issr = self.issr_params
        T = self.met_params.air_temperature

        # Saturation specific humidity at each pressure level [kg kg⁻¹]
        p_Pa = self.met_levels * 100.0          # hPa → Pa, shape (n_level,)
        qs = q_sat_ice(T, p_Pa)                  # shape (n_level,)

        # ISSR centroid in local Cartesian metres
        ref_lon = float(self.sim_params.lon_bounds[0])
        ref_lat = float(self.sim_params.lat_bounds[0])
        cx, cy = lonlat_to_m(
            np.array([issr.issr_centroid[0]]),
            np.array([issr.issr_centroid[1]]),
            ref_lon, ref_lat,
        )
        cx, cy = float(cx[0]), float(cy[0])
        cz = issr.issr_centroid[2]              # altitude [m]

        # Displacement arrays with broadcasting shapes:
        # (n_lon, 1, 1), (1, n_lat, 1), (1, 1, n_level)
        dx = self.met_lons_m[:, None, None] - cx
        dy = self.met_lats_m[None, :, None] - cy
        dz = self.met_alts[None, None, :] - cz

        M = np.exp(
            -(dx ** 2) / (2.0 * issr.sigma_parallel ** 2)
            - (dy ** 2) / (2.0 * issr.sigma_perp ** 2)
            - (dz ** 2) / (2.0 * issr.sigma_z ** 2)
        )  # shape (n_lon, n_lat, n_level)

        # q = q_background + Δq · Gaussian mask
        q_bg = issr.rhi_bg * qs                  # (n_level,)
        dq = (issr.rhi_peak - issr.rhi_bg) * qs  # (n_level,)
        q_3d = q_bg[None, None, :] + dq[None, None, :] * M
        q_3d = np.clip(q_3d, 1e-9, None)

        # Broadcast over time → (n_lon, n_lat, n_level, n_time)
        n_time = len(self.times_con)
        return np.broadcast_to(q_3d[..., None], q_3d.shape + (n_time,)).copy()

    # ------------------------------------------------------------------
    # Trajectory generation
    # ------------------------------------------------------------------

    def traj_gen(self, alt: float | None = None) -> Flight:
        """
        Generate a straight east-west flight trajectory.

        Waypoints are evenly spaced in longitude from ``lon_bounds[0]`` to
        ``lon_bounds[1]`` at the latitude midpoint of the domain.

        Parameters
        ----------
        alt : float, optional
            Cruise altitude [m]. Defaults to ``fl_params.fl_target_alt``
            or the midpoint of ``sim_params.alt_bounds``.

        Returns
        -------
        Flight
        """
        sp = self.sim_params
        fp = self.fl_params

        if alt is None:
            alt = fp.fl_target_alt
        if alt is None:
            alt = 0.5 * (sp.alt_bounds[0] + sp.alt_bounds[1])

        lat_mid = 0.5 * (sp.lat_bounds[0] + sp.lat_bounds[1])
        n_wp = len(self.times_fl)

        df = pd.DataFrame(
            {
                "longitude": np.linspace(fp.fl_lon_bounds[0], fp.fl_lon_bounds[1], n_wp),
                "latitude": np.full(n_wp, lat_mid),
                "altitude": np.full(n_wp, float(alt)),
                "time": self.times_fl,
            }
        )

        return Flight(
            data=df,
            attrs={"aircraft_type": fp.ac_type, "flight_id": "synthetic_fl_001"},
            fuel=SAFBlend(fp.pct_blend),
        )

    # ------------------------------------------------------------------
    # Meteorological data generation
    # ------------------------------------------------------------------

    def gen_met(self) -> tuple[MetDataset, MetDataset]:
        """
        Generate synthetic pressure-level and radiation MetDatasets.

        The ISSR is embedded as a Gaussian ``specific_humidity`` perturbation
        over a horizontally and temporally uniform background atmosphere.

        Radiation uses a nighttime proxy (solar = 0 W m⁻²; representative OLR
        values). For accurate RF calculations substitute real ERA5 radiation.

        Returns
        -------
        met : MetDataset
            Pressure-level variables required by CoCiP.
        rad : MetDataset
            Single-level radiation variables required by CoCiP.
        """
        mp = self.met_params
        n_lo = len(self.met_lons)
        n_la = len(self.met_lats)
        n_lv = len(self.met_levels)
        n_t = len(self.times_con)
        shape4 = (n_lo, n_la, n_lv, n_t)
        shape3 = (n_lo, n_la, n_t)
        times = self.times_con.values

        q_4d = self._gen_issr_field()

        ds = xr.Dataset(
            {
                "air_temperature": (
                    ["longitude", "latitude", "level", "time"],
                    np.full(shape4, mp.air_temperature),
                ),
                "specific_humidity": (
                    ["longitude", "latitude", "level", "time"],
                    q_4d,
                ),
                "eastward_wind": (
                    ["longitude", "latitude", "level", "time"],
                    np.full(shape4, mp.eastward_wind),
                ),
                "northward_wind": (
                    ["longitude", "latitude", "level", "time"],
                    np.full(shape4, mp.northward_wind),
                ),
                "lagrangian_tendency_of_air_pressure": (
                    ["longitude", "latitude", "level", "time"],
                    np.full(shape4, mp.lagrangian_tendency_of_air_pressure),
                ),
                # Clear-sky synthetic atmosphere: no cirrus cloud
                "tau_cirrus": (
                    ["longitude", "latitude", "level", "time"],
                    np.zeros(shape4),
                ),
            },
            coords={
                "longitude": self.met_lons,
                "latitude": self.met_lats,
                "level": self.met_levels,   # hPa, descending order
                "time": times,
            },
        )
        met = MetDataset(ds)

        # Nighttime radiation proxy: solar = 0, representative OLR values.
        # Units are set to "W m**-2" so CoCiP skips the J/m² accumulation
        # conversion and uses the values directly as instantaneous fluxes.
        # The dataset-level attrs mimic ERA5 reanalysis metadata so that
        # pycontrails uses the correct variable names (top_net_*_radiation).
        rad_vars = {
            "top_net_thermal_radiation": -200.0,   # W m⁻² (negative = upward)
            "top_net_solar_radiation": 0.0,        # nighttime → no solar
            "surface_net_thermal_radiation": -50.0,
            "surface_net_solar_radiation": 0.0,
        }
        ds_rad = xr.Dataset(
            {
                name: xr.DataArray(
                    np.full(shape3, value, dtype=np.float32),
                    dims=["longitude", "latitude", "time"],
                    attrs={"units": "W m**-2"},
                )
                for name, value in rad_vars.items()
            },
            coords={
                "longitude": self.met_lons,
                "latitude": self.met_lats,
                "time": times,
            },
            attrs={
                "provider": "ECMWF",
                "dataset": "ERA5",
                "product": "reanalysis",
            },
        )
        rad = MetDataset(ds_rad.expand_dims({"level": [-1]}))

        return met, rad

    # ------------------------------------------------------------------
    # Aircraft performance and emissions
    # ------------------------------------------------------------------

    def ac_perf(self, fl: Flight, met: MetDataset | None = None) -> Flight:
        """
        Compute aircraft performance with the Poll-Schumann (PS) model.

        Parameters
        ----------
        fl : Flight
        met : MetDataset, optional
            Without met the PS model falls back to the International Standard
            Atmosphere.

        Returns
        -------
        Flight
            With additional columns: ``fuel_flow``, ``thrust``,
            ``engine_efficiency``, ``true_airspeed``, etc.
        """
        return PSFlight(met=met).eval(fl)

    def emissions(self, fl: Flight, met: MetDataset | None = None) -> Flight:
        """
        Estimate aircraft emissions with the Pycontrails Emissions model.

        Parameters
        ----------
        fl : Flight
            Must contain PS model output columns.
        met : MetDataset, optional

        Returns
        -------
        Flight
            With additional columns: ``nvpm_ei_n``, ``nox_ei``, etc.
        """
        return Emissions(met=met).eval(fl)

    # ------------------------------------------------------------------
    # SAF assignment
    # ------------------------------------------------------------------

    def assign_saf(self, fl: Flight, pct_blend: float) -> Flight:
        """
        Return a copy of *fl* with a new SAF blend percentage applied.

        Parameters
        ----------
        fl : Flight
        pct_blend : float
            SAF blend ratio [%, 0–100].

        Returns
        -------
        Flight
        """
        fl_saf = fl.copy()
        fl_saf.fuel = SAFBlend(pct_blend)
        return fl_saf

    # ------------------------------------------------------------------
    # CoCiP
    # ------------------------------------------------------------------

    def run_cocip(
        self,
        fl: Flight,
        met: MetDataset,
        rad: MetDataset,
        **cocip_kwargs,
    ) -> tuple[Flight, pd.DataFrame | None]:
        """
        Run the CoCiP contrail lifecycle model.

        Parameters
        ----------
        fl : Flight
            Must contain aircraft-performance and emissions columns.
        met : MetDataset
            Pressure-level met (from :meth:`gen_met`).
        rad : MetDataset
            Radiation met (from :meth:`gen_met`).
        **cocip_kwargs
            Additional keyword arguments forwarded to :class:`Cocip`.

        Returns
        -------
        fl_out : Flight
            Flight with CoCiP persistent-contrail flags.
        contrail : pd.DataFrame or None
            Lagrangian contrail segments; ``None`` if no persistent contrails
            formed.
        """
        cocip = Cocip(
            met=met,
            rad=rad,
            dt_integration=np.timedelta64(1, "h"),  # match 1-hour met time step
            max_age=self.sim_params.t_con[2],        # stop tracking at end of met domain
            **cocip_kwargs,
        )
        fl_out = cocip.eval(fl)
        return fl_out, cocip.contrail

    # ------------------------------------------------------------------
    # Strategy runner
    # ------------------------------------------------------------------

    def run_strategy(
        self,
        strategy: Literal["S0", "S1", "S2", "S3"],
        met: MetDataset | None = None,
        rad: MetDataset | None = None,
        **cocip_kwargs,
    ) -> dict:
        """
        Run a complete simulation for one named avoidance strategy.

        ====  ============================================================
        S0    Baseline  – fly through ISSR at cruise altitude.
        S1    Descend   – fly ``fl_alt_delta`` m below the ISSR centroid.
        S2    Climb     – fly ``fl_alt_delta`` m above the ISSR centroid.
        S3    Combined  – descend (as S1) with a 100 % SAF blend.
        ====  ============================================================

        Parameters
        ----------
        strategy : {"S0", "S1", "S2", "S3"}
        met, rad : MetDataset, optional
            Pre-computed datasets; generated fresh if not supplied.
        **cocip_kwargs
            Forwarded to :class:`Cocip`.

        Returns
        -------
        dict
            Keys: ``strategy``, ``altitude``, ``pct_blend``, ``ef``,
            ``rf_lw_mean``, ``fuel_burn_mean``, ``fl``, ``contrail``.
        """
        fp = self.fl_params
        issr = self.issr_params

        alt_cruise = fp.fl_target_alt or 0.5 * (
            self.sim_params.alt_bounds[0] + self.sim_params.alt_bounds[1]
        )

        strategy_map: dict[str, tuple[float, float]] = {
            "S0": (alt_cruise,
                   fp.pct_blend),
            "S1": (issr.issr_centroid[2] - fp.fl_alt_delta,
                   fp.pct_blend),
            "S2": (issr.issr_centroid[2] + fp.fl_alt_delta,
                   fp.pct_blend),
            "S3": (issr.issr_centroid[2] - fp.fl_alt_delta,
                   100.0),
        }
        if strategy not in strategy_map:
            raise ValueError(
                f"Unknown strategy {strategy!r}. Choose from {list(strategy_map)}."
            )
        alt, pct_blend = strategy_map[strategy]

        if met is None or rad is None:
            met, rad = self.gen_met()

        fl = self.traj_gen(alt=alt)
        if pct_blend != fp.pct_blend:
            fl = self.assign_saf(fl, pct_blend)

        fl = self.ac_perf(fl, met)
        fl = self.emissions(fl, met)
        fl_out, contrail = self.run_cocip(fl, met, rad, **cocip_kwargs)

        ef = (
            float(contrail["ef"].sum())
            if contrail is not None and "ef" in contrail.columns
            else 0.0
        )
        rf_lw = (
            float(contrail["rf_lw"].mean())
            if contrail is not None and "rf_lw" in contrail.columns
            else 0.0
        )
        fuel_burn = (
            float(fl_out.dataframe["fuel_flow"].mean())
            if "fuel_flow" in fl_out.dataframe.columns
            else None
        )

        return {
            "strategy": strategy,
            "altitude": alt,
            "pct_blend": pct_blend,
            "ef": ef,
            "rf_lw_mean": rf_lw,
            "fuel_burn_mean": fuel_burn,
            "fl": fl_out,
            "contrail": contrail,
        }

    def run_all_strategies(self, **cocip_kwargs) -> dict[str, dict]:
        """
        Run all four strategies in sequence, sharing one met/rad dataset.

        Parameters
        ----------
        **cocip_kwargs
            Forwarded to :class:`Cocip` for every strategy.

        Returns
        -------
        dict mapping strategy name → results dict (see :meth:`run_strategy`).
        """
        met, rad = self.gen_met()
        results: dict[str, dict] = {}
        for s in ("S0", "S1", "S2", "S3"):
            print(f"Running strategy {s} …")
            results[s] = self.run_strategy(s, met=met, rad=rad, **cocip_kwargs)
        return results

    def compare_strategies(
        self,
        results: dict[str, dict] | None = None,
        **cocip_kwargs,
    ) -> pd.DataFrame:
        """
        Tabulate a summary DataFrame for all four strategies.

        Parameters
        ----------
        results : dict, optional
            Output of :meth:`run_all_strategies`. Computed fresh if not given.
        **cocip_kwargs
            Forwarded to :meth:`run_all_strategies` if *results* is ``None``.

        Returns
        -------
        pd.DataFrame
            Indexed by strategy name.
        """
        if results is None:
            results = self.run_all_strategies(**cocip_kwargs)

        rows = [
            {
                "Strategy": s,
                "Altitude [m]": r["altitude"],
                "SAF blend [%]": r["pct_blend"],
                "Total EF [J]": r["ef"],
                "Mean LW RF [W m⁻²]": r["rf_lw_mean"],
                "Mean fuel flow [kg s⁻¹]": r["fuel_burn_mean"],
            }
            for s, r in results.items()
        ]
        return pd.DataFrame(rows).set_index("Strategy")

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def preprocess(self) -> None:
        """
        Run the full preprocessing pipeline and cache outputs as attributes.

        Steps
        -----
        1. Generate the synthetic met and radiation datasets.
        2. Generate the baseline (S0) flight trajectory.
        3. Compute aircraft performance using the PS model.
        4. Estimate emissions.

        Outputs stored as ``self.fl``, ``self.met``, ``self.rad``.
        """
        print("Generating synthetic met/rad …")
        self.met, self.rad = self.gen_met()

        print("Generating flight trajectory …")
        fl = self.traj_gen()

        print("Running PS aircraft-performance model …")
        fl = self.ac_perf(fl, self.met)

        print("Estimating emissions …")
        fl = self.emissions(fl, self.met)

        self.fl = fl
        print(
            "Preprocessing complete.\n"
            f"  Waypoints  : {len(fl)}\n"
            f"  Altitude   : {fl.dataframe['altitude'].iloc[0]:.0f} m\n"
            f"  Met domain : "
            f"lon=[{self.met_lons[0]:.2f}, {self.met_lons[-1]:.2f}] "
            f"lat=[{self.met_lats[0]:.2f}, {self.met_lats[-1]:.2f}] "
            f"lev=[{self.met_levels[-1]:.0f}, {self.met_levels[0]:.0f}] hPa"
        )

    # ------------------------------------------------------------------
    # Required Model.eval() implementation
    # ------------------------------------------------------------------

    def eval(self, source: Flight | None = None, **params) -> dict[str, dict]:
        """
        Evaluate all four avoidance strategies.

        Satisfies the pycontrails :class:`Model` abstract interface.
        Equivalent to calling :meth:`run_all_strategies`.

        Parameters
        ----------
        source : Flight, optional
            Ignored; trajectories are generated internally.
        **params
            Forwarded to :meth:`run_all_strategies`.

        Returns
        -------
        dict mapping strategy name → results dict.
        """
        return self.run_all_strategies(**params)

    # ------------------------------------------------------------------
    # Visualisation helpers
    # ------------------------------------------------------------------

    def plot_rhi(
        self,
        alt: float | None = None,
        time_idx: int = 0,
        ax: plt.Axes | None = None,
    ) -> plt.Axes:
        """
        Plot a horizontal RHi map at a given altitude and time index.

        The ISSR boundary (RHi = 1) is drawn as a dashed black contour.

        Parameters
        ----------
        alt : float, optional
            Altitude slice [m]. Defaults to the ISSR centroid altitude.
        time_idx : int
            Index into ``self.times_con``.
        ax : matplotlib.axes.Axes, optional

        Returns
        -------
        matplotlib.axes.Axes
        """
        if alt is None:
            alt = self.issr_params.issr_centroid[2]

        T = self.met_params.air_temperature
        alt_idx = int(np.argmin(np.abs(self.met_alts - alt)))
        p_Pa = self.met_levels[alt_idx] * 100.0
        qs = q_sat_ice(T, p_Pa)

        q_4d = self._gen_issr_field()
        rhi = q_4d[:, :, alt_idx, time_idx] / qs   # (n_lon, n_lat)

        if ax is None:
            _, ax = plt.subplots(figsize=(9, 4))

        cf = ax.contourf(
            self.met_lons, self.met_lats, rhi.T,
            levels=np.linspace(0.5, 1.5, 21),
            cmap="RdBu_r", extend="both",
        )
        ax.contour(
            self.met_lons, self.met_lats, rhi.T,
            levels=[1.0], colors="k", linewidths=1.5, linestyles="--",
        )
        plt.colorbar(cf, ax=ax, label="RHi [–]")
        ax.set_xlabel("Longitude [°]")
        ax.set_ylabel("Latitude [°]")
        ax.set_title(
            f"RHi at {alt / 1e3:.1f} km ({self.met_levels[alt_idx]:.0f} hPa)"
            f"  |  t = {self.times_con[time_idx]}"
        )
        return ax

    def plot_rhi_vertical(
        self,
        lon: float | None = None,
        time_idx: int = 0,
        ax: plt.Axes | None = None,
    ) -> plt.Axes:
        """
        Plot an altitude profile of RHi at the ISSR centroid longitude.

        Parameters
        ----------
        lon : float, optional
            Longitude [°]. Defaults to the ISSR centroid longitude.
        time_idx : int
            Index into ``self.times_con``.
        ax : matplotlib.axes.Axes, optional

        Returns
        -------
        matplotlib.axes.Axes
        """
        if lon is None:
            lon = self.issr_params.issr_centroid[0]

        T = self.met_params.air_temperature
        p_Pa = self.met_levels * 100.0
        qs = q_sat_ice(T, p_Pa)

        lon_idx = int(np.argmin(np.abs(self.met_lons - lon)))
        lat_mid = len(self.met_lats) // 2

        q_4d = self._gen_issr_field()
        rhi = q_4d[lon_idx, lat_mid, :, time_idx] / qs   # (n_level,)

        if ax is None:
            _, ax = plt.subplots(figsize=(5, 6))

        ax.plot(rhi, self.met_alts / 1e3, "b-o", ms=4)
        ax.axvline(1.0, color="k", linestyle="--", linewidth=1.5, label="RHi = 1")
        ax.set_xlabel("RHi [–]")
        ax.set_ylabel("Altitude [km]")
        ax.set_title(f"RHi vertical profile  (lon = {lon:.2f}°)")
        ax.legend()
        return ax

    def plot_strategy_comparison(
        self,
        results: dict[str, dict] | None = None,
        **cocip_kwargs,
    ) -> plt.Figure:
        """
        Bar chart comparing total EF and mean fuel flow across strategies.

        Parameters
        ----------
        results : dict, optional
            Output of :meth:`run_all_strategies`. Computed if not given.
        **cocip_kwargs
            Forwarded to :meth:`run_all_strategies` if *results* is ``None``.

        Returns
        -------
        matplotlib.figure.Figure
        """
        if results is None:
            results = self.run_all_strategies(**cocip_kwargs)

        strategies = list(results.keys())
        efs = [results[s]["ef"] for s in strategies]
        fuels = [
            results[s]["fuel_burn_mean"] or 0.0
            for s in strategies
        ]

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        colours = ["C0", "C1", "C2", "C3"]
        axes[0].bar(strategies, efs, color=colours)
        axes[0].set_xlabel("Strategy")
        axes[0].set_ylabel("Total EF [J]")
        axes[0].set_title("Contrail energy forcing")

        axes[1].bar(strategies, fuels, color=colours)
        axes[1].set_xlabel("Strategy")
        axes[1].set_ylabel("Mean fuel flow [kg s⁻¹]")
        axes[1].set_title("Fuel burn")

        fig.suptitle("Avoidance strategy comparison", fontweight="bold")
        fig.tight_layout()
        return fig


# ---------------------------------------------------------------------------
# Quick-start entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Customise any parameters here before running
    sim_params = SimParams(
        lon_bounds=(0.0, 8.0),        # extra room for eastward wind advection
        lat_bounds=(0.0, 2.0),
        alt_bounds=(10000.0, 14000.0),  # covers ±1500 m avoidance from 12 km ISSR
        hres_sim=0.25,
        vres_sim=500.0,
    )
    issr_params = ISSRParams(
        issr_centroid=(2.0, 1.0, 12000.0),
        sigma_parallel=150_000.0,
        sigma_perp=50_000.0,
        sigma_z=3000.0,
        rhi_bg=0.75,
        rhi_peak=1.20,
    )

    sim = ISSRAvoidance(sim_params=sim_params, issr_params=issr_params)

    # --- Visualise the synthetic ISSR before running CoCiP ---
    print("Plotting ISSR field …")
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    sim.plot_rhi(ax=axes[0])
    sim.plot_rhi_vertical(ax=axes[1])
    fig.tight_layout()
    plt.savefig("issr_field.png", dpi=150)
    print("Saved issr_field.png")

    # --- Run full preprocessing (PS model + emissions) ---
    sim.preprocess()

    # --- Run all four avoidance strategies ---
    results = sim.run_all_strategies()

    # --- Print comparison table ---
    df = sim.compare_strategies(results)
    print("\nStrategy comparison:")
    print(df.to_string())

    # --- Save comparison figure ---
    fig = sim.plot_strategy_comparison(results)
    plt.savefig("strategy_comparison.png", dpi=150)
    print("Saved strategy_comparison.png")

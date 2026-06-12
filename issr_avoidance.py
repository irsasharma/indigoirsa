import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Literal, Optional
from dataclasses import asdict, dataclass, field, fields, is_dataclass

from pycontrails import Flight, MetDataset, models
from pycontrails.core.models import Model
from pycontrails.core.fuel import Fuel, SAFBlend
from pycontrails.models.emissions import Emissions
from pycontrails.models.ps_model import PSFlight
from pycontrails.physics import constants, geo, thermo, units

from pycontrails.models.cocip import Cocip

@dataclass
class SimParams:
    """Default simulation parameters."""
    # Temporal domain
    
    # flight time - sampled waypoints at e.g. 2 min intervals for 1 hr
    t_fl: tuple[pd.Timestamp, pd.Timedelta, pd.Timedelta] = field(
        default_factory=lambda: (
            pd.to_datetime("2025-01-20 13:00:00"),
            pd.Timedelta(minutes=2),
            pd.Timedelta(hours=1),
        )
    )  # (start time, time step, run time)

    # contrail time - the time vector for the contrail simulation, representing the sampling 
    t_con: tuple[pd.Timestamp, pd.Timedelta, pd.Timedelta] = field(
        default_factory=lambda: (
            pd.to_datetime("2025-01-20 13:00:00"),
            pd.Timedelta(minutes=2),
            pd.Timedelta(hours=24),
        )
    )  # (start time, time step, run time)

    #  spatial domain
    ref_lat: float = 0.0  # reference latitude for local Cartesian grid [deg]
    lon_bounds: tuple[float, float] = (0.0, 1.0)  # lon bounds [deg]
    alt_bounds: tuple[float, float] = (12000, 13000)  # alt bounds [m]
    hres_sim: float = 0.01  # horizontal resolution [deg]
    vres_sim: float = 500  # vertical resolution [m]

@dataclass
class FlParams:
    """Default flight/fleet parameters."""
    # Synthetic focal-point trajectory controls
    ac_type: str = "A320" # aircraft type for performance and emissions calculations
    pct_blend: float = 0.0 # percentage of SAF blend in fuel (0-100)
    
    fl_speed: float | list[float] = 230.0           # m/s
    fl_target_alt: float | list[float] | None = None  # m; None → midpoint of alt_bounds
    fl_div_frac: float | list[float] = 0.5            # m; distance between parallel flight paths in synthetic formation
    fl_rocd: float | list[float] = 0.0             # m/s; positive = climb, negative = descent

@dataclass
class ContrailParams:
    """Default plume dispersion parameters."""
    depth: float = 50.0  # initial plume depth, [m]
    width: float = 50.0  # initial plume width, [m]
    verbose_outputs: bool = False  # print verbose outputs

@dataclass
class MetParams:
    """Default meteorological parameters."""
    eastward_wind: float | None = 0.0  # m/s
    northward_wind: float | None = 0.0  # m/s
    lagrangian_tendency_of_air_pressure: float | None = 0.0  # Pa/s
    air_temperature: float | None = 220.0  # K

@dataclass
class ISSRParams:
    """Default parameters for the ISSR model representation."""
    issr_centroid: tuple[float, float] = (0.5, 12500)  # (lat, alt) of ISSR centroid [deg, m]

class ISSRAvoidance(Model):

    def __init__(self,
                sim_params: SimParams,
                fl_params: FlParams,
                contrail_params: ContrailParams,
                met_params: MetParams,
                issr_params: ISSRParams):
        super().__init__()

        # Build spatial grid from current bounds
        self._build_grid(sim_params)

        # Generate time vectors
        self.times_fl = pd.date_range(
            start=sim_params.t_fl[0],
            end=sim_params.t_fl[0] + sim_params.t_fl[2],
            freq=sim_params.t_fl[1],
        )

        self.times_con = pd.date_range(
            start=sim_params.t_con[0],
            end=sim_params.t_con[0] + sim_params.t_con[2],
            freq=sim_params.t_con[1],
        )
        
        all_params = {
            "sim_params": sim_params,
            "fl_params": fl_params,
            "contrail_params": contrail_params,
            "met_params": met_params,
            "issr_params": issr_params,
        }

        # Set the model parameters
        self.sim_params = sim_params
        self.fl_params = fl_params
        self.contrail_params = contrail_params
        self.met_params = met_params
        self.issr_params = issr_params
        self.all_params = all_params

    def _build_grid(self, sim_params):
        """Build coarse grid vectors and meter axes from sim_params bounds.

        Also builds padded met grid arrays (``met_lons``, ``met_lats``,
        ``met_levels``) that extend one cell beyond the BOXM domain on
        each side so that DryAdvection can interpolate at flight
        waypoints near the domain boundary.
        """
        hres = sim_params.hres_sim
        vres = sim_params.vres_sim
        _eps = 1e-9  # tolerance so a one-cell domain (start == stop) still yields one point

        self.lats = np.arange(
            sim_params.lat_bounds[0] + hres / 2,
            sim_params.lat_bounds[1] + _eps,
            hres,
        )
        self.lons = np.arange(
            sim_params.lon_bounds[0] + hres / 2,
            sim_params.lon_bounds[1] + _eps,
            hres,
        )
        self.alts = np.arange(
            sim_params.alt_bounds[0] + vres / 2,
            sim_params.alt_bounds[1] + _eps,
            vres,
        )

        # Padded met grid: one extra cell on each side for DryAdvection
        self.met_lats = np.arange(
            sim_params.lat_bounds[0] - hres / 2,
            sim_params.lat_bounds[1] + hres + _eps,
            hres,
        )
        self.met_lons = np.arange(
            sim_params.lon_bounds[0] - hres / 2,
            sim_params.lon_bounds[1] + hres + _eps,
            hres,
        )
        self.met_alts = np.arange(
            sim_params.alt_bounds[0] - vres / 2,
            sim_params.alt_bounds[1] + vres + _eps,
            vres,
        )
        self.met_levels = units.m_to_pl(self.met_alts)

        self.levels = units.m_to_pl(self.alts)

        # Convert 1D lon/lat axes to meter axes
        ref_lon = float(self.lons.min())
        ref_lat = float(self.lats.min())
        self.lons_m, _ = lonlat_to_m(
            self.lons,
            np.full_like(self.lons, ref_lat, dtype=float),
            ref_lon,
            ref_lat,
        )
        _, self.lats_m = lonlat_to_m(
            np.full_like(self.lats, ref_lon, dtype=float),
            self.lats,
            ref_lon,
            ref_lat,
        )

    def traj_gen(self) -> list[Flight]:
        """Generate flight trajectory points. Supports loading and plotting all test flights if requested."""
        fl_params = self.gpat.fl_params

        # traj will by default be a single flight along the center of the domain, with entry and exit points of diversion to be calculated from fl_params input.



    def gen_met(self) -> MetDataset:
        """Generate meteorology data."""

        met_params = self.gpat.met_params

        time_bounds = (self.times_sim[0], self.times_sim[-1])

        era5pl = ERA5(
            time=time_bounds,
            variables=Cocip.met_variables + Cocip.optional_met_variables,
            pressure_levels=pressure_levels,
        )
        era5sl = ERA5(time=time_bounds, variables=Cocip.rad_variables)


        # # Step 1: Create with STANDARD names for MetDataset validation
        # met_standard = xr.Dataset(
        #     data_vars={
        #         "eastward_wind": (
        #             ("time", "level", "latitude", "longitude"),
        #             np.full((len(self.gpat.times_sim), len(met_levels), len(met_lats), len(met_lons)), met_params.eastward_wind),
        #         ),
        #         "northward_wind": (
        #             ("time", "level", "latitude", "longitude"),
        #             np.full((len(self.gpat.times_sim), len(met_levels), len(met_lats), len(met_lons)), met_params.northward_wind),
        #         ),
        #         "lagrangian_tendency_of_air_pressure": (
        #             ("time", "level", "latitude", "longitude"),
        #             np.full((len(self.gpat.times_sim), len(met_levels), len(met_lats), len(met_lons)), met_params.lagrangian_tendency_of_air_pressure),
        #         ),
        #     },
        #     coords={
        #         "longitude": met_lons,
        #         "latitude": met_lats,
        #         "level": met_levels,
        #         "time": pd.to_datetime(self.gpat.times_sim.values).strftime("%Y-%m-%dT%H:%M:%SZ"),
        #     },
        # )

        # Step 2: Initialize MetDataset (validates standard names)
        # met = MetDataset(met_standard)
        
        # calculate solar zenith angle
        # met.data["sza"] = (
        #     ("latitude", "longitude", "time"),
        #     self._calc_sza(
        #         met["latitude"].data.values, met["longitude"].data.values, met["time"].data.values
        #     ),
        # )

        return met

    def assign_saf(self, pct_blend: float):
        saf=SAFBlend(pct_blend)
        fl_saf=fl.copy()

        fl_saf.attrs["aircraft_type"] = "A320"
        fl_saf.fuel = saf



    def preprocess(self):
        # Generate flight trajectory points
        self.fl = self.setup.traj_gen()

        self._build_grid(self.sim_params)

        # Generate meteorological data
        self.met = self.setup.gen_met()

        # Calculate aircraft performance using PS Model
        self.fl = self.setup.ac_perf()
        print("Aircraft performance calculated using PS Model.")

        # Estimate emissions using Pycontrails Emissions Model
        self.fl = self.setup.emissions()
        print("Emissions estimated using Pycontrails Emissions Model.")

        # Simulate contrail formation, persistence and climate impact using Pycontrails CoCiP model.
        if self.fl_params.sim_plumes:
            self.fl, self.pl = self.setup.sim_plumes()
            print("Plume dispersion simulated using Pycontrails Dry Advection Model.")

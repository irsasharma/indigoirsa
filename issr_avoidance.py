import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Literal, Optional
from dataclasses import asdict, dataclass, field, fields, is_dataclass

from pycontrails import Flight, MetDataset, models
from pycontrails.core.models import Model
from pycontrails.models.emissions import Emissions
from pycontrails.models.ps_model import PSFlight
from pycontrails.physics import constants, geo, thermo, units

from pycontrails.models.cocip import Cocip

@dataclass
class SimParams:
    """Default simulation parameters."""
    # Temporal domain
    
    # flight time
    t_fl: tuple[pd.Timestamp, pd.Timedelta, pd.Timedelta] = field(
        default_factory=lambda: (
            pd.to_datetime("2025-01-20 13:00:00"),
            pd.Timedelta(minutes=2),
            pd.Timedelta(hours=1),
        )
    )  # (start time, time step, run time)

    # contrail time
    t_contrail: tuple[pd.Timestamp, pd.Timedelta, pd.Timedelta] = field(
        default_factory=lambda: (
            pd.to_datetime("2025-01-20 13:00:00"),
            pd.Timedelta(minutes=2),
            pd.Timedelta(hours=24),
        )
    )  # (start time, time step, run time)

    #  spatial domain
    lat_bounds: tuple[float, float] = (0.0, 1.0)  # lat bounds [deg]
    lon_bounds: tuple[float, float] = (0.0, 1.0)  # lon bounds [deg]
    alt_bounds: tuple[float, float] = (12000, 13000)  # alt bounds [m]
    hres_sim: float = 0.01  # horizontal resolution [deg]
    vres_sim: float = 500  # vertical resolution [m]

@dataclass
class FlParams:
    """Default flight/fleet parameters."""
    # Synthetic focal-point trajectory controls
    n_ac: int = 1
    ac_type: str = "A320"
    fl_heading: float | list[float] = 90.0          # deg, 0=N, 90=E
    fl_speed: float | list[float] = 230.0           # m/s
    fl_altitude: float | list[float] | None = None  # m; None → midpoint of alt_bounds

    fl_t_cross_s: float | list[float] | None = 1800.0
    fl_sep_time_s: float = 0.0

    fl_entry_time_s: float | list[float] | None = None

    fl_focal_lat: float | None = None
    fl_focal_lon: float | None = None
    fl_focal_alt: float | None = None  # m; None → midpoint of alt_bounds

    fl_cross_track_offset_m: float | list[float] = 0.0
    fl_altitude_offset_m: float | list[float] = 0.0

    domain_padding_m: float = 0.0

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

class ISSRAvoidance(Model):

    def __init__(self,
                sim_params: SimParams,
                fl_params: Optional[FlParams] = None,
                contrail_params: Optional[ContrailParams] = None,
                met_params: Optional[MetParams] = None):
        super().__init__()

        # Apply defaults for Eulerian-only / param-sweep mode (no flights/plumes needed)
        if fl_params is None:
            fl_params = FlParams(n_ac=0, mode="synthetic")
        if contrail_params is None:
            contrail_params = ContrailParams()
        if met_params is None:
            met_params = MetParams()

        # param_axes mode is always plume-free
        if sim_params.param_axes is not None:
            fl_params.n_ac = 0

        # Build spatial grid from current bounds
        self._build_grid(sim_params)

        # Generate time vectors
        if fl_params.n_ac > 0:
            self.times_fl = pd.date_range(
                start=sim_params.t_fl[0],
                end=sim_params.t_fl[0] + sim_params.t_fl[2],
                freq=sim_params.t_fl[1],
            )

            self.times_pl = pd.date_range(
                start=sim_params.t_pl[0],
                end=sim_params.t_sim[0] + sim_params.t_sim[2],
                freq=sim_params.t_pl[1],
            )
        else:
            self.times_fl = None
            self.times_pl = None

        self.times_sim = pd.date_range(
            start=sim_params.t_sim[0],
            end=sim_params.t_sim[0] + sim_params.t_sim[2],
            freq=sim_params.t_sim[1],
        )

        self.times_out = pd.date_range(
            start=sim_params.t_out[0],
            end=sim_params.t_out[0] + sim_params.t_out[2],
            freq=sim_params.t_out[1],
        )

        # Set up paths and job ID
        self.run_path = sim_params.run_path
        self.data_path = sim_params.data_path
        self.job_id = sim_params.job_id

        # If running in direct mode with a specified flight file, derive job_id from the filename if not already set.
        if fl_params.mode == "opensky" and fl_params.file is not None:
            self.job_id = os.path.splitext(os.path.basename(fl_params.file))[0]

        sim_params.date_created = pd.Timestamp.now()

        # Grab species numbers from the input files for later use in indexing model outputs
        chem_params.species_emi_num = grab_species_num(self.run_path, chem_params.species_emi)
        if fl_params.n_ac > 0:
            chem_params.species_pl_num = grab_species_num(self.run_path, chem_params.species_pl)
        else:
            chem_params.species_pl_num = np.array([], dtype=int)
        chem_params.species_out_num = grab_species_num(self.run_path, chem_params.species_out)
        chem_params.species_boxm_num = grab_species_num_boxm(self.run_path)

        # Validate species and time hierarchies
        validate_species_hierarchy(chem_params)
        if fl_params.n_ac > 0:
            validate_time_hierarchy(sim_params)

        # Per-run directory: data/<config_family>/<job_id>/
        self.run_dir = self.data_path + self.job_id + "/"

        # Shared global inputs: data/glob/
        self.inputs_glob = str(Path(self.run_path) / "data" / "glob") + "/"

        # Backward-compatible aliases, if other code still uses these names
        self.inputs_job = self.run_dir
        self.outputs_job = self.run_dir

        # If job dirs exist, clear and recreate
        if os.path.exists(self.inputs_job):
            shutil.rmtree(self.inputs_job)

        if os.path.exists(self.outputs_job):
            shutil.rmtree(self.outputs_job)

        os.makedirs(self.inputs_job, exist_ok=True)
        os.makedirs(self.outputs_job, exist_ok=True)      

        all_params = {
            "sim_params": sim_params,
            "fl_params": fl_params,
            "contrail_params": contrail_params,
            "met_params": met_params,
        }

        # Set the model parameters
        self.sim_params = sim_params
        self.fl_params = fl_params
        self.contrail_params = contrail_params
        self.met_params = met_params
        self.all_params = all_params

        # Set up the model classes for preprocessing and running GPAT
        self.setup = GPATSetup(self)
        self.run = GPATRun(self)

    def _build_grid(self, sim_params):
        """Build coarse grid vectors and meter axes from sim_params bounds.

        Also builds padded met grid arrays (``met_lons``, ``met_lats``,
        ``met_levels``) that extend one cell beyond the BOXM domain on
        each side so that DryAdvection can interpolate at flight
        waypoints near the domain boundary.
        """
        if sim_params.param_axes is not None:
            self._build_param_grid(sim_params)
            return

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

        # generate synthetic formation flight
        if fl_params.mode == "synthetic":
            return self._traj_gen_synthetic_focal()

        # grab data from opensky
        if fl_params.mode == "opensky":
            return self._traj_gen_opensky()

    def gen_met(self) -> MetDataset:
        """Generate meteorology data.

        Uses the padded met grid (``met_lons``, ``met_lats``,
        ``met_levels``) so that DryAdvection can interpolate at flight
        waypoints near the BOXM domain boundary.
        """
        if self.gpat.sim_params.param_axes is not None:
            return self._gen_met_param_sweep()

        met_params = self.gpat.met_params
        met_lons = self.gpat.met_lons
        met_lats = self.gpat.met_lats
        met_levels = self.gpat.met_levels

        # Step 1: Create with STANDARD names for MetDataset validation
        met_standard = xr.Dataset(
            data_vars={
                "eastward_wind": (
                    ("time", "level", "latitude", "longitude"),
                    np.full((len(self.gpat.times_sim), len(met_levels), len(met_lats), len(met_lons)), met_params.eastward_wind),
                ),
                "northward_wind": (
                    ("time", "level", "latitude", "longitude"),
                    np.full((len(self.gpat.times_sim), len(met_levels), len(met_lats), len(met_lons)), met_params.northward_wind),
                ),
                "lagrangian_tendency_of_air_pressure": (
                    ("time", "level", "latitude", "longitude"),
                    np.full((len(self.gpat.times_sim), len(met_levels), len(met_lats), len(met_lons)), met_params.lagrangian_tendency_of_air_pressure),
                ),
            },
            coords={
                "longitude": met_lons,
                "latitude": met_lats,
                "level": met_levels,
                "time": pd.to_datetime(self.gpat.times_sim.values).strftime("%Y-%m-%dT%H:%M:%SZ"),
            },
        )

        # Step 2: Initialize MetDataset (validates standard names)
        met = MetDataset(met_standard)

        month = self.gpat.times_sim[0].month

        # Step 3: Load and interpolate climatology with standard names
        air_temperature = (
            xr.open_dataarray(self.gpat.inputs_glob + "air_temperature.nc", engine="netcdf4")
            .sel(month=month - 1)
            .interp(
                longitude=met_lons,
                latitude=met_lats,
                level=met_levels,
                method="linear",
                kwargs={"fill_value": "extrapolate"},
            )
            .broadcast_like(met.data["eastward_wind"])
        )

        h2o_concs = (
            xr.open_dataarray(self.gpat.inputs_glob + "h2o_concs.nc", engine="netcdf4")
            .sel(month=month - 1)
            .interp(
                longitude=met_lons,
                latitude=met_lats,
                level=met_levels,
                method="linear",
                kwargs={"fill_value": "extrapolate"},
            )
            .broadcast_like(met.data["eastward_wind"])
        )
        N_A = 6.022e23  # Avogadro's number
        
        # Add temp and H2O to met dataset
        met.data["air_temperature"] = air_temperature
        met.data["H2O"] = h2o_concs.transpose("latitude", "longitude", "level", "time")

        # Calculate specific humidity and relative humidity
        rho_d = met["air_pressure"].data / (constants.R_d * met["air_temperature"].data)
        met.data["specific_humidity"] = met.data["H2O"] * constants.M_d / (N_A * rho_d * 1e-6)
        met.data["relative_humidity"] = thermo.rhi(
            met.data["specific_humidity"], met.data["air_temperature"], met.data["air_pressure"]
        )

        # Calculate number density of air (M) to feed into box model calcs
        met.data["M"] = (N_A / constants.M_d) * rho_d * 1e-6  # [molecules / cm^3]
        met.data["M"] = met.data["M"].transpose("latitude", "longitude", "level", "time")

        # Calculate O2 and N2 number concs based on M
        met.data["O2"] = 2.079e-01 * met.data["M"]
        met.data["N2"] = 7.809e-01 * met.data["M"]

        # calculate solar zenith angle
        met.data["sza"] = (
            ("latitude", "longitude", "time"),
            self._calc_sza(
                met["latitude"].data.values, met["longitude"].data.values, met["time"].data.values
            ),
        )

        return met

    def assign_saf(self):
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

"""
run_issr.py
===========
Runner script for the ISSR contrail avoidance simulation.
Edit the parameter blocks below, then run:

    python run_issr.py
"""

from issr_avoidance import ContrailSimulator, SimParams, FlParams, MetParams, ISSRParams
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

sim_params = SimParams(
    t_start=pd.Timestamp("2025-01-20 18:00"),
    dt_fl=pd.Timedelta(minutes=2),
    duration_fl=pd.Timedelta(hours=1),
    dt_met=pd.Timedelta(hours=1),
    duration_met=pd.Timedelta(hours=6),
    lon_bounds=(-2.0, 6.0),
    lat_bounds=(-2.0, 4.0),
    alt_bounds=(8_000.0, 16_000.0),
    hres=0.25,
    vres=500.0,
)

fl_params = FlParams(
    ac_type="A320",
    pct_blend=0.0,        # baseline SAF blend [%]
    target_alt=None,      # None -> domain midpoint
    alt_delta=1_000.0,     # avoidance altitude shift [m]
    fl_lon_bounds=(-2.0, 6.0),
    rocd=5.0,             # rate of climb/descent [m/s]
    avoid_frac=0.5,       # fraction of ISSR to avoid [0-1]
)

met_params = MetParams(
    air_temperature=220.0,
    eastward_wind=0.0,
    northward_wind=3.0,
    lagrangian_tendency_of_air_pressure=0.0,
)

issr_params = ISSRParams(
    centroid=(2.0, 1.0, 12_000.0),   # (lon deg, lat deg, alt m)
    sigma_parallel=150_000.0,          # along-track half-width [m]
    sigma_perp=50_000.0,               # across-track half-width [m]
    sigma_z=500.0,                   # vertical half-width [m]
    rhi_bg=0.75,
    rhi_peak=1.20,
)

# ---------------------------------------------------------------------------
# Flight cases
# ---------------------------------------------------------------------------

flight_cases = pd.DataFrame([
    {
        "case": "baseline",
        "alt": 12_000.0,
        "alt_target": None,
        "avoidance": "none",
        "pct_blend": 0.0,
    },
    {
        "case": "descend_500m",
        "alt": 12_000.0,
        "alt_target": 11_500.0,
        "avoidance": "descend",
        "pct_blend": 0.0,
    },
    {
        "case": "descend_1000m",
        "alt": 12_000.0,
        "alt_target": 11_000.0,
        "avoidance": "descend",
        "pct_blend": 0.0,
    },
    {
        "case": "descend_1500m",
        "alt": 12_000.0,
        "alt_target": 10_500.0,
        "avoidance": "descend",
        "pct_blend": 0.0,
    },
    {
        "case": "climb_500m",
        "alt": 12_000.0,
        "alt_target": 12_500.0,
        "avoidance": "climb",
        "pct_blend": 0.0,
    },
    {
        "case": "climb_1000m",
        "alt": 12_000.0,
        "alt_target": 13_000.0,
        "avoidance": "climb",
        "pct_blend": 0.0,
    },
    {
        "case": "climb_1500m",
        "alt": 12_000.0,
        "alt_target": 13_500.0,
        "avoidance": "climb",
        "pct_blend": 0.0,
    }
    # {
    #     "case": "baseline_100saf",
    #     "alt": 12_000.0,
    #     "alt_target": None,
    #     "avoidance": "none",
    #     "pct_blend": 100.0,
    # },
    # {
    #     "case": "descend_500m_100saf",
    #     "alt": 12_000.0,
    #     "alt_target": 11_500.0,
    #     "avoidance": "descend",
    #     "pct_blend": 100.0,
    # },
])

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

sim = ContrailSimulator(
    sim_params=sim_params,
    fl_params=fl_params,
    met_params=met_params,
    issr_params=issr_params,
)

results = sim.run_flights(flight_cases)
summary = sim.summarise_results(results)

print("\nSimulation summary:")
print(summary.to_string())

sim.save_results(results, summary)

fig = sim.plot_3d(results, output_html="results/issr_3d.html")

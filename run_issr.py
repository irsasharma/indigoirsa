"""
run_issr.py
===========
Runner script for the ISSR contrail avoidance simulation.
Edit the parameter blocks below, then run:

    python run_issr.py
"""

from issr_avoidance import ContrailSimulator, SimParams, FlParams, MetParams, ISSRParams
import pandas as pd
from dataclasses import replace
from itertools import product
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

sim_params = SimParams(
    t_start=pd.Timestamp("2025-01-01 18:00"),
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
    },
    {
        "case": "descend_500m",
        "alt": 12_000.0,
        "alt_target": 11_500.0,
        "avoidance": "descend",
    },
    {
        "case": "descend_1000m",
        "alt": 12_000.0,
        "alt_target": 11_000.0,
        "avoidance": "descend",
    },
    {
        "case": "descend_1500m",
        "alt": 12_000.0,
        "alt_target": 10_500.0,
        "avoidance": "descend",
    },
    {
        "case": "climb_500m",
        "alt": 12_000.0,
        "alt_target": 12_500.0,
        "avoidance": "climb",
    },
    {
        "case": "climb_1000m",
        "alt": 12_000.0,
        "alt_target": 13_000.0,
        "avoidance": "climb",
    },
    {
        "case": "climb_1500m",
        "alt": 12_000.0,
        "alt_target": 13_500.0,
        "avoidance": "climb",
    },
])

# ---------------------------------------------------------------------------
# Batch configuration
# ---------------------------------------------------------------------------

# Set one or more values per key to sweep combinations.
# Supported keys:
#   - fl_<field on FlParams>
#   - issr_<field on ISSRParams>
batch_axes = {
    "issr_rhi_peak": [0.8, 1.0, 1.20, 1.40],
    "issr_sigma_z": [500.0, 1000.0, 2000.0, 4000.0],
    "avoid_frac": [0.25, 0.5, 0.75],
}


def _slug(value):
    text = str(value)
    cleaned = []
    for ch in text:
        if ch.isalnum() or ch in "-_.":
            cleaned.append(ch)
        else:
            cleaned.append("-")
    slug = "".join(cleaned).strip("-")
    return slug or "value"


def iter_batch_scenarios(axes):
    if not axes:
        return [dict()]

    keys = list(axes.keys())
    values = [axes[k] for k in keys]

    for key, vals in axes.items():
        if not isinstance(vals, (list, tuple)) or len(vals) == 0:
            raise ValueError(f"batch_axes['{key}'] must be a non-empty list/tuple")

    scenarios = []
    for combo in product(*values):
        scenarios.append(dict(zip(keys, combo)))
    return scenarios


def apply_batch_scenario(base_cases, base_fl_params, base_issr_params, scenario):
    cases = base_cases.copy(deep=True)
    fl_local = base_fl_params
    issr_local = base_issr_params

    for key, value in scenario.items():
        if key.startswith("fl_"):
            field = key.removeprefix("fl_")
            if not hasattr(fl_local, field):
                raise ValueError(f"Unknown FlParams field in batch_axes: '{field}'")
            fl_local = replace(fl_local, **{field: value})
            continue

        if key.startswith("issr_"):
            field = key.removeprefix("issr_")
            if not hasattr(issr_local, field):
                raise ValueError(f"Unknown ISSRParams field in batch_axes: '{field}'")
            issr_local = replace(issr_local, **{field: value})
            continue

        raise ValueError(
            f"Unsupported batch key '{key}'. Use 'fl_<field>' or 'issr_<field>'."
        )

    return cases, fl_local, issr_local

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

scenarios = iter_batch_scenarios(batch_axes)
run_stamp = pd.Timestamp.now().strftime("%Y%m%d-%H%M%S")
batch_root = Path("results") / f"batch_{run_stamp}"
batch_root.mkdir(parents=True, exist_ok=True)

all_summaries = []

print(f"\nRunning {len(scenarios)} batch scenario(s)...")

for idx, scenario in enumerate(scenarios, start=1):
    scenario_items = [f"{k}-{_slug(v)}" for k, v in scenario.items()]
    scenario_name = "__".join(scenario_items) if scenario_items else "default"
    scenario_dir = batch_root / f"{idx:03d}_{scenario_name}"
    scenario_dir.mkdir(parents=True, exist_ok=True)

    scenario_cases, scenario_fl_params, scenario_issr_params = apply_batch_scenario(
        flight_cases,
        fl_params,
        issr_params,
        scenario,
    )

    print(f"\n[{idx}/{len(scenarios)}] Scenario: {scenario_name}")

    sim = ContrailSimulator(
        sim_params=sim_params,
        fl_params=scenario_fl_params,
        met_params=met_params,
        issr_params=scenario_issr_params,
    )

    results = sim.run_flights(scenario_cases)
    summary = sim.summarise_results(results)

    for key, value in scenario.items():
        summary[key] = value

    print(summary.to_string())

    sim.save_results(results, summary, output_dir=str(scenario_dir))

    with_case = summary.reset_index()
    with_case.insert(0, "scenario", scenario_name)
    all_summaries.append(with_case)

batch_summary = pd.concat(all_summaries, ignore_index=True)
batch_summary_path = batch_root / "batch_summary.csv"
batch_summary.to_csv(batch_summary_path, index=False)

print("\nBatch complete")
print(f"Output root: {batch_root}")
print(f"Combined summary: {batch_summary_path}")

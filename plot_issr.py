"""
plot_issr.py
============
Post-processing and figures for the synthetic-ISSR contrail-avoidance
simulation produced by run_issr.py.

Usage
-----
    python plot_issr.py                  # all figures + debug table
    python plot_issr.py --no-show        # save PDFs only
    python plot_issr.py --debug          # debug tables, no figures
    python plot_issr.py --outdir figures # set output directory

Figures
-------
  fig1_issr_cross_section.pdf   – ISSR RHi field (lon/alt) + flight paths
  fig2_rhi_along_track.pdf      – along-track RHi per case
  fig3_ef_summary.pdf           – total EF bar chart (log scale)
  fig4_contrail_width.pdf       – contrail width vs age per case
  fig5_ef_sensitivity.pdf       – EF vs altitude offset from ISSR centroid
  fig6_contrail_spatial.pdf     – contrail waypoints (lon/lat, EF colour)
  fig7_ef_vs_fuel.pdf           – EF reduction vs fuel penalty trade-off
"""

from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import TwoSlopeNorm

PAPER_RC = {
    "font.family": "sans-serif", "font.size": 10,
    "axes.titlesize": 11, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.linestyle": "--",
    "grid.linewidth": 0.4, "grid.alpha": 0.5, "lines.linewidth": 1.6,
}

CASE_COLOURS = {
    "baseline": "#1f77b4", "baseline_100saf": "#aec7e8",
    "descend_500m": "#d62728", "descend_1000m": "#ff7f0e",
    "descend_1500m": "#8c564b", "climb_500m": "#2ca02c",
    "climb_1000m": "#9467bd", "climb_1500m": "#e377c2",
    "descend_500m_100saf": "#bcbd22",
}

def _colour(case, cases):
    return CASE_COLOURS.get(case, plt.cm.tab10(
        (cases.index(case) if case in cases else 0) / 10))

def _label(case):
    return {
        "baseline": "Baseline", "baseline_100saf": "Baseline 100% SAF",
        "descend_500m": "Descend 500 m", "descend_1000m": "Descend 1000 m",
        "descend_1500m": "Descend 1500 m", "climb_500m": "Climb 500 m",
        "climb_1000m": "Climb 1000 m", "climb_1500m": "Climb 1500 m",
        "descend_500m_100saf": "Descend 500 m 100% SAF",
    }.get(case, case.replace("_", " ").title())


def load_results(results_dir):
    sp = results_dir / "summary.csv"
    if not sp.exists():
        sys.exit(f"[ERROR] summary.csv not found in {results_dir}")
    summary = pd.read_csv(sp, index_col=0)
    flights, contrails = {}, {}
    for case in summary.index:
        fp = results_dir / f"flight_{case}.csv"
        cp = results_dir / f"contrail_{case}.csv"
        if fp.exists():
            flights[case] = pd.read_csv(fp)
        contrails[case] = pd.read_csv(cp) if cp.exists() else None
    rp = results_dir / "rhi_field.npy"
    return dict(summary=summary, flights=flights, contrails=contrails,
                rhi_field=np.load(rp) if rp.exists() else None)


def print_debug(data):
    s = data["summary"]
    print("\n" + "=" * 70)
    print("SIMULATION SUMMARY")
    print("=" * 70)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 140)
    print(s.to_string())
    if "baseline" in s.index:
        ref_ef = s.loc["baseline", "total_ef_J"]
        ref_ff = s.loc["baseline", "mean_fuel_flow_kg_s"]
        print("\n" + "-" * 70)
        print("NORMALISED (relative to baseline)")
        print("-" * 70)
        rows = []
        for c in s.index:
            ef = s.loc[c, "total_ef_J"]
            ff = s.loc[c, "mean_fuel_flow_kg_s"]
            rows.append({"case": c,
                "EF [TJ]": ef / 1e12,
                "EF change [%]": (ef - ref_ef) / abs(ref_ef) * 100,
                "fuel [kg/s]": ff,
                "fuel change [%]": (ff - ref_ff) / abs(ref_ff) * 100,
                "contrail_pts": int(s.loc[c, "contrail_points"])})
        print(pd.DataFrame(rows).set_index("case").to_string())
    print("=" * 70 + "\n")


def _issr_coords(rhi, flights):
    n_lon, n_lat, n_alt = rhi.shape
    lons = np.concatenate([f["longitude"].values for f in flights.values()])
    lats = np.concatenate([f["latitude"].values for f in flights.values()])
    alts = np.concatenate([f["altitude"].values for f in flights.values()])
    lon0, lon1 = lons.min(), lons.max()
    lat0, lat1 = lats.min(), lats.max()
    alt0, alt1 = alts.min(), alts.max()
    hres = (lon1 - lon0) / max(n_lon - 2, 1)
    hlat = (lat1 - lat0) / max(n_lat - 2, 1)
    vres = (alt1 - alt0) / max(n_alt - 2, 1)
    met_lons = np.linspace(lon0 - hres / 2, lon1 + hres / 2, n_lon)
    met_lats = np.linspace(lat0 - hlat / 2, lat1 + hlat / 2, n_lat)
    met_alts = np.linspace(alt0 - vres / 2, alt1 + vres / 2, n_alt)
    return met_lons, met_lats, met_alts


# ---- Figure 1: ISSR cross-section + flight paths -------------------------

def fig_issr_cross_section(data, cases, outdir):
    rhi = data.get("rhi_field")
    if rhi is None:
        print("[SKIP] fig1 – rhi_field.npy not found"); return None
    met_lons, met_lats, met_alts = _issr_coords(rhi, data["flights"])
    lat_idx = len(met_lats) // 2
    rhi_slice = rhi[:, lat_idx, :]

    fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
    cf = ax.contourf(met_lons, met_alts / 1e3, rhi_slice.T,
                     levels=np.linspace(0.6, 1.35, 16),
                     cmap="RdYlBu_r", extend="both")
    ax.contour(met_lons, met_alts / 1e3, rhi_slice.T,
               levels=[1.0], colors="red", linewidths=1.5, linestyles="--")
    cb = fig.colorbar(cf, ax=ax, label="RHi")
    cb.ax.axhline(1.0, color="red", linewidth=1.0, linestyle="--")

    for case in cases:
        fl = data["flights"].get(case)
        if fl is None: continue
        ax.plot(fl["longitude"], fl["altitude"] / 1e3,
                color=_colour(case, cases), linewidth=1.6, label=_label(case))

    ax.set_xlabel("Longitude [°E]")
    ax.set_ylabel("Altitude [km]")
    ax.set_title("ISSR Relative Humidity over Ice (lon/alt cross-section, lat midpoint)\n"
                 "Red dashed = RHi 1.0 boundary;  coloured lines = flight paths")
    ax.legend(fontsize=8, loc="upper right")
    out = outdir / "fig1_issr_cross_section.pdf"
    fig.savefig(out); print(f"[SAVED] {out}")
    return fig


# ---- Figure 2: Along-track RHi -------------------------------------------

def fig_rhi_along_track(data, cases, outdir):
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    for case in cases:
        fl = data["flights"].get(case)
        if fl is None or "rhi" not in fl.columns: continue
        ax.plot(fl["longitude"], fl["rhi"],
                color=_colour(case, cases), label=_label(case), alpha=0.85)
    ax.axhline(1.0, color="red", linewidth=1.2, linestyle="--",
               label="RHi = 1.0 (ISSR onset)")
    ax.set_xlabel("Longitude [°E]")
    ax.set_ylabel("RHi along flight path")
    ax.set_title("Relative Humidity over Ice Along Each Flight Track\n"
                 "Avoidance strategies sample lower RHi outside the ISSR")
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    out = outdir / "fig2_rhi_along_track.pdf"
    fig.savefig(out); print(f"[SAVED] {out}")
    return fig


# ---- Figure 3: EF summary (log scale) ------------------------------------

def fig_ef_summary(data, cases, outdir):
    s      = data["summary"]
    ef_TJ  = [s.loc[c, "total_ef_J"] / 1e12 for c in cases]
    labels = [_label(c) for c in cases]
    colours= [_colour(c, cases) for c in cases]
    ref_ef = s.loc["baseline", "total_ef_J"] if "baseline" in s.index else s.iloc[0]["total_ef_J"]

    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    bars = ax.bar(range(len(cases)), ef_TJ, color=colours, edgecolor="white", linewidth=0.5)
    for i, (bar, val) in enumerate(zip(bars, [v * 1e12 for v in ef_TJ])):
        pct = (val - ref_ef) / abs(ref_ef) * 100
        txt = "ref" if i == 0 else f"{pct:+.0f}%"
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.15, txt,
                ha="center", va="bottom", fontsize=8)
    ax.set_yscale("log")
    ax.set_xticks(range(len(cases)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Total Contrail EF [TJ]  (log scale)")
    ax.set_title("Total Contrail Energy Forcing by Avoidance Strategy")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.3g}"))
    out = outdir / "fig3_ef_summary.pdf"
    fig.savefig(out); print(f"[SAVED] {out}")
    return fig



# ---- Figure 4: EF vs altitude offset from centroid ----------------------

def fig_ef_sensitivity(data, cases, outdir):
    s = data["summary"]
    if "baseline" not in s.index:
        print("[SKIP] fig4 – no baseline"); return None
    centroid_alt = float(s.loc["baseline", "altitude_m"])
    ref_ef = float(s.loc["baseline", "total_ef_J"])

    descend, climb = [], []
    for c in cases:
        if c == "baseline": continue
        tgt = s.loc[c, "target_altitude_m"]
        if pd.isna(tgt): continue
        offset = float(tgt) - centroid_alt
        ef     = float(s.loc[c, "total_ef_J"])
        (climb if offset > 0 else descend).append((abs(offset), ef))

    fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
    ax.axhline(ref_ef / 1e12, color=_colour("baseline", cases),
               linewidth=1.2, linestyle="--", label="Baseline")

    for pts, case_key, marker in [
            (descend, "descend_500m", "o"),
            (climb,   "climb_500m",   "s")]:
        if not pts: continue
        xs, ys = zip(*sorted(pts))
        ax.plot([0] + list(xs), [ref_ef / 1e12] + [v / 1e12 for v in ys],
                marker + "-", color=_colour(case_key, cases),
                label="Descend" if "descend" in case_key else "Climb")

    ax.set_yscale("log")
    ax.set_xlabel("Altitude Offset from ISSR Centroid [m]")
    ax.set_ylabel("Total Contrail EF [TJ]  (log scale)")
    ax.set_title("EF Sensitivity to Altitude Avoidance\n"
                 "Reflects the vertical Gaussian profile of the ISSR")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.3g}"))
    ax.legend()
    out = outdir / "fig4_ef_sensitivity.pdf"
    fig.savefig(out); print(f"[SAVED] {out}")
    return fig


# ---- Figure 5: Contrail spatial (lat/lon, EF colour) --------------------

def fig_contrail_spatial(data, cases, outdir):
    valid = [c for c in cases
             if data["contrails"].get(c) is not None
             and not data["contrails"][c].empty
             and "ef" in data["contrails"][c].columns]
    if not valid:
        print("[SKIP] fig5 – no contrail EF data"); return None

    ncols = min(3, len(valid))
    nrows = -(-len(valid) // ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4.5 * ncols, 3.5 * nrows),
                             sharex=True, sharey=True, layout="constrained")
    axes = np.atleast_1d(axes).ravel()

    all_ef = np.concatenate([data["contrails"][c]["ef"].values for c in valid])
    nonzero = all_ef[all_ef != 0]
    vabs = np.percentile(np.abs(nonzero), 95) if len(nonzero) else 1.0
    norm = TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs)
    cmap = mpl.colormaps["RdBu_r"]

    rhi = data.get("rhi_field")
    issr_footprint = None
    if rhi is not None:
        met_lons, met_lats, met_alts = _issr_coords(rhi, data["flights"])
        alt_idx = np.argmin(np.abs(met_alts - 12000.0))
        issr_footprint = (met_lons, met_lats, rhi[:, :, alt_idx])

    for ax, case in zip(axes, valid):
        con = data["contrails"][case]
        ax.scatter(con["longitude"], con["latitude"],
                   c=con["ef"].values, cmap=cmap, norm=norm,
                   s=3, linewidths=0, alpha=0.7, rasterized=True)
        if issr_footprint is not None:
            lo, la, rh = issr_footprint
            ax.contour(lo, la, rh.T, levels=[1.0],
                       colors="red", linewidths=0.8, linestyles="--", alpha=0.7)
        ax.set_title(_label(case), fontsize=9)
        ax.set_xlabel("Lon [°E]")
        ax.set_ylabel("Lat [°N]")

    for ax in axes[len(valid):]:
        ax.set_visible(False)

    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=axes[:len(valid)], label="EF per waypoint [J]",
                 fraction=0.015, pad=0.02)
    fig.suptitle("Contrail Waypoints and EF  (red dashed = ISSR boundary)", fontsize=10)
    out = outdir / "fig5_contrail_spatial.pdf"
    fig.savefig(out); print(f"[SAVED] {out}")
    return fig


# ---- Figure 6: EF reduction vs fuel penalty trade-off -------------------

def fig_ef_vs_fuel(data, cases, outdir):
    s = data["summary"]
    if "baseline" not in s.index:
        print("[SKIP] fig6 – no baseline"); return None
    ref_ef = float(s.loc["baseline", "total_ef_J"])
    ref_ff = float(s.loc["baseline", "mean_fuel_flow_kg_s"])

    fig, ax = plt.subplots(figsize=(6, 5), layout="constrained")
    for case in cases:
        ef  = float(s.loc[case, "total_ef_J"])
        ff  = float(s.loc[case, "mean_fuel_flow_kg_s"])
        x   = (ff - ref_ff) / abs(ref_ff) * 100
        y   = (ref_ef - ef) / abs(ref_ef) * 100
        ax.scatter(x, y, color=_colour(case, cases), s=90, zorder=3,
                   edgecolors="white", linewidths=0.5)
        ax.annotate(_label(case), (x, y), textcoords="offset points",
                    xytext=(6, 4), fontsize=7.5)

    ax.axhline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.4)
    ax.axvline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.4)
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    ax.fill_between([min(xlim[0], -1), 0], 0, max(ylim[1], 5),
                    alpha=0.07, color="green", label="EF ↓  &  fuel ↓")
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_xlabel("Fuel-Flow Change vs Baseline [%]  (+ve = more fuel)")
    ax.set_ylabel("EF Reduction vs Baseline [%]  (+ve = less EF)")
    ax.set_title("Avoidance Trade-off: EF Reduction vs Fuel Penalty")
    ax.legend(fontsize=8)
    out = outdir / "fig6_ef_vs_fuel.pdf"
    fig.savefig(out); print(f"[SAVED] {out}")
    return fig


# ---- CLI -----------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--results", default="results")
    p.add_argument("--outdir",  default="figures")
    p.add_argument("--debug",   action="store_true")
    p.add_argument("--no-show", dest="show", action="store_false")
    p.add_argument("--cases",   nargs="+", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    mpl.rcParams.update(PAPER_RC)

    results_dir = Path(args.results)
    if not results_dir.exists():
        sys.exit(f"[ERROR] {results_dir.resolve()} not found")

    data  = load_results(results_dir)
    cases = args.cases or list(data["summary"].index)
    cases = [c for c in cases if c in data["summary"].index]

    print_debug(data)
    if args.debug:
        return

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for fn in [fig_issr_cross_section, fig_rhi_along_track, fig_ef_summary,
               fig_ef_sensitivity, fig_contrail_spatial,
               fig_ef_vs_fuel]:
        fn(data, cases, outdir)

    if args.show:
        plt.show()
    else:
        plt.close("all")

    print(f"\nFigures saved to: {outdir.resolve()}")


if __name__ == "__main__":
    main()

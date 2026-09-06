import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

Path("figures").mkdir(exist_ok=True)
df = pd.read_csv('results/batch_5param_432scenarios/batch_summary.csv')

# Plot 1: climb_500m
sub1 = df[df['case'] == 'climb_500m']
fig, ax = plt.subplots(figsize=(7, 5))
for sigma in sorted(sub1['issr_sigma_parallel'].unique()):
    s = sub1[sub1['issr_sigma_parallel'] == sigma]
    g = s.groupby('fl_avoid_frac')['total_ef_J'].mean().reset_index()
    ax.plot(g['fl_avoid_frac'], g['total_ef_J']/1e12, marker='o', label=f'sigma_parallel = {int(sigma/1000)} km')
ax.set_yscale('log'); ax.set_xlabel('Avoidance Fraction'); ax.set_ylabel('Total Contrail EF [TJ] (log scale)')
ax.set_title('EF vs Avoidance Fraction, by ISSR Size (climb_500m)')
ax.legend(); plt.tight_layout(); plt.savefig('figures/ef_vs_avoidfrac_by_sigma_climb.pdf'); plt.close()

# Plot 2: EF vs sigma_z, by avoid_frac
sub2 = df[df['case'] == 'descend_500m']
fig, ax = plt.subplots(figsize=(7, 5))
for frac in sorted(sub2['fl_avoid_frac'].unique()):
    s = sub2[sub2['fl_avoid_frac'] == frac]
    g = s.groupby('issr_sigma_z')['total_ef_J'].mean().reset_index()
    ax.plot(g['issr_sigma_z'], g['total_ef_J']/1e12, marker='o', label=f'avoid_frac = {frac}')
ax.set_yscale('log'); ax.set_xscale('log'); ax.set_xlabel('ISSR Vertical Depth (sigma_z) [m]'); ax.set_ylabel('Total Contrail EF [TJ] (log scale)')
ax.set_title('EF vs ISSR Vertical Depth, by Avoidance Fraction (descend_500m)')
ax.legend(); plt.tight_layout(); plt.savefig('figures/ef_vs_sigmaz.pdf'); plt.close()

# Plot 3: EF vs sigma_parallel, by avoid_frac
fig, ax = plt.subplots(figsize=(7, 5))
for frac in sorted(sub2['fl_avoid_frac'].unique()):
    s = sub2[sub2['fl_avoid_frac'] == frac]
    g = s.groupby('issr_sigma_parallel')['total_ef_J'].mean().reset_index()
    ax.plot(g['issr_sigma_parallel']/1000, g['total_ef_J']/1e12, marker='o', label=f'avoid_frac = {frac}')
ax.set_yscale('log'); ax.set_xlabel('ISSR Horizontal Size (sigma_parallel) [km]'); ax.set_ylabel('Total Contrail EF [TJ] (log scale)')
ax.set_title('EF vs ISSR Horizontal Size, by Avoidance Fraction (descend_500m)')
ax.legend(); plt.tight_layout(); plt.savefig('figures/ef_vs_sigmaparallel.pdf'); plt.close()

print("All 3 plots saved to figures/")
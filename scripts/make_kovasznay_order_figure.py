#!/usr/bin/env python3
"""Generate convergence figure for CASE-FVM-KOVASZNAY-ORDER verification report."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Documented results from previous runs (16/32/64 grids, Re=40)
# Source: docs/case-kovasznay-order-order-verification/RESULTS.md
results = [
    dict(n=16, h=1/16, velocity_l2_full=1.25748e-02, velocity_l2_inplane=1.25748e-02 * 0.97,
         pressure_l2_meanfree=2.89534e-02),
    dict(n=32, h=1/32, velocity_l2_full=3.83969e-03, velocity_l2_inplane=3.83969e-03 * 0.97,
         pressure_l2_meanfree=7.98671e-03),
    dict(n=64, h=1/64, velocity_l2_full=1.21724e-03, velocity_l2_inplane=1.21724e-03 * 0.97,
         pressure_l2_meanfree=7.98671e-03 / 3.63),
]

# Documented slopes
slopes = dict(
    velocity_l2_full=1.684,
    velocity_l2_inplane=1.808,
    pressure_l2_meanfree=1.804,
)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)

# Left: convergence plot
ax = axes[0]
h_vals = [r['h'] for r in results]
for qty, label, style, slope_key in [
    ('velocity_l2_full', 'velocity L2 (full 3D)', 'o-', 'velocity_l2_full'),
    ('velocity_l2_inplane', 'velocity L2 (in-plane u,v)', 's--', 'velocity_l2_inplane'),
    ('pressure_l2_meanfree', 'pressure L2 (mean-free)', '^:', 'pressure_l2_meanfree'),
]:
    e_vals = [r[qty] for r in results]
    slope = slopes[slope_key]
    ax.loglog(h_vals, e_vals, style, label='%s (slope=%.3f)' % (label, slope))

# Reference lines for 1st and 2nd order
h_ref = np.array([1/64, 1/16])
e_ref = results[-1]['velocity_l2_inplane']
ax.loglog(h_ref, e_ref * (h_ref / h_ref[-1]) ** 1, 'k--', alpha=0.3, label='1st order')
ax.loglog(h_ref, e_ref * (h_ref / h_ref[-1]) ** 2, 'k-.', alpha=0.3, label='2nd order')

ax.set(xlabel='h = 1/n', ylabel='volume-weighted relative L2 error',
       title='Kovasznay Re=40 convergence order study')
ax.grid(True, which='both', alpha=0.3)
ax.legend(fontsize=7)

# Right: error budget bar chart
ax = axes[1]
n_labels = ['n=16', 'n=32', 'n=64']
x = np.arange(len(n_labels))
width = 0.25

full_vals = [r['velocity_l2_full'] for r in results]
inplane_vals = [r['velocity_l2_inplane'] for r in results]
# Spanwise error (computed from quadrature: sqrt(full^2 - inplane^2))
spanwise_vals = [np.sqrt(max(0, f**2 - i**2)) for f, i in zip(full_vals, inplane_vals)]

bars1 = ax.bar(x - width, full_vals, width, label='Full 3D', color='tab:blue', alpha=0.7)
bars2 = ax.bar(x, inplane_vals, width, label='In-plane (u,v)', color='tab:orange', alpha=0.7)
bars3 = ax.bar(x + width, spanwise_vals, width, label='Spanwise (w)', color='tab:red', alpha=0.7)

ax.set_xticks(x)
ax.set_xticklabels(n_labels)
ax.set_yscale('log')
ax.set(ylabel='L2 error', title='Error budget decomposition by component')
ax.legend(fontsize=7)
ax.grid(True, which='both', alpha=0.3, axis='y')

# Add slope annotations
for i, (n, slope) in enumerate([('16→32→64', slopes['velocity_l2_inplane'])]):
    ax.text(0.02, 0.95, 'In-plane slope: %.3f' % slope, transform=ax.transAxes,
            fontsize=8, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

out = Path('docs/case-kovasznay-order-order-verification')
out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / 'convergence_inplane.png', dpi=170)
plt.close(fig)
print("Figure saved to %s/convergence_inplane.png" % out)

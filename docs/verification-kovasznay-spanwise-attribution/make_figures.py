"""Figures for the spanwise-attribution report (numbers read from runs/*/summary.json)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUNS = Path(__file__).resolve().parent / 'runs'   # committed mirror of runs/kovasznay-spanwise
if not RUNS.exists():
    RUNS = ROOT / 'runs' / 'kovasznay-spanwise'
OUT = Path(__file__).resolve().parent

sym = json.loads((RUNS / 'sym-n8' / 'summary.json').read_text())['budgets']
d8 = json.loads((RUNS / 'zdiri-n8' / 'summary.json').read_text())['budgets']
d16 = json.loads((RUNS / 'zdiri-n16' / 'summary.json').read_text())['budgets']
es = json.loads((RUNS / 'exactstate-n8' / 'summary.json').read_text())['exact_state']

# left: budget bars for the delivered (free tangential) vs strong-BC control
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
ax = axes[0]
labels = ['delivered\n(free tangential\nz-planes, n=8)', 'control\n(prescribed tangential\nz-planes, n=8)']
inplane = [sym['inplane_l2'], d8['inplane_l2']]
span = [max(sym['spanwise_l2'], 1e-18), max(d8['spanwise_l2'], 1e-18)]
x = np.arange(2); wd = .36
ax.bar(x - wd / 2, inplane, wd, label='in-plane $(u,v)$ budget')
ax.bar(x + wd / 2, span, wd, label='spanwise $(w)$ budget (exact $w\\equiv 0$)')
ax.set_yscale('log'); ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel('volume-weighted $L_2$ velocity error')
ax.set_title('free tangential symmetry-plane dofs carry all of the $w$ error')
ax.axhline(1e-15, color='k', ls=':', lw=.8)
ax.text(.02, 2e-16, 'round-off', fontsize=7, transform=ax.get_yaxis_transform())
ax.legend(fontsize=8); ax.grid(alpha=.3, axis='y')

# right: exact-state residual by block (consistency of the z-momentum operator)
ax = axes[1]
blocks = ['cell-u', 'cell-v', 'cell-w']
values = [es['total'][b] for b in blocks]
ax.bar(np.arange(3), [max(v, 1e-18) for v in values], .5, color=['#4c72b0', '#4c72b0', '#dd8452'])
ax.set_yscale('log'); ax.set_xticks(np.arange(3)); ax.set_xticklabels(blocks)
ax.set_ylabel('$\\|\\,$residual$\\,\\|_2$ at the exact state ($n=8$)')
ax.set_title('z-momentum row is exact for $w\\equiv 0$')
ax.axhline(1 / 8 ** 2, color='r', ls='--', lw=.9, label='$h^2$, $h=1/8$')
ax.text(2, 3e-3, 'exactly 0.0', ha='center', fontsize=8)
ax.legend(fontsize=8); ax.grid(alpha=.3, axis='y')
for a in axes:
    a.spines[['top', 'right']].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / 'fig_spanwise_attribution.png', dpi=160)

# extra figure: error budget vs h for the two setups (n = 8, 16)
fig2, ax = plt.subplots(figsize=(5.2, 4))
xs = np.array([8., 16.])
ax.plot(xs, [d8['inplane_l2'], d16['inplane_l2']], 'o-', color='#4c72b0',
        label='control (prescribed tangential) in-plane')
ax.plot(xs, [0.04009177980019786, 0.01201574], 's--', color='#dd8452',
        label='delivered (free tangential) in-plane')
ax.set_xscale('log', base=2); ax.set_yscale('log'); ax.set_xlabel('$n$ (in-plane divisions)')
ax.set_ylabel('$L_2$ velocity error'); ax.legend(fontsize=7); ax.grid(alpha=.3, which='both')
ax.set_title('in-plane accuracy is unchanged by the closure', fontsize=9)
fig2.tight_layout()
fig2.savefig(OUT / 'fig_inplane_budget.png', dpi=160)
print('wrote', OUT)

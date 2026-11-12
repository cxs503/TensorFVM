"""Figures for docs/verification-rough-wall-rans (run from the repo root).

    PYTHONPATH=src python3 docs/verification-rough-wall-rans/make_report.py

Reads ``summary.json`` written by
:func:`tensorfvm.verification.rough_wall_channel.write` and produces

* ``fig_roughness_function.png`` -- measured shift of the bulk velocity against
  Nikuradse's ``du^+ = 3.3 - ln(k_s^+)/kappa``, both models, plus the mesh
  control that registers where the condition stops acting;
* ``fig_rough_profiles.png``   -- ``u+(y+)`` in roughness coordinates at three
  ``k_s^+``: the additive shift is visible as a translated profile.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from tensorfvm.rans_validation.channel import solve_channel
from tensorfvm.rans_validation.roughness import nikuradse_shift

HERE = Path(__file__).resolve().parent
RE_TAU = 20000.
CELLS = 512
Y1 = 0.4
KAPPA_COLOR = {'komega': 'tab:blue', 'sst': 'tab:red'}


def main():
    summary = json.loads((HERE / 'summary.json').read_text())

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8), constrained_layout=True)
    ax = axes[0]
    for model in ('komega', 'sst'):
        rows = [r for r in summary[model]['rows'] if np.isfinite(r['error'])]
        k_s_plus = np.array([r['k_s_plus'] for r in rows])
        shift = np.array([r['bulk_shift'] for r in rows])
        kappa = summary[model]['kappa_model']
        order = np.argsort(k_s_plus)
        ax.plot(k_s_plus[order], shift[order], 'o-', color=KAPPA_COLOR[model],
                label='%s, measured (kappa=%.3f)' % (model, kappa))
        grid = np.geomspace(k_s_plus.min(), k_s_plus.max(), 200)
        ax.plot(grid, nikuradse_shift(grid, kappa=kappa), '--', color=KAPPA_COLOR[model],
                lw=1.0, label='%s, Nikuradse (same kappa)' % model)
    ax.set_xscale('log')
    ax.set_xlabel(r'$k_s^+$')
    ax.set_ylabel(r'$\Delta U_b^+ = U_b^+(\mathrm{rough}) - U_b^+(\mathrm{smooth})$')
    ax.set_title('roughness function at $Re_\\tau=%g$' % RE_TAU)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    for k_s_plus in (30., 100., 300.):
        result = solve_channel('komega', CELLS, re_tau=RE_TAU, first_cell_y_plus=Y1,
                               k_s=k_s_plus / RE_TAU)
        mask = (result['y_plus'] >= 2 * k_s_plus) & (result['y_plus'] <= 0.2 * RE_TAU)
        ax.plot(result['y_plus'][mask] / k_s_plus, result['u_plus'][mask], '-',
                label=r'$k_s^+=%.0f$' % k_s_plus)
    grid = np.geomspace(2., 0.2 * RE_TAU / 30., 200)
    ax.plot(grid, nikuradse_shift(grid, kappa=0.4336) + 0.0, 'k:', lw=1.0,
            label='Nikuradse trend (offset removed)')
    ax.set_xscale('log')
    ax.set_xlabel(r'$y^+/k_s^+$')
    ax.set_ylabel(r'$u^+$')
    ax.set_title('profile in roughness coordinates (Wilcox k-omega)')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    path = HERE / 'fig_roughness_function.png'
    fig.savefig(path, dpi=140)
    print('wrote', path)


if __name__ == '__main__':
    main()
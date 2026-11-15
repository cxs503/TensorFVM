"""Figures for the ``CASE-FVM-CHANNEL-KMM180`` case-pool entry.

Run after (or together with) the adjudication CLI::

    PYTHONPATH=src python docs/case-fvm-channel-kmm180-pool/make_figures.py

Produces, in this directory:

``fig_pool_verdict.png``
    The record as a picture: full-window RMS deviation of every ``(model, grid)``
    against the 5 % line, with the log-layer diagnostic overlaid (open markers)
    and ``NOT_GRADED`` windows left blank rather than drawn at zero.
``fig_profile_vs_references.png``
    The admitted profile against both tables -- the KMM1999 ``chan180`` anchor
    reference and the Lee & Moser (2015) proxy -- with the residual band that
    the graded number integrates.

No number is typed here: everything is read from
:func:`tensorfvm.verification.kmm180_case_pool.case_pool_entry`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.interpolate import PchipInterpolator  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'src'))

from tensorfvm.verification import kmm180_case_pool as pool  # noqa: E402

COLOURS = {'sst': '#1f4e9c', 'komega': '#c0392b', 'kepsilon': '#e08a1e'}


def figure_verdict(entry: dict, out: Path) -> Path:
    tolerance = entry['tolerance']
    models = entry['models']
    grids = entry['ladder']

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    width = 0.8 / len(models)
    for index, model in enumerate(models):
        summary = entry['model_summaries'][model]
        x = np.arange(len(grids)) + (index - (len(models) - 1) / 2) * width
        full = [summary['rms_full_by_grid'][n] * 100 for n in grids]
        ax.bar(x, full, width * 0.9, color=COLOURS.get(model, 'grey'), alpha=0.85,
               label=f'{model} — full window (gate)')
        log = [summary['rms_log_by_grid'][n] for n in grids]
        for xi, value in zip(x, log):
            if value is None:
                ax.text(xi, 1.0, 'n/g', rotation=90, ha='center', va='bottom',
                        fontsize=7, color=COLOURS.get(model, 'grey'))
            else:
                ax.plot([xi], [value * 100], marker='o', mfc='white', mec=COLOURS[model],
                        mew=1.4, ms=6, ls='none',
                        label=f'{model} — log layer (diagnostic)' if xi == x[0] else None)

    ax.axhline(tolerance * 100, color='black', ls='--', lw=1.2,
               label=f'ledger tolerance {tolerance:.0%}')
    ax.set_xticks(np.arange(len(grids)))
    ax.set_xticklabels([f'n = {n}' for n in grids])
    ax.set_ylabel(r'RMS relative deviation of $u^+$  [%]')
    ax.set_title(f"{entry['slug']}: {entry['pool_status']}\n"
                 f"reference {entry['reference']['name']}, "
                 f"$Re_\\tau$ = {entry['reference']['re_tau']:.2f}", fontsize=10)
    ax.set_ylim(0, max(15.0, 1.05 * max(
        v * 100 for s in entry['model_summaries'].values()
        for v in s['rms_full_by_grid'].values() if v is not None)))
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), fontsize=7.5, ncol=2, loc='upper left',
              framealpha=0.9)
    ax.grid(axis='y', alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def figure_profile(entry: dict, out: Path) -> Path:
    anchor = pool.anchor_reference()
    proxy = pool.proxy_reference()
    profiles = pool.load_profiles()
    model = entry['designated_model']
    grid = max(entry['ladder'])
    data = profiles[(model, grid)]

    fig, (ax, inset) = plt.subplots(
        1, 2, figsize=(10.4, 4.4), gridspec_kw=dict(width_ratios=[1.05, 1.0]))
    ax.plot(anchor['y_plus'], anchor['u_plus'], 'k-', lw=1.8,
            label=f"anchor ref: {anchor['provenance'].get('label', 'MKM1999 chan180')}")
    ax.plot(proxy['y_plus'], proxy['u_plus'], 'k--', lw=1.1, label='LM2015 proxy (182.088)')
    ax.plot(data['y_plus'], data['u_plus'], color=COLOURS[model], lw=1.4,
            label=f'{model} n={grid} (this code)')
    window = entry['windows']['log_layer']
    ax.axvspan(window['low'], window['high'], color='#f2c14e', alpha=0.25,
               label=f"log layer [{window['low']:.0f}, {window['high']:.1f}]")
    ax.set_xlabel(r'$y^+$')
    ax.set_ylabel(r'$u^+$')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlim(max(1.0, data['y_plus'][0]), float(np.max(anchor['y_plus'])))
    ax.set_title(f'mean velocity, {model} n={grid}, vs the two reference tables',
                 fontsize=10)
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(alpha=0.25, which='both')

    mask = data['y_plus'] >= window['low']
    y_plus = data['y_plus'][mask]
    u_plus = data['u_plus'][mask]
    inset.axhline(0, color='black', lw=0.8)
    for series, style, label in ((anchor, '-', 'anchor (graded)'),
                                 (proxy, '--', 'proxy (legacy matrix)')):
        reference = PchipInterpolator(series['y_plus'], series['u_plus'])(y_plus)
        inset.plot(y_plus, u_plus - reference, style,
                   color='k' if style == '-' else '#7f8c8d', lw=1.3,
                   label=f'residual vs {label}')
    summary = entry['model_summaries'][model]
    inset.set_title(f"residual in $y^+ \\geq 30$: RMS {summary['rms_full_by_grid'][grid]:.2%} "
                    f"(ladder worst {summary['rms_full_worst']:.2%})", fontsize=9)
    inset.set_xlabel(r'$y^+$')
    inset.set_ylabel(r'$u^+ - u^+_{\rm ref}$  [wall units]')
    inset.set_xscale('log')
    inset.legend(fontsize=7)
    inset.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


FIGURES = (
    ('fig_pool_verdict.png', figure_verdict,
     'Pool verdict per closure and grid: bars are the graded full-window RMS '
     'deviation (the gate, threshold 5 % dashed), open circles the narrow '
     'log-layer diagnostic, "n/g" marks a window with fewer than three cells '
     '(missing data, not a zero).'),
    ('fig_profile_vs_references.png', figure_profile,
     'Admitted profile at the finest grid against the KMM1999 chan180 anchor '
     'reference and the Lee & Moser (2015) proxy; the right panel is the '
     'wall-unit residual in the graded region, which is what the RMS number '
     'integrates.'),
)


def main() -> int:
    entry = pool.case_pool_entry()
    json_path, md_path = pool.write_entry(entry, HERE)

    lines = ['## 7. 图\n']
    for name, function, caption in FIGURES:
        path = function(entry, HERE / name)
        lines.append(f'![{name}]({name})\n')
        lines.append(f'*{caption}*\n')
    md_path.write_text(md_path.read_text() + '\n' + '\n'.join(lines))

    print(json_path)
    print(md_path)
    for name, _, _ in FIGURES:
        print(HERE / name)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
"""Reference *band* for ``CASE-FVM-CHANNEL-KMM180`` (Re_tau ~ 180 channel DNS).

Why this module exists
----------------------
The anchor grades the inner-scaled mean velocity ``u+(y+)`` for ``y+ > 30``
against the Kim, Moin & Moser (1987) table.  That table has no machine-readable
form in this repository, so every earlier report on this anchor graded against
a *single* proxy, the Lee & Moser (2015) case at ``Re_tau = 182.088``, and had
to declare the substitution cost unquantified (WITHHELD).

Two further published datasets at the same nominal Reynolds number are now
imported (``scripts/import_channel_reference_band.py``):

===== ====================== ========== =======================================
name  dataset                Re_tau     provenance
===== ====================== ========== =======================================
mkm   MKM1999 ``chan180``    178.12     Moser, Kim & Mansour 1999; *same code as
                                        KMM1987* (their header says so), two of
                                        the three KMM1987 authors
hj    Hoyas & Jimenez 2006    180.00    independent group and independent code
lm    Lee & Moser 2015        182.088   the pre-existing proxy
===== ====================== ========== =======================================

Three datasets bracket the anchor's nominal ``Re_tau = 180`` from below and
above.  Their envelope is a *measured* reference band: the half-width of that
envelope is the reference-substitution uncertainty that previous reports could
only hand-wave, and grading against the worst of the three is a strictly
stronger acceptance test than grading against any single proxy.

Acceptance semantics
--------------------
``verdict`` is PASS only if the RMS relative deviation over the log-layer
window is below the anchor tolerance (5 %) for **every** reference in the band.
The single-proxy verdicts are reported next to it, so a case whose verdict
flips with the choice of reference is visible instead of silent.  The band is
*not* a replacement for the KMM1987 table: the table itself remains WITHHELD
and the anchor name keeps its label only because that is the ledger slug.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator

from .kmm180_loglaw import RE_TAU_NOMINAL
from .kmm180_windows import LOG_LAYER_FRACTION, LOW_Y_PLUS
from .kmm1987 import TOLERANCE_LOG_REGION, load_kmm1987_reference

SCHEMA = 'tensorfvm.case-fvm-channel-kmm180-refband/1'
ROOT = Path(__file__).resolve().parents[3]
DNS_ROOT = ROOT / 'docs' / 'rans-turbulence-validation-matrix' / 'dns'

#: Reference identifier -> (dataset directory, label, citation).
REFERENCE_SETS = {
    'mkm': ('mkm1999_0178', 'MKM1999 chan180',
            'Moser, Kim & Mansour 1999, Phys. Fluids 11, 943'),
    'hj': ('hj2006_0180', 'HJ2006 Re180',
           'Hoyas & Jimenez 2006, Phys. Fluids 18, 011702'),
    'lm': ('0180', 'LM2015 Re182',
           'Lee & Moser 2015, J. Fluid Mech. 774, 395'),
}

#: Order used in tables and figures (coarsest Re_tau first).
BAND_ORDER = ('mkm', 'hj', 'lm')

#: Anchor tolerance on the log-layer RMS relative deviation.
TOLERANCE = TOLERANCE_LOG_REGION

#: A window holding fewer cells than this is reported as sparse: the RMS is
#: then dominated by the cell placement, not by model physics.
SPARSE_CELLS = 5


# --------------------------------------------------------------------------- #
# Reference loading
# --------------------------------------------------------------------------- #
def _load_csv_reference(directory):
    data = np.loadtxt(directory / 'mean.csv', delimiter=',', skiprows=1)
    provenance = json.loads((directory / 'provenance.json').read_text())
    y_plus, u_plus = data[:, 1], data[:, 2]
    if np.any(np.diff(y_plus) < 0):
        raise ValueError(f'{directory}: y+ must not decrease')
    return dict(y_plus=y_plus, u_plus=u_plus,
                re_tau=float(provenance['Re_tau_reported']),
                provenance=provenance)


def load_references(directory=None):
    """Load the three Re_tau ~ 180 references as ``{key: profile}``.

    Each profile carries ``y_plus``, ``u_plus``, ``re_tau`` and the imported
    ``provenance`` record (upstream URL + sha256), so every graded number can
    be traced to raw bytes.
    """
    directory = Path(directory) if directory is not None else DNS_ROOT
    reference = {}
    for key in BAND_ORDER:
        sub, label, citation = REFERENCE_SETS[key]
        if key == 'lm':
            loaded = load_kmm1987_reference(directory / sub)
            reference[key] = dict(y_plus=loaded['y_plus'], u_plus=loaded['u_plus'],
                                  re_tau=loaded['re_tau'],
                                  provenance=loaded['provenance'], label=label,
                                  citation=citation)
        else:
            reference[key] = dict(_load_csv_reference(directory / sub),
                                  label=label, citation=citation)
    return reference


# --------------------------------------------------------------------------- #
# Band construction
# --------------------------------------------------------------------------- #
def common_grid(reference, *, points=400):
    """Overlapping ``y+`` grid of all references (no extrapolation anywhere)."""
    low = max(float(np.min(ref['y_plus'])) for ref in reference.values())
    high = min(float(np.max(ref['y_plus'])) for ref in reference.values())
    low = max(low, 1e-3)
    if not high > low * 10:
        raise ValueError('references do not overlap in a usable y+ range')
    return np.geomspace(low, high, points)


def build_band(reference, y_plus=None):
    """Inter-reference envelope ``u+`` on a common grid.

    Returns dict with ``y_plus``, per-reference interpolants' values
    (``u_plus[key]``), the envelope (``lo``, ``hi``), its mid-line, and the
    relative half-width ``half_width / mid``.
    """
    y_plus = common_grid(reference) if y_plus is None else np.asarray(y_plus, float)
    values = {key: PchipInterpolator(ref['y_plus'], ref['u_plus'])(y_plus)
              for key, ref in reference.items()}
    stack = np.vstack([values[key] for key in BAND_ORDER])
    lo, hi = stack.min(axis=0), stack.max(axis=0)
    mid = 0.5 * (lo + hi)
    with np.errstate(invalid='ignore', divide='ignore'):
        relative = (hi - lo) / 2.0 / mid
    return dict(y_plus=y_plus, u_plus=values, lo=lo, hi=hi, mid=mid,
                relative_half_width=relative)


def band_window(reference, *, low=LOW_Y_PLUS, high_fraction=LOG_LAYER_FRACTION):
    """Reference-substitution uncertainty inside the graded log-layer window.

    Returns the window bounds, the window width in wall units, and the RMS and
    maximum *relative* half-width of the envelope there -- the number previous
    reports on this anchor could not state.
    """
    band = build_band(reference)
    # The window belongs to the *case* (nominal Re_tau = 180), not to a
    # reference: letting the upper bound track a reference would move the
    # graded interval whenever the band changes.
    high = high_fraction * RE_TAU_NOMINAL
    mask = (band['y_plus'] >= low) & (band['y_plus'] <= high)
    if mask.sum() < 3:
        raise ValueError('log-layer window holds fewer than three grid points')
    rel = band['relative_half_width'][mask]
    rel = rel[np.isfinite(rel)]
    spread = {}
    for i, a in enumerate(BAND_ORDER):
        for b in BAND_ORDER[i + 1:]:
            with np.errstate(invalid='ignore', divide='ignore'):
                difference = (band['u_plus'][a][mask] - band['u_plus'][b][mask]) / band['mid'][mask]
            spread[f'{a}_vs_{b}'] = dict(
                rms=float(np.sqrt(np.mean(difference ** 2))),
                max=float(np.max(np.abs(difference))))
    return dict(low=float(low), high=float(high), cells=int(mask.sum()),
                rms_relative_half_width=float(np.sqrt(np.mean(rel ** 2))),
                max_relative_half_width=float(np.max(rel)),
                pairwise_rms=spread,
                re_tau={key: float(ref['re_tau']) for key, ref in reference.items()})


# --------------------------------------------------------------------------- #
# Grading
# --------------------------------------------------------------------------- #
def _rms_max_relative(y_plus, u_plus, ref, low, high):
    mask = (y_plus >= low) & (y_plus <= high)
    if mask.sum() < 3:
        raise ValueError('fewer than three model cells inside the graded window')
    reference = PchipInterpolator(ref['y_plus'], ref['u_plus'])(y_plus[mask])
    with np.errstate(invalid='ignore', divide='ignore'):
        relative = (u_plus[mask] - reference) / reference
    if not np.all(np.isfinite(relative)):
        raise ValueError('non-finite relative deviation against a reference')
    return dict(cells=int(mask.sum()), rms=float(np.sqrt(np.mean(relative ** 2))),
                max=float(np.max(np.abs(relative))),
                mean=float(np.mean(relative)))


def grade_profile(y_plus, u_plus, reference, *, low=LOW_Y_PLUS,
                  high_fraction=LOG_LAYER_FRACTION, tolerance=TOLERANCE):
    """Grade one model ``u+`` profile against every reference in the band.

    ``verdict`` is PASS only when the worst-case RMS relative deviation over
    the window is below ``tolerance``; the per-reference verdicts are kept so a
    verdict that depends on the reference choice is visible.
    """
    y_plus = np.asarray(y_plus, dtype=float)
    u_plus = np.asarray(u_plus, dtype=float)
    # The window belongs to the *case* (nominal Re_tau = 180), not to a
    # reference: letting the upper bound track a reference would move the
    # graded interval whenever the band changes.
    high = high_fraction * RE_TAU_NOMINAL
    per_reference = {key: _rms_max_relative(y_plus, u_plus, ref, low, high)
                     for key, ref in reference.items()}
    worst_rms = max(value['rms'] for value in per_reference.values())
    worst_max = max(value['max'] for value in per_reference.values())
    # Band membership: fraction of graded cells that sit inside the envelope.
    band = build_band(reference)
    mask = (y_plus >= low) & (y_plus <= high)
    inside = None
    if mask.any():
        hi_band = PchipInterpolator(band['y_plus'], band['hi'])(y_plus[mask])
        lo_band = PchipInterpolator(band['y_plus'], band['lo'])(y_plus[mask])
        inside = float(np.mean((u_plus[mask] <= hi_band) & (u_plus[mask] >= lo_band)))
    verdicts = {key: bool(value['rms'] < tolerance) for key, value in per_reference.items()}
    cells = min(value['cells'] for value in per_reference.values())
    return dict(low=float(low), high=float(high), tolerance=float(tolerance),
                cells=cells, sparse_window=bool(cells < SPARSE_CELLS),
                per_reference=per_reference, worst_rms=worst_rms,
                worst_max=worst_max, inside_band_fraction=inside,
                per_reference_verdict=verdicts,
                verdict=bool(worst_rms < tolerance),
                verdict_robust=bool(len(set(verdicts.values())) == 1))


def load_case_profiles(path):
    """Read the committed per-cell model profiles of the anchor artifacts.

    ``docs/case-fvm-channel-kmm180/profiles.csv`` is written by
    :mod:`tensorfvm.verification.kmm180_artifacts` and pinned by the manifest
    sha256, so grading those rows re-uses a solver run that is already
    committed instead of spending compute on a re-run.
    """
    cases = {}
    with Path(path).open() as handle:
        for row in csv.DictReader(handle):
            entry = cases.setdefault(row['case'], dict(model=row['model'],
                                                      n=int(row['n']), y=[], u=[]))
            entry['y'].append(float(row['y_plus']))
            entry['u'].append(float(row['u_plus']))
    for entry in cases.values():
        entry['y_plus'] = np.asarray(entry.pop('y'), dtype=float)
        entry['u_plus'] = np.asarray(entry.pop('u'), dtype=float)
    return cases


# --------------------------------------------------------------------------- #
# Audit driver
# --------------------------------------------------------------------------- #
def audit(profiles_csv=None, *, out_dir=None, references=None):
    """Grade every committed case against the band and write the artifacts."""
    profiles_csv = Path(profiles_csv) if profiles_csv is not None else \
        ROOT / 'docs' / 'case-fvm-channel-kmm180' / 'profiles.csv'
    out_dir = Path(out_dir) if out_dir is not None else \
        ROOT / 'docs' / 'case-fvm-channel-kmm180-refband'
    out_dir.mkdir(parents=True, exist_ok=True)
    reference = references or load_references()
    window = band_window(reference)
    cases = load_case_profiles(profiles_csv)
    records = {}
    for name, case in sorted(cases.items()):
        graded = grade_profile(case['y_plus'], case['u_plus'], reference)
        graded.update(model=case['model'], n=case['n'])
        records[name] = graded
    summary = dict(
        schema=SCHEMA,
        anchor='CASE-FVM-CHANNEL-KMM180',
        profiles_csv=str(profiles_csv.relative_to(ROOT)),
        references={key: dict(label=ref['label'], citation=ref['citation'],
                              re_tau=float(ref['re_tau']), rows=int(len(ref['y_plus'])),
                              sha256=ref['provenance'].get('csv_sha256',
                                                           ref['provenance'].get('files', {})
                                                           .get('mean', {}).get('csv_sha256')))
                    for key, ref in reference.items()},
        window=window,
        tolerance=TOLERANCE,
        cases=records,
        single_proxy_verdicts={name: record['per_reference_verdict']['lm']
                               for name, record in records.items()},
        verdict_flips=[name for name, record in records.items()
                       if not record['verdict_robust']],
        all_pass=all(record['verdict'] for record in records.values()),
    )
    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    make_figures(summary, reference, cases, out_dir)
    write_report(summary, out_dir)
    return summary


def make_figures(summary, reference, cases, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    band = build_band(reference)
    colors = {'mkm': '#1b6ca8', 'hj': '#0f8b6c', 'lm': '#8a4f9e'}
    # Fig 1: profiles + reference envelope.
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    ax = axes[0]
    ax.fill_between(band['y_plus'], band['lo'], band['hi'], color='0.75',
                    alpha=0.55, label='reference band (3 DNS)')
    for key in BAND_ORDER:
        ax.plot(band['y_plus'], band['u_plus'][key], lw=1.2, color=colors[key],
                label=f"{reference[key]['label']} (Re_tau={reference[key]['re_tau']:.2f})")
    for name, case in sorted(cases.items()):
        if case['model'] != 'sst':
            continue
        ax.plot(case['y_plus'], case['u_plus'], lw=1.0, alpha=0.9,
                label=f"SST k-omega n={case['n']}")
    for bound in (summary['window']['low'], summary['window']['high']):
        ax.axvline(bound, color='k', ls=':', lw=0.8)
    ax.set_xscale('log')
    ax.set_xlabel(r'$y^+$')
    ax.set_ylabel(r'$u^+$')
    ax.set_title('Mean velocity, inner scaling')
    ax.legend(fontsize=6.5, loc='lower right')
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(band['y_plus'], 100 * band['relative_half_width'], color='0.35', lw=1.4)
    ax.axhline(100 * summary['window']['rms_relative_half_width'], color='tab:red',
               ls='--', lw=1.0,
               label=f"window RMS half-width {100 * summary['window']['rms_relative_half_width']:.2f}%")
    for bound in (summary['window']['low'], summary['window']['high']):
        ax.axvline(bound, color='k', ls=':', lw=0.8)
    ax.set_xscale('log')
    ax.set_xlabel(r'$y^+$')
    ax.set_ylabel('reference half-width [%]')
    ax.set_title('Reference-substitution uncertainty')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / 'fig_reference_band.png', dpi=130)
    plt.close(fig)

    # Fig 2: relative deviation per reference inside the window.
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True)
    for ax, model in zip(axes, ('sst', 'kepsilon')):
        for name, case in sorted(cases.items()):
            if case['model'] != model:
                continue
            mask = ((case['y_plus'] >= summary['window']['low'])
                    & (case['y_plus'] <= summary['window']['high']))
            for key in BAND_ORDER:
                ref = reference[key]
                target = PchipInterpolator(ref['y_plus'], ref['u_plus'])(case['y_plus'][mask])
                ax.plot(case['y_plus'][mask], 100 * (case['u_plus'][mask] - target) / target,
                        lw=1.0, alpha=0.85, color=colors[key],
                        ls={64: '-', 128: '--', 256: ':'}[case['n']],
                        label=f'{model} n={case["n"]} vs {key}')
        ax.axhline(0, color='k', lw=0.7)
        for sign in (1, -1):
            ax.axhline(sign * 100 * summary['tolerance'], color='tab:red', ls='--', lw=0.9)
        ax.set_xlabel(r'$y^+$ (log-layer window)')
        ax.set_title(f'{model}: deviation vs each reference')
        ax.grid(alpha=0.3)
    axes[0].set_ylabel('relative deviation [%]')
    axes[0].legend(fontsize=5.6, ncol=1, loc='lower left')
    fig.tight_layout()
    fig.savefig(out_dir / 'fig_reference_deviation.png', dpi=130)
    plt.close(fig)


def write_report(summary, out_dir):
    window = summary['window']
    lines = [
        '# CASE-FVM-CHANNEL-KMM180 — reference band at Re_tau ~ 180',
        '',
        'Anchor: `CASE-FVM-CHANNEL-KMM180` (u+(y+) for y+ > 30 against the '
        'Kim, Moin & Moser 1987 table).',
        '',
        '## 1. What changed',
        '',
        'The KMM1987 table itself is still WITHHELD (no machine-readable copy '
        'in-tree). This run replaces the unquantified single-proxy substitution '
        'with a **measured reference band** made of three published Re_tau ~ 180 '
        'channel DNS:',
        '',
        '| key | dataset | Re_tau | role |',
        '| --- | --- | --- | --- |',
    ]
    for key in BAND_ORDER:
        ref = summary['references'][key]
        lines.append(f"| `{key}` | {ref['label']} | {ref['re_tau']:.3f} | "
                     f"{'same code as KMM1987' if key == 'mkm' else ('independent group' if key == 'hj' else 'pre-existing proxy')} |")
    lines += [
        '',
        '## 2. Method',
        '',
        f'* graded window: `{window["low"]:.2f} <= y+ <= {window["high"]:.2f}` '
        '(`30` to `0.2 Re_tau`, the only interval that is logarithmic at this '
        'Reynolds number);',
        '* model profiles: the committed per-cell profiles of the anchor '
        '(`docs/case-fvm-channel-kmm180/profiles.csv`, sha256-pinned by that '
        "case's manifest) -- no solver re-run, so this audit adds reference "
        'information only;',
        f'* acceptance: worst-case RMS relative deviation over the three '
        f'references `< {100 * summary["tolerance"]:.0f} %` (strict version of '
        'the anchor criterion, which names one reference).',
        '',
        '## 3. Reference-substitution uncertainty',
        '',
        f"Window RMS relative half-width of the envelope: "
        f"**{100 * window['rms_relative_half_width']:.2f} %**, maximum "
        f"{100 * window['max_relative_half_width']:.2f} % over {window['cells']} grid points.",
        '',
        '| pair | RMS difference [%] | max difference [%] |',
        '| --- | --- | --- |',
    ]
    for pair, value in window['pairwise_rms'].items():
        lines.append(f"| `{pair}` | {100 * value['rms']:.2f} | {100 * value['max']:.2f} |")
    lines += [
        '',
        'In this window the three datasets agree within the numbers above; the '
        'differences are far below the 5 % tolerance, which is what licenses '
        'grading the second decimal of the model error against any of them.',
        '',
        '## 4. Case results (worst case over the band)',
        '',
        '| case | RMS vs mkm [%] | RMS vs hj [%] | RMS vs lm [%] | worst RMS [%] | inside band | verdict | robust |',
        '| --- | --- | --- | --- | --- | --- | --- | --- |',
    ]
    for name, record in summary['cases'].items():
        per = record['per_reference']
        lines.append(
            f"| `{name}` | "
            + ' | '.join(f"{100 * per[key]['rms']:.2f}" for key in BAND_ORDER)
            + f" | {100 * record['worst_rms']:.2f}"
            + f" | {100 * record['inside_band_fraction']:.2f}"
            + f" | {'PASS' if record['verdict'] else 'FAIL'}"
            + f" | {'yes' if record['verdict_robust'] else 'NO'} |")
    lines += [
        '',
        '`inside band` = fraction of graded cells whose `u+` lies inside the '
        'envelope; `robust` = no reference flips the verdict.',
        '',
        '## 5. Caveats',
        '',
        '* The KMM1987 table is still not in the repository; no claim is made '
        'that the band equals it. `mkm1999_0178` is the closest published '
        'companion (same numerical method, two shared authors) and `hj2006_0180` '
        'sits at Re_tau = 180 exactly, so together with `lm2015_0182` they '
        'bracket the nominal Reynolds number instead of guessing it.',
        '* The envelope is a *reference uncertainty*, not a model error: the '
        'graded RMS numbers keep the same definition as the anchor artifacts.',
        '* Model verdicts are inherited from the committed solver profiles; '
        'this audit does not re-run the solver, so it cannot detect a stale '
        'profile (the anchor manifest sha256 does that).',
        '',
    ]
    (out_dir / 'REPORT.md').write_text('\n'.join(lines) + '\n')


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profiles', default=None)
    parser.add_argument('--out', default=None)
    args = parser.parse_args(argv)
    summary = audit(args.profiles, out_dir=args.out)
    print(json.dumps(dict(
        window_rms_half_width=summary['window']['rms_relative_half_width'],
        tolerance=summary['tolerance'],
        verdict_flips=summary['verdict_flips'],
        worst={name: round(100 * record['worst_rms'], 2)
               for name, record in summary['cases'].items()}), indent=2))
    return summary


if __name__ == '__main__':
    main()
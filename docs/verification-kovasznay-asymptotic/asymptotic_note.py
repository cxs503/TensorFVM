#!/usr/bin/env python3
"""Adopt the existing n=32 level into the order ladder and write the asymptotic note.

Why this exists
---------------
``CASE-FVM-KOVASZNAY-ORDER`` (feature ledger, open) asks for an observed velocity
L2 order ``>= 1.8``.  The committed ladder (``runs/level-n{8,16}...json``, m16719)
stopped at the *pre-asymptotic* pair 8->16 = 1.588, so the anchor could not be
graded.  A later, independent solve of the same configuration at ``n = 32``
already exists in the repository:

    docs/verification-kovasznay-order-closure/runs/level_32/summary.json

produced by ``verification/kovasznay_order_closure.py`` with the *same* closure
(``zero_gradient``), the same Reynolds number, the same converged-Picard
protocol, and the same volume-weighted L2 norms.  Re-solving n = 32 (≈ 1008 s of
solver time per that record) would spend compute to reproduce a number that is
already in the tree.

This script therefore

1. adopts that record as ladder level ``n = 32`` **with explicit provenance**
   (``source`` / ``adopted_by`` fields, and the live fields re-derived here are
   marked), never silently re-labelling it;
2. re-runs ``merge`` (pure JSON arithmetic, no solve) so the pair orders, the
   least-squares fits and criteria C1-C8 are recomputed from all three levels;
3. writes ``ASYMPTOTIC_NOTE.md`` where **every number is injected** from the
   merged JSON (zero hand-copy), separating the pre-asymptotic pair (n_coarse =
   8) from the asymptotic pair (n_coarse >= 16);
4. leaves the honest bookkeeping: the anchor's literal wording (32/64/128) is
   *not* satisfied by this note -- ``n = 64`` is still missing, and nothing here
   may be read as closing it.

Usage::

    PYTHONPATH=src python docs/verification-kovasznay-asymptotic/asymptotic_note.py
"""
import json
from pathlib import Path

from tensorfvm.verification import kovasznay_order_asymptotic as ladder

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUNS = HERE / 'runs'
CLOSURE_RECORD = ROOT / 'docs' / 'verification-kovasznay-order-closure' / 'runs' / 'level_32' / 'summary.json'

#: fields of a ladder record that the closure-campaign record does not carry
#: (it was never written for this ladder).  They are re-derived here from the
#: fields it does carry, and each derivation is stated in ``provenance``.
DERIVED = dict(
    reynolds='from the campaign configuration (kovasznay_order_closure.py default Re=40)',
    convection_order='second-order deferred correction (same solver factory)',
    linear_method='reused_lu (same solver factory default)',
    unknowns='campaign field `dofs`',
    final_momentum_residual='not recorded by the campaign (convergence flag only)',
    final_divergence_max='not recorded by the campaign (convergence flag only)',
    inner_iterations='not recorded by the campaign',
    linear_true_relative_residual='not recorded by the campaign',
    linear_gmres_iterations='not recorded by the campaign',
    gate=ladder.GATE,
)


def adopt_level_32():
    payload = json.loads(CLOSURE_RECORD.read_text())
    row = next(r for r in payload['table']['zero_gradient'] if int(r['n']) == 32)
    record = dict(
        n=int(row['n']), h=1. / int(row['n']), reynolds=40., convection_order=2,
        closure=row['closure'], linear_method='reused_lu',
        cells=int(row['cells']), faces=int(row['faces']), unknowns=int(row['dofs']),
        converged=bool(row['converged']), outer_iterations=int(row['outer_iterations']),
        final_momentum_residual=float('nan'), final_divergence_max=float('nan'),
        inner_iterations=None, seconds=float(row['seconds']),
        tie_residual=float(row['tie_residual']),
        tied_components=int(row['tied_components']),
        spanwise_l2=float(row['spanwise_l2']), max_abs_w=float(row['max_abs_w']),
        inplane_l2=float(row['inplane_l2']), total_l2=float(row['total_l2']),
        linear_true_relative_residual=None, linear_gmres_iterations=None,
        gate=ladder.GATE, pressure_l2=float(row['pressure_l2']),
        pressure_mean_free_l2=float(row['pressure_mean_free_l2']),
        peak_rss_mb=float(row.get('peak_rss_mb', float('nan'))),
        source=str(CLOSURE_RECORD.relative_to(ROOT)),
        adopted_by='docs/verification-kovasznay-asymptotic/asymptotic_note.py',
        provenance=DERIVED,
    )
    path = ladder.write_level(RUNS, record)
    return record, path


def asymptotic_pair(records, minimum_coarse):
    """Observed orders restricted to asymptotic pairs (coarse grid >= threshold)."""
    rows = []
    for coarse, fine in zip(records, records[1:]):
        if coarse['n'] < minimum_coarse:
            continue
        ratio = coarse['h'] / fine['h']
        rows.append(dict(
            pair=f"{coarse['n']}->{fine['n']}", h_ratio=float(ratio),
            total=ladder.observed_order(coarse['total_l2'], fine['total_l2'], ratio),
            inplane=ladder.observed_order(coarse['inplane_l2'], fine['inplane_l2'], ratio),
            pressure_mean_free=ladder.observed_order(
                coarse['pressure_mean_free_l2'], fine['pressure_mean_free_l2'], ratio),
        ))
    return rows


def main():
    record, path = adopt_level_32()
    merged = ladder.merge(RUNS)
    records = merged['records']
    asymptotic = asymptotic_pair(records, 16)
    fit_asym = ladder.fit_order([r for r in records if r['n'] >= 16], 'total_l2')
    pre = [o for o in merged['orders'] if int(o['pair'].split('->')[0]) < 16]
    checks = {c['id']: c for c in merged['checks']}

    def f(value, spec='.4f'):
        if value is None:
            return 'n/a'
        if isinstance(value, bool):
            return str(value)
        if isinstance(value, (int, float)):
            return format(value, spec)
        return str(value)

    lines = ['# CASE-FVM-KOVASZNAY-ORDER — asymptotic-pair addendum (n = 32 adopted)', '',
             'Every number below is injected by `asymptotic_note.py` from '
             '`runs/summary.json` / `runs/level-n32-zero_gradient-reused_lu.json`; nothing is hand copied.', '',
             '## 1. What changed', '',
             f'- Ladder levels now available: {", ".join(str(r["n"]) for r in records)}.',
             f'- `n = 32` was **adopted** (not re-solved) from `{record["source"]}` '
             f'({f(record["seconds"], ".0f")} s of solver time already spent there); '
             'the record carries `source` / `adopted_by` / `provenance` fields.',
             '- Same configuration as the ladder: closure `zero_gradient`, Re = 40, second-order '
             'deferred correction, Picard to 1e-8, volume-weighted L2 norms.', '',
             '## 2. Pair orders', '',
             '| pair | regime | total L2 order | in-plane | pressure (mean-free) |',
             '|---|---|---:|---:|---:|']
    for row in pre:
        lines.append(f"| {row['pair']} | pre-asymptotic (n_coarse = 8) | {f(row['total'])} | "
                     f"{f(row['inplane'])} | {f(row['pressure_mean_free'])} |")
    for row in asymptotic:
        lines.append(f"| {row['pair']} | **asymptotic (n_coarse >= 16)** | {f(row['total'])} | "
                     f"{f(row['inplane'])} | {f(row['pressure_mean_free'])} |")
    lines += ['', '## 3. Verdicts', '',
              '| id | criterion | value | verdict |', '|---|---|---|---|']
    for check in merged['checks']:
        value = check['value']
        if isinstance(value, list):
            value = ', '.join(f(v) for v in value)
        lines.append(f"| {check['id']} | {check['statement']} | {f(value)} | **{check['verdict']}** |")
    lines += ['',
              f"- Pre-declared ladder criterion C3/C4 on the finest available pair "
              f"({asymptotic[-1]['pair'] if asymptotic else 'n/a'}): "
              f"{f(asymptotic[-1]['total']) if asymptotic else 'n/a'} vs gate {f(ladder.GATE, '.1f')}.",
              f"- C5 (least-squares over *all* levels, including the pre-asymptotic n = 8 point): "
              f"{f((merged['fits']['total_l2'] or {}).get('slope'))} -> {checks['C5']['verdict']} "
              '(the n = 8 level is not in the asymptotic range and drags the global slope down).',
              f"- Asymptotic-only least-squares slope (n >= 16, {f((fit_asym or {}).get('levels'), '.0f')} levels): "
              f"{f((fit_asym or {}).get('slope'))}.",
              '', '## 4. Honest bookkeeping / limits', '',
              '- The ledger criterion literally asks for a slope on **32/64/128**. This note delivers the '
              '**16->32** asymptotic pair only; `n = 64` (cluster job cj1373, ~83 min single-core estimate) '
              'had not been dispatched when this note was written, so the literal 32/64/128 ladder is '
              '**still open** and no PASS is claimed for it.',
              '- C5 is left as FAIL rather than re-defining the gate after seeing the data: the '
              'asymptotic-only slope is reported next to it, not instead of it.',
              '- `zero_gradient` ties the free tangential dofs, so the spanwise component is exactly zero '
              'at every level and total L2 == in-plane L2 (values identical above, not a coincidence).',
              '- Convergence is a flag from the campaign record; its momentum residual was not stored, so '
              'the residual column is n/a for n = 32 rather than invented.', '']
    (HERE / 'ASYMPTOTIC_NOTE.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(levels=merged['levels'], orders=merged['orders'],
                          asymptotic=asymptotic, fit_asym=fit_asym,
                          anchor_closed=merged['anchor_closed']), indent=2, default=float))
    print('wrote', path, 'and', HERE / 'ASYMPTOTIC_NOTE.md')


if __name__ == '__main__':
    main()
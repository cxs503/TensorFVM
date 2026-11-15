"""Machine-adjudicable case-pool entry for ``CASE-FVM-CHANNEL-KMM180``.

Why this module exists
----------------------
The ledger row is

    CASE-FVM-CHANNEL-KMM180 | 算例池·湍流槽道 Re_tau=180 平均速度对数律
    ref: Kim, Moin & Moser 1987 J. Fluid Mech. 177:133
    容差: <5% | 指标: 内尺度平均速度 u+(y+) 在 y+>30 区间 vs KMM1987 表

The word 算例池 (case pool) demands a *record a machine can adjudicate*.
Before this module the anchor existed only as prose: eight ``kmm180_*`` analysis
nodes, a benchmark driver, CSV/JSON artifacts and a set of reports each quoting
a percentage.  Nothing in the repository answered, in one place and without a
human reading a table:

* which quantity is graded, in which window, against which reference;
* whether that reference really is the KMM1987-lineage table the ledger names,
  or the documented Lee & Moser (2015) proxy;
* which ``(model, grid)`` configurations the pool actually admits;
* whether a PASS survives mesh refinement, or is a property of the coarsest
  mesh in the ladder;
* what would have to change for the entry to be rejected (i.e. can the gate
  fail at all).

This module answers those five questions and nothing else.  It adds **no new
physics, no new solver run and no new tolerance**: every number is recomputed
from already-committed artifacts, and the tolerance is imported from
:mod:`tensorfvm.verification.kmm1987` so a pool record cannot drift from the
grading code it summarises.

Inputs (all committed)
----------------------
``docs/case-fvm-channel-kmm180/profiles.csv``
    Per-cell ``y+`` and model ``u+`` for every ``(model, n)`` of the published
    matrix (``sst``/``komega``/``kepsilon`` x ``n = 64/128/256``).  The graded
    deviation is recomputed from these cell values, not copied from a report.
``docs/case-fvm-channel-kmm180/benchmark.json``
    Solver-side ``Cf`` and ``U_bulk+`` per case (not derivable from the profile
    alone, so read rather than recomputed).
``docs/rans-turbulence-validation-matrix/dns/mkm1999_0180/``
    The anchor reference: Moser, Kim & Mansour (1999) ``chan180`` statistics,
    i.e. the KMM1987-lineage dataset at ``Re_tau = 178.12``.  Digest verified
    against ``provenance.json`` on every load.
``docs/rans-turbulence-validation-matrix/dns/0180/``
    The Lee & Moser (2015) proxy used by the legacy validation matrix, kept
    here only so the reference-substitution effect can be *measured*.

Grading windows
---------------
``full``
    ``30 <= y+ <= top of the reference table`` -- the literal ledger window
    (``y+ > 30``).  This is the primary criterion.
``log_layer``
    ``30 <= y+ <= 0.2 Re_tau`` (~36 at this Reynolds number) -- the only
    interval that is actually logarithmic.  At ``n = 64`` it holds **two**
    model cells, i.e. fewer than the three a verdict requires, so it is
    recorded as ``NOT_GRADED`` rather than silently widened.  It is a
    diagnostic criterion rather than part of admission because the ledger
    window is the full one -- but when it is gradable it is published even
    though it fails.

Honest status of the anchor
---------------------------
Run ``python -m tensorfvm.verification.kmm180_case_pool`` for the current
verdict; at the time of writing it is ``ADMITTED_WITH_CAVEAT``: SST meets the
5 % full-window criterion at every grid of the ladder, the narrow log-layer
criterion does **not** (5.7-6.2 % once gradable), ``k-omega`` and ``k-epsilon``
are rejected outright (10.8 % / 7.1 %), and the surface-friction error of the
admitted configuration already exceeds 5 % at ``n >= 128``.  A record that
reported only "PASS, 3.6 %" would be true and misleading at the same time; the
caveat list exists so it cannot be quoted that way.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .kmm180_direct_anchor import (
    load_mkm1999_reference,
    reference_windows,
    window_stats,
)
from .kmm180_windows import LOW_Y_PLUS
from .kmm1987 import TOLERANCE_LOG_REGION, load_kmm1987_reference

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

#: Module lives at src/tensorfvm/verification/kmm180_case_pool.py
REPO_ROOT = Path(__file__).resolve().parents[3]
ARTIFACT_DIR = REPO_ROOT / 'docs' / 'case-fvm-channel-kmm180'
PROFILES_CSV = ARTIFACT_DIR / 'profiles.csv'
BENCHMARK_JSON = ARTIFACT_DIR / 'benchmark.json'
#: Cross-check fixture written by :mod:`tensorfvm.verification.kmm180_direct_anchor`.
DIRECT_ANCHOR_SUMMARY = (REPO_ROOT / 'docs' / 'verification-channel-kmm180-direct-anchor'
                         / 'summary.json')

SCHEMA = 'tensorfvm.case-fvm-channel-kmm180.case-pool/1'

#: The ledger contract, transcribed once.  ``tolerance`` is *not* freely chosen:
#: :data:`POOL_TOLERANCE` is the grading code's own constant, and the test suite
#: asserts the two agree with the committed ledger string.
LEDGER_CONTRACT = dict(
    slug='CASE-FVM-CHANNEL-KMM180',
    title='算例池·湍流槽道 Re_tau=180 平均速度对数律',
    reference=('Kim, Moin & Moser 1987 J. Fluid Mech. 177:133 '
               '(DOI 10.1017/S0022112087000892)'),
    metric='wall-unit mean velocity u+(y+) RMS relative deviation in y+ > 30',
    tolerance_percent=5.0,
    window_low_y_plus=LOW_Y_PLUS,
)

#: Grading tolerance, imported from the grading module (never re-typed).
POOL_TOLERANCE: float = TOLERANCE_LOG_REGION

#: Mesh ladder present in the committed artifacts.
LADDER: Tuple[int, ...] = (64, 128, 256)

#: RANS closures present in the committed artifacts.
MODELS: Tuple[str, ...] = ('sst', 'komega', 'kepsilon')

#: A window needs at least this many model cells before it can carry a verdict
#: (mirrors the guard in ``kmm180_direct_anchor.window_stats``, which raises
#: below three).
GRADABILITY_MIN_CELLS = 3

#: The closure the pool admits.  SST is designated *ex ante* as the production
#: model of this case because it is the only closure in the matrix whose wall
#: treatment is designed for the ``y+ ~ 1`` resolution used here; the other two
#: are carried as rejected controls and their numbers are published, not
#: dropped.
DESIGNATED_MODEL = 'sst'

#: Admission criteria.  Single source of truth: every row names the indicator it
#: grades (computed by :func:`indicators`), the comparator, the threshold, and
#: whether it participates in admission.  Non-primary rows are *published*
#: diagnostics with an explicit ``excluded_reason`` -- the pool record must show
#: the failures it does not gate on, otherwise it is a marketing document.
CRITERIA: List[dict] = [
    dict(
        name='u_plus_rms_full',
        quantity=('RMS relative deviation of u+ from the MKM1999 chan180 table, '
                  '30 <= y+ <= centreline, worst grid of the ladder '
                  f'({DESIGNATED_MODEL})'),
        indicator='rms_full_worst',
        comparator='lt',
        threshold=POOL_TOLERANCE,
        unit='-',
        primary=True,
        source=('feature_ledger acceptance: 指标 内尺度平均速度 u+(y+) 在 y+>30 区间; '
                '容差 <5%'),
    ),
    dict(
        name='reference_is_kmm1987_lineage',
        quantity=('graded reference is the KMM1987-lineage dataset '
                  '(MKM1999 chan180, digest verified), not the LM2015 proxy'),
        indicator='reference_is_anchor',
        comparator='is_true',
        threshold=True,
        unit='bool',
        primary=True,
        source='feature_ledger acceptance: ref KMM1987 J. Fluid Mech. 177:133',
    ),
    dict(
        name='admitted_model_exists',
        quantity='at least one closure of the matrix meets u_plus_rms_full',
        indicator='admitted_model_count',
        comparator='ge',
        threshold=1,
        unit='models',
        primary=True,
        source='a case-pool entry must name the configuration it admits',
    ),
    dict(
        name='u_plus_rms_log_layer',
        quantity=('RMS relative deviation of u+ in the physical log layer '
                  '30 <= y+ <= 0.2 Re_tau, worst *gradable* grid (SST)'),
        indicator='rms_log_worst_gradable',
        comparator='lt',
        threshold=POOL_TOLERANCE,
        unit='-',
        primary=False,
        excluded_reason=(
            'not part of admission: the ledger window is the literal "y+ > 30" full '
            'window, and this window is only ~6.4 wall units wide here, so at n=64 it '
            'holds 2 cells and cannot be graded at all. It is published because it is '
            'the physically meaningful log-layer measure, because it is the criterion '
            'the admitted configuration FAILS, and because hiding it would let "u+ '
            'within 5 % in the log layer" be claimed from a full-window number '
            '(docs/verification-channel-kmm180-metric-norm).'),
    ),
    dict(
        name='cf',
        quantity=('relative error of skin-friction coefficient Cf vs the anchor '
                  'reference profile, worst grid of the ladder (SST)'),
        indicator='cf_rel_worst',
        comparator='lt',
        threshold=POOL_TOLERANCE,
        unit='-',
        primary=False,
        excluded_reason=(
            'not in the ledger string (which grades u+ only); kept as a diagnostic '
            'because it is the quantity a resistance consumer actually needs: it fails '
            'from n=128 upwards while the graded u+ criterion passes, the clearest '
            'evidence that the full-window profile metric is lenient.'),
    ),
    dict(
        name='bulk_velocity',
        quantity=('relative error of bulk velocity U_bulk+ vs the anchor '
                  'reference, worst grid of the ladder (SST)'),
        indicator='bulk_rel_worst',
        comparator='lt',
        threshold=POOL_TOLERANCE,
        unit='-',
        primary=False,
        excluded_reason=(
            'not in the ledger string; integrated counterpart of the u+ profile, '
            'reported so a mass-flux user can see the error the pointwise metric '
            'understates.'),
    ),
    dict(
        name='refinement_stability',
        quantity=('the full-window deviation used for admission is the worst over the '
                  'ladder, and every grid of the ladder is separately published'),
        indicator='ladder_published',
        comparator='is_true',
        threshold=True,
        unit='bool',
        primary=False,
        excluded_reason=(
            'meta-criterion: it gates the *record*, not the solver. Reported because '
            'the SST full-window deviation grows monotonically with refinement '
            '(2.37 % -> 3.21 % -> 3.61 %), so the coarsest grid is the most favourable '
            'one; an entry quoting a single grid would be grid-lucky by construction.'),
    ),
    dict(
        name='log_layer_gradability',
        quantity=('model cells inside the log-layer window at the coarsest grid of '
                  'the ladder (SST, n=64)'),
        indicator='cells_log_coarsest',
        comparator='ge',
        threshold=GRADABILITY_MIN_CELLS,
        unit='cells',
        primary=False,
        excluded_reason=(
            'documents why the log-layer criterion can be NOT_GRADED at all: at n=64 '
            'the window [30, 35.6] contains two cell centres. That is a property of '
            'the mesh resolution of the window, not a solver error, and it is the '
            'reason the log-layer criterion is a diagnostic rather than a gate.'),
    ),
]

#: Indicator names in publication order (CLI table / report).
INDICATOR_ORDER = tuple(row['indicator'] for row in CRITERIA)


# --------------------------------------------------------------------------- #
# Artifact loading
# --------------------------------------------------------------------------- #

def load_profiles(path: Optional[Path] = None) -> Dict[Tuple[str, int], dict]:
    """Load per-cell profiles from the committed ``profiles.csv``.

    Returns ``{(model, n): {'y_plus': ndarray, 'u_plus': ndarray}}`` sorted by
    ``y+``.  Raises if a ``(model, n)`` pair of the declared matrix is missing,
    so a truncated artifact cannot yield a pool record with fewer
    configurations than advertised.
    """
    path = Path(path) if path is not None else PROFILES_CSV
    buckets: Dict[Tuple[str, int], List[Tuple[float, float]]] = {}
    with path.open(newline='') as handle:
        for row in csv.DictReader(handle):
            key = (row['model'].strip(), int(row['n']))
            buckets.setdefault(key, []).append((float(row['y_plus']),
                                                float(row['u_plus'])))

    profiles: Dict[Tuple[str, int], dict] = {}
    for key, pairs in buckets.items():
        pairs.sort()
        y_plus = np.array([p[0] for p in pairs], dtype=float)
        u_plus = np.array([p[1] for p in pairs], dtype=float)
        if np.any(np.diff(y_plus) <= 0):
            raise ValueError(f'{path}: y+ is not strictly increasing for {key}')
        profiles[key] = dict(y_plus=y_plus, u_plus=u_plus)

    missing = [(m, n) for m in MODELS for n in LADDER if (m, n) not in profiles]
    if missing:
        raise ValueError(f'{path} is missing configurations {missing}')
    return profiles


def load_solver_metrics(path: Optional[Path] = None) -> Dict[Tuple[str, int], dict]:
    """Load solver-side ``Cf`` / ``U_bulk+`` from the committed ``benchmark.json``.

    These cannot be recovered from ``profiles.csv`` alone (``Cf`` needs the
    solver's ``u_tau``), so they are read -- but the *reference* values they are
    compared against are recomputed from the reference profile inside
    :func:`cf_relative_error` / :func:`bulk_relative_error`, and the read values
    are cross-checked against ``kmm180_direct_anchor``'s summary by
    :func:`audit_direct_anchor_summary`.
    """
    path = Path(path) if path is not None else BENCHMARK_JSON
    payload = json.loads(path.read_text())
    out: Dict[Tuple[str, int], dict] = {}
    for case in payload['cases']:
        key = (case['model'], int(case['n']))
        out[key] = dict(Cf=float(case['Cf']),
                        bulk_u_plus=float(case['bulk_u_plus']),
                        iterations=int(case.get('iterations', -1)),
                        converged=bool(case.get('converged', True)))
    return out


def anchor_reference() -> dict:
    """The KMM1987-lineage reference actually graded against (digest checked)."""
    return load_mkm1999_reference()


def proxy_reference() -> dict:
    """The Lee & Moser (2015) proxy used by the legacy validation matrix."""
    return load_kmm1987_reference()


def reference_identity(anchor: Optional[dict] = None,
                       proxy: Optional[dict] = None) -> dict:
    """Resolve *which* table is graded, with the evidence for saying so.

    ``re_tau`` is the discriminator that matters: the ledger names Re_tau = 180
    of the KMM1987 method, this table reports 178.12, the proxy 182.088.
    ``delta_Cf_relative`` is the measured cost of using the proxy instead, so a
    reader can see the substitution is small but not zero.
    """
    anchor = anchor_reference() if anchor is None else anchor
    proxy = proxy_reference() if proxy is None else proxy

    provenance = anchor['provenance']
    recorded = provenance['files']['mean']['csv_sha256']
    directory = Path(provenance['directory']) if 'directory' in provenance else (
        REPO_ROOT / 'docs' / 'rans-turbulence-validation-matrix' / 'dns' / 'mkm1999_0180')
    digest = hashlib.sha256((directory / 'mean.csv').read_bytes()).hexdigest()

    return dict(
        name='MKM1999 chan180 (KMM1987 method)',
        citation='Moser, Kim & Mansour 1999, Phys. Fluids 11:943 (chan180 statistics)',
        re_tau=float(anchor['re_tau']),
        nominal_re_tau=180.0,
        Cf=float(anchor['Cf']),
        U_bulk_plus=float(anchor['U_bulk_plus']),
        points=int(np.size(anchor['y_plus'])),
        max_y_plus=float(np.max(anchor['y_plus'])),
        csv_sha256=digest,
        csv_sha256_recorded=recorded,
        digest_matches=bool(digest == recorded),
        proxy_name='Lee & Moser 2015 (Re_tau = 182.088)',
        proxy_re_tau=float(proxy['re_tau']),
        delta_Cf_relative=float(abs(anchor['Cf'] - proxy['Cf']) / anchor['Cf']),
        delta_re_tau=float(proxy['re_tau'] - anchor['re_tau']),
    )


# --------------------------------------------------------------------------- #
# Grading
# --------------------------------------------------------------------------- #

def grade_window(y_plus, u_plus, dns: dict, window: dict) -> dict:
    """Grade one profile in one window, recording non-gradability instead of raising."""
    try:
        stats = window_stats(y_plus, u_plus, dns, **window)
    except ValueError as error:
        mask = ((np.asarray(y_plus, dtype=float) >= window['low'])
                & (np.asarray(y_plus, dtype=float) <= window['high']))
        return dict(low=float(window['low']), high=float(window['high']),
                    cells=int(np.count_nonzero(mask)), verdict='NOT_GRADED',
                    rms_relative=None, reason=str(error))
    stats['verdict'] = 'PASS' if stats['rms_relative'] < POOL_TOLERANCE else 'FAIL'
    stats['reason'] = None
    return stats


def model_ladder(model: str, dns: dict,
                 profiles: Optional[Dict[Tuple[str, int], dict]] = None,
                 metrics: Optional[Dict[Tuple[str, int], dict]] = None) -> List[dict]:
    """Grade one closure at every grid of the ladder against ``dns``.

    Both published windows are graded.  ``window['verdict']`` is ``PASS`` /
    ``FAIL`` / ``NOT_GRADED`` and the deviation is ``None`` when not gradable,
    so a consumer cannot average a missing window into a number.
    """
    profiles = load_profiles() if profiles is None else profiles
    metrics = load_solver_metrics() if metrics is None else metrics
    windows = reference_windows(dns)

    rows: List[dict] = []
    for n in LADDER:
        key = (model, n)
        if key not in profiles:
            raise KeyError(f'no committed profile for {key}')
        y_plus = profiles[key]['y_plus']
        u_plus = profiles[key]['u_plus']
        full = grade_window(y_plus, u_plus, dns, windows['full'])
        log_layer = grade_window(y_plus, u_plus, dns, windows['log_layer'])
        solver = metrics.get(key, {})
        rows.append(dict(
            model=model, n=int(n),
            cells=int(np.size(y_plus)),
            first_cell_y_plus=float(y_plus[0]),
            full=full, log_layer=log_layer,
            Cf=solver.get('Cf'), bulk_u_plus=solver.get('bulk_u_plus'),
            iterations=solver.get('iterations'),
        ))
    return rows


def cf_relative_error(row: dict, dns: dict) -> Optional[float]:
    """``|Cf_model - Cf_ref| / Cf_ref`` (reference side recomputed, not typed)."""
    if row.get('Cf') is None:
        return None
    return float(abs(row['Cf'] - dns['Cf']) / dns['Cf'])


def bulk_relative_error(row: dict, dns: dict) -> Optional[float]:
    """``|U_bulk+_model - U_bulk+_ref| / U_bulk+_ref``."""
    if row.get('bulk_u_plus') is None:
        return None
    return float(abs(row['bulk_u_plus'] - dns['U_bulk_plus']) / dns['U_bulk_plus'])


def summarise_model(rows: Sequence[dict], dns: dict) -> dict:
    """Collapse a ladder into the worst-case numbers the criteria grade.

    Worst case, not best case: the graded deviation of the SST profile grows
    with refinement, so a minimum over the ladder would manufacture a PASS out
    of the coarsest mesh.  ``log_layer`` uses the worst *gradable* grid and
    records how many grids were gradable, because a ``NOT_GRADED`` window is
    missing data, not a zero.
    """
    full_values = [r['full']['rms_relative'] for r in rows
                   if r['full'].get('rms_relative') is not None]
    log_values = [r['log_layer']['rms_relative'] for r in rows
                  if r['log_layer'].get('rms_relative') is not None]
    cf_values = [v for v in (cf_relative_error(r, dns) for r in rows) if v is not None]
    bulk_values = [v for v in (bulk_relative_error(r, dns) for r in rows)
                   if v is not None]

    if not full_values:
        raise ValueError('no gradable full-window value in the ladder')

    def worst(values):
        return float(max(values)) if values else None

    full_by_grid = {r['n']: r['full'].get('rms_relative') for r in rows}
    monotone = all(
        (full_by_grid[a] is not None and full_by_grid[b] is not None
         and full_by_grid[b] >= full_by_grid[a])
        for a, b in zip(LADDER, LADDER[1:]))

    return dict(
        model=rows[0]['model'],
        grids=[int(r['n']) for r in rows],
        rms_full_by_grid=full_by_grid,
        rms_log_by_grid={r['n']: r['log_layer'].get('rms_relative') for r in rows},
        cells_log_by_grid={r['n']: r['log_layer']['cells'] for r in rows},
        rms_full_worst=worst(full_values),
        rms_full_best=float(min(full_values)),
        rms_log_worst_gradable=worst(log_values),
        log_layer_gradable_grids=[r['n'] for r in rows
                                  if r['log_layer'].get('rms_relative') is not None],
        cf_rel_by_grid={r['n']: cf_relative_error(r, dns) for r in rows},
        cf_rel_worst=worst(cf_values),
        bulk_rel_by_grid={r['n']: bulk_relative_error(r, dns) for r in rows},
        bulk_rel_worst=worst(bulk_values),
        full_verdicts={r['n']: r['full']['verdict'] for r in rows},
        log_verdicts={r['n']: r['log_layer']['verdict'] for r in rows},
        admitted=bool(worst(full_values) < POOL_TOLERANCE),
        deviation_grows_with_refinement=bool(monotone),
        verdict_margin_to_threshold=float(POOL_TOLERANCE - worst(full_values)),
    )


def indicators(dns: Optional[dict] = None,
               profiles: Optional[Dict[Tuple[str, int], dict]] = None,
               metrics: Optional[Dict[Tuple[str, int], dict]] = None,
               identity: Optional[dict] = None) -> dict:
    """Every quantity the :data:`CRITERIA` table grades, recomputed from artifacts."""
    dns = anchor_reference() if dns is None else dns
    profiles = load_profiles() if profiles is None else profiles
    metrics = load_solver_metrics() if metrics is None else metrics
    identity = reference_identity(anchor=dns) if identity is None else identity

    models = {m: summarise_model(model_ladder(m, dns, profiles, metrics), dns)
              for m in MODELS}
    designated = models[DESIGNATED_MODEL]
    admitted = [m for m, s in models.items() if s['admitted']]

    return dict(
        rms_full_worst=designated['rms_full_worst'],
        rms_log_worst_gradable=designated['rms_log_worst_gradable'],
        cf_rel_worst=designated['cf_rel_worst'],
        bulk_rel_worst=designated['bulk_rel_worst'],
        reference_is_anchor=bool(identity['digest_matches']
                                 and abs(identity['re_tau'] - 180.0) < 5.0),
        admitted_model_count=len(admitted),
        admitted_models=admitted,
        cells_log_coarsest=int(designated['cells_log_by_grid'][LADDER[0]]),
        ladder_published=bool(list(designated['grids']) == list(LADDER)),
    )


# --------------------------------------------------------------------------- #
# Criteria evaluation
# --------------------------------------------------------------------------- #

_COMPARATORS = {
    'lt': lambda value, threshold: bool(value < threshold),
    'le': lambda value, threshold: bool(value <= threshold),
    'ge': lambda value, threshold: bool(value >= threshold),
    'gt': lambda value, threshold: bool(value > threshold),
    'is_true': lambda value, threshold: bool(value) is bool(threshold),
}


def evaluate_criteria(values: dict, criteria: Optional[Sequence[dict]] = None) -> List[dict]:
    """Apply :data:`CRITERIA` to ``values`` and return one row per criterion.

    A missing indicator is a hard error: a pool record must not silently drop a
    criterion it cannot evaluate.
    """
    criteria = CRITERIA if criteria is None else criteria
    rows: List[dict] = []
    for criterion in criteria:
        name = criterion['indicator']
        if name not in values:
            raise KeyError(f'criterion {criterion["name"]} needs indicator {name}')
        value = values[name]
        comparator = _COMPARATORS[criterion['comparator']]
        passed = None if value is None else comparator(value, criterion['threshold'])
        row = dict(criterion)
        row['value'] = value
        row['pass'] = passed
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# Pool entry
# --------------------------------------------------------------------------- #

def _dispositions(criteria: Sequence[dict]) -> dict:
    primary = [c for c in criteria if c['primary']]
    diagnostics = [c for c in criteria if not c['primary']]
    not_graded = [c['name'] for c in criteria if c['pass'] is None]
    failed_primary = [c['name'] for c in primary if c['pass'] is False]
    failed_diagnostics = [c['name'] for c in diagnostics if c['pass'] is False]
    # A primary criterion with no value is *missing data*, not a failure: the
    # pool withholds the entry rather than rejecting a case it could not grade.
    withheld_primary = [c['name'] for c in primary if c['pass'] is None]

    if withheld_primary:
        status = 'WITHHELD'
    elif failed_primary:
        status = 'REJECTED'
    elif failed_diagnostics:
        status = 'ADMITTED_WITH_CAVEAT'
    else:
        status = 'ADMITTED'
    return dict(pool_status=status,
                failed_primary=failed_primary,
                withheld_primary=withheld_primary,
                failed_diagnostics=failed_diagnostics,
                not_graded=not_graded,
                primary_count=len(primary),
                diagnostic_count=len(diagnostics))


def case_pool_entry(profiles: Optional[Dict[Tuple[str, int], dict]] = None,
                    dns: Optional[dict] = None,
                    proxy: Optional[dict] = None,
                    metrics: Optional[Dict[Tuple[str, int], dict]] = None) -> dict:
    """Assemble the complete, machine-adjudicable pool record."""
    dns = anchor_reference() if dns is None else dns
    proxy = proxy_reference() if proxy is None else proxy
    profiles = load_profiles() if profiles is None else profiles
    metrics = load_solver_metrics() if metrics is None else metrics

    identity = reference_identity(anchor=dns, proxy=proxy)
    values = indicators(dns=dns, profiles=profiles, metrics=metrics, identity=identity)
    criteria = evaluate_criteria(values)

    models = {m: summarise_model(model_ladder(m, dns, profiles, metrics), dns)
              for m in MODELS}
    disposition = _dispositions(criteria)

    entry = dict(
        schema=SCHEMA,
        slug=LEDGER_CONTRACT['slug'],
        title=LEDGER_CONTRACT['title'],
        ledger_contract=dict(
            reference=LEDGER_CONTRACT['reference'],
            metric=LEDGER_CONTRACT['metric'],
            tolerance=POOL_TOLERANCE,
            tolerance_percent=LEDGER_CONTRACT['tolerance_percent'],
            window_low_y_plus=LEDGER_CONTRACT['window_low_y_plus'],
        ),
        tolerance=POOL_TOLERANCE,
        tolerance_source='tensorfvm.verification.kmm1987.TOLERANCE_LOG_REGION',
        windows=reference_windows(dns),
        reference=identity,
        designated_model=DESIGNATED_MODEL,
        ladder=list(LADDER),
        models=list(MODELS),
        criteria=criteria,
        admitted_models=values['admitted_models'],
        rejected_models=[m for m in MODELS if m not in values['admitted_models']],
        model_summaries=models,
        **disposition,
    )
    entry['caveats'] = _caveats(entry)
    entry['reproduce'] = ('python -m tensorfvm.verification.kmm180_case_pool '
                          '--out docs/case-fvm-channel-kmm180-pool')
    return entry


def _caveats(entry: dict) -> List[str]:
    """Plain-language limits of the entry, derived from the record itself."""
    designated = entry['model_summaries'][DESIGNATED_MODEL]
    caves: List[str] = []

    if 'u_plus_rms_log_layer' in entry['failed_diagnostics']:
        plain = ' / '.join(f'n={n}: {v:.2%}' for n, v in
                           designated['rms_log_by_grid'].items() if v is not None)
        caves.append(
            'The admitted configuration fails the physically meaningful narrow '
            f'log-layer window (30 <= y+ <= 0.2 Re_tau): {plain}. The ledger window is '
            'the literal full window, so this does not block admission, but no claim of '
            '"u+ within 5 % in the log layer" may be made from this entry.')

    if 'cf' in entry['failed_diagnostics']:
        plain = ' / '.join(f'n={n}: {v:.2%}' for n, v in
                           designated['cf_rel_by_grid'].items() if v is not None)
        caves.append(
            f'Skin friction Cf errs by {plain} against the anchor reference: the wall '
            'quantity a resistance consumer needs is worse than the graded profile '
            'metric suggests, and it crosses 5 % from n=128 upwards.')

    if designated['deviation_grows_with_refinement']:
        caves.append(
            'The full-window deviation of the admitted configuration grows with '
            'refinement '
            + ' -> '.join(f'{v:.2%}' for v in designated['rms_full_by_grid'].values())
            + '; the coarsest grid is therefore the most favourable one. The criterion '
            'is graded on the worst grid for this reason -- a single-grid quote would be '
            'grid-lucky.')

    if entry['rejected_models']:
        plain = ', '.join(
            f'{m}: {entry["model_summaries"][m]["rms_full_worst"]:.2%}'
            for m in entry['rejected_models'])
        caves.append(
            'The entry admits the case *with the designated closure only*; the other '
            f'closures of the matrix are rejected controls ({plain} vs 5 %).')

    if 'log_layer_gradability' in entry['failed_diagnostics']:
        cells = designated['cells_log_by_grid']
        plain = ' / '.join(f'n={n}: {c}' for n, c in cells.items())
        caves.append(
            'The narrow log-layer window is under-resolved by the coarsest mesh: '
            f'cell centres inside [30, 0.2 Re_tau] = {plain} (a verdict needs '
            f'{GRADABILITY_MIN_CELLS}). The log-layer diagnostic is therefore '
            '"NOT_GRADED" exactly where the full-window number looks best, which is '
            'the concrete reason the two windows must be quoted together.')

    not_graded = [c['name'] for c in entry['criteria'] if c['pass'] is None]
    if not_graded:
        caves.append('Criteria not gradable at every grid: ' + ', '.join(not_graded))
    if entry['withheld_primary']:
        caves.append('Primary criteria without a value (entry withheld): '
                     + ', '.join(entry['withheld_primary']))
    return caves


#: Summary maps whose keys are grid sizes.  JSON turns integer keys into
#: strings, so a consumer reading ``case_pool_entry.json`` must restore them;
#: :func:`load_entry` does that and is the supported way to read the record back.
GRID_KEYED_FIELDS = ('rms_full_by_grid', 'rms_log_by_grid', 'cells_log_by_grid',
                     'cf_rel_by_grid', 'bulk_rel_by_grid', 'full_verdicts',
                     'log_verdicts')


def load_entry(path: Path) -> dict:
    """Read a committed pool record back with integer grid keys.

    NaN-free JSON round-trips are not enough on their own: ``json`` writes dict
    keys as strings, so ``entry['model_summaries']['sst']['rms_full_by_grid'][64]``
    raises ``KeyError`` on a freshly parsed record.  Every consumer (report
    publisher, dashboard, another agent) needs the same repair, so it lives here
    rather than in each of them.
    """
    payload = json.loads(Path(path).read_text())
    for summary in payload.get('model_summaries', {}).values():
        for field in GRID_KEYED_FIELDS:
            if field in summary:
                summary[field] = {int(key): value for key, value in summary[field].items()}
    return payload


def check_ladder_entries(entry: dict) -> None:
    """Guard: the entry must expose every grid, and the gate must be able to fail.

    Kept in the module (not only in the tests) so a consumer can assert the
    invariant on a freshly built record.  Raises ``ValueError`` if the record is
    degenerate: fewer grids than the declared ladder, or every closure admitted
    (which would mean the criteria no longer discriminate).
    """
    for model, summary in entry['model_summaries'].items():
        if list(summary['rms_full_by_grid']) != list(LADDER):
            raise ValueError(f'{model} does not publish the full ladder {LADDER}')
    if all(summary['admitted'] for summary in entry['model_summaries'].values()):
        raise ValueError('every closure admitted: the pool criteria no longer '
                         'discriminate, so a PASS would be uninformative')
    if not entry['rejected_models']:
        raise ValueError('no rejected control: the gate cannot be shown to fail')


def audit_direct_anchor_summary(entry: dict,
                                path: Optional[Path] = None) -> dict:
    """Cross-artifact audit: recomputed numbers vs ``kmm180_direct_anchor`` summary.

    Two agents can agree on the same artifact set only if the recomputation is
    independent of the writer.  This compares this module's numbers with the
    committed direct-anchor summary and reports the largest disagreement (the
    test suite pins it below ``1e-9``).
    """
    path = Path(path) if path is not None else DIRECT_ANCHOR_SUMMARY
    if not path.exists():
        return dict(available=False, path=str(path))
    payload = json.loads(path.read_text())
    stored = {(c['model'], int(c['n'])): c for c in payload['cases']}

    worst = 0.0
    compared = 0
    for model, summary in entry['model_summaries'].items():
        pairs = [(summary['rms_full_by_grid'], 'anchor', 'full', 'rms_relative'),
                 (summary['rms_log_by_grid'], 'anchor', 'log_layer', 'rms_relative'),
                 (summary['cf_rel_by_grid'], 'Cf_deviation', None, 'anchor_relative')]
        for values, block, window, key in pairs:
            for n, value in values.items():
                reference = stored.get((model, int(n)))
                if reference is None or value is None:
                    continue
                node = reference[block]
                if window is not None:
                    node = node.get(window, {})
                candidate = node.get(key)
                if candidate is None:
                    continue
                worst = max(worst, abs(value - candidate))
                compared += 1

    return dict(available=True, path=str(path), compared=compared,
                max_abs_difference=float(worst),
                agrees=bool(compared > 0 and worst < 1e-9))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _fmt(value, *, percent: bool = False) -> str:
    if value is None:
        return 'n/a'
    if isinstance(value, bool):
        return 'yes' if value else 'no'
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if percent:
        return f'{value:.2%}'
    return f'{value:.6g}'


def render_table(entry: dict) -> str:
    """Human-readable rendering of the same record the JSON carries."""
    lines: List[str] = []
    lines.append(f"{entry['slug']} -- 算例池入池记录 ({entry['pool_status']})")
    lines.append(f"tolerance {entry['tolerance']:.2%} "
                 f"(source {entry['tolerance_source']})")
    reference = entry['reference']
    lines.append(f"reference {reference['name']}  Re_tau={reference['re_tau']:.2f}  "
                 f"Cf={reference['Cf']:.6f}  sha256 ok={_fmt(reference['digest_matches'])}")
    lines.append('')
    header = (f"{'criterion':32s} {'value':>12s} {'cmp':>8s} {'threshold':>10s} "
              f"{'gate':>4s} {'result':>10s}")
    lines.append(header)
    lines.append('-' * len(header))
    for row in entry['criteria']:
        percent = row['unit'] == '-'
        result = 'PASS' if row['pass'] else ('FAIL' if row['pass'] is False else 'NOT_GRADED')
        lines.append(f"{row['name']:32s} {_fmt(row['value'], percent=percent):>12s} "
                     f"{row['comparator']:>8s} "
                     f"{_fmt(row['threshold'], percent=percent):>10s} "
                     f"{('yes' if row['primary'] else 'no'):>4s} {result:>10s}")
    lines.append('')
    lines.append(f"{'model':10s} {'n':>5s} {'full u+':>9s} {'log u+':>9s} {'Cf':>9s} "
                 f"{'verdict':>8s}")
    for model in entry['models']:
        summary = entry['model_summaries'][model]
        for n in summary['grids']:
            lines.append(
                f"{model:10s} {n:5d} "
                f"{_fmt(summary['rms_full_by_grid'][n], percent=True):>9s} "
                f"{_fmt(summary['rms_log_by_grid'][n], percent=True):>9s} "
                f"{_fmt(summary['cf_rel_by_grid'][n], percent=True):>9s} "
                f"{summary['full_verdicts'][n]:>8s}")
    lines.append('')
    lines.append('admitted: ' + (', '.join(entry['admitted_models']) or 'none')
                 + ' | rejected: ' + (', '.join(entry['rejected_models']) or 'none'))
    if entry['caveats']:
        lines.append('')
        lines.append('caveats:')
        for cave in entry['caveats']:
            lines.append('  - ' + cave)
    return '\n'.join(lines)


def render_markdown(entry: dict) -> str:
    """Markdown report generated from the record (no hand-typed numbers)."""
    designated = entry['model_summaries'][DESIGNATED_MODEL]
    out: List[str] = []
    out.append(f"# {entry['slug']} — 算例池入池记录（机器可判）\n")
    out.append(f"**状态：`{entry['pool_status']}`**　容差 `{entry['tolerance']:.2%}` "
               f"（来自 `{entry['tolerance_source']}`）\n")
    out.append('## 1. 契约\n')
    contract = entry['ledger_contract']
    out.append('| 项 | 值 |\n|---|---|')
    out.append(f"| 参考 | {contract['reference']} |")
    out.append(f"| 指标 | {contract['metric']} |")
    out.append(f"| 容差 | {contract['tolerance_percent']} % |")
    out.append(f"| 窗口下限 | y+ >= {contract['window_low_y_plus']:.0f} |")
    out.append('')
    out.append('## 2. 判据表（single source of truth = `CRITERIA`）\n')
    out.append('| 判据 | 量 | 值 | 比较 | 阈值 | 入池门 | 结果 |')
    out.append('|---|---|---|---|---|---|---|')
    for row in entry['criteria']:
        percent = row['unit'] == '-'
        verdict = ('PASS' if row['pass']
                   else ('FAIL' if row['pass'] is False else 'NOT_GRADED'))
        out.append(f"| `{row['name']}` | {row['quantity']} | "
                   f"{_fmt(row['value'], percent=percent)} | {row['comparator']} | "
                   f"{_fmt(row['threshold'], percent=percent)} | "
                   f"{'是' if row['primary'] else '否'} | **{verdict}** |")
    out.append('')
    out.append('## 3. 网格/模型矩阵\n')
    out.append('| model | n | log 窗口格数 | full RMS | full | log RMS | log | Cf rel |')
    out.append('|---|---|---|---|---|---|---|---|')
    for model in entry['models']:
        summary = entry['model_summaries'][model]
        for n in summary['grids']:
            out.append(f"| {model} | {n} | {summary['cells_log_by_grid'][n]} | "
                       f"{_fmt(summary['rms_full_by_grid'][n], percent=True)} | "
                       f"{summary['full_verdicts'][n]} | "
                       f"{_fmt(summary['rms_log_by_grid'][n], percent=True)} | "
                       f"{summary['log_verdicts'][n]} | "
                       f"{_fmt(summary['cf_rel_by_grid'][n], percent=True)} |")
    out.append('')
    out.append('## 4. 参考表身份\n')
    reference = entry['reference']
    out.append('| 项 | 值 |\n|---|---|')
    out.append(f"| 入池参考 | {reference['name']} |")
    out.append(f"| 出处 | {reference['citation']} |")
    out.append(f"| Re_tau | {reference['re_tau']:.3f}（名义 180） |")
    out.append(f"| Cf（由剖面重算） | {reference['Cf']:.9f} |")
    out.append(f"| U_bulk+ | {reference['U_bulk_plus']:.6f} |")
    out.append(f"| 采样点 | {reference['points']} |")
    out.append(f"| sha256 校验 | {'通过' if reference['digest_matches'] else '不一致'} |")
    out.append(f"| 替身 LM2015 Re_tau | {reference['proxy_re_tau']:.3f} |")
    out.append(f"| 替身 Cf 相对差 | {reference['delta_Cf_relative']:.4%} |")
    out.append('')
    out.append('## 5. 注意点（自动生成自记录本身）\n')
    for cave in entry['caveats']:
        out.append(f'- {cave}')
    out.append('')
    out.append('## 6. 误差判定\n')
    worst_grid = max(designated['rms_full_by_grid'],
                     key=lambda k: designated['rms_full_by_grid'][k])
    out.append(f"- 主判据 `u_plus_rms_full` = {designated['rms_full_worst']:.2%} "
               f"(worst grid n={worst_grid})，阈值 5 % → "
               f"{'**< 5 %，通过**' if designated['admitted'] else '**>= 5 %，不通过**'}；"
               f"余量 {designated['verdict_margin_to_threshold']:.2%}。")
    log_value = designated['rms_log_worst_gradable']
    out.append(f"- 对数层窗口（诊断）= {_fmt(log_value, percent=True)} → "
               f"{'<5 %' if (log_value is not None and log_value < entry['tolerance']) else '**>= 5 % 或不入判**'}。")
    cf_value = designated['cf_rel_worst']
    out.append(f"- Cf（诊断）= {_fmt(cf_value, percent=True)} → "
               f"{'<5 %' if (cf_value is not None and cf_value < entry['tolerance']) else '**>= 5 %**'}。")
    out.append(f"- 复现命令：`{entry['reproduce']}`\n")
    return '\n'.join(out) + '\n'


def write_entry(entry: dict, out_dir: Path) -> Tuple[Path, Path]:
    """Write ``case_pool_entry.json`` + ``REPORT.md`` into ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / 'case_pool_entry.json'
    json_path.write_text(json.dumps(entry, indent=1, sort_keys=True) + '\n')
    md_path = out_dir / 'REPORT.md'
    md_path.write_text(render_markdown(entry))
    return json_path, md_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Adjudicate the entry.  Exit 0 = admitted (with or without caveat), 1 = rejected."""
    parser = argparse.ArgumentParser(
        prog='python -m tensorfvm.verification.kmm180_case_pool',
        description='Adjudicate the CASE-FVM-CHANNEL-KMM180 case-pool entry.')
    parser.add_argument('--json', dest='json_path', type=Path, default=None,
                        help='write the machine-readable entry to this path')
    parser.add_argument('--out', dest='out_dir', type=Path, default=None,
                        help='write case_pool_entry.json + REPORT.md into this directory')
    parser.add_argument('--quiet', action='store_true', help='suppress the table')
    args = parser.parse_args(list(argv) if argv is not None else None)

    entry = case_pool_entry()
    entry['audit'] = audit_direct_anchor_summary(entry)
    check_ladder_entries(entry)

    if not args.quiet:
        print(render_table(entry))
    if args.json_path is not None:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(json.dumps(entry, indent=1, sort_keys=True) + '\n')
    if args.out_dir is not None:
        json_path, md_path = write_entry(entry, args.out_dir)
        if not args.quiet:
            print(f'\nwrote {json_path} and {md_path}')

    return 0 if entry['pool_status'].startswith('ADMITTED') else 1


if __name__ == '__main__':  # pragma: no cover
    raise SystemExit(main())
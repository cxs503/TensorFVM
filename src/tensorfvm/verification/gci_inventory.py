"""Cross-implementation audit of the grid-convergence-index estimators in the tree.

Why this module exists
---------------------
The repository grew the same verification metric in several places: Roache's
grid-convergence index (GCI) is re-derived independently in
``verification/richardson.py`` (the canonical tool), ``verification/multi_grid.py``,
``verification/ladder.py``, ``verification/annular_convergence.py``,
``immersed/study.py`` and ``rans_ship_resistance/speed_sweep.py``.  A *tool*
that is wrong does not fail a test of the *solver*; it silently publishes a
confident number.  This module therefore audits the estimators against series
whose exact limit is known, and grades each estimator by the only question that
matters for a verification claim:

    does the reported band cover the actual fine-grid error, and does the
    estimator refuse to report a confident band where the series carries no
    order signal?

Metric convention (uniform across implementations)
--------------------------------------------------
Every GCI estimator here is *relative to the fine-grid value*: the claim it
supports is ``|fine - exact| <= GCI * |fine|``.  That is the failure test used
below, ``error <= gci * |fine|``.  A tiny GCI band that excludes the true error
is an **overconfident** record: the number looks like verification and is not.

Regimes
-------
``asymptotic``          one-term ``C h^p`` decay (p = 1, 2) -- every estimator
                        must recover ``p`` and cover the error.
``pre_asymptotic``      a fourth-order term of the same size as the second-order
                        one -- the observed order is polluted.
``near_cancellation``   ``(v3-v2)/(v2-v1) = 1e3``: the two finest grids nearly
                        cancel.  Observed ``p ~ 10`` is an artefact; an
                        estimator without an order-plausibility cap reports a
                        band about 1e-6 wide over an error of 1e-6.
``sign_alternating``    the metric does not decay monotonically.
``stagnant``            the three grids are identical (trivially converged).

Run
---
``python -m tensorfvm.verification.gci_inventory --json out.json --markdown out.md``
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

THEORY_ORDER = 2.0
SPACINGS = (1.0 / 4.0, 1.0 / 2.0, 1.0)   # fine -> coarse, uniform ratio 2


# --------------------------------------------------------------------------- #
# Regime battery (generated, never hand-copied)
# --------------------------------------------------------------------------- #
def series(exact, amplitude, order, spacings=SPACINGS, *, secondary=0.0,
           secondary_order=None):
    """``f(h) = exact + amplitude h^order (+ secondary h^secondary_order)``."""
    h = np.asarray(spacings, dtype=float)
    values = exact + amplitude * h ** order
    if secondary:
        values = values + secondary * h ** secondary_order
    return tuple(float(v) for v in values)


@dataclass
class Regime:
    name: str
    values: tuple
    exact: float
    note: str


def regimes():
    """Synthetic refinement series with a known limit."""
    out = [
        Regime('asymptotic_p1', series(1.0, 0.4, 1), 1.0,
               'first-order decay, ratio 2'),
        Regime('asymptotic_p2', series(1.0, 0.4, 2), 1.0,
               'second-order decay, ratio 2'),
        Regime('pre_asymptotic', series(1.0, 0.5, 2, secondary=2.0,
                                        secondary_order=4), 1.0,
               'fourth-order term as large as the second-order one'),
        # near cancellation: (v3-v2)/(v2-v1) = 1e3
        Regime('near_cancellation', (1.0 + 1e-6, 1.0 + 2e-6, 1.0 + 1.002e-3),
               1.0, '(v3-v2)/(v2-v1) = 1e3, observed p ~ 10, true error 1e-6'),
        Regime('sign_alternating', (1.001, 0.999, 1.002), 1.0,
               'non-monotone metric: no one-sided C h^p decay'),
        Regime('stagnant', (1.0, 1.0, 1.0), 1.0,
               'identical grids: exactly converged, no order signal'),
    ]
    return out


# --------------------------------------------------------------------------- #
# Adapters: every estimator reduced to (order, gci, extrapolated, flags)
# --------------------------------------------------------------------------- #
@dataclass
class Reading:
    implementation: str
    regime: str
    status: str                 # ok | flagged | raises | not_run
    order: float = float('nan')
    gci: float = float('nan')   # relative to the fine-grid value
    extrapolated: float = float('nan')
    covered: bool = False       # GCI covers the actual fine-grid error
    confident: bool = False     # finite, narrow band reported as converged
    detail: dict = field(default_factory=dict)


def _first(mapping, keys):
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def _canonical(f, m, c):
    from .richardson import richardson_gci
    st = richardson_gci(f, m, c, SPACINGS, theory_order=THEORY_ORDER)
    return dict(order=st.order_used, gci=st.gci, extrapolated=st.extrapolated,
                flags=dict(convergence=st.convergence, fallback_reason=st.fallback_reason,
                           asymptotic=st.asymptotic, observed_order=st.order))


def _multi_grid(f, m, c):
    from .multi_grid import multi_grid_order
    st = multi_grid_order(SPACINGS, (f, m, c), theory_order=THEORY_ORDER)
    return dict(order=st.order_used, gci=st.gci, extrapolated=st.extrapolated,
                flags=dict(convergence=st.convergence, fallback_reason=st.fallback_reason,
                           asymptotic=st.asymptotic, observed_order=st.order))


def _ladder(f, m, c):
    from .ladder import gci_three_grid
    out = gci_three_grid(f, m, c, ratio=2.0)
    order = out.get('observed_order')
    return dict(order=float('nan') if order is None else float(order),
                gci=float(out['gci_fine']), extrapolated=float('nan'),
                flags=dict(monotone=out.get('monotone'),
                           oscillatory=out.get('oscillatory')))


def _annular(f, m, c):
    from .annular_convergence import observed_order as annular_order
    # ``annular_convergence.observed_order`` takes its levels coarse -> fine.
    out = annular_order((c, m, f), 2.0)
    order = _first(out, ('order',))
    gci = _first(out, ('gci_fine_relative', 'gci_fine', 'gci'))
    extrap = _first(out, ('richardson_extrapolated', 'extrapolated'))
    return dict(order=float('nan') if order is None else float(order),
                gci=float('nan') if gci is None else float(gci),
                extrapolated=float('nan') if extrap is None else float(extrap),
                flags=dict(consistent=out.get('order_consistent_with_coarser_triple')))


def _immersed(f, m, c):
    from ..immersed.study import richardson
    out = richardson(f, m, c)
    return dict(order=float(out['observed_order']),
                gci=float(out['gci_fine_pct']) / 100.0,
                extrapolated=float(out['extrapolated']),
                flags=dict(ratio_eps32_eps21=out.get('ratio_eps32_eps21')))


def _ship_resistance(f, m, c):
    from ..rans_ship_resistance.speed_sweep import grid_convergence_index
    out = grid_convergence_index(f, m, c)
    return dict(order=float(out['observed_order']), gci=float(out['gci_fine']),
                extrapolated=float(out['extrapolated']),
                flags=dict(monotonic=out.get('monotonic'),
                           relative_change=out.get('relative_change')))


IMPLEMENTATIONS = {
    'verification.richardson.richardson_gci': _canonical,
    'verification.multi_grid.multi_grid_order': _multi_grid,
    'verification.ladder.gci_three_grid': _ladder,
    'verification.annular_convergence.observed_order': _annular,
    'immersed.study.richardson': _immersed,
    'rans_ship_resistance.speed_sweep.grid_convergence_index': _ship_resistance,
}


def _honest(flags):
    """True when the estimator itself says the series carries no usable order."""
    if flags.get('asymptotic') is False:
        return True
    if flags.get('convergence') in ('inconsistent', 'order-implausible'):
        return True
    if flags.get('fallback_reason') not in (None, 'none'):
        return True
    if flags.get('monotone') is False or flags.get('consistent') is False:
        return True
    return False


def evaluate(name, adapter, regime):
    f, m, c = regime.values
    try:
        out = adapter(f, m, c)
    except Exception as exc:                       # noqa: BLE001 - audit records it
        return Reading(name, regime.name, 'raises', detail=dict(error=repr(exc)[:160]))
    order, gci, extrap = out['order'], out['gci'], out['extrapolated']
    flags = out.get('flags', {})
    if not math.isfinite(gci):
        return Reading(name, regime.name, 'no_band', order=order, gci=gci,
                       extrapolated=extrap, detail=flags)
    error = abs(f - regime.exact)
    covered = gci * abs(f) >= error * (1.0 - 1e-12)
    # "Overconfident" = a finite band that misses the true fine-grid error while
    # the estimator reports nothing that disqualifies the band.  That record is
    # indistinguishable from a real verification result downstream.
    honest = _honest(flags)
    status = 'ok' if covered else ('flagged' if honest else 'overconfident')
    return Reading(name, regime.name, status, order=order, gci=gci,
                   extrapolated=extrap, covered=bool(covered),
                   confident=bool((not covered) and not honest),
                   detail=flags)


def run():
    rows = [evaluate(name, adapter, regime)
            for name, adapter in IMPLEMENTATIONS.items()
            for regime in regimes()]
    summary = {}
    for name in IMPLEMENTATIONS:
        mine = [r for r in rows if r.implementation == name]
        summary[name] = dict(
            overconfident=[r.regime for r in mine if r.status == 'overconfident'],
            raises=[r.regime for r in mine if r.status == 'raises'],
            covered=[r.regime for r in mine if r.covered],
            not_run=[r.regime for r in mine if r.status == 'not_run'],
            no_band=[r.regime for r in mine if r.status == 'no_band'])
    return dict(theory_order=THEORY_ORDER, spacings=list(SPACINGS),
                implementations=list(IMPLEMENTATIONS), rows=[r.__dict__ for r in rows],
                summary=summary)


def markdown(payload):
    lines = ['| implementation | regime | status | p | GCI/fine | covers error |',
             '|---|---|---|---|---|---|']
    for r in payload['rows']:
        order = 'n/a' if not math.isfinite(r['order']) else f"{r['order']:.3f}"
        gci = 'n/a' if not math.isfinite(r['gci']) else f"{r['gci']:.3e}"
        lines.append(f"| `{r['implementation']}` | {r['regime']} | {r['status']} | "
                     f"{order} | {gci} | {'yes' if r['covered'] else 'no'} |")
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--json', type=Path, default=None)
    parser.add_argument('--markdown', type=Path, default=None)
    args = parser.parse_args(argv)
    payload = run()
    text = json.dumps(payload, indent=1)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text)
    if args.markdown:
        args.markdown.write_text(markdown(payload) + '\n')
    print(markdown(payload))
    for name, data in payload['summary'].items():
        print(f"\n{name}: overconfident={data['overconfident']} raises={data['raises']} "
              f"covered={data['covered']}")
    return payload


if __name__ == '__main__':
    main()
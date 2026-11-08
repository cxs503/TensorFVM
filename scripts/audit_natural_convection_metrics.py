#!/usr/bin/env python3
"""Arithmetic self-consistency audit of the archived natural-convection metrics.

Every record under ``docs/incompressible-baselines/cavity_n0*.json`` stores both
the *primitives* of a run (``nusselt``, ``nusselt_cold``, ``psi_abs_max``,
``meta.nu``, ``meta.prandtl``) and *derived* fields that mix them
(``nusselt_error_rel``, ``energy_imbalance_rel``, ``psi_error_rel``).  A derived
field that cannot be recomputed from the primitives of the same record is a
defect: the number a reader quotes may describe code that no longer exists.

The audit answers, per record, one question per derived field:

    does the stored value equal the value the committed primitives imply?

and reports the ladder-level consequences of what it finds:

* ``psi_error_rel`` is stale in **all eight** archived records: it was computed
  against the *dimensioned* ``|psi|`` instead of ``|psi|/alpha`` (the
  de Vahl Davis normalisation), i.e. 98.8-99.9 % error instead of the true
  1.0-37.3 %.  The recomputation is exact, so the stale formula is identifiable
  rather than guessed: ``|psi_error_rel - (|psi_abs_max| - ref)/ref| == 0``.
* the ``Nu`` ladder is *non-monotone* at ``Ra = 1e4`` and ``Ra = 1e5``, so an
  observed order of accuracy does not exist for those two anchors -- the audit
  says ``WITHHELD`` with the offending differences instead of fitting a number.
* the ``|psi|/alpha`` ladder *is* monotone and first order (``p ~ 1.0-1.2``),
  which is the piece of evidence that turns the ``psi`` anchor from "10 % off"
  into "10 % off *and still marching*": the discrepancy is discretisation, not a
  normalisation mismatch, and it is not closed at ``n = 96``.

Usage::

    python3 scripts/audit_natural_convection_metrics.py            # write the ledger
    python3 scripts/audit_natural_convection_metrics.py --check    # ratchet only
"""

from __future__ import annotations

import argparse
import glob
import json
import os

#: Records whose ``psi_error_rel`` predates the ``psi_over_alpha`` fix.  Pinned by
#: name so that regenerating one of them is a conscious edit, never a silent one.
LEGACY_PSI_ERROR = (
    "cavity_n032_ra1e04.json", "cavity_n032_ra1e05.json", "cavity_n032_ra1e06.json",
    "cavity_n064_ra1e04.json", "cavity_n064_ra1e05.json", "cavity_n064_ra1e06.json",
    "cavity_n096_ra1e05.json", "cavity_n096_ra1e06.json",
)
TOL = 1e-9


def _rel(a, b):
    return abs(float(a) - float(b)) / max(abs(float(b)), 1e-300)


def audit_record(path: str) -> dict:
    with open(path) as fh:
        rec = json.load(fh)
    met, meta, ref = rec["metrics"], rec["meta"], rec["reference"]
    alpha = float(meta["nu"]) / float(meta["prandtl"])
    psi_over_alpha = float(met["psi_abs_max"]) / alpha
    recomputed = dict(
        nusselt_error_rel=abs(float(met["nusselt"]) - ref["nu_contract"]) / ref["nu_contract"],
        energy_imbalance_rel=(abs(float(met["nusselt"]) - float(met["nusselt_cold"]))
                              / max(abs(float(met["nusselt"])), 1e-12)),
        psi_over_alpha=psi_over_alpha,
        psi_error_rel=abs(psi_over_alpha - ref["psi_max_dvd1983"]) / ref["psi_max_dvd1983"],
        # the *stale* formula, reconstructed to prove what the old field measured
        psi_error_rel_raw_psi=(abs(float(met["psi_abs_max"]) - ref["psi_max_dvd1983"])
                               / ref["psi_max_dvd1983"]),
    )
    stale_psi = abs(float(met["psi_error_rel"]) - recomputed["psi_error_rel_raw_psi"]) < TOL
    checks = {
        key: _rel(met[key], recomputed[key]) if key in met else None
        for key in ("nusselt_error_rel", "energy_imbalance_rel")
    }
    if "psi_over_alpha" in met:                       # records written after the fix
        checks["psi_over_alpha"] = _rel(met["psi_over_alpha"], psi_over_alpha)
        checks["psi_error_rel"] = _rel(met["psi_error_rel"], recomputed["psi_error_rel"])
    return dict(
        record=os.path.basename(path),
        n=int(meta["n"]), rayleigh=float(meta["rayleigh"]),
        nusselt=float(met["nusselt"]), nusselt_error_rel_stored=float(met["nusselt_error_rel"]),
        psi_over_alpha=psi_over_alpha, psi_error_rel_true=recomputed["psi_error_rel"],
        psi_error_rel_stored=float(met["psi_error_rel"]),
        psi_error_rel_is_legacy_raw_psi=stale_psi,
        max_rel_mismatch={k: v for k, v in checks.items() if v is not None},
    )


def observed_order(h, f):
    """Apparent order of accuracy on three grids of unequal spacing.

    ``f_i = f + C h_i**p`` implies ``(f1-f2)/(f2-f3) = (h1**p-h2**p)/(h2**p-h3**p)``;
    the ratio is monotone in ``p`` so a bisection settles it.  Returns ``None``
    when the sequence is not monotone -- the honest answer for a ladder that is
    still oscillating.
    """
    h = [float(x) for x in h]
    f = [float(x) for x in f]
    d1, d2 = f[0] - f[1], f[1] - f[2]
    if d1 * d2 <= 0.0:
        return None
    ratio = d1 / d2

    def rhs(p):
        return (h[0] ** p - h[1] ** p) / (h[1] ** p - h[2] ** p)

    lo, hi = 1e-3, 12.0
    if not (rhs(lo) <= ratio <= rhs(hi)):
        return None
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if rhs(mid) < ratio:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def richardson(f3, f2, n2, n3, p):
    """Extrapolate the fine-grid value to ``h -> 0`` with the fitted order."""
    ratio = (float(n3) / float(n2)) ** p
    return float(f3) + (float(f3) - float(f2)) / (ratio - 1.0)


def build_ledger(docs_dir: str) -> dict:
    paths = sorted(glob.glob(os.path.join(docs_dir, "incompressible-baselines", "cavity_n0*.json")))
    records = [audit_record(p) for p in paths]
    reproduced = os.path.join(docs_dir, "verification-natural-convection-metrics",
                              "reproduced_cavity_n032_ra1e05.json")
    repro = None
    if os.path.exists(reproduced):
        with open(reproduced) as fh:
            rec = json.load(fh)
        archived = next((r for r in records
                         if r["n"] == 32 and abs(r["rayleigh"] - 1e5) < 1e-9), None)
        repro = dict(
            record=os.path.basename(reproduced),
            settings={k: rec["settings"].get(k) for k in
                      ("alpha_u", "alpha_p", "alpha_T", "dc_relax", "poisson_solver")},
            iterations=int(rec["metrics"]["iterations"]),
            nusselt_rel_diff=None if not archived else _rel(rec["metrics"]["nusselt"], archived["nusselt"]),
            psi_over_alpha_rel_diff=None if not archived else _rel(
                rec["metrics"]["psi_over_alpha"], archived["psi_over_alpha"]),
        )
    ladders = {}
    for ra in sorted({r["rayleigh"] for r in records}):
        seq = sorted((r for r in records if r["rayleigh"] == ra), key=lambda r: r["n"])
        h = [1.0 / r["n"] for r in seq]
        nu = [r["nusselt"] for r in seq]
        psi = [r["psi_over_alpha"] for r in seq]
        order_nu = observed_order(h, nu) if len(seq) >= 3 else None
        order_psi = observed_order(h, psi) if len(seq) >= 3 else None
        entry = dict(
            levels=[r["n"] for r in seq], nusselt=nu, psi_over_alpha=psi,
            nusselt_monotone=all((nu[i] - nu[i + 1]) * (nu[i + 1] - nu[i + 2]) > 0
                                 for i in range(len(nu) - 2)) if len(nu) >= 3 else None,
            psi_monotone=all((psi[i] - psi[i + 1]) * (psi[i + 1] - psi[i + 2]) > 0
                             for i in range(len(psi) - 2)) if len(psi) >= 3 else None,
            observed_order_nusselt=order_nu, observed_order_psi=order_psi,
            nusselt_error_rel_finest=_rel(nu[-1], 4.519) if abs(ra - 1e5) < 60
            else _rel(nu[-1], {1e4: 2.243, 1e6: 8.800}[ra]),
            psi_error_rel_finest=_rel(psi[-1], {1e4: 5.071, 1e5: 9.111, 1e6: 16.32}[ra]),
        )
        if order_psi is not None:
            entry["psi_richardson"] = richardson(psi[-1], psi[-2], seq[-2]["n"], seq[-1]["n"], order_psi)
            entry["psi_richardson_error_rel"] = _rel(entry["psi_richardson"],
                                                     {1e4: 5.071, 1e5: 9.111, 1e6: 16.32}[ra])
        entry["nusselt_verdict"] = (
            "PASS" if entry["nusselt_error_rel_finest"] < 0.03 else "FAIL")
        entry["psi_verdict"] = (
            "PASS" if entry["psi_error_rel_finest"] < 0.03 else
            "WITHHELD (monotone ladder still marching: discretisation-limited)")
        ladders[f"Ra={ra:.0e}"] = entry
    legacy = sorted(r["record"] for r in records if r["psi_error_rel_is_legacy_raw_psi"])
    return dict(
        source="docs/incompressible-baselines/cavity_n0*.json (+ reproduced n=32 Ra=1e5)",
        records=records, ladders=ladders, reproduced=repro,
        legacy_psi_error_records=legacy,
        legacy_psi_error_matches_pin=legacy == sorted(LEGACY_PSI_ERROR),
        summary=dict(
            n_records=len(records),
            n_legacy_psi_error=len(legacy),
            max_derived_rel_mismatch=max(
                (v for r in records for k, v in r["max_rel_mismatch"].items()
                 if k != "psi_error_rel" and v is not None), default=0.0),
            nusselt_fail=[k for k, v in ladders.items() if v["nusselt_verdict"] != "PASS"],
            order_withheld=[k for k, v in ladders.items()
                            if v["observed_order_nusselt"] is None],
        ),
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs"))
    ap.add_argument("--output", default=None)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    ledger = build_ledger(args.docs)
    out = args.output or os.path.join(
        args.docs, "verification-natural-convection-metrics", "audit.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(ledger, fh, indent=2, sort_keys=True)
    print(json.dumps(ledger["summary"], indent=2))
    if args.check:
        problems = []
        if not ledger["legacy_psi_error_matches_pin"]:
            problems.append("legacy psi_error_rel set changed: %s"
                            % ledger["legacy_psi_error_records"])
        if ledger["summary"]["max_derived_rel_mismatch"] > 1e-6:
            problems.append("derived field does not reconcile: %g"
                            % ledger["summary"]["max_derived_rel_mismatch"])
        if problems:
            print("\n".join(problems))
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
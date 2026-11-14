"""Causal (prefix-only) audit of the convergence-diagnosis forecast.

Why this exists
---------------
``docs/xval-round2/ROUND2.md`` reports **0.00 % forecast error on all six
decaying cases** and credits the round-2 superlinear extrapolation.  Two things
in that report do not survive an adversarial read:

1. The harness (:func:`tensorfvm.verification.convergence_xval_round2.run_case_v2`)
   runs the solver to its *reference stopping iteration* and only then asks the
   diagnosis for a forecast.  At that point the diagnosis reports ``converged``
   and returns ``forecast_iterations = k`` -- the number it "predicts" is the
   number of iterations already recorded.  The reported 0.00 % is therefore an
   identity, not a forecast: **the metric cannot fail**.
2. The ``v1`` column of the artifact is ``n/a`` for every case, so the quoted
   "v1 had 307 % error" baseline is not reproduced anywhere in the artifact.

What this script measures
-------------------------
Each stored residual history is replayed one iteration at a time: at prefix
``k`` the diagnosis sees only ``residuals[:k+1]``.  Its forecast is scored
against the stopping iteration that the OpenFOAM-10 rule eventually produced for
the *whole* history -- a value the diagnosis had not seen.  Three arms:

* ``v1``   -- :class:`ConvergenceMonitor` (constant geometric tail rate);
* ``v2``   -- ``_diagnose_v2(..., backtest=False)``, i.e. the round-2 code as
  shipped (linear extrapolation of ``log rate`` blended 70/30);
* ``v2g``  -- ``_diagnose_v2(..., backtest=True)``: the same extrapolation, but
  only when it wins a causal hold-out test against the constant-rate model
  (``_walk_forward_accepts``).

Outputs
-------
``causal_forecast_audit.json`` / ``.md`` -- generated, never hand-copied.

Pure numpy/CPU, a few seconds, no solver call, no cluster: it reads the
committed histories in ``docs/convergence-diagnosis-v2/histories``.

    PYTHONPATH=src python docs/xval-round2/causal_forecast_audit.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from tensorfvm.convergence_diagnosis import (ConvergenceMonitor,
                                             DiagnosisConfig,
                                             openfoam_converged)
from tensorfvm.verification.convergence_xval_round2 import (_diagnose_v2,
                                                            _tail_rate)

ROOT = Path(__file__).resolve().parents[2]
HISTORY_DIR = ROOT / "docs" / "convergence-diagnosis-v2" / "histories"
OUT_DIR = ROOT / "docs" / "xval-round2"

#: diagnosis configuration, identical to ``run_case_v2`` / CASES_V2
CFG_KWARGS = dict(max_iter=2000, window=5, min_points=6, ratio_spread_tol=0.20,
                  min_decades=1.0, rate_stability_tol=0.05)

#: cases the round-2 suite expects NOT to converge (forecast error undefined)
NON_DECAYING = ("cd-pe10-cg-misuse", "poisson-n32-richardson")

#: relative leads (fraction of the stopping iteration) at which skill is scored
LEAD_FRACTIONS = (0.5, 0.25, 0.1)

ARMS = ("v1", "v2", "v2g")


def load_histories() -> Dict[str, dict]:
    return {p.stem: json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(HISTORY_DIR.glob("*.json"))}


def true_stop_iteration(residuals: Sequence[float], tolerance: float,
                        rel_tol: float) -> Optional[int]:
    """Stopping iteration the OF-10 rule produces for the *full* history."""
    if not residuals:
        return None
    r0 = residuals[0]
    for i in range(1, len(residuals)):
        if openfoam_converged(residuals[i], r0, tolerance, rel_tol):
            return i
    return None


def v1_forecasts(iterations: Sequence[int], residuals: Sequence[float],
                 cfg: DiagnosisConfig) -> List[dict]:
    """Replay the v1 monitor causally: one record() per iteration."""
    mon = ConvergenceMonitor(cfg)
    out = []
    for k, r in zip(iterations, residuals):
        d = mon.record(k, r)
        out.append({"k": k, "verdict": d.verdict, "ready": bool(d.forecast_ready),
                    "forecast": d.forecast_iterations, "rate": d.rate,
                    "spread": None, "superlinear": False, "gate": None})
    return out


def v2_forecasts(iterations: Sequence[int], residuals: Sequence[float],
                 cfg: DiagnosisConfig, backtest: bool) -> List[dict]:
    """Replay the round-2 diagnosis causally (rate_history built from prefix only)."""
    out: List[dict] = []
    for k in range(len(residuals)):
        prefix = residuals[: k + 1]
        # only rates already observed (indices < k) can enter the rate history
        rate_history: List[Tuple[int, float]] = [
            (iterations[i], out[i]["rate"])
            for i in range(len(out))
            if out[i]["rate"] is not None
            and (out[i]["spread"] is None or out[i]["spread"] <= cfg.ratio_spread_tol)
        ]
        diag = _diagnose_v2(prefix, cfg, rate_history, cfg.max_iter, backtest=backtest)
        rate, spread = _tail_rate(prefix, cfg.window)
        out.append({"k": k, "verdict": diag["verdict"],
                    "ready": bool(diag["forecast_ready"]),
                    "forecast": diag["forecast_iterations"], "rate": rate,
                    "spread": spread,
                    "superlinear": bool(diag.get("superlinear_fit")),
                    "gate": diag.get("superlinear_gate")})
    return out


def first_causal_forecast(series: Sequence[dict]) -> Optional[dict]:
    """First prefix at which the gate opens -- the earliest honest forecast."""
    for row in series:
        if row["ready"] and row["forecast"] is not None:
            return row
    return None


def lead_time_table(series: Sequence[dict], stop: int) -> List[dict]:
    """Forecast error as a function of how far ahead of ``stop`` it was issued."""
    rows = []
    for row in series:
        k, fc = row["k"], row["forecast"]
        if not row["ready"] or fc is None or k >= stop:
            continue
        rows.append({"k": k, "lead": stop - k, "lead_fraction": (stop - k) / stop,
                     "forecast": fc, "error": abs(fc - stop) / stop,
                     "signed": (fc - stop) / stop})
    return rows


def error_at_lead(series: Sequence[dict], stop: int, fraction: float) -> Optional[dict]:
    """Forecast issued when ``fraction`` of the run is still to come."""
    target = int(round(stop * (1.0 - fraction)))
    for row in series:
        if row["k"] >= target and row["ready"] and row["forecast"] is not None:
            if row["k"] >= stop:
                return None
            return {"k": row["k"], "forecast": row["forecast"],
                    "error": abs(row["forecast"] - stop) / stop,
                    "lead": stop - row["k"]}
    return None


def summarise_case(key: str, hist: dict) -> dict:
    iterations = [h["iteration"] for h in hist["history"]]
    residuals = [h["residual"] for h in hist["history"]]
    tol, rel = hist["tolerance"], hist["rel_tol"]
    cfg = DiagnosisConfig(tolerance=tol, rel_tol=rel, **CFG_KWARGS)

    stop = true_stop_iteration(residuals, tol, rel)

    arms = {
        "v1": v1_forecasts(iterations, residuals, cfg),
        "v2": v2_forecasts(iterations, residuals, cfg, backtest=False),
        "v2g": v2_forecasts(iterations, residuals, cfg, backtest=True),
    }

    row = {
        "case": key, "solver": hist["solver"], "n_samples": len(residuals),
        "stopping_iteration": stop, "decaying": key not in NON_DECAYING,
        "arms": {},
    }
    for arm, series in arms.items():
        a: dict = {
            "verdict_at_end": series[-1]["verdict"],
            "first": first_causal_forecast(series),
            "superlinear_triggers": sum(1 for r in series if r["superlinear"]),
            "gate_accepts": sum(1 for r in series
                                if r["gate"] and r["gate"].get("accepted")),
            "gate_tests": sum(1 for r in series if r["gate"]),
        }
        if stop:
            first = a["first"]
            if first is not None:
                a["first_error"] = abs(first["forecast"] - stop) / stop
                a["first_lead"] = stop - first["k"]
            lt = lead_time_table(series, stop)
            a["lead_time"] = lt
            if lt:
                errs = sorted(r["error"] for r in lt)
                a["median_error"] = errs[len(errs) // 2]
                a["max_error"] = errs[-1]
            for frac in LEAD_FRACTIONS:
                a[f"at_lead_{frac}"] = error_at_lead(series, stop, frac)
        row["arms"][arm] = a
    return row


def _pct(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{100.0 * x:.1f}%"


def _cell(a: dict, field: str) -> str:
    """``forecast (error)`` for a per-lead entry."""
    entry = a.get(field)
    if not entry:
        return "n/a"
    return f"{entry['forecast']} ({_pct(entry['error'])})"


def render_markdown(rows: List[dict], meta: dict) -> str:
    lines = [
        "# Causal (prefix-only) forecast audit — convergence diagnosis round 2", "",
        "Generated by `docs/xval-round2/causal_forecast_audit.py` from the committed",
        "residual histories in `docs/convergence-diagnosis-v2/histories`.",
        "`forecast (error)` = iterations predicted for the stop, and the relative",
        "error against the stopping iteration the OF-10 rule actually produced.", "",
        "## Arm 1 — earliest prefix at which the confidence gate opens", "",
        "| case | stop iter | v1 | v2 (as shipped) | v2g (hold-out gated) |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        stop = r["stopping_iteration"]
        if not stop:
            lines.append(f"| {r['case']} | none (no convergence) | "
                         f"{r['arms']['v1']['verdict_at_end']} | "
                         f"{r['arms']['v2']['verdict_at_end']} | "
                         f"{r['arms']['v2g']['verdict_at_end']} |")
            continue
        cells = []
        for arm in ARMS:
            a = r["arms"][arm]
            f = a.get("first")
            cells.append("never ready" if not f
                         else f"{f['forecast']} (+{a['first_lead']}) = {_pct(a.get('first_error'))}")
        lines.append(f"| {r['case']} | {stop} | " + " | ".join(cells) + " |")

    lines += ["", "## Arm 2 — skill vs forecast lead (fraction of the run still to come)", ""]
    header = "| case | stop iter | arm | " + " | ".join(f"lead {int(f*100)}%" for f in LEAD_FRACTIONS) + " |"
    lines += [header, "|---|---|---|" + "---|" * len(LEAD_FRACTIONS)]
    for r in rows:
        if not r["stopping_iteration"]:
            continue
        for arm in ARMS:
            a = r["arms"][arm]
            lines.append(f"| {r['case']} | {r['stopping_iteration']} | {arm} | "
                         + " | ".join(_cell(a, f"at_lead_{f}") for f in LEAD_FRACTIONS) + " |")

    lines += ["", "## Arm 3 — error summary over all causal prefixes", "",
              "| case | arm | median |err| | max |err| | superlinear fired | gate accepted/tests |",
              "|---|---|---|---|---|---|"]
    for r in rows:
        if not r["stopping_iteration"]:
            continue
        for arm in ARMS:
            a = r["arms"][arm]
            lines.append(f"| {r['case']} | {arm} | {_pct(a.get('median_error'))} | "
                         f"{_pct(a.get('max_error'))} | {a['superlinear_triggers']} | "
                         f"{a['gate_accepts']}/{a['gate_tests']} |")

    lines += ["", "## Meta", "", "```json", json.dumps(meta, indent=2), "```"]
    return "\n".join(lines) + "\n"


def main() -> dict:
    histories = load_histories()
    if not histories:
        raise SystemExit(f"no histories under {HISTORY_DIR}")
    rows = [summarise_case(k, h) for k, h in histories.items()]
    decaying = [r for r in rows if r["decaying"] and r["stopping_iteration"]]

    def count(pred) -> int:
        return sum(1 for r in decaying if pred(r))

    meta = {
        "n_cases": len(rows),
        "n_decaying": len(decaying),
        "criterion": "|forecast - true stopping iteration| / stopping iteration <= 15%",
        "first_causal_forecast_within_15pct": {
            arm: count(lambda r, a=arm: (r["arms"][a].get("first_error") is not None
                                         and r["arms"][a]["first_error"] <= 0.15))
            for arm in ARMS},
        "worst_first_causal_error": {
            arm: max((r["arms"][arm].get("first_error") or 0.0 for r in decaying), default=None)
            for arm in ARMS},
        "within_15pct_at_lead_25pct": {
            arm: count(lambda r, a=arm: (r["arms"][a].get("at_lead_0.25") is not None
                                         and r["arms"][a]["at_lead_0.25"]["error"] <= 0.15))
            for arm in ARMS},
        "ex_post_forecast_is_identity": True,
        "note": ("the ex-post metric used by run_case_v2 (forecast issued after the "
                 "stopping iteration was already recorded) is an identity: "
                 "forecast == k == reference; it is not a forecast and must not be "
                 "quoted as forecast accuracy"),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "causal_forecast_audit.json").write_text(
        json.dumps({"cases": rows, "meta": meta}, indent=2), encoding="utf-8")
    (OUT_DIR / "causal_forecast_audit.md").write_text(render_markdown(rows, meta),
                                                      encoding="utf-8")
    print(json.dumps(meta, indent=2))
    return {"cases": rows, "meta": meta}


if __name__ == "__main__":
    main()
"""Four-arm, strictly-causal comparison of stopping-iteration forecasts.

Protocol
--------
For every stored residual history the true stopping iteration is the first index
at which the OpenFOAM-10 ``checkConvergence`` disjunction holds on the *whole*
history.  Each arm is then replayed one iteration at a time (prefix
``residuals[:k+1]`` only) and scored as ``|forecast - stop| / stop``.

Scoring is **strict**: the error at lead ``L`` is taken at the single prefix
``k = round(stop * (1 - L))``.  An arm that has no forecast at that prefix is
recorded as *not published* -- it gets no credit for publishing later, which is
what made the round-2 "0.00 %" artifact look good.

Arms
----
* ``v1``  -- :class:`ConvergenceMonitor` (constant geometric tail rate);
* ``v2``  -- round-2 code as shipped (linear extrapolation of ``log rate``, 70/30 blend);
* ``v2g`` -- the same with the round-2 hold-out gate (``backtest=True``);
* ``v3``  -- :mod:`tensorfvm.verification.convergence_forecast_v3` (curvature-gated
  quadratic log-residual model + anchored extrapolation + prediction interval).

    PYTHONPATH=src python docs/xval-round2/causal_forecast_v3.py
"""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from tensorfvm.convergence_diagnosis import DiagnosisConfig
from tensorfvm.verification.convergence_forecast_v3 import (CONFIDENCES,
                                                            QUAD_WINDOW, T_CRIT,
                                                            causal_series)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
AUDIT = HERE / "causal_forecast_audit.py"
OUT_JSON = HERE / "causal_forecast_v3.json"
OUT_MD = HERE / "causal_forecast_v3.md"

LEADS = (0.5, 0.25, 0.1)
ARMS = ("v1", "v2", "v2g", "v3")
PASS = 0.15


def load_audit_module():
    spec = importlib.util.spec_from_file_location("causal_forecast_audit", AUDIT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _pct(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{100.0 * x:.1f}%"


def _at_lead_strict(series: List[dict], stop: int, frac: float) -> dict:
    """Forecast published at *exactly* the prefix ``round(stop*(1-frac))``."""
    k = int(round(stop * (1.0 - frac)))
    row = series[k]
    return {"k": k, "published": bool(row["ready"] and row["forecast"] is not None),
            "forecast": row["forecast"],
            "error": (abs(row["forecast"] - stop) / stop
                      if row["ready"] and row["forecast"] is not None else None),
            "interval": row.get("interval")}


def case_row(key: str, hist: dict, audit) -> dict:
    iterations = [h["iteration"] for h in hist["history"]]
    residuals = [h["residual"] for h in hist["history"]]
    cfg = DiagnosisConfig(tolerance=hist["tolerance"], rel_tol=hist["rel_tol"],
                          **audit.CFG_KWARGS)
    stop = audit.true_stop_iteration(residuals, hist["tolerance"], hist["rel_tol"])
    series = {
        "v1": audit.v1_forecasts(iterations, residuals, cfg),
        "v2": audit.v2_forecasts(iterations, residuals, cfg, backtest=False),
        "v2g": audit.v2_forecasts(iterations, residuals, cfg, backtest=True),
        "v3": causal_series(residuals, iterations, cfg),
    }
    row: dict = {"case": key, "solver": hist["solver"], "n_samples": len(residuals),
                 "stopping_iteration": stop,
                 "decaying": key not in audit.NON_DECAYING
                 and stop is not None,
                 "arms": {}}
    for arm, ser in series.items():
        entry: dict = {"verdict_at_end": ser[-1]["verdict"],
                       "publish_rate": sum(1 for s in ser if s["ready"]) / len(ser)}
        if arm == "v3":
            entry["quadratic_fraction"] = (
                sum(1 for s in ser if s.get("model") == "quadratic") / len(ser))
        if stop:
            first = audit.first_causal_forecast(ser)
            entry["first_publish_k"] = first["k"] if first else None
            entry["first_publish_error"] = (abs(first["forecast"] - stop) / stop
                                            if first else None)
            entry["leads"] = {f"{f}": _at_lead_strict(ser, stop, f) for f in LEADS}
            lt = audit.lead_time_table(ser, stop)
            if lt:
                errs = sorted(r["error"] for r in lt)
                entry["median_prefix_error"] = errs[len(errs) // 2]
                entry["max_prefix_error"] = errs[-1]
        row["arms"][arm] = entry
    row["cliff_evidence"] = _cliff_evidence(residuals, stop) if stop else None
    return row


def _cliff_evidence(residuals: List[float], stop: int) -> dict:
    """Ratio profile just before the stop: is the crossing a terminal breakdown?"""
    y = np.log(np.asarray(residuals, dtype=float))
    k = int(round(stop * 0.75))
    tail = np.exp(np.diff(y[max(k - 9, 0): k + 1]))
    end = np.exp(np.diff(y[max(stop - 8, 0): stop + 1]))
    return {"k_at_lead25": k, "ratios_at_lead25": [float(x) for x in tail],
            "ratios_final_steps": [float(x) for x in end],
            "ratio_min_final_steps": float(np.min(end)) if end.size else None}


def coverage(rows: List[dict], conf: str) -> Optional[dict]:
    """Fraction of published v3 prefixes whose interval contains the true stop."""
    tot = hit = 0
    gaps: List[float] = []
    for r in rows:
        if not r["decaying"]:
            continue
        stop = r["stopping_iteration"]
        ser = r["arms"]["v3"]
        # rebuild intervals from the JSON-stored leads is not enough: recompute
        for entry in ser.get("intervals", []):
            tot += 1
            if entry["lo"] <= stop <= entry["hi"]:
                hit += 1
    return {"n": tot, "hit": hit, "coverage": (hit / tot if tot else None)} if tot else None


def summarise(rows: List[dict]) -> dict:
    decaying = [r for r in rows if r["decaying"]]
    meta: dict = {"n_decaying": len(decaying), "leads": list(LEADS),
                  "pass_threshold": PASS, "arms": {},
                  "protocol": "strict same-prefix scoring; unpublished = no credit"}
    for arm in ARMS:
        firsts = [r["arms"][arm]["first_publish_error"] for r in decaying
                  if r["arms"][arm].get("first_publish_error") is not None]
        meta["arms"][arm] = {
            "first_publish_scored": len(firsts),
            "first_publish_passed_15pct": sum(1 for f in firsts if f <= PASS),
            "first_publish_passed_50pct": sum(1 for f in firsts if f <= 0.5),
            "first_publish_worst": max(firsts) if firsts else None,
            "publish_rate_mean": float(np.mean([r["arms"][arm]["publish_rate"]
                                                for r in decaying])),
            "leads": {},
        }
        for frac in LEADS:
            entries = [r["arms"][arm]["leads"][f"{frac}"] for r in decaying]
            pub = [e for e in entries if e["published"]]
            errs = [e["error"] for e in pub]
            meta["arms"][arm]["leads"][f"{frac}"] = {
                "n_cases": len(entries), "n_published": len(pub),
                "passed_15pct": sum(1 for e in errs if e <= PASS),
                "worst": max(errs) if errs else None,
                "median": sorted(errs)[len(errs) // 2] if errs else None,
                "conservative_passed_15pct": sum(1 for e in errs if e <= PASS),
            }
    return meta


def render(rows: List[dict], meta: dict, cov: Dict[str, dict]) -> str:
    lines = [
        "# Causal stopping-iteration forecast: arm v3 vs v1/v2/v2g",
        "",
        "Generated by `docs/xval-round2/causal_forecast_v3.py` from the committed",
        "histories in `docs/convergence-diagnosis-v2/histories` (no solver run).",
        "Scoring is **strict**: the error at lead `L` is taken at the single prefix",
        "`k = round(stop*(1-L))`; an arm without a forecast there is *not published*",
        "and gets no credit for publishing later.",
        "",
        "## 1 — Error at fixed lead (same prefix for every arm)",
        "",
        "| case | stop | arm | lead 50% | lead 25% | lead 10% |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        if not r["decaying"]:
            continue
        for arm in ARMS:
            cells = []
            for frac in LEADS:
                e = r["arms"][arm]["leads"][f"{frac}"]
                cells.append("not published" if not e["published"]
                             else f"{e['forecast']} ({_pct(e['error'])})")
            lines.append(f"| {r['case']} | {r['stopping_iteration']} | {arm} | "
                         + " | ".join(cells) + " |")

    lines += ["", "## 2 — Pass counts at fixed lead (published / worst published)",
              "",
              "| arm | lead 50% | lead 25% | lead 10% | publish rate |",
              "|---|---|---|---|---|"]
    for arm in ARMS:
        cells = []
        for frac in LEADS:
            s = meta["arms"][arm]["leads"][f"{frac}"]
            cells.append(f"{s['passed_15pct']}/{s['n_published']} (worst {_pct(s['worst'])})")
        lines.append(f"| {arm} | " + " | ".join(cells)
                     + f" | {100*meta['arms'][arm]['publish_rate_mean']:.0f}% |")

    lines += ["", "## 3 — First published forecast (each arm's own gate)",
              "",
              "| arm | scored | ≤15% | ≤50% | worst |",
              "|---|---|---|---|---|"]
    for arm in ARMS:
        s = meta["arms"][arm]
        lines.append(f"| {arm} | {s['first_publish_scored']} | {s['first_publish_passed_15pct']} "
                     f"| {s['first_publish_passed_50pct']} | {_pct(s['first_publish_worst'])} |")

    lines += ["", "## 4 — v3 prediction-interval coverage (all published prefixes)", "",
              "| nominal | prefixes | contains true stop | coverage |", "|---|---|---|---|"]
    for conf, c in cov.items():
        lines.append(f"| {conf} | {c['n']} | {c['hit']} | {_pct(c['coverage'])} |")

    lines += ["", "## 5 — Cliff evidence at the lead-25% decision point", "",
              "Ratio profile of the residual just before the stopping iteration; a",
              "ratio rising towards 1 and then collapsing means the crossing is a",
              "terminal breakdown, which no smooth model of `log residual` can see.",
              "",
              "| case | ratios at lead 25% | ratios over the final 8 steps |",
              "|---|---|---|"]
    for r in rows:
        if not r["decaying"]:
            continue
        ev = r["cliff_evidence"]
        lines.append(f"| {r['case']} | " + " ".join(f"{x:.3f}" for x in ev["ratios_at_lead25"])
                     + " | " + " ".join(f"{x:.3f}" for x in ev["ratios_final_steps"]) + " |")

    lines += ["", "## Meta", "", "```json", json.dumps(meta, indent=2), "```", ""]
    return "\n".join(lines)


def main() -> dict:
    audit = load_audit_module()
    histories = audit.load_histories()
    if not histories:
        raise SystemExit(f"no histories under {audit.HISTORY_DIR}")
    rows = [case_row(k, h, audit) for k, h in histories.items()]
    meta = summarise(rows)
    # interval coverage needs every published prefix, so replay v3 once more
    cov: Dict[str, dict] = {}
    for conf in CONFIDENCES:
        tot = hit = 0
        for key, hist in histories.items():
            row = next(r for r in rows if r["case"] == key)
            if not row["decaying"]:
                continue
            residuals = [h["residual"] for h in hist["history"]]
            iterations = [h["iteration"] for h in hist["history"]]
            cfg = DiagnosisConfig(tolerance=hist["tolerance"], rel_tol=hist["rel_tol"],
                                  **audit.CFG_KWARGS)
            ser = causal_series(residuals, iterations, cfg)
            stop = row["stopping_iteration"]
            for s in ser:
                iv = (s.get("interval") or {}).get(f"{conf:.2f}")
                if not s["ready"] or s["k"] >= stop or not iv:
                    continue
                tot += 1
                hit += int(iv["lo"] <= stop <= iv["hi"])
        cov[f"{conf:.2f}"] = {"n": tot, "hit": hit,
                              "coverage": (hit / tot) if tot else None}
    payload = {"meta": meta, "coverage": cov, "cases": rows,
               "v3_config": {"window": QUAD_WINDOW, "t_crit": T_CRIT}}
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    OUT_MD.write_text(render(rows, meta, cov), encoding="utf-8")
    for arm in ARMS:
        s = meta["arms"][arm]
        print(f"{arm:4s} " + " ".join(
            f"lead{int(f*100)}% {s['leads'][f'{f}']['passed_15pct']}/"
            f"{s['leads'][f'{f}']['n_published']} worst {_pct(s['leads'][f'{f}']['worst'])}"
            for f in LEADS)
            + f" | first {s['first_publish_passed_15pct']}/{s['first_publish_scored']}"
            + f" worst {_pct(s['first_publish_worst'])}")
    for conf, c in cov.items():
        print(f"interval {conf}: coverage {_pct(c['coverage'])} over {c['n']} prefixes")
    return payload


if __name__ == "__main__":
    main()
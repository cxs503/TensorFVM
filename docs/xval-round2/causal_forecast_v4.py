"""Five-arm, strictly-causal comparison including v4 calibrated intervals.

Arms
----
* ``v1`` -- :class:`ConvergenceMonitor` (constant geometric tail rate);
* ``v3`` -- curvature-gated quadratic ``log residual`` model + parameter-only
  interval (``docs/xval-round2/FORECAST-V3.md``);
* ``v4`` -- walk-forward model selection (v1 family vs v3 family, chosen by the
  realised multi-step error of the *same* solve) + interval calibrated with the
  walk-forward model error (:mod:`tensorfvm.verification.convergence_forecast_v4`).

Scoring (unchanged from the v3 audit, so numbers are comparable)
---------------------------------------------------------------
The truth is the first index at which the OpenFOAM-10 ``checkConvergence``
disjunction holds on the *whole* history.  The error at lead ``L`` is taken at
the single prefix ``k = round(stop * (1 - L))``; an arm that publishes nothing
there is scored as *not published*.

Interval coverage is reported two ways:

* **strict lead** -- does the interval at that one prefix contain the stop;
* **all prefixes** -- over every prefix that published an interval (this is the
  convention under which v3 reported 11.8 % / 19.4 % at nominal 68 % / 90 %).

    PYTHONPATH=src python docs/xval-round2/causal_forecast_v4.py
"""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from tensorfvm.convergence_diagnosis import DiagnosisConfig
from tensorfvm.verification import convergence_forecast_v3 as v3mod
from tensorfvm.verification import convergence_forecast_v4 as v4mod

HERE = Path(__file__).resolve().parent
AUDIT = HERE / "causal_forecast_audit.py"
OUT_JSON = HERE / "causal_forecast_v4.json"
OUT_MD = HERE / "causal_forecast_v4.md"

LEADS = (0.5, 0.25, 0.1)
ARMS = ("v1", "v3", "v4")
CONFS = ("0.68", "0.90")
PASS = 0.15


def load_audit_module():
    spec = importlib.util.spec_from_file_location("causal_forecast_audit", AUDIT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _at_lead_strict(series: List[dict], stop: int, frac: float) -> dict:
    k = int(round(stop * (1.0 - frac)))
    k = min(max(k, 0), len(series) - 1)
    row = series[k]
    ready = bool(row["ready"] and row["forecast"] is not None)
    out = {"k": k, "published": ready,
           "forecast": row["forecast"],
           "error": (abs(row["forecast"] - stop) / stop if ready else None)}
    for field in ("interval",):
        ivs = row.get(field) or {}
        out[field] = {c: ({"inside": bool(iv["lo"] <= stop <= iv["hi"]),
                          "rel_width": (iv["hi"] - iv["lo"]) / stop,
                          "lo": iv["lo"], "hi": iv["hi"]}
                          if iv else None) for c, iv in ivs.items()}
    return out


#: an interval wider than this (relative to the true stop) is called *degenerate*:
#: at that width it no longer tells a scheduler more than "keep waiting".
INFORMATIVE_WIDTH = 0.5


def _coverage_all_prefixes(series: List[dict], stop: int, conf: str,
                           field: str) -> dict:
    """Coverage over every prefix that published an interval (v3 convention).

    Also reports coverage restricted to *informative* intervals (relative width
    <= :data:`INFORMATIVE_WIDTH`) and the share of degenerate ones, so a wide but
    useless band cannot be passed off as calibration.
    """
    hit = tot = 0
    widths: List[float] = []
    i_hit = i_tot = 0
    unbounded = 0
    for row in series:
        if not row["ready"]:
            continue
        iv = (row.get(field) or {}).get(conf)
        if not iv:
            continue
        tot += 1
        inside = iv["lo"] <= stop <= iv["hi"]
        hit += 1 if inside else 0
        width = (iv["hi"] - iv["lo"]) / stop
        widths.append(width)
        unbounded += 1 if iv.get("unbounded_hi") else 0
        if width <= INFORMATIVE_WIDTH:
            i_tot += 1
            i_hit += 1 if inside else 0
    return {"n": tot, "covered": hit,
            "coverage": (hit / tot) if tot else None,
            "median_rel_width": float(np.median(widths)) if widths else None,
            "informative_n": i_tot, "informative_covered": i_hit,
            "informative_coverage": (i_hit / i_tot) if i_tot else None,
            "informative_share": (i_tot / tot) if tot else None,
            "unbounded_share": (unbounded / tot) if tot else None}


def case_row(key: str, hist: dict, audit) -> dict:
    iterations = [h["iteration"] for h in hist["history"]]
    residuals = [h["residual"] for h in hist["history"]]
    cfg = DiagnosisConfig(tolerance=hist["tolerance"], rel_tol=hist["rel_tol"],
                          **audit.CFG_KWARGS)
    stop = audit.true_stop_iteration(residuals, hist["tolerance"], hist["rel_tol"])
    errors_out: dict = {}
    series = {
        "v1": audit.v1_forecasts(iterations, residuals, cfg),
        "v3": v3mod.causal_series(residuals, iterations, cfg),
        "v4": v4mod.causal_series(residuals, iterations, cfg,
                                  errors_out=errors_out),
    }
    row: dict = {"case": key, "solver": hist["solver"], "n_samples": len(residuals),
                 "stopping_iteration": stop,
                 "decaying": key not in audit.NON_DECAYING and stop is not None,
                 "arms": {}}
    for arm, ser in series.items():
        entry: dict = {"verdict_at_end": ser[-1]["verdict"],
                       "publish_rate": sum(1 for s in ser if s["ready"]) / len(ser)}
        if stop:
            first = audit.first_causal_forecast(ser)
            entry["first_publish_k"] = first["k"] if first else None
            entry["first_publish_error"] = (abs(first["forecast"] - stop) / stop
                                            if first else None)
            entry["leads"] = {f"{f}": _at_lead_strict(ser, stop, f) for f in LEADS}
        if arm in ("v3", "v4"):
            field = "interval"
            if stop:
                entry["coverage_all_prefixes"] = {
                    c: _coverage_all_prefixes(ser, stop, c, field) for c in CONFS}
        if arm == "v4":
            fams = [s["selection"]["family"] for s in ser if s.get("selection")]
            entry["family_share"] = {
                "constant": sum(1 for f in fams if f == "constant") / max(len(fams), 1),
                "v3": sum(1 for f in fams if f == "v3") / max(len(fams), 1)}
            entry["quadratic_share"] = (
                sum(1 for s in ser if s.get("model") == "quadratic") / len(ser))
        row["arms"][arm] = entry
    err = errors_out.get("errors")
    if err is not None:
        row["scale_law"] = {
            fam: {"exponent": err.exponent(fam),
                  "q_unit_68": err.unit_quantile(fam, 0.68),
                  "q_unit_90": err.unit_quantile(fam, 0.90),
                  "profile": {str(h): v for h, v in
                              sorted(err.growth_profile(fam).items())[:: max(
                                  1, len(err.growth_profile(fam)) // 12)]}}
            for fam in v4mod.FAMILIES}
    return row


def aggregate(rows: List[dict]) -> dict:
    decaying = [r for r in rows if r["decaying"]]
    out: dict = {"n_decaying": len(decaying), "by_lead": {}, "first_publish": {}}
    for arm in ARMS:
        out["first_publish"][arm] = {
            "pass": sum(1 for r in decaying
                        if r["arms"][arm]["first_publish_error"] is not None
                        and r["arms"][arm]["first_publish_error"] <= PASS),
            "worst": max((r["arms"][arm]["first_publish_error"] or 0.0)
                         for r in decaying),
        }
    for lead in LEADS:
        cell: dict = {}
        for arm in ARMS:
            errs, pub = [], 0
            for r in decaying:
                d = r["arms"][arm]["leads"][f"{lead}"]
                if d["published"]:
                    pub += 1
                    errs.append(d["error"])
            cell[arm] = {"published": pub, "pass": sum(1 for e in errs if e <= PASS),
                         "worst": max(errs) if errs else None,
                         "median": float(np.median(errs)) if errs else None}
        # coverage: strict lead + all prefixes
        cov: dict = {}
        for arm, field in (("v3", "interval"), ("v4", "interval")):
            per_conf: dict = {}
            for c in CONFS:
                inside = tot = 0
                widths = []
                for r in decaying:
                    d = r["arms"][arm]["leads"][f"{lead}"]
                    iv = d.get(field, {}).get(c)
                    if not iv:
                        continue
                    tot += 1
                    inside += 1 if iv["inside"] else 0
                    widths.append(iv["rel_width"])
                per_conf[c] = {"n": tot, "covered": inside,
                               "coverage": (inside / tot) if tot else None,
                               "median_rel_width": (float(np.median(widths))
                                                    if widths else None)}
            cov[arm] = per_conf
        cell["coverage_strict_lead"] = cov
        out["by_lead"][f"{lead}"] = cell
    # all-prefix coverage (v3's reported convention)
    allp: dict = {}
    for arm in ("v3", "v4"):
        per_conf = {}
        for c in CONFS:
            hit = tot = 0
            widths = []
            for r in decaying:
                agg = r["arms"][arm]["coverage_all_prefixes"][c]
                hit += agg["covered"]
                tot += agg["n"]
                if agg["median_rel_width"] is not None:
                    widths.append(agg["median_rel_width"])
            inf_hit = sum(r["arms"][arm]["coverage_all_prefixes"][c]
                          ["informative_covered"] for r in decaying)
            inf_tot = sum(r["arms"][arm]["coverage_all_prefixes"][c]
                          ["informative_n"] for r in decaying)
            unb = sum(r["arms"][arm]["coverage_all_prefixes"][c]["unbounded_share"] * r
                      ["arms"][arm]["coverage_all_prefixes"][c]["n"] for r in decaying)
            per_conf[c] = {"n": tot, "covered": hit,
                           "coverage": (hit / tot) if tot else None,
                           "median_case_rel_width": (float(np.median(widths))
                                                     if widths else None),
                           "informative_n": inf_tot, "informative_covered": inf_hit,
                           "informative_coverage": ((inf_hit / inf_tot)
                                                    if inf_tot else None),
                           "informative_share": (inf_tot / tot) if tot else None,
                           "unbounded_share": (unb / tot) if tot else None}
        allp[arm] = per_conf
    out["coverage_all_prefixes"] = allp
    return out


def _pct(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{100.0 * x:.1f}%"


def render_markdown(rows: List[dict], agg: dict, meta: dict) -> str:
    lines = ["# Causal stopping-iteration forecast v4 — calibrated intervals", ""]
    lines.append(f"* cases: {len(rows)} ({agg['n_decaying']} decaying)")
    lines.append(f"* acceptance (ledger): point-forecast error <= {PASS:.0%}")
    lines.append("* protocol: strict causal prefix replay, error taken at one prefix "
                 "per lead; coverage measured at that prefix and over all prefixes")
    lines.append("")
    lines.append("## Point forecast by lead (pass = error <= 15 %)")
    lines.append("")
    lines.append("| lead | " + " | ".join(ARMS) + " |")
    lines.append("|---" * (len(ARMS) + 1) + "|")
    for lead in LEADS:
        cell = agg["by_lead"][f"{lead}"]
        lines.append("| " + f"{lead:.0%}" + " | " + " | ".join(
            f"{cell[a]['pass']}/{agg['n_decaying']} (worst {_pct(cell[a]['worst'])})"
            for a in ARMS) + " |")
    fp = agg["first_publish"]
    lines.append("| first publish | " + " | ".join(
        f"{fp[a]['pass']}/{agg['n_decaying']} (worst {_pct(fp[a]['worst'])})"
        for a in ARMS) + " |")
    lines.append("")
    lines.append("## Interval coverage and width")
    lines.append("")
    lines.append("| scope | arm | nominal | measured coverage | median width / stop |")
    lines.append("|---|---|---|---|---|")
    for lead in LEADS:
        cell = agg["by_lead"][f"{lead}"]
        for arm in ("v3", "v4"):
            for c in CONFS:
                d = cell["coverage_strict_lead"][arm][c]
                lines.append(f"| lead {lead:.0%} | {arm} | {float(c):.0%} | "
                             f"{_pct(d['coverage'])} (n={d['n']}) | "
                             f"{_pct(d['median_rel_width'])} |")
    for arm in ("v3", "v4"):
        for c in CONFS:
            d = agg["coverage_all_prefixes"][arm][c]
            lines.append(f"| all prefixes | {arm} | {float(c):.0%} | "
                         f"{_pct(d['coverage'])} (n={d['n']}) | "
                         f"{_pct(d['median_case_rel_width'])} |")
    lines.append("")
    lines.append("## Per case, lead 25 % and lead 10 %")
    lines.append("")
    lines.append("| case | stop | v1@25 | v3@25 | v4@25 | v1@10 | v3@10 | v4@10 | "
                 "v4 iv68@25 | v3 iv68@25 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        if not r["decaying"]:
            continue

        def cell(arm, lead):
            d = r["arms"][arm]["leads"][f"{lead}"]
            return _pct(d["error"]) if d["published"] else "n/p"

        def iv(arm, field, lead):
            d = r["arms"][arm]["leads"][f"{lead}"]
            x = (d.get(field) or {}).get("0.68")
            if not x:
                return "none"
            return f"[{x['lo']}, {x['hi']}]{'*' if x['inside'] else ''}"

        lines.append(
            f"| {r['case']} | {r['stopping_iteration']} | "
            f"{cell('v1', 0.25)} | {cell('v3', 0.25)} | {cell('v4', 0.25)} | "
            f"{cell('v1', 0.1)} | {cell('v3', 0.1)} | {cell('v4', 0.1)} | "
            f"{iv('v4', 'interval', 0.25)} | {iv('v3', 'interval', 0.25)} |")
    lines.append("")
    lines.append("`*` marks an interval that contains the true stopping iteration.")
    return "\n".join(lines) + "\n"


def main() -> dict:
    audit = load_audit_module()
    hist = audit.load_histories()
    rows = [case_row(k, h, audit) for k, h in hist.items()]
    agg = aggregate(rows)
    meta = {"leads": list(LEADS), "arms": list(ARMS), "pass": PASS,
            "confidences": list(CONFS)}
    payload = {"meta": meta, "aggregate": agg, "cases": rows}
    OUT_JSON.write_text(json.dumps(payload, indent=2))
    OUT_MD.write_text(render_markdown(rows, agg, meta))
    print(json.dumps(agg, indent=2))
    return payload


if __name__ == "__main__":
    main()
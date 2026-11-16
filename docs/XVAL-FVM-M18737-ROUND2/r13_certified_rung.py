"""Round-13 driver: the missing ``nx = 12800`` rung of the graded family.

Why this file exists
--------------------
``CERTIFIED-NX6400-ERRATA.md`` §⑤-4 and ``CLOSURE-graded-refinement-verified.md`` §③
left exactly two items open on the anchor ``XVAL-FVM-M18737-ROUND2``:

1. ``missing_rungs = [12800]`` for the graded (MUSCL-Hancock/MC/HLLC) family;
2. the CFL 0.4 vs 0.8 admission control (``cfl_sensitivity`` printed as *None*).

Both are closed here, under the criteria frozen **before** any number landed in
``PREDECLARATION-r13.md`` (E0 artifact genuineness, E1/E1r CFL admission, E2 prediction
band, E3 certified gate, E4 honest ceiling under the five error-bar conventions of
``ref25600_r11_robust.py``, E5 what was predicted).

The script never runs the solver: it reads the stored rungs produced by
``ladder/wc_ladder.py`` (CFL 0.4 and the CFL 0.8 variants) plus the WENO5 reference
rungs, and re-integrates them.  Metrics come from ``xval/wc_graded_refinement.py``
(same window / interpolation / normalisation as every earlier round) and the error-bar
conventions from ``ref25600_r11_robust.py`` -- both imported, never copied, so a change
in those definitions propagates instead of silently diverging.

Usage::

    PYTHONPATH=src python docs/XVAL-FVM-M18737-ROUND2/r13_certified_rung.py \
        --json docs/XVAL-FVM-M18737-ROUND2/r13_certified_rung.json \
        --markdown /tmp/r13.md
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]                         # repo root
ROOT = HERE.parent / "woodward-colella-double-blast"
DATA = ROOT / "ladder" / "data"
ARTIFACTS = ROOT / "artifacts"

#: graded (TensorFVM MUSCL/HLLC) rungs, coarse -> fine.  A rung may live under the
#: canonical ``hllc`` tag (CFL 0.4) or the ``hllc_cfl0.8`` variant, which is only
#: admitted once E1 shows that a CFL change alone does not move the L1 metric.
GRADED = (800, 1600, 3200, 6400, 12800)
GRADED_TAGS = ("hllc", "hllc_cfl0.8")

#: reference (WENO5/SSP-RK3/Rusanov) rungs.  ``weno5_cfl0.8`` is the same instrument
#: with the time step relaxed; it exists because CFL 0.4 at 25600 exceeds the job clock.
REF_FINEST = 12800                              # the rung the graded family is scored on
REF_EXTENDED = 25600

#: reference-ladder rung selection, coarse -> fine.  Each entry lists the tag
#: preferences for that rung (first available wins).  The default keeps the R14
#: behaviour verbatim (6400 rung = frozen CFL 0.4 instrument); Round-15
#: (``PREDECLARATION-r15.md``) may override the 6400 entry *only after* its
#: E1r6400 same-grid CFL admission gate passes, so the ladder can run on one
#: instrument and the E4 ``licensed`` check (all rungs same CFL) can hold.
REF_LADDER_TAGS = ((6400, ("weno5",)),
                   (REF_FINEST, ("weno5_cfl0.8", "weno5")),
                   (REF_EXTENDED, ("weno5_cfl0.8", "weno5")))

#: frozen bars (owner ledger / certified gate / the R11 sub-bars)
BAR_LEDGER = 15.0
BAR_THREE = 3.0
BAR_CERTIFIED = 1.0
CFL_ADMISSION = 0.2                             # %, pre-registered in PREDECLARATION-r13

#: E2 pre-registered prediction (PREDECLARATION-r13 §③ E2)
PREDICTED_L1_12800 = 0.449
PREDICTION_BAND = 0.15


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


GH = _load_module(ROOT / "xval" / "wc_graded_refinement.py", "wc_graded_refinement")
R11 = _load_module(HERE / "ref25600_r11_robust.py", "wc_ref25600_r11")


# --------------------------------------------------------------------------- #
# profile IO
# --------------------------------------------------------------------------- #
def profile_path(nx: int, tag: str) -> Path:
    return DATA / f"ladder_profile_nx{nx}_{tag}.npz"


def metrics_path(nx: int, tag: str) -> Path:
    return DATA / f"ladder_metrics_nx{nx}_{tag}.json"


def load_profile(nx: int, tag: str) -> Optional[Dict]:
    p = profile_path(nx, tag)
    if not p.exists():
        return None
    d = np.load(p)
    x = np.asarray(d["x_got"] if "x_got" in d else d["x"], float).ravel()
    rho = np.asarray(d["rho_got"] if "rho_got" in d else d["rho"], float).ravel()
    o = np.argsort(x)
    return {"x": x[o], "rho": rho[o], "tag": f"nx{nx}_{tag}",
            "source": str(p.relative_to(REPO))}


def load_metrics(nx: int, tag: str) -> Optional[Dict]:
    p = metrics_path(nx, tag)
    if not p.exists():
        return None
    return json.loads(p.read_text())


def first_available(nx: int, tags) -> Optional[Dict]:
    for tag in tags:
        prof = load_profile(nx, tag)
        if prof is not None:
            prof["metrics"] = load_metrics(nx, tag)
            return prof
    alt = ARTIFACTS / f"wc_blast_profile_nx{nx}.npz"
    if alt.exists():
        d = np.load(alt)
        x = np.asarray(d["x_got"] if "x_got" in d else d["x"], float).ravel()
        rho = np.asarray(d["rho_got"] if "rho_got" in d else d["rho"], float).ravel()
        o = np.argsort(x)
        return {"x": x[o], "rho": rho[o], "tag": f"nx{nx}_artifacts",
                "source": str(alt.relative_to(REPO)), "metrics": None}
    return None


def instrument_cfl(tag: str) -> float:
    """CFL implied by a rung tag: ``*_cfl0.8`` relaxed step, otherwise the frozen 0.4."""
    return 0.8 if "cfl0.8" in tag else 0.4


# --------------------------------------------------------------------------- #
# independent re-integration (so the verdict is not the script grading itself)
# --------------------------------------------------------------------------- #
def l1_percent_independent(xa, ra, xb, rb, window=(0.4, 1.0), npts: int = 40001) -> float:
    """L1 on ``window`` by resampling onto a fixed uniform stencil + rectangle rule.

    Different from ``GH.l1_percent`` (which integrates on the graded rung's own
    abscissae with the trapezoid rule): a metric artefact would have to survive both.
    """
    xs = np.linspace(window[0], window[1], npts)
    a = np.interp(xs, xa, ra)
    b = np.interp(xs, xb, rb)
    return 100.0 * float(np.abs(a - b).sum() / np.abs(b).sum())


# --------------------------------------------------------------------------- #
# E0: artifact genuineness
# --------------------------------------------------------------------------- #
def _mass_drift(m: Dict):
    """Largest finite drift in the metrics payload (``mass_energy_drift`` is a dict)."""
    raw = m.get("mass_drift")
    if raw is None:
        raw = m.get("mass_energy_drift")
    if isinstance(raw, dict):
        vals = [abs(float(x)) for x in raw.values()
                if isinstance(x, (int, float)) and math.isfinite(float(x))]
        return max(vals) if vals else None
    return raw


def genuineness(nx: int, prof: Dict) -> Dict:
    """E0a/E0b/E0c: the stored rung really is a run of *this* grid at *this* CFL."""
    m = prof.get("metrics") or {}
    x, rho = prof["x"], prof["rho"]
    dx = float(np.median(np.diff(x)))
    out = {
        "tag": prof["tag"],
        "source": prof["source"],
        "cells": int(x.size),
        "dx": dx,
        "dx_expect": 1.0 / nx,
        "dx_rel_error": abs(dx - 1.0 / nx) / (1.0 / nx),
        "rho_min": float(rho.min()),
        "rho_max": float(rho.max()),
        "all_finite": bool(np.all(np.isfinite(rho))),
        "positive": bool(np.all(rho > 0.0)),
        "steps": m.get("steps"),
        "seconds_solver": m.get("seconds_solver"),
        "cfl": m.get("cfl"),
        "mass_drift": _mass_drift(m),
        "positivity_fixes": m.get("positivity_fixes"),
    }
    checks = {
        "E0a_dx": out["dx_rel_error"] < 1e-12,
        "E0a_cells": out["cells"] in (nx, nx - 1, nx // 2),
        "E0b_positive_finite": out["positive"] and out["all_finite"],
        "E0b_steps_seconds": bool(out["steps"]) and bool(out["seconds_solver"]),
        "E0c_mass_drift": (out["mass_drift"] is not None
                           and float(out["mass_drift"]) < 1e-10),
    }
    if out["cfl"] is not None:
        # the CFL in the metrics payload must be the one the tag advertises: a rung
        # named ``*_cfl0.8`` that actually ran at 0.4 would silently mix instruments.
        checks["E0c_cfl_matches_tag"] = (
            abs(float(out["cfl"]) - instrument_cfl(prof["tag"])) < 1e-12)
    # rungs archived under artifacts/ carry no metrics sidecar; their metric-based
    # checks are *not measured* rather than passed, and they do not gate this round.
    out["metrics_available"] = bool(m)
    if not m:
        checks = {k: v for k, v in checks.items()
                  if k in ("E0a_dx", "E0b_positive_finite")}
    out["checks"] = checks
    out["pass"] = all(checks.values())
    return out


# --------------------------------------------------------------------------- #
# the verdict
# --------------------------------------------------------------------------- #
def compute() -> Dict:
    ref = load_profile(REF_FINEST, "weno5")
    if ref is None:
        raise FileNotFoundError(f"missing reference rung {profile_path(REF_FINEST, 'weno5')}")
    ref_coarse = load_profile(6400, "weno5")

    graded: Dict[int, Dict] = {}
    for nx in GRADED:
        prof = first_available(nx, GRADED_TAGS)
        if prof is not None:
            graded[nx] = prof
    missing = [nx for nx in GRADED if nx not in graded]

    # ---- E0 genuineness of every graded rung (all are read by the verdict) ----
    gates = {str(nx): genuineness(nx, graded[nx]) for nx in sorted(graded)}
    gates_ok = all(g["pass"] for g in gates.values())

    # ---- E1 / E1r: CFL admission controls (same grid, CFL 0.4 vs 0.8) ----
    def cfl_pair(nx: int, canonical: str, relaxed: str) -> Optional[Dict]:
        a, b = load_profile(nx, canonical), load_profile(nx, relaxed)
        if a is None or b is None:
            return None
        d = GH.l1_percent(a["x"], a["rho"], b["x"], b["rho"])
        d_peak = GH.peak_percent(a["x"], a["rho"], b["x"], b["rho"])
        return {"nx": nx, "canonical_tag": a["tag"], "relaxed_tag": b["tag"],
                "l1_percent": d, "peak_percent": d_peak,
                "admitted": bool(d < CFL_ADMISSION),
                "threshold_percent": CFL_ADMISSION}

    e1 = cfl_pair(6400, "hllc", "hllc_cfl0.8")
    e1r = cfl_pair(REF_FINEST, "weno5", "weno5_cfl0.8")
    admission_ok = bool(e1 and e1["admitted"])

    # ---- graded ladder vs the self-certified WENO5@12800 ----
    ladder: Dict[int, Dict] = {}
    for nx in sorted(graded):
        p = graded[nx]
        ladder[nx] = {
            "tag": p["tag"],
            "l1_percent": GH.l1_percent(p["x"], p["rho"], ref["x"], ref["rho"]),
            "l1_percent_independent": l1_percent_independent(p["x"], p["rho"],
                                                            ref["x"], ref["rho"]),
            "peak_percent": GH.peak_percent(p["x"], p["rho"], ref["x"], ref["rho"]),
            "mass_percent": GH.mass_percent(p["x"], p["rho"], ref["x"], ref["rho"]),
            "peak_density": float(np.max(p["rho"])),
        }

    orders = {}
    ks = sorted(ladder)
    for a, b in zip(ks, ks[1:]):
        orders[f"{a}->{b}"] = {
            "coarse_l1": ladder[a]["l1_percent"],
            "fine_l1": ladder[b]["l1_percent"],
            "p": GH._order(ladder[a]["l1_percent"], ladder[b]["l1_percent"]),
            "monotone": bool(ladder[b]["l1_percent"] < ladder[a]["l1_percent"]),
        }

    finest_pair = f"{ks[-2]}->{ks[-1]}" if len(ks) >= 2 else None
    p_obs = orders[finest_pair]["p"] if finest_pair in orders else float("nan")
    e_inf = (float("nan") if not np.isfinite(p_obs) else
             GH._asymptote_fixed_order(ladder[ks[-2]]["l1_percent"],
                                       ladder[ks[-1]]["l1_percent"], p_obs))
    floor_detected = bool(np.isfinite(e_inf) and e_inf > 0.0)
    floor = {
        "finest_pair": finest_pair,
        "order": p_obs,
        "e_inf_percent_raw": None if not np.isfinite(e_inf) else e_inf,
        "floor_detected": floor_detected,
        "floor_class": ("NO_FLOOR_DETECTED" if not floor_detected else
                        ("GRID_LIMITED" if e_inf < BAR_CERTIFIED else "SCHEME_LIMITED")),
        "predicted_l1_percent": {
            str(2 * ks[-1]): float(ladder[ks[-1]]["l1_percent"] * (2.0) ** (-p_obs))
        } if np.isfinite(p_obs) else {},
    }

    # ---- E2: was the pre-registered prediction hit? ----
    l1_finest = ladder[ks[-1]]["l1_percent"]
    band = (PREDICTED_L1_12800 * (1 - PREDICTION_BAND),
            PREDICTED_L1_12800 * (1 + PREDICTION_BAND))
    e2 = {
        "predicted_percent": PREDICTED_L1_12800,
        "band_percent": [band[0], band[1]],
        "measured_percent": l1_finest,
        "deviation_percent_of_prediction": 100.0 * (l1_finest - PREDICTED_L1_12800)
        / PREDICTED_L1_12800,
        "hit": bool(band[0] <= l1_finest <= band[1]),
        "order_within_band": bool(np.isfinite(p_obs) and p_obs <= 1.15),
        "monotone": bool(orders.get(finest_pair, {}).get("monotone", False)),
    }

    # ---- E3: certified gate (<1 %) on the finest rung ----
    peak_finest = ladder[ks[-1]]["peak_percent"]
    e3 = {
        "nx": ks[-1],
        "l1_below_1pct": bool(l1_finest < BAR_CERTIFIED),
        "peak_below_1pct": bool(peak_finest < BAR_CERTIFIED),
        "certified_under_1pct": bool(l1_finest < BAR_CERTIFIED
                                    and peak_finest < BAR_CERTIFIED),
        "scope": "vs self-certified WENO5@%d, NOT vs the exact solution" % REF_FINEST,
    }

    # ---- E4: honest ceiling with a reference bar taken at the *new* rung ----
    ref_ladder: List[Dict] = []
    for nx, tags in REF_LADDER_TAGS:
        prof = next((p for p in (load_profile(nx, t) for t in tags) if p is not None), None)
        if prof is None:
            continue
        row = {"nx": nx, "tag": prof["tag"], "cfl": instrument_cfl(prof["tag"])}
        if ref_ladder:
            prev = ref_ladder[-1]
            prev_prof = load_profile(prev["nx"], prev["tag"].split("_", 1)[1])
            row["l1_vs_prev_percent"] = GH.l1_percent(prev_prof["x"], prev_prof["rho"],
                                                      prof["x"], prof["rho"])
            row["p_vs_prev"] = (GH._order(prev["l1_vs_prev_percent"], row["l1_vs_prev_percent"])
                                if prev.get("l1_vs_prev_percent") else None)
        ref_ladder.append(row)

    bars = None
    e4: Dict = {"status": "not_measured",
                "reason": "reference-side pair (12800 -> 25600) not available"}
    if len(ref_ladder) >= 3:
        e_pair = ref_ladder[-1]["l1_vs_prev_percent"]
        e_prev = ref_ladder[-2]["l1_vs_prev_percent"]
        p_ref = GH._order(e_prev, e_pair)
        r_seq = [b["l1_vs_prev_percent"] / a["l1_vs_prev_percent"]
                 for a, b in zip(ref_ladder[1:], ref_ladder[2:])]
        bars = R11.bar_conventions(e_pair, p_ref, r_seq[-1] if r_seq else None, r_seq)
        asym = R11.asymptotic_check(e_pair, e_prev)
        asym["order"] = p_ref
        asym["order_ratio"] = (p_ref / asym["formal_order_weno5"]
                               if asym["formal_order_weno5"] else None)
        asym["asymptotic"] = bool(asym.get("monotone")
                                 and p_ref >= 0.5 * asym["formal_order_weno5"])
        rows = [{"nx": nx, "l1_percent": ladder[nx]["l1_percent"], "measured": True}
                for nx in ks]
        regrade = R11.regrade_under_bars(bars, rows)
        crit = R11.critical_order(bars, l1_finest)
        e4 = {
            "status": "measured",
            "reference_pair": ref_ladder,
            "e_percent": e_pair,
            "p_ref": p_ref,
            "ratio_sequence": r_seq,
            "asymptotic": asym,
            "bars": bars,
            "rows": regrade["rows"],
            "finest": regrade["finest"],
            "verdicts": regrade["verdicts"],
            "critical_order": crit,
            "b0_below_1pct": bool((regrade["finest"]["upper_bound_percent"].get("B0")
                                   or math.inf) < BAR_CERTIFIED),
            "caveat": ("reference-side 12800 rung is the CFL 0.8 variant "
                       f"(admitted={bool(e1r and e1r['admitted'])})"),
            "licensed": bool(len({r["cfl"] for r in ref_ladder}) == 1),
            "license_note": ("全部参考档同一 CFL 口径 ⇒ eps_ref 可直接作误差棒" if
                             len({r["cfl"] for r in ref_ladder}) == 1 else
                             "参考档跨 CFL 口径（混合仪器）⇒ eps_ref 记为探针："
                             "CFL 侧影响以已测 CFL 敏感性为上界（cfl_scaling_local.json，"
                             "0.048–0.056 % 量级），远小于该对差商，故 eps_ref 的一阶值是可信的，"
                             "但不得当作已认证的误差棒"),
        }

    # ---- assembly ----
    worst_indep = max(abs(ladder[nx]["l1_percent"] - ladder[nx]["l1_percent_independent"])
                      for nx in ks)
    verdict = {
        "status": ("measured" if (gates_ok and admission_ok and REF_FINEST in ks) else
                   "not_measured"),
        "gates_all_pass": gates_ok,
        "cfl_admission_pass": admission_ok,
        "missing_rungs": missing,
        "E0_genuineness": gates,
        "E1_cfl_graded": e1,
        "E1r_cfl_reference": e1r,
        "ladder": {str(nx): ladder[nx] for nx in ks},
        "orders": orders,
        "floor": floor,
        "E2_prediction": e2,
        "E3_certified": e3,
        "E4_honest_ceiling": e4,
        "independent_check_max_abs_diff_percent": worst_indep,
        "improvement_3200_to_finest_percent": (
            100.0 * (ladder[3200]["l1_percent"] - l1_finest) / ladder[3200]["l1_percent"]
            if 3200 in ladder else None),
        "bars": {"ledger": BAR_LEDGER, "three": BAR_THREE, "certified": BAR_CERTIFIED},
    }
    # one-line adjudication, kept in the artifact so the report cannot drift from it
    if verdict["status"] != "measured":
        verdict["adjudication"] = "本轮不裁决：入场门或工件门未过（见 E0/E1）"
    else:
        tags = []
        if e3["certified_under_1pct"]:
            tags.append(f"CERTIFIED_UNDER_1PCT_AT_NX{ks[-1]}")
        tags.append(floor["floor_class"])
        if e4.get("status") == "measured":
            passed = [k for k in ("B0", "B1", "B2", "B3", "B4")
                      if (e4["finest"]["upper_bound_percent"].get(k) or math.inf) < BAR_CERTIFIED]
            tags.append("HONEST_UNDER_1PCT_" + ("_".join(passed) if passed else "NONE"))
        verdict["adjudication"] = " | ".join(tags)
    return verdict


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #
def _f(x, nd=4):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "—"
    return f"{x:.{nd}f}"


def _eps(bar, key: str = "eps_percent"):
    """Read ``eps_percent``/``assumption`` from a convention entry (float-tolerant)."""
    if isinstance(bar, dict):
        return bar.get(key)
    return bar if key == "eps_percent" else "—"


def _gate_row(name: str, d: Dict) -> str:
    if not d:
        return f"| {name} | — | — | — | — | not_measured |"
    return (f"| {name} | {d['nx']} | `{d['canonical_tag']}` vs `{d['relaxed_tag']}` | "
            f"{_f(d['l1_percent'])} | {_f(d['peak_percent'])} | "
            f"{'✅ 准入' if d['admitted'] else '❌ 拒绝'} |")


def render(v: Dict) -> str:
    ks = sorted(int(k) for k in v["ladder"])
    gate_rows = "\n".join([_gate_row("E1（被定级族）", v.get("E1_cfl_graded")),
                           _gate_row("E1r（参考族）", v.get("E1r_cfl_reference"))])
    rows = "\n".join(
        f"| {nx} | {_f(v['ladder'][str(nx)]['l1_percent'])} | "
        f"{_f(v['ladder'][str(nx)]['peak_percent'])} | "
        f"{_f(v['ladder'][str(nx)]['mass_percent'])} | "
        f"{'✅' if v['ladder'][str(nx)]['l1_percent'] < 1.0 else '❌'} | "
        f"`{v['ladder'][str(nx)]['tag']}` |" for nx in ks)
    orders = "\n".join(
        f"| {k} | {_f(o['coarse_l1'])} | {_f(o['fine_l1'])} | {_f(o['p'])} | "
        f"{'yes' if o['monotone'] else 'no'} |" for k, o in v["orders"].items())
    e4 = v["E4_honest_ceiling"]
    if e4.get("status") == "measured":
        conv_head = "| 口径 | eps_ref [%] | 承重假设 |\n|---|---|---|\n" + "\n".join(
            f"| {k} | {_f(_eps(e4['bars'][k]), 4)} | {_eps(e4['bars'][k], 'assumption')} |"
            for k in ("B0", "B1", "B2", "B3", "B4") if k in e4["bars"])
        bound_head = ("| nx | L1 [%] | B0 | B1 | B2 | B3 | B4 |\n|---|---|---|---|---|---|---|"
                      + "\n" + "\n".join(
                          f"| {r['nx']} | {_f(r['l1_percent'])} | "
                          + " | ".join(_f(r["upper_bound_percent"].get(k))
                                       for k in ("B0", "B1", "B2", "B3", "B4")) + " |"
                          for r in e4["rows"]))
        surv = ("| 口径 | <1.5 % | <1 % | 总账 |\n|---|---|---|---|\n" + "\n".join(
            f"| {k} | {d['E4']} | {d['E5']} | {d['ledger']} |"
            for k, d in e4["verdicts"].items()))
    else:
        conv_head = bound_head = surv = f"_未测_：{e4.get('reason')}"
    return f"""# Round-13：被定级族的 nx=12800 档、CFL 0.8 入场门与诚实上限

> 锚点 `XVAL-FVM-M18737-ROUND2`，验收 <15 %。判据先于数字冻结于 `PREDECLARATION-r13.md`。

## ① 工件门（E0）

| 档 | cells | dx 相对误差 | ρmin | ρmax | steps | 秒 | cfl | mass drift | 通过 |
|---|---|---|---|---|---|---|---|---|---|
""" + "\n".join(
        f"| `{g['tag']}` | {g['cells']} | {g['dx_rel_error']:.2e} | {g['rho_min']:.4f} | "
        f"{g['rho_max']:.4f} | {g['steps']} | {_f(g['seconds_solver'], 1)} | {g['cfl']} | "
        f"{g['mass_drift']} | {'✅' if g['pass'] else '❌'} |"
        for g in v["E0_genuineness"].values()) + f"""

## ② CFL 入场门（E1 / E1r，阈值 {CFL_ADMISSION} %）

| 门 | 档 | 口径对 | L1 [%] | 峰值 [%] | 准入 |
|---|---|---|---|---|---|
""" + gate_rows + f"""
## ③ 被定级族逐档（对自认证 WENO5@{REF_FINEST}）

| nx | L1 [%] | 峰值 [%] | 窗质量 [%] | L1 <1 % | 档 tag |
|---|---|---|---|---|---|
{rows}

| 相邻档 | 粗档 L1 | 细档 L1 | 观测阶 p | 单调 |
|---|---|---|---|---|
{orders}

地板判定：`{v['floor']['floor_class']}`（最细对 {v['floor']['finest_pair']}，
`e_inf = {_f(v['floor']['e_inf_percent_raw'])}`，p = {_f(v['floor']['order'])}）。

## ④ E2 预注册预测命中

| 项 | 值 |
|---|---|
| 预测 L1({ks[-1]}) | {_f(v['E2_prediction']['predicted_percent'], 3)} % |
| 预注册带 ±15 % | {_f(v['E2_prediction']['band_percent'][0], 3)}–{_f(v['E2_prediction']['band_percent'][1], 3)} % |
| 实测 | **{_f(v['E2_prediction']['measured_percent'])} %** |
| 偏差 | {_f(v['E2_prediction']['deviation_percent_of_prediction'], 1)} % |
| 命中 | {'✅' if v['E2_prediction']['hit'] else '❌'} |

## ⑤ E4 诚实上限（参考侧误差棒取新档位）

{conv_head}

{bound_head}

{surv}

## ⑥ 裁决

`{v['adjudication']}`
"""

# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def figures(v: Dict, outdir: Path) -> List[Dict]:
    """Three panels: the ladder, the honest ceiling per convention, the CFL controls."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    made: List[Dict] = []
    ks = sorted(int(k) for k in v["ladder"])
    es = [v["ladder"][str(nx)]["l1_percent"] for nx in ks]

    fig, ax = plt.subplots(1, 3, figsize=(15.0, 4.4))
    ax[0].loglog(ks, es, "o-", color="#1f77b4", label="MUSCL/HLLC (measured)")
    if v["floor"]["predicted_l1_percent"]:
        nx_next = int(next(iter(v["floor"]["predicted_l1_percent"])))
        ax[0].loglog([ks[-1], nx_next],
                     [es[-1], v["floor"]["predicted_l1_percent"][str(nx_next)]],
                     "s--", color="#999999", label="power-law prediction")
    ax[0].axhline(1.0, color="crimson", ls=":", label="<1 % certified bar")
    ax[0].axhline(v["bars"]["ledger"], color="k", ls="-.", lw=0.8, label="<15 % ledger")
    ax[0].set_xlabel("nx")
    ax[0].set_ylabel("L1 vs WENO5@%d [%%]" % REF_FINEST)
    ax[0].set_title("(a) graded ladder incl. nx=12800")
    ax[0].grid(True, which="both", alpha=0.3)
    ax[0].legend(fontsize=7)

    e4 = v["E4_honest_ceiling"]
    if e4.get("status") == "measured":
        keys = ["B0", "B1", "B2", "B3", "B4"]
        for key, col in zip(keys, ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]):
            ys = [r["upper_bound_percent"].get(key) for r in e4["rows"]]
            xs = [r["nx"] for r in e4["rows"] if r["upper_bound_percent"].get(key) is not None]
            yv = [y for y in ys if y is not None]
            if yv:
                ax[1].loglog(xs, yv, "s-", color=col, label=key)
        ax[1].loglog([r["nx"] for r in e4["rows"]], [r["l1_percent"] for r in e4["rows"]],
                     "ko--", label="point estimate")
        ax[1].axhline(1.0, color="crimson", ls=":", label="<1 %")
        ax[1].axhline(1.5, color="#bbbbbb", ls="--", lw=0.8, label="<1.5 %")
        ax[1].set_xlabel("nx (graded)")
        ax[1].set_ylabel("L1 + eps_ref [%%]")
        ax[1].set_title("(b) honest upper bound per error-bar convention")
        ax[1].grid(True, which="both", alpha=0.3)
        ax[1].legend(fontsize=7)
    else:
        ax[1].text(0.5, 0.5, "not measured", ha="center", va="center")
        ax[1].set_axis_off()

    labels, vals = [], []
    for name, d in (("E1 graded\nnx=6400", v["E1_cfl_graded"]),
                    ("E1r reference\nnx=12800", v["E1r_cfl_reference"])):
        if d:
            labels.append(name)
            vals.append(d["l1_percent"])
    if vals:
        ax[2].bar(labels, vals, color="#4c72b0")
        ax[2].axhline(CFL_ADMISSION, color="crimson", ls=":", label="admission %.1f %%" % CFL_ADMISSION)
        ax[2].set_ylabel("L1 [%]  CFL 0.4 vs 0.8, same grid")
        ax[2].set_title("(c) CFL admission controls")
        ax[2].legend(fontsize=7)
    else:
        ax[2].text(0.5, 0.5, "not measured", ha="center", va="center")
        ax[2].set_axis_off()

    fig.suptitle("XVAL-FVM-M18737-ROUND2 R13: nx=12800 rung, CFL admission, honest ceiling")
    fig.tight_layout()
    p = outdir / "fig_r13_certified_rung.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    made.append({"path": str(p), "name": p.name,
                 "caption": "(a) 被定级族 L1 阶梯（含 nx=12800 实测）与幂律预测、1 % / 15 % 判据线；"
                            "(b) 五种误差棒口径下的诚实上界（L1 + eps_ref，参考侧取新档位）；"
                            "(c) CFL 0.4 vs 0.8 的同格准入控制与 0.2 % 阈值。"})
    return made


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", type=Path, default=HERE / "r13_certified_rung.json")
    ap.add_argument("--markdown", type=Path, default=None)
    ap.add_argument("--figures", type=Path, default=HERE / "figures")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    v = compute()
    if not args.no_figures:
        v["figures"] = figures(v, args.figures)
    args.json.write_text(json.dumps(v, indent=2) + "\n")
    md = render(v)
    if args.markdown:
        args.markdown.write_text(md)
    print(json.dumps({"status": v["status"], "adjudication": v["adjudication"],
                      "missing_rungs": v["missing_rungs"],
                      "finest_l1": v["ladder"][str(max(int(k) for k in v["ladder"]))]["l1_percent"]
                      if v["ladder"] else None,
                      "json": str(args.json)}))


if __name__ == "__main__":
    main()

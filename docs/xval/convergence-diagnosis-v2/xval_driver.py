"""XVAL-FVM-CONVERGENCE-DIAGNOSIS-V2 driver.

Cross-validates TensorFVM's ``convergence-diagnosis-v2`` estimators
(:mod:`tensorfvm.verification.sod1978_convergence`: adjacent-grid successive
rates and the least-squares "fitted exponent") against the mature three-grid GCI
procedure of the open-source / standards practice (Roache 1994; Celik, Ghia,
Roache, Freitas, Coleman & Raad, *ASME J. Fluids Eng.* **130** (2008) 078001 --
the reporting procedure followed by OpenFOAM/SU2-style post-processing and, in
the marine domain, by ITTC 7.5-03-01-01).

Three faces participate, and no number in the artifact is typed by hand:

``ref``
    :mod:`gci_reference` -- independent implementation written from the
    published formulas; imports nothing from ``tensorfvm``, so agreement cannot
    be an artefact of shared code.
``repo_gci``
    ``tensorfvm.ship_freesurface.mesh_convergence.richardson_extrapolation`` --
    the GCI implementation already in the tree, used to show the independent
    face is not a peculiar re-derivation.
``got``
    the v2 estimators, imported from the module under cross-validation and
    applied to exactly the same data (v2 fits the *error* series of a wave
    position, so the comparison gives it exactly that series).

Predeclared before any number was inspected:

* estimators are interchangeable iff the observed order agrees within 15 %
  (owner precision standard) -- and that requirement is only imposed where the
  sequence is demonstrably in the asymptotic range (overlapping-triplet
  consistency <= 15 %, the non-vacuous form of Celik eq. (17));
* the same 15 % applies to derived quantities (extrapolated functional, GCI);
* ``Fs = 1.25`` (three grids); ``band = 0.15``.

Design point the cross-validation turned up: the published eq. (17) ratio
``GCI_medium/(r^p GCI_fine)`` is an *identity* when ``p`` comes from the same
triplet -- it cannot detect a non-asymptotic sequence.  The driver therefore
also runs a four-grid (100/200/400/800) overlapping-triplet consistency test,
which can.

Run::

    PYTHONPATH=src python docs/xval/convergence-diagnosis-v2/xval_driver.py --out <dir>

Runtime ~4 min on one core, deterministic (fixed grids, CFL, no randomness).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))

from gci_reference import (  # noqa: E402  (independent reference face)
    ASYMPTOTIC_BAND,
    FS_3GRID,
    crosscheck_series,
    gci,
    gci_absolute,
    limit_form,
    richardson_extrapolated,
    triplet_consistency,
)

THRESHOLDS: Dict[str, float] = {
    "p_rel_diff": 0.15,
    "value_rel_diff": 0.15,
    "asymptotic_band": ASYMPTOTIC_BAND,
    "fs": FS_3GRID,
}

POSITION_FUNCTIONALS = ("shock_x", "contact_x")
ERROR_FUNCTIONALS = ("l1_rho_full", "l1_rho_cellavg", "l1_rho_plateaus",
                     "smooth_wave_l1")


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _finite(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def _num(x) -> float:
    return float(x) if isinstance(x, (int, float)) else float("nan")


def _rel_diff(a, b) -> float:
    if not (_finite(a) and _finite(b)) or float(b) == 0.0:
        return float("nan")
    return abs(float(a) - float(b)) / abs(float(b))


# ----------------------------------------------------------------------
# data acquisition (measurements only -- no diagnosis here)
# ----------------------------------------------------------------------
def _sod_row(nx: int, order: int, cfl: float, t_end: float) -> Dict:
    from tensorfvm.verification.sod1978_convergence import measure

    row = measure(nx, order=order, cfl=cfl, t_end=t_end)
    return {
        "nx": int(nx), "dx": float(row["dx"]), "order": int(order),
        "cfl": float(cfl), "t_end": float(row["t_end"]),
        "steps": int(row["steps"]), "seconds": float(row["seconds"]),
        "l1_rho_full": float(row["l1_rho_full"]),
        "l1_rho_cellavg": float(row["l1_rho_cellavg"]),
        "l1_rho_plateaus": float(row["l1_rho_plateaus"]),
        "l1_rho_restricted": float(row["l1_rho_restricted"]),
        "shock_x": float(row["waves"]["shock_x"]),
        "contact_x": float(row["waves"]["contact_x"]),
        "shock_exact": float(row["exact_waves"]["shock_x"]),
        "contact_exact": float(row["exact_waves"]["contact_x"]),
        "positivity_fixes": int(row["positivity_fixes"]),
        "budget_drift_mass": float(row["budget_drift_mass"]),
    }


def _smooth_row(nx: int, order: int, cfl: float = 0.4) -> Dict:
    from tensorfvm.verification.sod1978 import run_smooth_wave

    run = run_smooth_wave(nx, order=order, cfl=cfl)
    return {"nx": int(nx), "dx": 1.0 / nx, "order": int(order),
            "smooth_wave_l1": float(run["l1_rho"]), "steps": int(run["steps"])}


def acquire(grids: Sequence[int], order: int = 2, cfl: float = 0.4,
            log=print) -> Dict:
    t0 = time.time()
    sod_rows, smooth_rows = [], []
    for nx in grids:
        row = _sod_row(nx, order, cfl, 0.14)
        sod_rows.append(row)
        log(f"[sod o{order} cfl={cfl}] nx={nx} L1={row['l1_rho_full']:.4e} "
            f"shock={row['shock_x']:.8f} ({row['seconds']:.1f}s)")
    for nx in grids:
        row = _smooth_row(nx, order, cfl)
        smooth_rows.append(row)
        log(f"[smooth o{order} cfl={cfl}] nx={nx} L1={row['smooth_wave_l1']:.4e}")
    return {"grids": list(grids), "order": order, "cfl": cfl,
            "sod_rows": sod_rows, "smooth_rows": smooth_rows,
            "wall_seconds": round(time.time() - t0, 1)}


def merge_sweeps(*sweeps: Dict) -> Dict:
    """Union of sweeps with identical physics, one row per grid (deduplicated)."""
    order = sweeps[0]["order"]
    cfl = sweeps[0]["cfl"]
    sod: Dict[int, Dict] = {}
    smooth: Dict[int, Dict] = {}
    for sw in sweeps:
        assert sw["order"] == order and sw["cfl"] == cfl, "physics mismatch"
        for r in sw["sod_rows"]:
            sod[int(r["nx"])] = r
        for r in sw["smooth_rows"]:
            smooth[int(r["nx"])] = r
    return {"grids": sorted(sod), "order": order, "cfl": cfl,
            "sod_rows": [sod[k] for k in sorted(sod)],
            "smooth_rows": [smooth[k] for k in sorted(smooth)],
            "wall_seconds": 0.0,
            "assembled_from": [list(sw["grids"]) for sw in sweeps]}


# ----------------------------------------------------------------------
# diagnosis: ref (independent), repo_gci (in-tree GCI), got (v2 estimators)
# ----------------------------------------------------------------------
def _v2_estimators(errors: Sequence[float], nx: Sequence[int]) -> Dict:
    """Apply the v2 estimators *from the module under XVAL* to ``errors``."""
    from tensorfvm.verification.sod1978_convergence import (
        fitted_exponent,
        successive_orders,
    )

    rows = [{"nx": int(n), "q": float(v)} for n, v in zip(nx, errors)]
    pairs = successive_orders(rows, "q")
    return {
        "successive": pairs,
        "successive_finest": float(pairs[-1]["rate"]) if pairs else float("nan"),
        "loglog_exponent_finest3": fitted_exponent(rows, "q"),
    }


def _functionals_for_sweep(sweep: Dict) -> Dict[str, Dict[str, object]]:
    """Physics of each cross-validated functional: what ref fits, what v2 fits."""
    rows = sweep["sod_rows"]
    smooth = {int(r["nx"]): r for r in sweep["smooth_rows"]}
    out: Dict[str, Dict[str, object]] = {}
    for name in POSITION_FUNCTIONALS:
        exact = float(rows[-1][f"{name.split('_')[0]}_exact"])
        phi = [float(r[name]) for r in rows]
        out[name] = {"phi": phi, "err": [abs(v - exact) for v in phi], "exact": exact}
    for name in ERROR_FUNCTIONALS:
        if name == "smooth_wave_l1":
            phi = [float(smooth[int(r["nx"])]["smooth_wave_l1"]) for r in rows]
            out[name] = {"phi": phi, "err": list(phi), "exact": 0.0}
        else:
            phi = [float(r[name]) for r in rows]
            out[name] = {"phi": phi, "err": list(phi), "exact": 0.0}
    return out


def diagnose(sweep: Dict) -> Dict:
    from tensorfvm.ship_freesurface.mesh_convergence import richardson_extrapolation

    rows = sweep["sod_rows"]
    nx = [int(r["nx"]) for r in rows]
    h = [float(r["dx"]) for r in rows]
    functionals = _functionals_for_sweep(sweep)
    out: Dict[str, Dict] = {}
    for name, spec in functionals.items():
        phi = [float(v) for v in spec["phi"]]
        err = [float(v) for v in spec["err"]]
        ref = crosscheck_series(phi, h, fs=THRESHOLDS["fs"],
                                band=THRESHOLDS["asymptotic_band"])
        p_ref = _num(ref["observed_order"])
        r = float(ref["r21"])
        # declared from the physics of the functional (an error norm has a zero
        # limit by construction); limit_form() is kept only as a diagnostic
        lf = "nonzero_limit" if name in POSITION_FUNCTIONALS else "zero_limit"
        lf_heuristic = limit_form(phi, h, p_ref)
        got = _v2_estimators(err, nx)
        p_lsq = _num(got["loglog_exponent_finest3"])
        p_succ = _num(got["successive_finest"])
        # same triplet as the reference face (the finest three), otherwise the
        # two implementations would be reading different grids on >=4-grid sweeps
        repo = richardson_extrapolation(phi[-3], phi[-2], phi[-1], r=r,
                                        Fs=THRESHOLDS["fs"])
        entry: Dict = {
            "phi": phi, "err_fitted_by_v2": err, "h": h, "nx": nx,
            "exact_limit": spec["exact"],
            "limit_form": lf,
            "limit_form_heuristic": lf_heuristic,
            "ref": ref, "got": got,
            "agreement_ref_vs_got": {
                "p_ref": p_ref,
                "p_got_loglog": p_lsq,
                "p_got_successive": p_succ,
                "rel_diff_loglog": _rel_diff(p_lsq, p_ref),
                "rel_diff_successive": _rel_diff(p_succ, p_ref),
                "pass_loglog": bool(_finite(_rel_diff(p_lsq, p_ref))
                                    and _rel_diff(p_lsq, p_ref) <= THRESHOLDS["p_rel_diff"]),
                "pass_successive": bool(_finite(_rel_diff(p_succ, p_ref))
                                        and _rel_diff(p_succ, p_ref) <= THRESHOLDS["p_rel_diff"]),
            },
            "agreement_ref_vs_repo_gci": {
                "p_ref": p_ref, "p_repo_gci": _num(repo.p),
                "rel_diff_p": _rel_diff(_num(repo.p), p_ref),
                "extrapolated_ref": _num(ref["extrapolated"]),
                "extrapolated_repo_gci": _num(repo.S_extrapolated),
                "rel_diff_extrapolated": _rel_diff(_num(repo.S_extrapolated),
                                                   _num(ref["extrapolated"])),
                "gci_ref": _num(ref["gci_fine"]),
                "gci_repo_gci": _num(repo.GCI_fine),
                "rel_diff_gci": _rel_diff(_num(repo.GCI_fine), _num(ref["gci_fine"])),
                "class_ref": str(ref["convergence_class"]),
                "class_repo_gci": str(repo.convergence_type),
            },
        }
        ext_ref = _num(ref["extrapolated"])
        ext_from_got = (richardson_extrapolated(phi[-3:], r, p_lsq)
                        if _finite(p_lsq) and p_lsq > 0 else float("nan"))
        gci_rel_ref = _num(ref["gci_fine"])
        gci_abs_ref = gci_absolute(phi[-3:], r, p_ref, fs=THRESHOLDS["fs"]) \
            if _finite(p_ref) and p_ref > 0 else float("nan")
        gci_from_got = (gci(phi[-3:], r, p_lsq, fs=THRESHOLDS["fs"])
                        if _finite(p_lsq) and p_lsq > 0 else float("nan"))
        gci_abs_from_got = (gci_absolute(phi[-3:], r, p_lsq, fs=THRESHOLDS["fs"])
                            if _finite(p_lsq) and p_lsq > 0 else float("nan"))
        derived = {
            "extrapolated_ref": ext_ref,
            "extrapolated_from_got_exponent": ext_from_got,
            "rel_diff_extrapolated": _rel_diff(ext_from_got, ext_ref),
            "gci_relative_ref": gci_rel_ref,
            "gci_absolute_ref": gci_abs_ref,
            "gci_absolute_from_got_exponent": gci_abs_from_got,
            "gci_relative_from_got_exponent": gci_from_got,
            "relative_form_applicable": lf == "nonzero_limit",
        }
        if lf == "nonzero_limit" and _finite(ext_ref) and _finite(gci_rel_ref):
            ex = float(spec["exact"])
            derived["exact"] = ex
            derived["relerr_fine"] = _rel_diff(phi[-1], ex)
            derived["relerr_extrapolated_ref"] = _rel_diff(ext_ref, ex)
            derived["gci_covers_exact"] = bool(
                abs(ext_ref - ex) <= gci_rel_ref * abs(ext_ref))
        entry["derived_values"] = derived
        out[name] = entry
    return out


# ----------------------------------------------------------------------
# committed v2 artifact: re-diagnose its own numbers, show the verdict flip
# ----------------------------------------------------------------------
def committed_diagnosis(path: Path) -> Dict:
    art = json.loads(path.read_text(encoding="utf-8"))
    rows = art["tables"]["order2"]
    nx = [int(r["nx"]) for r in rows]
    h = [float(r["dx"]) for r in rows]
    shock_exact = float(rows[-1]["exact_waves"]["shock_x"])
    contact_exact = float(rows[-1]["exact_waves"]["contact_x"])
    series = {
        "shock_position_error": [abs(r["waves"]["shock_x"] - shock_exact) for r in rows],
        "contact_position_error": [abs(r["waves"]["contact_x"] - contact_exact) for r in rows],
    }
    # reference face on the same numbers
    ref = {k: crosscheck_series(v, h, fs=THRESHOLDS["fs"],
                                band=THRESHOLDS["asymptotic_band"])
           for k, v in series.items()}
    # before: what the published artifact recorded and read in its verdict
    before = {
        "grids": nx,
        "contact_exponent_finest3": art["wave_rates"]["order2"]["wave_error.contact_x"]
        ["fitted_exponent_finest3"],
        "shock_exponent_finest3": art["wave_rates"]["order2"]["wave_error.shock_x"]
        ["fitted_exponent_finest3"],
        "contact_successive": [p["rate"] for p in art["wave_rates"]["order2"]
                               ["wave_error.contact_x"]["successive"]],
        "shock_successive": [p["rate"] for p in art["wave_rates"]["order2"]
                             ["wave_error.shock_x"]["successive"]],
        "sublinear_convergence_observed": art["verdicts"]["sublinear_convergence_observed"],
    }
    # after: guarded exponents of the patched module, identical numbers
    from tensorfvm.verification.convergence_crosscheck import effective_exponent

    after: Dict[str, object] = {}
    for label, key in (("contact", "contact_position_error"),
                       ("shock", "shock_position_error")):
        got = effective_exponent(series[key], h)
        after[f"{label}_exponent"] = _num(got["value"])
        after[f"{label}_source"] = got["source"]
        after[f"{label}_loglog_admissible"] = bool(got["valid"])
        after[f"{label}_class"] = got["class"]
        after[f"{label}_reason"] = got["reason"]
    after["sublinear_convergence_observed"] = bool(
        art["verdicts"]["c3_full_exponent_finest3_order2"] < 1.0
        and _finite(after["contact_exponent"]) and _finite(after["shock_exponent"])
        and after["contact_exponent"] < after["shock_exponent"])
    after["ordering_holds_with_finest_successive"] = bool(
        before["contact_successive"][-1] < before["shock_successive"][-1])
    after["ordering_holds_with_effective_exponent"] = bool(
        _finite(after["contact_exponent"]) and _finite(after["shock_exponent"])
        and after["contact_exponent"] < after["shock_exponent"])
    return {
        "artifact": str(path.relative_to(ROOT)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "before": before,
        "after": after,
        "reference": {
            k: {"class": v["convergence_class"],
                "observed_order": _num(v["observed_order"]),
                "asymptotic_ratio": _num(v["asymptotic_ratio"]),
                "gci_relative_fine": _num(v["gci_fine"])}
            for k, v in ref.items()},
        "series": series,
    }


# ----------------------------------------------------------------------
# summary / markdown / figures
# ----------------------------------------------------------------------
def summary_rows(sweeps: Dict[str, Dict], sweep_cfgs: Dict[str, Dict]) -> List[Dict]:
    out: List[Dict] = []
    for label, diag in sweeps.items():
        for name, e in diag.items():
            agr = e["agreement_ref_vs_got"]
            ref = e["ref"]
            dv = e["derived_values"]
            consistency = ref.get("triplet_consistency") or {}
            out.append({
                "sweep": label,
                "grids": sweep_cfgs[label]["grids"],
                "functional": name,
                "limit_form": e["limit_form"],
                "class_ref": ref["convergence_class"],
                "p_ref": agr["p_ref"],
                "p_got_loglog": agr["p_got_loglog"],
                "p_got_successive": agr["p_got_successive"],
                "rel_diff_loglog": agr["rel_diff_loglog"],
                "rel_diff_successive": agr["rel_diff_successive"],
                "pass_loglog": agr["pass_loglog"],
                "pass_successive": agr["pass_successive"],
                "asymptotic_ratio": _num(ref["asymptotic_ratio"]),
                "triplet_p_coarse": _num(consistency.get("p_triplet_coarse")),
                "triplet_p_fine": _num(consistency.get("p_triplet_fine")),
                "triplet_consistent": consistency.get("consistent"),
                "gci_relative_ref": dv["gci_relative_ref"],
                "gci_absolute_ref": dv["gci_absolute_ref"],
                "gci_absolute_from_got": dv["gci_absolute_from_got_exponent"],
                "extrapolated_ref": dv["extrapolated_ref"],
                "extrapolated_from_got": dv["extrapolated_from_got_exponent"],
                "rel_diff_extrapolated": dv["rel_diff_extrapolated"],
                "relerr_fine": dv.get("relerr_fine"),
                "relerr_extrapolated_ref": dv.get("relerr_extrapolated_ref"),
                "gci_covers_exact": dv.get("gci_covers_exact"),
            })
    return out


def _fmt(x, nd=3) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, bool):
        return "✅" if x else "❌"
    if isinstance(x, (int, float)):
        if not math.isfinite(float(x)):
            return "n/a"
        return f"{float(x):.{nd}f}"
    return str(x)


def _pct(x) -> str:
    if x is None or not isinstance(x, (int, float)) or not math.isfinite(float(x)):
        return "n/a"
    return f"{float(x) * 100:.2f}%"


def render_markdown(artifact: Dict) -> str:
    L: List[str] = []
    a = L.append
    a("# XVAL-FVM-CONVERGENCE-DIAGNOSIS-V2 — ref vs got（程序生成，勿手改）")
    a("")
    a("回归命令：`PYTHONPATH=src python docs/xval/convergence-diagnosis-v2/xval_driver.py "
      "--out docs/xval/convergence-diagnosis-v2`")
    a("")
    a(f"- 生成时刻 `{artifact['generated_at']}`；环境 {artifact['environment']}")
    a("- ref = 独立实现 `gci_reference.py`（Roache 1994 / Celik et al. 2008 三网格 GCI，"
      "不 import 本仓任何模块）")
    a("- got = 本仓 `tensorfvm.verification.sod1978_convergence` 的相邻对速率与 "
      "log-log 最小二乘指数（v2 语义：对**误差**序列拟合）")
    a(f"- 预声明：p 相对差 ≤ {artifact['thresholds']['p_rel_diff']:.0%}；导出量相对差 ≤ "
      f"{artifact['thresholds']['value_rel_diff']:.0%}；重叠三重组一致性 ≤ "
      f"{artifact['thresholds']['asymptotic_band']:.0%}；Fs={artifact['thresholds']['fs']}")
    a("")
    a("## A. 观测阶（ref vs got）")
    a("")
    a("| sweep（网格） | 功能量 | 极限形式 | ref 收敛类 | p_ref | p_got(lsq) | p_got(相邻对) | "
      "Δlsq | Δsucc | lsq | succ | 单三重组 eq.17 比 |")
    a("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in artifact["summary"]:
        a(f"| {r['sweep']} `{r['grids']}` | {r['functional']} | {r['limit_form']} | "
          f"{r['class_ref']} | {_fmt(r['p_ref'])} | {_fmt(r['p_got_loglog'])} | "
          f"{_fmt(r['p_got_successive'])} | {_pct(r['rel_diff_loglog'])} | "
          f"{_pct(r['rel_diff_successive'])} | {_fmt(r['pass_loglog'])} | "
          f"{_fmt(r['pass_successive'])} | {_fmt(r['asymptotic_ratio'], 2)} |")
    a("")
    a("## B. 重叠三重组一致性（≥4 网格，非空转的渐近区检验）")
    a("")
    a("| sweep | 功能量 | p(粗三重组) | p(细三重组) | 相对差 | 渐近区一致 |")
    a("|---|---|---|---|---|---|")
    for r in artifact["summary"]:
        if r["triplet_p_coarse"] is None and r["triplet_consistent"] is None:
            continue
        a(f"| {r['sweep']} | {r['functional']} | {_fmt(r['triplet_p_coarse'])} | "
          f"{_fmt(r['triplet_p_fine'])} | "
          f"{_fmt(None if r['triplet_consistent'] is None else abs(r['triplet_p_fine'] - r['triplet_p_coarse']) / abs(r['triplet_p_coarse']) if (r['triplet_p_coarse'] and math.isfinite(r['triplet_p_coarse']) and math.isfinite(r['triplet_p_fine'])) else float('nan'), 3)} | "
          f"{_fmt(r['triplet_consistent'])} |")
    a("")
    a("## C. 导出量（Richardson 外推 / GCI 带 / 对解析真值）")
    a("")
    a("| sweep | 功能量 | 外推(ref) | 外推(用 got 的 p) | Δ | GCI(abs, ref) | GCI(abs, got p) | "
      "细网格相对误差 | 外推相对误差 | GCI 覆盖真值 |")
    a("|---|---|---|---|---|---|---|---|---|---|")
    for r in artifact["summary"]:
        if r["limit_form"] != "nonzero_limit":
            continue
        a(f"| {r['sweep']} | {r['functional']} | {_fmt(r['extrapolated_ref'], 6)} | "
          f"{_fmt(r['extrapolated_from_got'], 6)} | {_pct(r['rel_diff_extrapolated'])} | "
          f"{_fmt(r['gci_absolute_ref'], 6)} | {_fmt(r['gci_absolute_from_got'], 6)} | "
          f"{_pct(r['relerr_fine'])} | {_pct(r['relerr_extrapolated_ref'])} | "
          f"{_fmt(r['gci_covers_exact'])} |")
    a("")
    a("零极限功能量（L1 范数类）只比观测阶：它们的相对 GCI 是消失量的百分比，"
      "单三重组 eq.17 比退化为 `r^-p`，两者都不承载信息（见 REPORT ⑤）。")
    a("")
    com = artifact["committed_v2_artifact"]
    if com:
        a("## D. 已入库 v2 工件（`docs/repro-fvm-sod1978-convergence/study.json`）的重诊断")
        a("")
        a(f"- 工件 sha256 `{com['sha256'][:16]}…`，网格 `{com['before']['grids']}`（r=2）")
        a(f"- published 激波位置 log-log 指数 = `{com['before']['shock_exponent_finest3']:.4f}`；"
          f"相邻对速率 = `{com['before']['shock_successive'][-1]:.4f}`")
        a(f"- ref 对同一序列的收敛类 = `{com['reference']['shock_position_error']['class']}`"
          f"（拒绝给出阶：p=NaN）")
        a(f"- 判据翻转：`sublinear_convergence_observed` "
          f"before={com['before']['sublinear_convergence_observed']} → "
          f"after={com['after']['sublinear_convergence_observed']}")
        a(f"- contact 有效指数 `{com['after']['contact_exponent']:.4f}`"
          f"（来源 {com['after']['contact_source']}），"
          f"shock 有效指数 `{com['after']['shock_exponent']:.4f}`"
          f"（来源 {com['after']['shock_source']}）")
        a("")
    return "\n".join(L)


def make_figures(outdir: Path, artifact: Dict) -> List[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:                                        # pragma: no cover
        return []
    names: List[str] = []
    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    combined = artifact["sweeps"]["combined_100_200_400_800"]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    for ax, name, title in ((axes[0], "shock_x", "shock position error |x_s - x_s,exact|"),
                            (axes[1], "l1_rho_plateaus", "L1 error, plateau windows")):
        e = combined[name]
        h = np.array(e["h"], dtype=float)
        phi = np.array(e["phi"], dtype=float)
        err = np.array(e["err_fitted_by_v2"], dtype=float)
        ax.loglog(h, err, "o-", color="tab:blue", label="measured error")
        p_ref = e["agreement_ref_vs_got"]["p_ref"]
        p_lsq = e["agreement_ref_vs_got"]["p_got_loglog"]
        hh = np.array([h[-1], h[0]])
        for p, lab, ls, col in ((p_ref, f"ref triplet p={p_ref:.2f}", "--", "tab:red"),
                                (p_lsq, f"v2 lsq p={p_lsq:.2f}", ":", "tab:green")):
            if math.isfinite(p) and p > 0:
                scale = err[-1] / h[-1] ** p
                ax.loglog(hh, scale * hh ** p, ls, color=col, label=lab)
        ax.set_xlabel("h"); ax.set_ylabel("error"); ax.set_title(title)
        ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    png = figdir / "xval_error_vs_h.png"
    fig.savefig(png, dpi=150); plt.close(fig)
    names.append(str(png.relative_to(outdir)))

    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    xs = np.linspace(0.0, 3.2, 60)
    ax.fill_between(xs, xs * 0.85, xs * 1.15, color="grey", alpha=0.2,
                    label="±15 % band")
    ax.plot(xs, xs, "k-", lw=1, label="identity")
    for r in artifact["summary"]:
        if not (math.isfinite(r["p_ref"]) and math.isfinite(r["p_got_loglog"])):
            continue
        ok = r["pass_loglog"]
        ax.plot(r["p_ref"], r["p_got_loglog"], "o" if ok else "s",
                color="tab:green" if ok else "tab:red", ms=6)
        ax.annotate(f"{r['functional'].replace('_', '')[:14]}@{r['sweep'].split('_')[0]}",
                    (r["p_ref"], r["p_got_loglog"]), fontsize=5.5,
                    xytext=(3, -4), textcoords="offset points")
    ax.set_xlabel("observed order — reference (GCI triplet)")
    ax.set_ylabel("observed order — v2 (log-log LSQ)")
    ax.set_title("estimator agreement (green = within 15 %)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    png = figdir / "xval_order_agreement.png"
    fig.savefig(png, dpi=150); plt.close(fig)
    names.append(str(png.relative_to(outdir)))

    if artifact.get("committed_v2_artifact"):
        com = artifact["committed_v2_artifact"]
        fig, ax = plt.subplots(figsize=(5.6, 4.2))
        h = np.array([1.0 / n for n in com["before"]["grids"]])
        for key, lab, col in (("shock_position_error", "shock position", "tab:red"),
                              ("contact_position_error", "contact position", "tab:blue")):
            err = np.array(com["series"][key], dtype=float)
            ax.loglog(h, err, "o-", color=col, label=lab)
        ax.set_xlabel("h"); ax.set_ylabel("position error")
        ax.set_title("committed v2 artifact, 50/100/200: non-monotone shock sequence")
        ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=8)
        fig.tight_layout()
        png = figdir / "xval_committed_artifact.png"
        fig.savefig(png, dpi=150); plt.close(fig)
        names.append(str(png.relative_to(outdir)))
    return names


# ----------------------------------------------------------------------
# verdicts
# ----------------------------------------------------------------------
def _verdicts(artifact: Dict) -> Dict:
    summary = artifact["summary"]
    combined_rows = [r for r in summary if r["sweep"].startswith("combined")]
    consistent = [r for r in combined_rows if r["triplet_consistent"] is True]
    inconsistent = [r for r in combined_rows if r["triplet_consistent"] is False]
    # decision rule (predeclared): agreement is required where the sequence is
    # in the asymptotic range (overlapping triplets agree within 15 %)
    required = consistent
    passes = [r for r in required if r["pass_loglog"] and r["pass_successive"]]
    fails = [r for r in required if not (r["pass_loglog"] and r["pass_successive"])]

    ref_agree: List[bool] = []
    ref_class_mismatch: List[str] = []
    for _, diag in artifact["sweeps"].items():
        for _, e in diag.items():
            cmp = e["agreement_ref_vs_repo_gci"]
            if _finite(cmp["p_ref"]) and _finite(cmp["p_repo_gci"]):
                ref_agree.append(cmp["rel_diff_p"] <= THRESHOLDS["p_rel_diff"])
            elif _finite(cmp["p_ref"]) != _finite(cmp["p_repo_gci"]):
                ref_class_mismatch.append(f"{name}")

    out = {
        "n_rows": len(summary),
        "note_non_vacuous_test": "single-triplet Celik eq.(17) ratio is an identity; "
                                 "the asymptotic-range claim above uses the "
                                 "overlapping-triplet consistency on 4 grids",
        "asymptotic_range_functionals": [f"{r['sweep']}/{r['functional']}"
                                         for r in consistent],
        "outside_asymptotic_range": [f"{r['sweep']}/{r['functional']}"
                                     for r in inconsistent],
        "agreement_required_rows": len(required),
        "agreement_pass_rows": len(passes),
        "agreement_fail_rows": [f"{r['sweep']}/{r['functional']}" for r in fails],
        "estimators_interchangeable_in_asymptotic_range": bool(len(fails) == 0),
        "both_reference_implementations_agree": bool(all(ref_agree)) if ref_agree else None,
        "n_numeric_comparisons_two_references": len(ref_agree),
        "reference_class_disagreements": ref_class_mismatch,
        "non_asymptotic_rows_where_single_triplet_eq17_looks_fine": [
            f"{r['sweep']}/{r['functional']}" for r in inconsistent
            if _finite(r["asymptotic_ratio"])
            and abs(r["asymptotic_ratio"] - 1.0) <= THRESHOLDS["asymptotic_band"]],
    }
    com = artifact.get("committed_v2_artifact")
    if com:
        out["committed_artifact"] = {
            "grids": com["before"]["grids"],
            "shock_loglog_exponent_published": com["before"]["shock_exponent_finest3"],
            "reference_class_shock": com["reference"]["shock_position_error"]["class"],
            "reference_order_shock": com["reference"]["shock_position_error"]["observed_order"],
            "shock_loglog_admissible": com["after"]["shock_loglog_admissible"],
            "effective_shock_exponent": com["after"]["shock_exponent"],
            "effective_contact_exponent": com["after"]["contact_exponent"],
            "verdict_before": com["before"]["sublinear_convergence_observed"],
            "verdict_after": com["after"]["sublinear_convergence_observed"],
            "ordering_robust": com["after"]["ordering_holds_with_finest_successive"]
            and com["after"]["ordering_holds_with_effective_exponent"],
        }
    return out


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def run(outdir: Path, log=print) -> Dict:
    started = time.time()
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "artifacts").mkdir(exist_ok=True)

    log("[acquire] main triplet 100/200/400 (order 2, CFL 0.4)")
    main = acquire((100, 200, 400), order=2, cfl=0.4, log=log)
    log("[acquire] fine triplet 200/400/800 (order 2, CFL 0.4)")
    fine = acquire((200, 400, 800), order=2, cfl=0.4, log=log)
    log("[acquire] CFL sensitivity 100/200/400 (order 2, CFL 0.2)")
    cfl_low = acquire((100, 200, 400), order=2, cfl=0.2, log=log)
    log("[acquire] first-order scheme 100/200/400 (order 1, CFL 0.4)")
    first = acquire((100, 200, 400), order=1, cfl=0.4, log=log)

    combined = merge_sweeps(main, fine)
    sweeps_cfg = {
        "main_100_200_400": main,
        "fine_200_400_800": fine,
        "combined_100_200_400_800": combined,
        "cfl0.2_100_200_400": cfl_low,
        "order1_100_200_400": first,
    }
    log("[diagnose] ref vs got vs repo_gci")
    sweeps = {k: diagnose(v) for k, v in sweeps_cfg.items()}

    committed_path = ROOT / "docs" / "repro-fvm-sod1978-convergence" / "study.json"
    committed = committed_diagnosis(committed_path) if committed_path.exists() else None

    artifact = {
        "anchor": "XVAL-FVM-CONVERGENCE-DIAGNOSIS-V2",
        "mission": 17360,
        "case": "Sod (1978) shock tube + periodic entropy wave (design-order control)",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": {"python": platform.python_version(),
                        "machine": platform.machine(),
                        "numpy": np.__version__},
        "thresholds": THRESHOLDS,
        "reference": {
            "procedure": "three-grid GCI (Roache 1994; Celik et al. 2008, "
                         "ASME J. Fluids Eng. 130:078001; ITTC 7.5-03-01-01)",
            "implementation": "docs/xval/convergence-diagnosis-v2/gci_reference.py "
                              "(independent; imports nothing from tensorfvm)",
            "second_implementation": "tensorfvm.ship_freesurface.mesh_convergence."
                                     "richardson_extrapolation (pre-existing in tree)",
        },
        "got": "tensorfvm.verification.sod1978_convergence "
               "(successive rates + log-log fitted exponent on the error series)",
        "sweeps_config": {k: {"grids": v["grids"], "order": v["order"], "cfl": v["cfl"],
                             "wall_seconds": v["wall_seconds"],
                             **({"assembled_from": v["assembled_from"]}
                                if "assembled_from" in v else {})}
                          for k, v in sweeps_cfg.items()},
        "raw": {k: v["sod_rows"] + v["smooth_rows"] for k, v in sweeps_cfg.items()},
        "sweeps": sweeps,
        "committed_v2_artifact": committed,
        "wall_seconds": round(time.time() - started, 1),
        "repro_cmd": "PYTHONPATH=src python docs/xval/convergence-diagnosis-v2/"
                     "xval_driver.py --out docs/xval/convergence-diagnosis-v2",
    }
    artifact["summary"] = summary_rows(sweeps, artifact["sweeps_config"])
    artifact["figures"] = make_figures(outdir, artifact)
    artifact["verdicts"] = _verdicts(artifact)

    (outdir / "artifacts" / "xval_convergence_diagnosis.json").write_text(
        json.dumps(artifact, indent=2, default=_json_default))
    (outdir / "artifacts" / "xval_table.md").write_text(render_markdown(artifact))
    log(f"[done] {artifact['wall_seconds']}s -> {outdir}")
    return artifact


def _json_default(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    raise TypeError(str(type(obj)))


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(HERE), help="output directory")
    args = ap.parse_args(argv)
    art = run(Path(args.out))
    print(json.dumps(art["verdicts"], indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
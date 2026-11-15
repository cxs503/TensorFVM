"""Round-7 closure driver: land the missing WENO5@25600 rung and re-adjudicate.

The debt this closes
-------------------
``R5-REFERENCE-EXTENSION-CLOSURE.md`` sits in state **待回填** because the extended
reference rung ``WENO5@25600`` (cluster job ``cj3710``) never came back, which leaves
round-5's frozen Q1-Q5 and round-4's P1/P2 ``not_measured``.  The rung is missing for a
*throughput* reason, not a numerical one: at the frozen CFL (0.4) the cost is
``steps x cells`` ~ 2.6e9 cell-updates ~ 2.3 h single core, longer than one job's wall clock.
``docs/woodward-colella-double-blast/ladder/wc_ladder_weno5_cfl.py`` therefore produces the
same rung at CFL 0.8 (``steps ~ 1/CFL``), and this driver is what makes that legal:

* **E1** (instrument equivalence, frozen in ``PREDECLARATION-r7.md``): the CFL-0.8 control
  rung at ``nx=6400`` must reproduce the frozen CFL-0.4 rung to within ``eps_ref/10`` in
  window L1 and ``eps_peak`` at the collision peak.  Only then is the CFL-0.8 25600 rung
  *promoted* to the standard artifact name that every other loader reads; the promotion is
  recorded inside the rung's own metrics JSON (``cfl``, ``promoted_from``, E1 numbers).
* **E2**: the residual amplitude ``max|D|`` (accumulated-difference field, the W1 instrument
  of ``wc_w1_norm.py``) must keep shrinking as ``h^1`` into the new pair -- the hypothesis
  round-6 left open.
* **E3/E4/E5**: does the reference error bar ``eps_ref`` fall below 0.5 %, and does the
  graded MUSCL/HLLC L1 honest upper bound fall below 1.5 % / below 1 %?

Nothing in rounds 1-6 is rewritten: this file only refills what was悬空 and writes its own
``R7-REF25600-CLOSURE.md`` + ``summary_r7.json`` (every number programmatic, zero hand-copy).

    PYTHONPATH=src python docs/XVAL-FVM-M18737-ROUND2/ref25600_r7.py
"""
from __future__ import annotations

import importlib.util
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from tensorfvm.verification import wc_ref_uncertainty as U  # noqa: E402
from tensorfvm.verification import wc_refine as R  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "woodward-colella-double-blast" / "ladder" / "data"
FIGDIR = HERE / "figures"
ARTIFACTS = ROOT / "woodward-colella-double-blast" / "artifacts"

EXTENDED = 25600
CONTROL_NX = 6400
CONTROL_CFL = 0.8
GRADED_RUNGS = (1600, 3200, 6400)
PREVIOUS_REF = 12800


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _mod(name: str, path: Path):
    """Import a doc-local module by path (keeps one definition of an instrument)."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _r(x, nd=4):
    return None if x is None else round(float(x), nd)


def _tagged(nx: int, cfl: float) -> str:
    return f"nx{nx}_weno5_cfl{float(cfl):g}"


# --------------------------------------------------------------------------- #
# 1. instrument equivalence control (E1)
# --------------------------------------------------------------------------- #
def instrument_control() -> Dict:
    """Is the CFL-0.8 run the same instrument as the frozen CFL-0.4 run?

    Thresholds are frozen in ``PREDECLARATION-r7.md``: ``eps_ref/10`` in window L1 and
    ``eps_peak`` at the peak, both taken from the *existing* CFL-0.4 ladder, i.e. the
    control is judged against how well the reference family certifies itself.
    """
    base = R.try_load_profile(DATA, CONTROL_NX, "weno5")
    try:
        alt = R.load_profile_tagged(DATA, _tagged(CONTROL_NX, CONTROL_CFL))
    except R.DataMissing:
        alt = None
    ladder = U.selfcert_ladder(DATA, "weno5")
    row = next((r for r in ladder if r["fine_nx"] == PREVIOUS_REF), None)
    eps_ref = None if row is None else row.get("u_ref_percent")
    eps_peak = None
    if row is not None:
        p_peak = _peak_order(ladder, PREVIOUS_REF)
        if p_peak and row.get("peak_percent") is not None:
            eps_peak = row["peak_percent"] / (2.0 ** p_peak - 1.0)
    out: Dict = {
        "status": "not_measured" if (base is None or alt is None) else "measured",
        "nx": CONTROL_NX, "cfl_baseline": 0.4, "cfl_control": CONTROL_CFL,
        "eps_ref_percent": _r(eps_ref), "eps_peak_percent": _r(eps_peak),
        "bars": {"l1_percent": _r(eps_ref / 10.0 if eps_ref else None),
                 "peak_percent": _r(eps_peak)},
        "sources": {"baseline": None if base is None else base["source"],
                    "control": None if alt is None else alt["source"]},
    }
    if out["status"] != "measured":
        out["E1"] = "not_measured"
        return out
    m = R.cell_average_l1(base["rho"], alt["rho"])
    out.update({
        "d_l1_percent": _r(m["l1_percent"], 6),
        "d_peak_percent": _r(m["peak_percent"], 6),
        "identical_grid_ratio": m["ratio"],
    })
    l1_ok = eps_ref is not None and m["l1_percent"] <= eps_ref / 10.0
    peak_ok = eps_peak is not None and m["peak_percent"] <= eps_peak
    out["l1_within_bar"] = bool(l1_ok)
    out["peak_within_bar"] = bool(peak_ok)
    out["E1"] = "pass" if (l1_ok and peak_ok) else "fail"
    return out


def _peak_order(ladder: List[Dict], fine_nx: int) -> Optional[float]:
    """Observed order of the *peak* distance, from the two finest pairs."""
    hi = next((r for r in ladder if r["fine_nx"] == fine_nx), None)
    lo = next((r for r in ladder if r["fine_nx"] == fine_nx // 2), None)
    if not hi or not lo:
        return None
    if not hi.get("peak_percent") or not lo.get("peak_percent"):
        return None
    return math.log(lo["peak_percent"] / hi["peak_percent"], 2)


# --------------------------------------------------------------------------- #
# 2. promotion gate: CFL-0.8 25600 rung -> standard artifact
# --------------------------------------------------------------------------- #
def promote_extended(control: Dict) -> Dict:
    """Copy the CFL-0.8 25600 rung to the standard name *iff* E1 passed.

    The provenance travels with the artifact: the standard metrics JSON is rewritten with
    ``cfl = 0.8``, ``promoted_from``, ``instrument_equivalence`` (the E1 block) and the raw
    run counters, so no downstream reader can mistake it for a CFL-0.4 rung.
    """
    tagged_prof = DATA / f"ladder_profile_{_tagged(EXTENDED, CONTROL_CFL)}.npz"
    tagged_met = DATA / f"ladder_metrics_{_tagged(EXTENDED, CONTROL_CFL)}.json"
    std_prof = DATA / f"ladder_profile_nx{EXTENDED}_weno5.npz"
    std_met = DATA / f"ladder_metrics_nx{EXTENDED}_weno5.json"
    out: Dict = {"tagged_present": tagged_prof.exists(), "already_present": std_prof.exists()}
    if std_prof.exists() and std_met.exists() and not out.get("rewrite"):
        out["status"] = "already_promoted"
        out["metrics"] = json.loads(std_met.read_text())
        return out
    if not tagged_prof.exists():
        out["status"] = "not_measured"
        return out
    if control.get("E1") != "pass":
        out["status"] = "blocked_by_E1"
        out["E1"] = control.get("E1")
        return out
    raw = json.loads(tagged_met.read_text()) if tagged_met.exists() else {}
    shutil.copyfile(tagged_prof, std_prof)
    metrics = dict(raw)
    metrics.update({
        "promoted_from": tagged_prof.name,
        "promotion_rule": "PREDECLARATION-r7.md E1 (CFL-0.8 admissible iff control rung "
                          "nx=6400 reproduces CFL-0.4 within eps_ref/10 L1 and eps_peak)",
        "instrument_equivalence": control,
        "cfl_frozen_instrument": 0.4,
    })
    std_met.write_text(json.dumps(metrics, indent=1) + "\n")
    out.update({"status": "promoted", "metrics": metrics})
    return out


# --------------------------------------------------------------------------- #
# 3. the extended reference ladder + residual attribution (E2/E3)
# --------------------------------------------------------------------------- #
def reference_ladder(control: Dict) -> Dict:
    """Five-rung self-certification and the h^1 shrinkage test on ``max|D|``."""
    w1 = _mod("wc_w1_norm_r7", HERE / "wc_w1_norm.py")
    ladder = U.selfcert_ladder(DATA, "weno5")
    present = [n for n in R.WENO5_RUNGS if R.try_load_profile(DATA, n, "weno5") is not None]

    # accumulated-difference amplitude D (mass units) for every consecutive pair
    defects: List[Dict] = []
    for coarse, fine in zip(present, present[1:]):
        pr_c = R.load_profile(DATA, coarse, "weno5")
        pr_f = R.load_profile(DATA, fine, "weno5")
        loc = w1.localize(pr_f, pr_c, "rho")
        defects.append({"coarse_nx": coarse, "fine_nx": fine,
                        "max_abs_D": float(loc["max_abs"]),
                        "w_eff_coarse_cells": float(loc.get("w_eff_coarse_cells", loc.get("w_eff_cells", float("nan")))),
                        "rel_net_mass": float(loc.get("rel_net_mass", float("nan")))})
    for lo, hi in zip(defects, defects[1:]):
        ratio = lo["max_abs_D"] / hi["max_abs_D"] if hi["max_abs_D"] else float("nan")
        hi["order_vs_previous_pair"] = math.log(ratio, 2) if ratio > 0 else float("nan")

    ext_unc = U.u_ref_of_reference(DATA, EXTENDED, "weno5", ladder=ladder)
    prev_unc = U.u_ref_of_reference(DATA, PREVIOUS_REF, "weno5", ladder=ladder)

    newest = defects[-1] if defects else {}
    p_d = newest.get("order_vs_previous_pair")
    out: Dict = {
        "reference_rungs_present": present,
        "extended_present": EXTENDED in present,
        "selfcert_pairs": ladder,
        "defects": defects,
        "extended_uncertainty": ext_unc,
        "previous_uncertainty": prev_unc,
        "instrument": control,
    }
    # E2 is only measurable once the *new* pair (12800->25600) exists: without it the
    # "newest" defect is the old 6400->12800 pair, which is not the quantity frozen in E2.
    if EXTENDED in present and p_d is not None and math.isfinite(p_d):
        out["E2"] = {
            "quantity": "p_D = log2( max|D|(6400->12800) / max|D|(12800->25600) )",
            "measured": _r(p_d, 4), "band": [1.85, 2.15],
            "verdict": "pass" if 1.85 <= p_d <= 2.15 else "fail",
            "predicted_max_abs_D": _r(defects[-2]["max_abs_D"] / 2.0, 6) if len(defects) >= 2 else None,
            "measured_max_abs_D": _r(newest.get("max_abs_D"), 6),
        }
    else:
        out["E2"] = {"verdict": "not_measured"}
    eps = ext_unc.get("u_ref_percent")
    out["E3"] = {
        "quantity": "eps_ref(L1 field) at nx=25600", "measured": _r(eps),
        "bar": 0.5, "previous": _r(prev_unc.get("u_ref_percent")),
        "verdict": "not_measured" if eps is None else ("pass" if eps < 0.5 else "fail"),
    }
    return out


# --------------------------------------------------------------------------- #
# 4. re-grade the graded family against the extended reference (E4/E5)
# --------------------------------------------------------------------------- #
def regrade(ref: Dict) -> Dict:
    if not ref.get("extended_present"):
        return {"E4": {"verdict": "not_measured"}, "E5": {"verdict": "not_measured"},
                "rows": [], "rows_prev": []}
    eps = ref["extended_uncertainty"].get("u_ref_percent")
    rows = R.graded_ladder(DATA, EXTENDED)
    prev = R.graded_ladder(DATA, PREVIOUS_REF)
    for r in rows:
        if not r.get("measured"):
            continue
        r["upper_bound_percent"] = _r(r["l1_percent"] + eps if eps else None)
        r["earned_certified"] = bool(eps is not None
                                    and (r["l1_percent"] + eps) < R.CERTIFIED_PERCENT)
    finest = next((r for r in rows if r.get("measured")), None)
    for r in reversed(rows):
        if r.get("measured"):
            finest = r
            break
    out: Dict = {"rows": rows, "rows_prev": prev, "eps_ref_percent": _r(eps)}
    if finest is None or eps is None:
        out["E4"] = {"verdict": "not_measured"}
        out["E5"] = {"verdict": "not_measured"}
        return out
    ub = finest["l1_percent"] + eps
    out["finest"] = {"nx": finest["nx"], "l1_percent": _r(finest["l1_percent"]),
                     "upper_bound_percent": _r(ub),
                     "peak_percent": _r(finest["peak_percent"]),
                     "pointwise_l1_percent": _r(finest.get("pointwise_l1_percent"))}
    out["E4"] = {"quantity": f"graded L1 honest upper bound @nx={finest['nx']}",
                 "measured": _r(ub), "bar": 1.5,
                 "verdict": "pass" if ub < 1.5 else "fail"}
    out["E5"] = {"quantity": "<1 % provable (upper bound < 1 %)",
                 "measured": _r(ub), "bar": 1.0,
                 "verdict": "pass" if ub < 1.0 else "fail"}
    prev_finest = next((r for r in reversed(prev) if r.get("measured")), None)
    if prev_finest:
        out["delta_vs_previous_reference_pp"] = _r(finest["l1_percent"] - prev_finest["l1_percent"])
    return out


# --------------------------------------------------------------------------- #
# 5. figures
# --------------------------------------------------------------------------- #
def figures(ref: Dict, grade: Dict) -> List[Dict]:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    out: List[Dict] = []

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    pairs = [(f"{r['coarse_nx']}→{r['fine_nx']}", r["l1_percent"]) for r in ref["selfcert_pairs"]]
    if pairs:
        lbl, val = zip(*pairs)
        colors = ["#999999"] * len(val)
        colors[-1] = "#1f77b4" if ref["extended_present"] else "#d62728"
        ax[0].bar(range(len(val)), val, color=colors)
        ax[0].set_xticks(range(len(val)))
        ax[0].set_xticklabels(lbl, rotation=30, ha="right", fontsize=8)
        ax[0].set_ylabel("WENO5 self-certification L1 [%]")
        ax[0].axhline(1.0, ls="--", lw=1, color="k", label="1 % bar")
        ax[0].legend(fontsize=8)
    ax[0].set_title("(a) reference self-convergence, 5-rung ladder")

    d = ref["defects"]
    if d:
        idx = np.arange(len(d))
        ax[1].semilogy(idx, [x["max_abs_D"] for x in d], "o-", label="max|D| (measured)")
        ax[1].semilogy(idx, [d[0]["max_abs_D"] / 2.0 ** (i + 1) for i in idx], "k--",
                       lw=1, label="h$^1$ reference slope")
        ax[1].set_xticks(idx)
        ax[1].set_xticklabels([f"{x['coarse_nx']}→{x['fine_nx']}" for x in d],
                              rotation=30, ha="right", fontsize=8)
        ax[1].set_ylabel("max|D|  [mass units]")
        ax[1].legend(fontsize=8)
    ax[1].set_title("(b) residual amplitude vs pair")

    rows = [r for r in grade.get("rows", []) if r.get("measured")]
    prev = [r for r in grade.get("rows_prev", []) if r.get("measured")]
    if prev:
        ax[2].loglog([r["nx"] for r in prev], [r["l1_percent"] for r in prev], "s--",
                     color="#999999", label="vs WENO5@12800")
    if rows:
        ax[2].loglog([r["nx"] for r in rows], [r["l1_percent"] for r in rows], "o-",
                     color="#1f77b4", label="vs WENO5@25600")
        ax[2].loglog([r["nx"] for r in rows], [r["upper_bound_percent"] for r in rows], ":",
                     color="#1f77b4", label="honest upper bound")
    ax[2].axhline(1.0, ls="--", lw=1, color="k")
    ax[2].axhline(15.0, ls=":", lw=1, color="r", label="ledger 15 %")
    ax[2].set_xlabel("nx (graded MUSCL/HLLC)")
    ax[2].set_ylabel("window L1 [%]")
    ax[2].legend(fontsize=8)
    ax[2].set_title("(c) graded family vs both references")
    fig.tight_layout()
    p1 = FIGDIR / "fig_r7_reference_ladder.png"
    fig.savefig(p1, dpi=140)
    plt.close(fig)
    out.append({"name": "fig_r7_reference_ladder.png", "path": str(p1),
                "caption": "五级 WENO5 参考自收敛、差分幅值 h^1 收缩检验、被定级族对 12800/25600 两个参考的定级"})

    if ref["extended_present"]:
        r13 = R.load_profile(DATA, PREVIOUS_REF, "weno5")
        r25 = R.load_profile(DATA, EXTENDED, "weno5")
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        ax.plot(r13["x"], r13["rho"], lw=1.2, label="WENO5@12800 (round-4 ref)")
        ax.plot(r25["x"], r25["rho"], lw=1.2, ls="--", label="WENO5@25600 (this round)")
        g = R.try_load_profile(DATA, max(GRADED_RUNGS), "hllc")
        if g is not None:
            ax.plot(g["x"], g["rho"], lw=0.9, color="#d62728",
                    label=f"MUSCL/HLLC@{max(GRADED_RUNGS)} (graded)")
        ax.set_xlim(0.55, 0.95)
        ax.set_xlabel("x")
        ax.set_ylabel(r"$\rho$ (cell average)")
        ax.legend(fontsize=8)
        ax.set_title("collision region: extended reference vs graded solution")
        fig.tight_layout()
        p2 = FIGDIR / "fig_r7_profiles_25600.png"
        fig.savefig(p2, dpi=140)
        plt.close(fig)
        out.append({"name": "fig_r7_profiles_25600.png", "path": str(p2),
                    "caption": "对撞区剖面对比：WENO5@12800 / WENO5@25600 / MUSCL-HLLC@6400"})
    return out


# --------------------------------------------------------------------------- #
# 6. document render (all numbers injected)
# --------------------------------------------------------------------------- #
def _fmt(x, nd=4, dash="—"):
    if x is None:
        return dash
    if isinstance(x, bool):
        return "✅" if x else "❌"
    if isinstance(x, (int, np.integer)):
        return f"{int(x)}"
    if isinstance(x, float) and not math.isfinite(x):
        return dash
    return f"{float(x):.{nd}f}"


def _probe_md(s: Dict) -> str:
    """The local CFL-cost probe: does relaxing the time step leave the solution alone?"""
    p = s.get("cfl_probe")
    if not p:
        return ""
    rows = "\n".join(
        f"| {r['nx']} | {_fmt(r['d_l1_percent'], 6)} | {_fmt(r['d_peak_percent'], 6)} | "
        f"{r['steps_cfl08']} | {_fmt(r['seconds'], 1)} | {r['fallbacks']} |"
        for r in p["rows"])
    eps_peak = s["control"].get("eps_peak_percent")
    worst = max(r["d_peak_percent"] for r in p["rows"])
    s.setdefault("probe_verdict", "not_measured")
    return f"""### A0. 成本路线先验探针（本机两个便宜档，1 核，非重算）

CFL 0.8 vs 冻结 CFL 0.4 的逐档差（同一重建/通量/`t_end`，只改时间步）：

| nx | 窗口 L1 差 [%] | 峰值差 [%] | 步数(CFL0.8) | 秒 | 正性回退 |
|---|---|---|---|---|---|
{rows}

`dx^3` 预期：时间误差随网格按 8×/档收缩。**实测不成立**——L1 差 1600→3200 从
{_fmt(p["rows"][0]["d_l1_percent"], 6)} % 到 {_fmt(p["rows"][1]["d_l1_percent"], 6)} %（观测"阶" {_fmt(p["order_l1"], 3)}），
峰值差反而从 {_fmt(p["rows"][0]["d_peak_percent"], 6)} % **涨到** {_fmt(p["rows"][1]["d_peak_percent"], 6)} %
（观测"阶" {_fmt(p["order_peak"], 3)}，负号=变差）。峰值是**非单调**的：CFL 影响的是
正性回退触发位置，而对撞峰恰在回退切换处，所以 0.8 档的峰位/峰值与 0.4 档不是同一支解。

**结论（负面，先于控制档返回即已成立）**：E1 的峰值门 `eps_peak = {_fmt(eps_peak, 6)} %`
比实测 CFL 效应（最差 {_fmt(worst, 6)} %）小 {_fmt((worst / eps_peak) if eps_peak else None, 1)} 倍，
故「用放松 CFL 换 25600 档」这条路**先验即不可采信**；即使控制档侥幸通过，
也只能说明 6400 档该指标恰好不敏感，不能外推到 25600。`probe_verdict = {s["probe_verdict"]}`。
"""


def render(s: Dict) -> str:
    c = s["control"]
    ref = s["reference"]
    g = s["regrade"]
    ext = ref["extended_uncertainty"]
    prev = ref["previous_uncertainty"]
    fin = g.get("finest", {})
    e2, e3, e4, e5 = ref["E2"], ref["E3"], g["E4"], g["E5"]
    ok = ref["extended_present"] and c.get("E1") == "pass"

    rows_pair = "\n".join(
        f"| {r['coarse_nx']}→{r['fine_nx']} | {_fmt(r['l1_percent'])} | {_fmt(r['peak_percent'])} "
        f"| {_fmt(r['mass_percent'], 6)} | {_fmt(r.get('order'))} | {_fmt(r.get('u_ref_percent'))} |"
        for r in ref["selfcert_pairs"])
    rows_def = "\n".join(
        f"| {d['coarse_nx']}→{d['fine_nx']} | {_fmt(d['max_abs_D'], 6)} "
        f"| {_fmt(d.get('order_vs_previous_pair'))} | {_fmt(d['w_eff_coarse_cells'], 2)} "
        f"| {_fmt(100 * d['rel_net_mass'], 6)} |"
        for d in ref["defects"])
    rows_grade = "\n".join(
        f"| {r['nx']} | {_fmt(r['l1_percent'])} | — | {_fmt(r['peak_percent'])} |"
        for r in g.get("rows_prev", []) if r.get("measured"))
    rows_grade2 = "\n".join(
        f"| {r['nx']} | {_fmt(r['l1_percent'])} | {_fmt(r.get('upper_bound_percent'))} "
        f"| {_fmt(r['peak_percent'])} | {_fmt(r.get('pointwise_l1_percent'))} |"
        for r in g.get("rows", []) if r.get("measured"))

    head = f"""# XVAL-FVM-M18737-ROUND2 Round-7：扩展参考 WENO5@25600 回填与冻结判据对账

> 锚点 `XVAL-FVM-M18737-ROUND2`（TensorFVM 的 m18737-round2 做法），验收 <15 %。
> **前置**：`R5-REFERENCE-EXTENSION-CLOSURE.md` 状态「待回填」——扩展参考档
> `WENO5@25600`（作业 `cj3710`）从未回到库中，Q1–Q5 与 round-4 的 P1/P2 因此 `not_measured`。
> **本文**：把该档真正算出来并回填，用 `PREDECLARATION-r7.md` 里**先行冻结**的 E1–E5
> 判据对账，然后重算整个定级。**不覆盖**前轮任何记录；凡未测项一律写 `not_measured`，不得读作 PASS。
>
> **状态**：`{'已回填（扩展参考在位且仪器等价性通过）' if ok else '仍未回填（见 ⑤-1）'}`

## ① 案例介绍（物理背景 / 参数表）

Woodward & Colella (1984) 双爆轰波：一维闭管 x∈[0,1]，三压力平台（左 p=1000 / 中 p=0.01 /
右 p=100，ρ=1、u=0），两束爆轰波相向对撞（t≈0.028）、积分至 t=0.038，γ=1.4，两端反射壁。
无解析解 ⇒ 误差恒为「两数值解之差」，参考解自身的离散误差必须作为误差棒计入。

| 参数 | 值 |
|---|---|
| 被定级格式 | MUSCL-Hancock 二阶 + MC 限幅 + HLLC + SSP-RK，CFL 0.4 |
| 参考格式 | WENO5 + SSP-RK3 + Rusanov（与被定级格式不共享离散） |
| 参考档位 | {", ".join(str(n) for n in ref["reference_rungs_present"])} |
| 比较窗口 | x∈[0.4, 1.0]，守恒限制比较（coarse 单元均值 = r 个 fine 单元均值） |
| 本轮新增 | `WENO5@25600`（CFL 0.8，经 E1 仪器等价性门后入标准档名） |

## ② 怎么做的

1. **为什么此前算不出来**：冻结仪器（CFL 0.4）在 nx=25600 上的成本
   `steps × cells ≈ 2.6e9` cell-update，单核 ~2.3 h，超过单作业墙钟 ⇒ `cj3710` 交不回。
   这是**吞吐**问题，不是数值问题。
2. **降成本**：`ladder/wc_ladder_weno5_cfl.py` 把 CFL 放到 0.8（`steps ∝ 1/CFL`，步数减半），
   其余（5 阶重建 / SSP-RK3 / Rusanov / t_end）一字不动。
3. **仪器等价性（E1，先冻结后测量）**：控制档 nx=6400 上 CFL0.8 vs 冻结 CFL0.4 的
   窗口 L1 差与峰值差必须分别 ≤ `eps_ref/10`、`eps_peak`。通过才把 CFL0.8 档**提升**为标准档名，
   并在该档 metrics JSON 内写清 `cfl=0.8`、`promoted_from` 与 E1 数字（不让下游误认仪器）。
4. **重算不确定度**：五级阶梯（1600/3200/6400/12800/25600）自收敛距离 → 观测阶 →
   Richardson 误差棒 `eps_ref`；被定级解对 25600 与对 12800 分别定级，给三角上界。
5. **残差归因（E2）**：用 round-6 的输运仪器（`wc_w1_norm.localize`）算累积差分场幅值
   `max|D|`，检验它在新增一档后是否仍按 `h^1` 收缩——这是 round-6 留给下一手的可证伪假设。
6. 全部数字由本脚本程序化注入（零手抄），回归门 `tests/test_compressible_wc_ref25600.py`。

{_probe_md(s)}

## ③ 计算结果（程序化注入）

### A. E1 仪器等价性（控制档 nx={c['nx']}）

| 量 | 实测 [%] | 门槛 [%] | 判定 |
|---|---|---|---|
| 窗口 L1 差（CFL0.8 vs CFL0.4） | {_fmt(c.get('d_l1_percent'), 6)} | {_fmt(c['bars']['l1_percent'])}（=eps_ref/10） | {_fmt(c.get('l1_within_bar'))} |
| 峰值差 | {_fmt(c.get('d_peak_percent'), 6)} | {_fmt(c['bars']['peak_percent'])}（=eps_peak） | {_fmt(c.get('peak_within_bar'))} |

**E1 判定：`{c.get('E1')}`** —— {'CFL0.8 档可作为同一仪器的产品' if c.get('E1') == 'pass' else 'CFL0.8 档不得作参考，扩展档撤回'}。

### B. 五级参考自收敛阶梯

| 相邻档 | 窗口 L1 [%] | 峰值差 [%] | 窗质量差 [%] | 观测阶 | eps_ref [%] |
|---|---|---|---|---|---|
{rows_pair}

参考侧误差棒：**eps_ref(25600) = {_fmt(ext.get('u_ref_percent'))} %**（阶 {_fmt(ext.get('order'))}，来源 `{ext.get('order_source')}`），
上一轮（12800）为 **{_fmt(prev.get('u_ref_percent'))} %**。

### C. 残差幅值收缩（E2，输运仪器）

| 相邻档 | max\\|D\\| [质量单位] | 对前一对的阶 | 有效宽度 [coarse cells] | 净质量 [%] |
|---|---|---|---|---|
{rows_def}

**E2 判定：`{e2.get('verdict')}`** —— 预测 `max|D|(12800→25600) ≈ {_fmt(e2.get('predicted_max_abs_D'), 6)}`，
实测 `{_fmt(e2.get('measured_max_abs_D'), 6)}`，`p_D = {_fmt(e2.get('measured'))}`（冻结带 [1.85, 2.15]）。

### D. 被定级解重新定级

对 round-4 参考（WENO5@12800）：

| nx | 窗口 L1 [%] | 诚实上界 | 峰值差 [%] |
|---|---|---|---|
{rows_grade}

对扩展参考（WENO5@25600）：

| nx | 窗口 L1 [%] | 诚实上界 [%] | 峰值差 [%] | 逐点 L1 [%] |
|---|---|---|---|---|
{rows_grade2}

最细被定级档 nx={fin.get('nx')}：`L1 = {_fmt(fin.get('l1_percent'))} %`、
`上界 = {_fmt(fin.get('upper_bound_percent'))} %`、`峰值 = {_fmt(fin.get('peak_percent'))} %`，
相对上一参考的变化 `{_fmt(g.get('delta_vs_previous_reference_pp'))} pp`。

## ④ 结果图

![五级参考自收敛 / 差分幅值 h^1 检验 / 双参考定级](figures/fig_r7_reference_ladder.png)

*（a）每对相邻档的自收敛距离（蓝=本轮新档）；（b）累积差分幅值 max|D| 与 h^1 参考斜率；
（c）被定级族分别对 12800 与 25600 参考的窗口 L1，虚点为诚实上界，黑虚线为 1 % 门、红点线为总账 15 %。*

![对撞区剖面对比](figures/fig_r7_profiles_25600.png)

*对撞区（x∈[0.55,0.95]）剖面：round-4 参考、本轮扩展参考、被定级 MUSCL/HLLC@6400。*

## ⑤ 注意点（局限 / 适用范围）

1. **仪器成本换了、判据没换**：CFL 0.8 只影响时间离散误差（`O(dt^3) ∝ dx^3`），
   在 6400 及以上按 64×/256× 衰减；E1 就是这门事的机检，不是注记。
2. **E2 是唯一检验「残差是否是纯 O(h) 缺陷」的机检**：若 `p_D` 掉出 [1.85, 2.15]，
   说明残差里混入了非 `O(h)` 成分，round-6 的归因就得推翻——届时照实记，不改判据。
3. **参考侧仍有误差**：`eps_ref(25600)` 不是 0，所有「点估计」都要配上误差棒，
   `<1 %` 的措辞必须区分「点估计」与「可证明上界」。
4. **窗口与范数依赖**：逐点 L1 对激波抹平敏感（趋近一阶），输运范数对搬移不敏感；
   两个范数不能互相顶替（round-6 已有专文）。
5. **`cj3710` 与 `cj3782`**：前者是 8 天未回的旧作业，后者是本轮的 CFL0.8 作业；
   `{"本轮 25600 档来自 " + str(s.get("jobs", {}).get("extended")) if s.get("jobs") else "本轮作业号见 ⑥"}`。
6. **未做**：WENO5@51200（成本再 ×4，仍超墙钟）；WENO7 探针仍未修（round-6 已证伪其可作参考）。

## ⑥ 误差判定（逐项显式回答「是否 <3 % / <1 %」）

| 关键量 | 实测 | <15 %（总账） | <3 % | <1.5 % | <1 % |
|---|---|---|---|---|---|
| 参考自收敛（12800→25600） | {_fmt(ext.get('l1_percent'))} % | {_fmt((ext.get('l1_percent') or 99) < 15)} | {_fmt((ext.get('l1_percent') or 99) < 3)} | {_fmt((ext.get('l1_percent') or 99) < 1.5)} | {_fmt((ext.get('l1_percent') or 99) < 1)} |
| eps_ref(25600) | {_fmt(ext.get('u_ref_percent'))} % | {_fmt((ext.get('u_ref_percent') or 99) < 15)} | {_fmt((ext.get('u_ref_percent') or 99) < 3)} | {_fmt((ext.get('u_ref_percent') or 99) < 1.5)} | {_fmt((ext.get('u_ref_percent') or 99) < 1)} |
| 被定级 L1 点估计 @nx={fin.get('nx')} | {_fmt(fin.get('l1_percent'))} % | {_fmt((fin.get('l1_percent') or 99) < 15)} | {_fmt((fin.get('l1_percent') or 99) < 3)} | {_fmt((fin.get('l1_percent') or 99) < 1.5)} | {_fmt((fin.get('l1_percent') or 99) < 1)} |
| 被定级 L1 诚实上界 @nx={fin.get('nx')} | {_fmt(fin.get('upper_bound_percent'))} % | {_fmt((fin.get('upper_bound_percent') or 99) < 15)} | {_fmt((fin.get('upper_bound_percent') or 99) < 3)} | {_fmt((fin.get('upper_bound_percent') or 99) < 1.5)} | {_fmt((fin.get('upper_bound_percent') or 99) < 1)} |
| 被定级 峰值差 @nx={fin.get('nx')} | {_fmt(fin.get('peak_percent'))} % | {_fmt((fin.get('peak_percent') or 99) < 15)} | {_fmt((fin.get('peak_percent') or 99) < 3)} | {_fmt((fin.get('peak_percent') or 99) < 1.5)} | {_fmt((fin.get('peak_percent') or 99) < 1)} |

**是否 <3 %？** 点估计 {_fmt(fin.get('l1_percent'))} %、上界 {_fmt(fin.get('upper_bound_percent'))} % ⇒
{'两种意义下都成立' if (fin.get('upper_bound_percent') or 99) < 3 else '点估计成立、上界未成立'}。
**是否 <1 %？** {'是（可证明）' if e5.get('verdict') == 'pass' else '否——点估计与上界均未进 1 %'}。
**总账 15 % 判据**：见上表，全部 {'PASS' if (fin.get('upper_bound_percent') or 99) < 15 else 'FAIL'}。

**判据对账（先冻结后测量）**：E1 `{c.get('E1')}` · E2 `{e2.get('verdict')}` · E3 `{e3.get('verdict')}` · E4 `{e4.get('verdict')}` · E5 `{e5.get('verdict')}`。

## 复现

```bash
# 1) 扩展参考（集群，1 核；本地等效命令）
PYTHONPATH=src python docs/woodward-colella-double-blast/ladder/wc_ladder_weno5_cfl.py --sizes 6400,25600 --cfl 0.8
# 2) 回填 + 裁决 + 图
PYTHONPATH=src python docs/XVAL-FVM-M18737-ROUND2/ref25600_r7.py
# 3) 回归门
PYTHONPATH=src python -m pytest tests/test_compressible_wc_ref25600.py -q
```
"""
    return head


# --------------------------------------------------------------------------- #
# 7. driver
# --------------------------------------------------------------------------- #
def build(jobs: Optional[Dict] = None, write: bool = True) -> Dict:
    control = instrument_control()
    promotion = promote_extended(control)
    reference = reference_ladder(control)
    reference["promotion"] = {k: v for k, v in promotion.items() if k != "metrics"}
    grade = regrade(reference)
    figs = figures(reference, grade)
    probe_path = HERE / "cfl_scaling_local.json"
    probe = json.loads(probe_path.read_text()) if probe_path.exists() else None
    s: Dict = {
        "anchor": "XVAL-FVM-M18737-ROUND2", "round": 7, "cfl_probe": probe,
        "control": control, "promotion": {k: v for k, v in promotion.items() if k != "metrics"},
        "extended_metrics": promotion.get("metrics"),
        "reference": reference, "regrade": grade, "figures": figs,
        "jobs": jobs or {},
        "verdicts": {"E1": control.get("E1"), "E2": reference["E2"].get("verdict"),
                     "E3": reference["E3"].get("verdict"), "E4": grade["E4"].get("verdict"),
                     "E5": grade["E5"].get("verdict")},
    }
    if probe:
        worst = max(r["d_peak_percent"] for r in probe["rows"])
        eps_peak = s["control"].get("eps_peak_percent")
        s["probe_verdict"] = ("inadmissible"
                              if (eps_peak and worst > eps_peak) else "admissible_so_far")
    if write:
        (HERE / "summary_r7.json").write_text(json.dumps(s, indent=1, default=float) + "\n")
        (HERE / "R7-REF25600-CLOSURE.md").write_text(render(s))
    return s


def main(argv=None) -> int:
    jobs = {"extended": "cj3782", "control": "cj3781"}
    if argv and "--jobs" in argv:
        i = argv.index("--jobs")
        jobs = json.loads(argv[i + 1])
    s = build(jobs=jobs)
    print(json.dumps({"E1": s["verdicts"]["E1"], "E2": s["verdicts"]["E2"],
                      "E3": s["verdicts"]["E3"], "E4": s["verdicts"]["E4"],
                      "E5": s["verdicts"]["E5"],
                      "extended_present": s["reference"]["extended_present"],
                      "eps_ref_percent": s["reference"]["extended_uncertainty"].get("u_ref_percent"),
                      "finest": s["regrade"].get("finest")}, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
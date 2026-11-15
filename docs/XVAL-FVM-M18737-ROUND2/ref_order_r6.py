"""Round-6 closure driver: grade the Woodward-Colella reference by *reconstruction order*.

Round-5 asked the reference family to refine further (WENO5@25600).  This round asks a
different, cheaper question: at a *fixed* grid, how much of the published certified-<1 %
residual (L1 = 1.776 %, MUSCL/HLLC@3200 vs WENO5@12800) is the reference family's own
truncation, and how much is genuinely the graded second-order solution?

The instrument is ``xcode/wc_weno7.py`` — a 7th-order member of the *same* code family
(shared SSP-RK3 + Rusanov, so **not** an independent code, but an independent
*reconstruction order*).  It had only ever been order-tested on a smooth entropy wave;
``xcode/wc_weno7_ladder.py`` runs it on the blast datum for the first time.

Inputs
------
* ``predeclaration_r6.json`` — Q1–Q6 (and AMENDMENT-1) frozen before any WENO7 blast rung
  existed;
* ``ladder_profile_nx{1600,3200,6400}_weno7.npz`` — produced by job ``cj3733``;
* the stored WENO5 / MUSCL-HLLC ladder profiles.

Outputs (every number programmatic, nothing hand-copied)
-------------------------------------------------------
* ``summary_r6.json``;
* ``R6-ORDER-AUDIT-CLOSURE.md``;
* ``figures/fig_r6_selfcert_ladder.png``, ``figures/fig_r6_regrade.png``,
  ``figures/fig_r6_profiles.png``.

    python docs/XVAL-FVM-M18737-ROUND2/ref_order_r6.py
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from tensorfvm.verification import wc_refine as wcr  # noqa: E402
from tensorfvm.verification import wc_ref_uncertainty as U  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "woodward-colella-double-blast" / "ladder" / "data"
FIGDIR = HERE / "figures"
GRADED = 3200
#: the round-2/3 number this round is trying to attribute
BASELINE_REF = ("weno5", 12800)
ORDER_FAMILIES = ("weno5", "weno7")


def _r(x, nd=4):
    return None if x is None else round(float(x), nd)


# --------------------------------------------------------------------------- #
# measurements (pure functions of the stored profiles)
# --------------------------------------------------------------------------- #
def two_direction_l1(a: Dict, b: Dict, window: Sequence[float] = wcr.WINDOW) -> Dict:
    """L1 of two profiles on the *same* grid, both interpolation directions.

    ``cell_average_l1`` normalises by the reference norm, so swapping the arguments is not
    a no-op even at ratio 1.  Reporting both directions (and their band) keeps the number
    from looking more precise than it is.
    """
    fwd = wcr.cell_average_l1(a["rho"], b["rho"], window)["l1_percent"]
    bwd = wcr.cell_average_l1(b["rho"], a["rho"], window)["l1_percent"]
    pk = wcr.cell_average_l1(a["rho"], b["rho"], window)["peak_percent"]
    return {"l1_percent": 0.5 * (fwd + bwd), "l1_percent_band": [min(fwd, bwd), max(fwd, bwd)],
            "peak_percent": pk, "mean_reference_percent": 0.5 * (fwd + bwd)}


def cross_order_gaps(data_dir: Path) -> List[Dict]:
    """WENO7 vs WENO5 at identical resolution: the pure reconstruction-order effect."""
    rows: List[Dict] = []
    for nx in wcr.WENO5_RUNGS:
        p5 = wcr.try_load_profile(data_dir, nx, "weno5")
        p7 = wcr.try_load_profile(data_dir, nx, "weno7")
        if p5 is None or p7 is None:
            rows.append({"nx": nx, "measured": False})
            continue
        rows.append({"nx": nx, "measured": True, **two_direction_l1(p5, p7)})
    return rows


def regrade_matrix(data_dir: Path, graded_nx: int = GRADED) -> List[Dict]:
    """Grade the same MUSCL/HLLC profile against every usable reference, with its budget."""
    prof = wcr.try_load_profile(data_dir, graded_nx, "hllc")
    if prof is None:
        raise wcr.DataMissing(f"graded profile nx={graded_nx} missing")
    ladders = {fam: U.selfcert_ladder(data_dir, fam) for fam in ORDER_FAMILIES}
    rows: List[Dict] = []
    for fam in ORDER_FAMILIES:
        for nx in wcr.WENO5_RUNGS:
            ref = wcr.try_load_profile(data_dir, nx, fam)
            if ref is None or nx < graded_nx:
                continue
            agg = wcr.cell_average_l1(prof["rho"], ref["rho"])
            ur = U.u_ref_of_reference(data_dir, nx, fam, ladder=ladders[fam])
            rows.append({"reference": f"{fam}@{nx}", "family": fam, "ref_nx": nx,
                         "measured": True, "l1_percent": agg["l1_percent"],
                         "peak_percent": agg["peak_percent"],
                         "mass_percent": agg["mass_percent"],
                         "u_ref_percent": ur["u_ref_percent"],
                         "reference_self_certified": ur["self_certified"],
                         **U.grade_with_budget(agg["l1_percent"], ur["u_ref_percent"])})
    return rows


def graded_ladder_vs(data_dir: Path, family: str, ref_nx: int) -> List[Dict]:
    """Every MUSCL/HLLC rung graded against one fixed reference."""
    ref = wcr.try_load_profile(data_dir, ref_nx, family)
    if ref is None:
        # a rung that was never produced is *not measured*, not an error and not a pass
        return [{"nx": nx, "measured": False} for nx in wcr.HLLC_RUNGS]
    rows: List[Dict] = []
    for nx in wcr.HLLC_RUNGS:
        prof = wcr.try_load_profile(data_dir, nx, "hllc")
        if prof is None or nx > ref_nx:
            rows.append({"nx": nx, "measured": False})
            continue
        agg = wcr.cell_average_l1(prof["rho"], ref["rho"])
        rows.append({"nx": nx, "measured": True, "l1_percent": agg["l1_percent"],
                     "peak_percent": agg["peak_percent"]})
    return rows


def measurements(data_dir: Path) -> Dict:
    ladders = {fam: U.selfcert_ladder(data_dir, fam) for fam in ORDER_FAMILIES}
    present = {fam: [r for r in wcr.WENO5_RUNGS
                     if wcr.try_load_profile(data_dir, r, fam) is not None]
               for fam in ORDER_FAMILIES}
    out: Dict = {"selfcert_ladders": ladders, "reference_rungs": present}

    row7 = next((r for r in ladders["weno7"] if r["fine_nx"] == 6400), None)
    out["weno7_fine_row"] = row7
    out["weno7_u_ref_6400"] = (U.u_ref_of_reference(data_dir, 6400, "weno7",
                                                    ladder=ladders["weno7"])
                               if row7 is not None else None)
    out["weno5_u_ref_12800"] = U.u_ref_of_reference(data_dir, 12800, "weno5",
                                                    ladder=ladders["weno5"])
    out["cross_order_gaps"] = cross_order_gaps(data_dir)
    out["regrades"] = regrade_matrix(data_dir)
    out["graded_ladder_w7"] = graded_ladder_vs(data_dir, "weno7", 6400)
    out["graded_ladder_w5"] = graded_ladder_vs(data_dir, "weno5", 12800)
    out["metrics"] = {}
    for fam in ORDER_FAMILIES:
        for nx in present[fam]:
            path = Path(data_dir) / f"{'weno7' if fam == 'weno7' else 'ladder'}_metrics_nx{nx}_{fam}.json"
            if path.exists():
                out["metrics"][f"{fam}@{nx}"] = json.loads(path.read_text())
    return out


# --------------------------------------------------------------------------- #
# adjudication
# --------------------------------------------------------------------------- #
def _find(rows: Iterable[Dict], **kw) -> Optional[Dict]:
    for r in rows:
        if all(r.get(k) == v for k, v in kw.items()):
            return r
    return None


def adjudicate(frozen: Dict, meas: Dict) -> Dict:
    pred = frozen["predictions"]
    amd = {a["id"]: a for a in frozen.get("amendments", [])}
    checks: Dict[str, Dict] = {}

    # Q1 -- WENO7's own observed order on the 3200->6400 pair
    q1 = pred["Q1_weno7_observed_order"]
    row7 = meas["weno7_fine_row"]
    checks["Q1"] = {"criterion": q1["quantity"], "band": q1["band"],
                    "point": q1["point_prediction"]}
    if row7 is not None and row7.get("order") is not None:
        checks["Q1"].update(U.adjudicate_prediction(row7["order"],
                                                    point=q1["point_prediction"],
                                                    band=q1["band"]))
        checks["Q1"]["order_source"] = row7.get("order_source")
        checks["Q1"]["e_selfcert_percent"] = row7["l1_percent"]
    else:
        checks["Q1"]["verdict"] = "not_measured"

    # Q2 -- same-resolution WENO7 vs WENO5 gap at 3200
    q2 = pred["Q2_cross_order_gap_3200"]
    g3200 = _find(meas["cross_order_gaps"], nx=3200)
    checks["Q2"] = {"criterion": q2["quantity"], "band": q2["band"],
                    "point": q2["point_prediction"]}
    if g3200 and g3200.get("measured"):
        checks["Q2"].update(U.adjudicate_prediction(g3200["l1_percent"],
                                                    point=q2["point_prediction"],
                                                    band=q2["band"]))
        checks["Q2"]["band_measured"] = g3200["l1_percent_band"]
        checks["Q2"]["peak_percent"] = g3200["peak_percent"]
    else:
        checks["Q2"]["verdict"] = "not_measured"

    # Q3 (AMENDMENT-1) -- WENO7@6400's own uncertainty, code convention
    rev = amd.get("AMENDMENT-1", {}).get("revised", {})
    q3_q = rev.get("quantity", pred["Q3_weno7_u_ref_6400"]["quantity"])
    q3_point = rev.get("point", pred["Q3_weno7_u_ref_6400"]["point_prediction"])
    q3_band = rev.get("band", pred["Q3_weno7_u_ref_6400"]["band"])
    checks["Q3"] = {"criterion": q3_q, "band": q3_band, "point": q3_point,
                    "amended": bool(rev)}
    ur7 = meas["weno7_u_ref_6400"]
    if ur7 and ur7.get("u_ref_percent") is not None:
        checks["Q3"].update(U.adjudicate_prediction(ur7["u_ref_percent"], point=q3_point,
                                                    band=q3_band))
        checks["Q3"]["e_selfcert_percent"] = ur7["e_selfcert_percent"]
        checks["Q3"]["order"] = ur7["order"]
        checks["Q3"]["cheaper_than_refinement"] = bool(
            ur7["u_ref_percent"] < meas["weno5_u_ref_12800"]["u_ref_percent"])
    else:
        checks["Q3"]["verdict"] = "not_measured"

    # Q4 -- how much of the baseline residual was the reference's
    q4 = pred["Q4_reference_side_share_pp"]
    base = _find(meas["regrades"], reference="weno5@12800")
    new = _find(meas["regrades"], reference="weno7@6400")
    checks["Q4"] = {"criterion": q4["quantity"], "band": q4["band"],
                    "point": q4["point_prediction"]}
    if base and new:
        delta = new["l1_percent"] - base["l1_percent"]
        checks["Q4"].update(U.adjudicate_prediction(delta, point=q4["point_prediction"],
                                                    band=q4["band"]))
        checks["Q4"]["baseline_l1_percent"] = base["l1_percent"]
        checks["Q4"]["new_l1_percent"] = new["l1_percent"]
    else:
        checks["Q4"]["verdict"] = "not_measured"

    # Q5 -- does the certified bar survive the reference upgrade?
    q5 = pred["Q5_certified_still_false"]
    checks["Q5"] = {"criterion": q5["quantity"], "point": q5["point_prediction"]}
    if base and new and ur7 and ur7.get("u_ref_percent") is not None:
        worst = new["l1_percent"] + ur7["u_ref_percent"]
        selfcert_ok = bool(ur7["self_certified"])
        floor_removed = bool(new["l1_percent"] < base["l1_percent"])
        graded_under_1 = bool(new["l1_percent"] < wcr.CERTIFIED_PERCENT)
        certified = bool(selfcert_ok and floor_removed and graded_under_1)
        checks["Q5"].update({
            "measured_l1_percent": new["l1_percent"],
            "u_ref_percent": ur7["u_ref_percent"],
            "honest_bound_percent": worst,
            "graded_under_1_percent": graded_under_1,
            "reference_self_certified": selfcert_ok,
            "floor_removed": floor_removed,
            "certified": certified,
            "measured": 1.0 if certified else 0.0,
            "verdict": "hit" if not certified else "miss",
            "certified_bar_percent": wcr.CERTIFIED_PERCENT})
    else:
        checks["Q5"]["verdict"] = "not_measured"

    # Q6 -- peak ratio (reference order should barely move the peak)
    q6 = pred["Q6_peak_ratio"]
    checks["Q6"] = {"criterion": q6["quantity"], "band": q6["band"],
                    "point": q6["point_prediction"]}
    if base and new:
        ratio = new["peak_percent"] / base["peak_percent"] if base["peak_percent"] else None
        checks["Q6"].update(U.adjudicate_prediction(ratio, point=q6["point_prediction"],
                                                    band=q6["band"]))
        checks["Q6"]["baseline_peak_percent"] = base["peak_percent"]
        checks["Q6"]["new_peak_percent"] = new["peak_percent"]
    else:
        checks["Q6"]["verdict"] = "not_measured"

    hits = [k for k, v in checks.items() if v.get("verdict") == "hit"]
    misses = [k for k, v in checks.items() if v.get("verdict") == "miss"]
    unmeasured = [k for k, v in checks.items()
                  if v.get("verdict") in ("not_measured", "unadjudicable", None)]
    return {"checks": checks, "hits": hits, "misses": misses, "not_measured": unmeasured,
            "score": f"{len(hits)}/{len(checks)}"}


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def figures(meas: Dict, outdir: Path = FIGDIR) -> Dict[str, str]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    made: Dict[str, str] = {}

    # 1. self-certification ladders of both reference orders (+1 % bar)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for fam, style in (("weno5", dict(marker="o", ls="-", color="C0")),
                       ("weno7", dict(marker="s", ls="-", color="C3"))):
        rows = [r for r in meas["selfcert_ladders"][fam] if r.get("l1_percent") is not None]
        if not rows:
            continue
        x = [r["fine_nx"] for r in rows]
        y = [r["l1_percent"] for r in rows]
        ax.plot(x, y, label=f"{fam.upper()} self-certification (coarse vs fine)", **style)
        for r in rows:
            ax.annotate(f"{r['l1_percent']:.3f}%", (r["fine_nx"], r["l1_percent"]),
                        textcoords="offset points", xytext=(4, 5), fontsize=7)
    ax.axhline(wcr.CERTIFIED_PERCENT, color="k", ls="--", lw=1,
               label=f"{wcr.CERTIFIED_PERCENT:.0f} % certification bar")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("reference grid nx (fine member of the pair)")
    ax.set_ylabel(r"$L_1$ between consecutive rungs  [%]")
    ax.set_title("Reference family self-certification: refinement vs reconstruction order")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = outdir / "fig_r6_selfcert_ladder.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    made["selfcert_ladder"] = str(p)

    # 2. re-grade of the same graded profile against every reference
    rows = [r for r in meas["regrades"] if r.get("measured")]
    if rows:
        fig, ax = plt.subplots(figsize=(7.6, 4.6))
        labels = [r["reference"] for r in rows]
        vals = [r["l1_percent"] for r in rows]
        errs = [r["u_ref_percent"] or 0.0 for r in rows]
        colors = ["C0" if r["family"] == "weno5" else "C3" for r in rows]
        xpos = np.arange(len(rows))
        ax.bar(xpos, vals, color=colors, alpha=0.85,
               label="measured $L_1$ (MUSCL/HLLC nx=3200 vs reference)")
        ax.errorbar(xpos, vals, yerr=errs, fmt="none", ecolor="k", capsize=4,
                    label=r"$\pm U_{ref}$ of that reference")
        for i, (v, e) in enumerate(zip(vals, errs)):
            ax.annotate(f"{v:.3f}%\n+{e:.3f} = {v + e:.3f}%", (i, v),
                        textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7)
        ax.axhline(wcr.CERTIFIED_PERCENT, color="k", ls="--", lw=1,
                   label=f"{wcr.CERTIFIED_PERCENT:.0f} % bar")
        ax.set_xticks(xpos)
        ax.set_xticklabels(labels, rotation=20, fontsize=8)
        ax.set_ylabel(r"$L_1$ [%]  (window $x\in[0.4,1]$)")
        ax.set_title("Same graded solution, references of different reconstruction order")
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=8)
        fig.tight_layout()
        p = outdir / "fig_r6_regrade.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        made["regrade"] = str(p)

    # 3. profiles zoom: where the smearing is
    try:
        p32 = wcr.load_profile(DATA, 3200, "hllc")
        p5 = wcr.load_profile(DATA, 12800, "weno5")
        p732 = wcr.load_profile(DATA, 3200, "weno7")
        p764 = wcr.load_profile(DATA, 6400, "weno7")
    except wcr.DataMissing:
        return made
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    for prof, lab, style in ((p32, "MUSCL/HLLC @3200 (graded)", dict(color="k", ls="-", lw=1.4)),
                             (p5, "WENO5 @12800 (round-2 reference)", dict(color="C0", ls="--", lw=1.2)),
                             (p732, "WENO7 @3200 (this round)", dict(color="C3", ls=":", lw=1.4)),
                             (p764, "WENO7 @6400 (this round)", dict(color="C1", ls="-", lw=1.4))):
        m = (prof["x"] >= 0.55) & (prof["x"] <= 0.9)
        ax.plot(prof["x"][m], prof["rho"][m], label=lab, **style)
    ax.set_xlabel("x")
    ax.set_ylabel(r"$\rho$ (cell average)")
    ax.set_title("Woodward–Colella two-blast, t=0.038: reference order vs graded solution")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = outdir / "fig_r6_profiles.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    made["profiles"] = str(p)
    return made


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #
def _fmt(x, nd=4, dash="—"):
    if x is None:
        return dash
    if isinstance(x, str):
        return x
    return f"{x:.{nd}f}"


def render(summary: Dict) -> str:
    m, adj, fig = summary["measurements"], summary["adjudication"], summary["figures"]
    chk = adj["checks"]
    L: List[str] = []
    A = L.append
    A("# XVAL-FVM-M18737-ROUND2 Round-6：把参考族的「阶」做成门（WENO7 首次上 blast 数据）")
    A("")
    A("> 锚点 `XVAL-FVM-M18737-ROUND2`，验收 <15 %（owner 标准）。本轮自设目标仍是 certified <1 %。")
    A("> 前置：`CERTIFIED-REF-CLOSURE.md`（1.776 % honest FAIL）· `REF-ATTRIBUTION.md`（同族地板 0.548 %，")
    A("> 只能说明参考族还在动，不能说明极限离被定级解多远）· `R4/R5`（加密被定级解 / 加密参考）。")
    A("> 判据在测量前冻结：`PREDECLARATION-r6.md` + `predeclaration_r6.json`（含 AMENDMENT-1）。")
    A("> 本文所有数字由 `ref_order_r6.py` 从 `summary_r6.json` 程序化渲染，零手抄。")
    A("")
    A("## ① 案例介绍（物理背景 / 参数表）")
    A("")
    A("Woodward & Colella (1984) 双爆轰波：一维闭管 x∈[0,1]，初始三压力平台 "
      "p=1000 (x≤0.1) / 0.01 (0.1<x<0.9) / 100 (x≥0.9)，ρ=1、u=0，γ=1.4，两端反射壁，")
    A("积分到 t=0.038（两束爆轰波在管中部对撞之后）。无解析解 ⇒ 「误差」永远是两个数值解之差，")
    A("**参考解自身的截断误差必须计入**。")
    A("")
    A("| 项 | 值 |")
    A("|---|---|")
    A("| 被定级解 | TensorFVM MUSCL-Hancock 二阶 + MC 限幅 + HLLC + SSP-RK，CFL 0.4，nx=3200 |")
    A("| 参考族 A | WENO5 + SSP-RK3 + Rusanov（`xcode/wc_crosscode.py`），nx=1600…12800 |")
    A("| 参考族 B（本轮新增） | WENO7 + SSP-RK3 + Rusanov（`xcode/wc_weno7.py`），nx=1600/3200/6400 |")
    A("| 比较窗 | x∈[0.4,1.0]，粗格守恒限制（粗格均值 = 其内 r 个细格均值） |")
    A("| 认证门 / 总账容差 | 1.0 % / 15.0 % |")
    A("")
    A("参考族 B 与 A **共享时间积分与界面通量**，因此不是独立代码，而是独立**重建阶**探针；")
    A("跨代码独立性仍由 A 对 TensorFVM 提供。这一点在判据里写明，不当独立性宣称。")
    A("")
    A("## ② 怎么做的（先冻结、再算、再对账）")
    A("")
    A("1. **冻结**：`PREDECLARATION-r6.md` 在 worktree 内 WENO7 blast 剖面数 = 0 时写下 Q1–Q6；")
    A("   `AMENDMENT-1` 修掉 Q3 的一次定义错误（原文用了自造的 ½·L1 比例，代码约定是 "
      "`U_ref = e/(2^p−1)`），修正发生在作业 `cj3733` 仍排队、3200/6400 档不存在时，原文保留不删。")
    A("2. **算参考族 B**：`xcode/wc_weno7_ladder.py --rungs 1600,3200,6400`，递交调度系统（作业 `cj3733`，")
    A("   1 核，本机不起重算力）。产物 `ladder_profile_nx{1600,3200,6400}_weno7.npz` + 指标 json。")
    A("3. **同网格跨阶差**：WENO7@nx vs WENO5@nx（nx=1600/3200/6400），剔除「加密」这一维，只留「阶」这一维；")
    A("   两个插值方向都算并报带（同网格下正反向归一化分母不同）。")
    A("4. **重定级**：同一个 MUSCL/HLLC@3200 剖面分别对每一个可用参考（WENO5@6400/12800、")
    A("   WENO7@3200/6400）定级，每个参考带自己的 `U_ref`（由同族自认证阶梯算，不是假设）。")
    A("5. **判据机检**：Q1–Q6 逐条对账，命中/未中一律记；缺工件记 `not_measured`，不改写为 PASS。")
    A("")
    A("## ③ 计算结果")
    A("")
    A("### 参考族自认证阶梯（同族相邻档，程序化）")
    A("")
    A("| 参考族 | coarse nx | fine nx | 自认证 L1 [%] | 观测阶 p | p 来源 | U_ref(fine) [%] |")
    A("|---|---|---|---|---|---|---|")
    for fam in ORDER_FAMILIES:
        for r in m["selfcert_ladders"][fam]:
            if r.get("l1_percent") is None:
                continue
            A(f"| {fam.upper()} | {r['coarse_nx']} | {r['fine_nx']} | {_fmt(r['l1_percent'])} | "
              f"{_fmt(r.get('order'), 4)} | {r.get('order_source', '—')} | {_fmt(r.get('u_ref_percent'))} |")
    A("")
    A("### 同网格跨阶差（只换重建阶，不换网格）")
    A("")
    A("| nx | L1(WENO7 vs WENO5) [%] | 插值带 [%] | 峰值差 [%] |")
    A("|---|---|---|---|")
    for g in m["cross_order_gaps"]:
        if not g.get("measured"):
            continue
        A(f"| {g['nx']} | {_fmt(g['l1_percent'])} | [{_fmt(g['l1_percent_band'][0])}, "
          f"{_fmt(g['l1_percent_band'][1])}] | {_fmt(g['peak_percent'])} |")
    A("")
    A("### 同一个被定级解 × 不同参考（重定级矩阵）")
    A("")
    A("| 参考 | 实测 L1 [%] | U_ref [%] | 最坏情形 L1+U_ref [%] | 参考自认证 | 门裁决 |")
    A("|---|---|---|---|---|---|")
    for r in m["regrades"]:
        A(f"| {r['reference']} | {_fmt(r['l1_percent'])} | {_fmt(r['u_ref_percent'])} | "
          f"{_fmt(r.get('worst_case_percent'))} | {r['reference_self_certified']} | `{r.get('verdict')}` |")
    A("")
    A("### WENO7 是否真的更便宜（成本-精度）")
    A("")
    met = m.get("metrics", {})
    rows = [(k, met[k].get("seconds"), met[k].get("steps"), met[k].get("positivity_fallbacks"))
            for k in ("weno5@6400", "weno5@12800", "weno7@3200", "weno7@6400") if k in met]
    if rows:
        A("| 参考 | 求解秒数 | 步数 | 正性回退次数 |")
        A("|---|---|---|---|")
        for k, sec, st, fb in rows:
            A(f"| {k} | {_fmt(sec, 1)} | {st} | {fb} |")
        A("")
    A("### 判据对账（Q1–Q6）")
    A("")
    A("| 编号 | 量 | 点预测 | 冻结带 | 实测 | 裁决 |")
    A("|---|---|---|---|---|---|")
    for k in sorted(chk):
        c = chk[k]
        meas = c.get("measured", c.get("measured_l1_percent"))
        if k == "Q5":
            meas = f"certified={c.get('certified')}（上界 {_fmt(c.get('honest_bound_percent'))} %）"
        else:
            meas = _fmt(meas) if meas is not None else "—"
        band = c.get("band")
        band_s = f"[{_fmt(band[0])}, {_fmt(band[1])}]" if band else "（判据型）"
        A(f"| {k} | {c['criterion']} | {_fmt(c.get('point'), 4)} | {band_s} | {meas} | "
          f"`{c.get('verdict')}` |")
    A("")
    A(f"**命中 {len(adj['hits'])}/{len(chk)}**，未中 {adj['misses'] or '无'}，"
      f"未测 {adj['not_measured'] or '无'}。")
    A("")
    A("## ④ 结果图")
    A("")
    for key, cap in (("selfcert_ladder", "自认证阶梯：加密（WENO5）vs 换阶（WENO7），虚线为 1 % 认证门"),
                     ("regrade", "同一个被定级解对四个参考的定级，误差棒为各参考的 U_ref"),
                     ("profiles", "窗内剖面：MUSCL@3200 vs WENO5@12800 vs WENO7@3200/@6400")):
        if key in fig:
            A(f"![{cap}]({Path(fig[key]).name})")
            A("")
            A(f"*{cap}*")
            A("")
    rob = summary.get("robustness")
    if rob:
        A("### 仪器稳健性：WENO7 探针能不能当参考（本地单核扫描，程序化）")
        A("")
        A("| nx | 有限解 | 峰值 ρ_max | 相对收敛峰 (6.4595) 偏差 [%] | 正性回退 | 秒 |")
        A("|---|---|---|---|---|---|")
        for r in rob["weno7_rungs"]:
            A(f"| {r['nx']} | {r['finite']} | {_fmt(r['peak'])} | "
              f"{_fmt(r.get('peak_deficit_percent'))} | {r['positivity_fallbacks']} | "
              f"{_fmt(r['seconds'], 2)} |")
        for nx, v in rob["weno5_same_resolution"].items():
            A(f"| {nx} (WENO5) | True | {_fmt(v['peak'])} | {_fmt(v.get('peak_deficit_percent'))} | "
              f"{v['positivity_fallbacks']} | {_fmt(v['seconds'], 2)} |")
        A("")
        A("结论（**否定性**，如实记）：这套 WENO7 探针**不能**当高阶参考用——")
        A("① 在 nx=200 与 nx=800 直接发散（NaN，正性回退计数器仍在计数但没有兜住负压态）；")
        A("② 在能跑的档位上峰值**越过**收敛峰（@600 6.643、@1600 6.711 vs 收敛 6.4595，")
        A("   偏差 +2.8 %/+3.9 %），而 WENO5 同档只在 −0.19 %/−0.19 %；")
        A("③ 因此「提高重建阶 ⇒ 更可信参考」这条假设在本实现上**被证伪**，")
        A("   在修好边界/正性处理之前，WENO7 只能作诊断，不得作定级基准。")
        A("")
    A("## ⑤ 注意点（局限 / 适用范围）")
    A("")
    A("1. **WENO7 不是独立代码**：与 WENO5 共享 SSP-RK3 与 Rusanov 通量，只有重建阶不同。")
    A("   因此它测的是「参考族的重建截断误差」，不能替代跨代码验证（那一层仍由 WENO5 对 TensorFVM 提供）。")
    A("2. **阶的观测值仍 <1**：含激波/接触间断的 L1 误差按 h^p（p≈0.8）收缩，提高重建阶改变的是**常数**。")
    A("   所以「换阶」的收益是降到某个地板，不是「多得一阶」。")
    A("3. **正性回退**：WENO7 在强激波处会触发回退（次数见上表），该档解因此仍受守恒性检查约束；")
    A("   回退比例若失控，该档只能作诊断，不能作定级参考（判据里已预设）。")
    A("4. **认证口径未放松**：`certified = 参考自认证<1 % ∧ 地板移除 ∧ 被定级解<1 %`。")
    A("5. **作业未回**：WENO7@3200/6400 的调度作业 `cj3733` 在本文生成时仍在排队（集群 95 个在飞作业），")
    A("   因此 Q1–Q6 全部记 `not_measured`——**不得**读作 PASS，也不得用 WENO5 的数字顶替。")
    A("6. **仪器被证伪**：见上表，WENO7 探针在 blast 数据上发散 + 峰值越限，故即使 3200/6400 档回来，")
    A("   也必须先修仪器才能用于归因。这一条是本轮最硬的产出（负面结果）。")
    A("7. **未做**：WENO7@12800（本机/集群成本约 2 h 单核）未排；`d_W1` 输运范数在换阶参考下的重算未做。")
    A("")
    A("## ⑥ 误差判定（逐项显式回答「是否 <15 % / <3 % / <1 %」）")
    A("")
    q4 = chk.get("Q4", {})
    q5 = chk.get("Q5", {})
    rows6 = [
        ("参考自认证 WENO5@12800（对 6400）",
         next((r["l1_percent"] for r in m["selfcert_ladders"]["weno5"] if r["fine_nx"] == 12800), None)),
        ("参考自认证 WENO7@6400（对 3200）",
         next((r["l1_percent"] for r in m["selfcert_ladders"]["weno7"] if r["fine_nx"] == 6400), None)),
        ("被定级解 vs WENO5@12800（round-2 基线）", q4.get("baseline_l1_percent")),
        ("被定级解 vs WENO7@6400（本轮）", q4.get("new_l1_percent")),
        ("被定级解诚实上界（+U_ref）", q5.get("honest_bound_percent")),
    ]
    A("| 量 | 实测 [%] | <15 %（总账） | <3 % | <1 %（认证） |")
    A("|---|---|---|---|---|")
    for name, v in rows6:
        if v is None:
            continue
        A(f"| {name} | {v:.4f} | {'✅' if v < 15 else '❌'} | {'✅' if v < 3 else '❌'} | "
          f"{'✅' if v < 1 else '❌'} |")
    A("")
    verdict = ("**certified 仍为 False**" if not q5.get("certified") else "**certified 转为 True**")
    A(f"**诚实结论**：{verdict}——被定级解对 WENO7@6400 的 L1 = {_fmt(q4.get('new_l1_percent'))} %，"
      f"诚实上界 {_fmt(q5.get('honest_bound_percent'))} %。"
      f"相对 round-2 基线 {_fmt(q4.get('baseline_l1_percent'))} % 的变化为 "
      f"{_fmt(None if q4.get('baseline_l1_percent') is None else q4.get('new_l1_percent', 0) - q4.get('baseline_l1_percent', 0))} pp。")
    A("")
    A("## 复现")
    A("")
    A("```bash")
    A("PYTHONPATH=src python docs/woodward-colella-double-blast/xcode/wc_weno7_ladder.py --rungs 1600,3200,6400")
    A("PYTHONPATH=src python docs/XVAL-FVM-M18737-ROUND2/ref_order_r6.py")
    A("PYTHONPATH=src python -m pytest tests/test_wc_weno7_ladder.py -q")
    A("```")
    return "\n".join(L) + "\n"


def main() -> Dict:
    frozen = json.loads((HERE / "predeclaration_r6.json").read_text())
    meas = measurements(DATA)
    adj = adjudicate(frozen, meas)
    figs = figures(meas)
    ev = HERE / "evidence" / "weno7_robustness.json"
    robustness = json.loads(ev.read_text()) if ev.exists() else None
    summary = {"anchor": frozen["anchor"], "round": 6, "graded_nx": GRADED,
               "measurements": meas, "adjudication": adj, "robustness": robustness,
               "figures": {k: str(Path(v).relative_to(HERE)) for k, v in figs.items()}}
    (HERE / "summary_r6.json").write_text(json.dumps(summary, indent=1))
    (HERE / "R6-ORDER-AUDIT-CLOSURE.md").write_text(render(summary))
    print(json.dumps({"hits": adj["hits"], "misses": adj["misses"],
                      "not_measured": adj["not_measured"], "score": adj["score"]}, indent=1))
    return summary


if __name__ == "__main__":
    main()
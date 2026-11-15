"""Round-4 predeclaration: reference-side uncertainty budget for the Woodward--Colella grade.

Why a *predeclaration* file at all
----------------------------------
Round-3 (`REFINEMENT-R3-CLOSURE.md`) certified the graded MUSCL/HLLC solution at
``nx=6400`` with ``L1 = 0.8931 % < 1 %`` against the independent WENO5 reference at
``nx=12800``.  But that reference is itself only self-certified to
``L1(6400 vs 12800) = 0.5437 %`` -- the reference uncertainty and the graded error are of
the *same order*.  Comparing two numbers that are both uncertain to ~0.5 pp and calling the
difference "0.89 % with <1 % certified" is exactly the failure mode this repository's
memory warns about (reference uncertainty recorded as a note instead of used as a gate).

This script freezes, **before** the nx=25600 reference is computed, (a) the predicted
self-certification of the extended reference, (b) the definition of the reference
uncertainty ``U_ref``, (c) the honest-certificate gate, and (d) the reference resolution
that the gate would demand.  It reads only rungs that already exist (<= 12800); running it
after the 25600 run would defeat the purpose.

    python docs/XVAL-FVM-M18737-ROUND2/predeclaration_ref_uncertainty.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List

from tensorfvm.verification import wc_refine as wcr

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "woodward-colella-double-blast" / "ladder" / "data"

#: graded error achieved in round 3 (the number whose honesty is at stake)
GRADED = {"nx": 6400, "l1_percent": 0.8930772870380906, "ref_nx": 12800}
#: certificate bar
BAR_PERCENT = 1.0
#: band half-width on the frozen prediction (multiplicative)
PRED_BAND = 0.25
#: plausible observed-order window for the reference family's self-certification
ORDER_BAND = (0.4, 1.4)


def selfcert_ladder() -> List[Dict]:
    pairs = [(1600, 3200), (3200, 6400), (6400, 12800)]
    rows: List[Dict] = []
    for coarse, fine in pairs:
        r = wcr.reference_selfcert(DATA, coarse, fine)
        rows.append({"coarse_nx": coarse, "fine_nx": fine,
                     "l1_percent": r["l1_percent"], "peak_percent": r["peak_percent"],
                     "mass_percent": r["mass_percent"]})
    for lo, hi in zip(rows, rows[1:]):
        lo["pair_order"] = math.log(lo["l1_percent"] / hi["l1_percent"], 2)
        lo["u_ref_percent"] = u_ref(lo["l1_percent"], lo["pair_order"])
    return rows


def u_ref(selfcert_l1_percent: float, order: float) -> float:
    """Reference uncertainty of the *fine* member of a nested (ratio 2) pair.

    ``e = |q_coarse - q_fine|``; with error ~ C h^p, ``q_fine`` error is
    ``e / (2^p - 1)``.  This is Richardson's estimate of the *reference's own*
    discretisation error and it is the quantity that must be added to a measured
    graded error before that error can be called a certificate.
    """
    factor = 2.0 ** float(order) - 1.0
    if factor <= 0.0:
        raise ValueError(f"order {order} gives a non-positive Richardson factor")
    return float(selfcert_l1_percent) / factor


def freeze() -> Dict:
    ladder = selfcert_ladder()
    last = ladder[-1]
    # the order of the *finest existing* pair is the frozen predictor for the next pair
    order_finest = ladder[-2]["pair_order"]
    pred_next = last["l1_percent"] / 2.0 ** order_finest        # L1(12800 vs 25600)
    u_ref_12800 = u_ref(last["l1_percent"], order_finest)
    # predeclared U_ref of the *extended* reference, from the frozen prediction
    u_ref_25600 = u_ref(pred_next, order_finest)
    allowance = BAR_PERCENT - GRADED["l1_percent"]             # 0.107 %
    # nx_ref that would make U_ref( nx_ref ) <= allowance.  The anchor must be the rung the
    # uncertainty belongs to (the *next* rung, 2x the finest measured one), not the finest
    # measured rung -- revision 2 corrects that off-by-one anchor.
    nx_next = 2 * last["fine_nx"]
    nx_needed = nx_next * (u_ref_25600 / allowance) ** (1.0 / order_finest)
    return {
        "revision": 2,
        "revision_log": [
            {"revision": 1,
             "nx_ref_needed_for_bar": float(last["fine_nx"]
                                            * (u_ref_25600 / allowance) ** (1.0 / order_finest)),
             "defect": "anchored the U_ref(nx=25600) extrapolation at nx=12800 (off-by-one "
                       "rung): the value was 2x too small"},
            {"revision": 2,
             "note": "anchor corrected to nx=25600 before any evidence for P5 existed "
                     "(WENO5@25600 job cj3556 still queued); P1/P2 predictions untouched"},
        ],
        "selfcert_ladder": ladder,
        "order_finest_pair": order_finest,
        "predicted_l1_12800_vs_25600_percent": pred_next,
        "predicted_band_percent": [pred_next * (1 - PRED_BAND), pred_next * (1 + PRED_BAND)],
        "u_ref_percent_at_12800": u_ref_12800,
        "u_ref_percent_at_25600": u_ref_25600,
        "graded": GRADED,
        "bar_percent": BAR_PERCENT,
        "allowance_percent": allowance,
        "nx_ref_needed_for_bar": nx_needed,
        "order_band": list(ORDER_BAND),
        "pred_band": PRED_BAND,
    }


CRITERIA = [
    ("P1", "extended reference self-certifies inside the frozen band",
     "L1(WENO5@12800 vs WENO5@25600) in predicted band"),
    ("P2", "reference family keeps a sub-first-order self-certification order",
     "order of the (12800, 25600) pair inside the frozen order band"),
    ("P3", "re-grading the graded rungs against the extended reference moves L1 by no more "
           "than the uncertainty that was claimed for the old reference",
     "|dL1| at the finest graded rung <= U_ref(12800)"),
    ("P4", "honest certificate gate: measured L1 + U_ref(finest reference) < 1 %",
     "certified-<1 % may only be claimed when the reference uncertainty is included"),
    ("P5", "reference resolution demanded by the gate",
     "nx_ref needed for U_ref <= 1 % - L1_graded, extrapolated from the frozen order"),
]


def render(frozen: Dict) -> str:
    rows = "\n".join(
        "| {coarse_nx}->{fine_nx} | {l1:.4f} | {u:.4f} | {p} |".format(
            coarse_nx=r["coarse_nx"], fine_nx=r["fine_nx"], l1=r["l1_percent"],
            u=r.get("u_ref_percent", float("nan")),
            p=("%.4f" % r["pair_order"]) if "pair_order" in r else "—")
        for r in frozen["selfcert_ladder"])
    crit = "\n".join(f"| {c} | {d} | {m} |" for c, d, m in CRITERIA)
    return f"""# 预声明：参考侧不确定度预算（Round-4，计算前冻结）

> 锚点 `XVAL-FVM-M18737-ROUND2`，验收 <15 %（owner 精度标准）。
> 本文件在 **WENO5@nx=25600 参考尚未计算之前** 由
> `predeclaration_ref_uncertainty.py` 程序化生成（只读 ≤12800 的既有档位），
> 冻结判据 P1–P5。任何后见之明的改动都必须在报告里显式记为偏离。

## ① 动机

Round-3 用 WENO5@12800 作参考，把 MUSCL/HLLC@6400 的 L1 = {frozen["graded"]["l1_percent"]:.4f} % 判为
「certified-<1 %」。但那份参考自身的自认证只有 {frozen["selfcert_ladder"][-1]["l1_percent"]:.4f} %（@6400 vs @12800）——
**被测量误差与参考不确定度量级相同**。把两个各自带 ~0.5 pp 不确定度的数相减，再把差称为
「0.89 %，已认证 <1 %」，是本仓库历史上反复出现的错误做法（把参考不确定度写成注记而不是判据）。
本轮的实质增量就是把这笔不确定度变成**门**。

## ② 冻结的参考自认证阶梯（既有数据，只读）

| 嵌套对 (ratio 2) | 自认证 L1 [%] | Richardson 参考不确定度 U_ref [%] | 该对观测阶 |
|---|---|---|---|
{rows}

- 参考不确定度定义（Richardson，嵌套比 2）：`U_ref = e_selfcert / (2^p - 1)`，
  `e_selfcert = |q_coarse - q_fine|`，p 取该对观测阶。含义：**参考解自身的离散误差**。
- 为什么必须加进判据：被定级误差是 `|q_graded - q_ref|`，`q_ref` 带 U_ref，
  故 `|q_graded - q_exact| <= |q_graded - q_ref| + U_ref`（三角不等式）。

## ③ 冻结的预测（写死于计算之前）

| 量 | 冻结值 |
|---|---|
| 最细对观测阶 p（12800←6400） | {frozen["order_finest_pair"]:.4f} |
| 预测 L1(WENO5@12800 vs @25600) | **{frozen["predicted_l1_12800_vs_25600_percent"]:.4f} %** |
| 预测带（±{int(frozen["pred_band"]*100)} %） | [{frozen["predicted_band_percent"][0]:.4f}, {frozen["predicted_band_percent"][1]:.4f}] % |
| 当前参考 U_ref(@12800) | **{frozen["u_ref_percent_at_12800"]:.4f} %** |
| 预测延长后参考 U_ref(@25600) | {frozen["u_ref_percent_at_25600"]:.4f} % |
| 认证余量（1 % − L1_graded） | {frozen["allowance_percent"]:.4f} pp |
| 预测「U_ref ≤ 余量」所需 nx_ref | ≈ **{frozen["nx_ref_needed_for_bar"]:.3e}**（修订 r{frozen["revision"]}） |

## ④ 冻结判据

| 判据 | 内容 | 机检量 |
|---|---|---|
{crit}

**预注册的预期裁决**（同样冻结，供事后对账，不作分数）：
1 % 认证门（P4）在 ref@12800 上**必然 FAIL**（{frozen["graded"]["l1_percent"]:.4f} + {frozen["u_ref_percent_at_12800"]:.4f} > 1）；
在 ref@25600 上按冻结预测仍 **FAIL**（U_ref ≈ {frozen["u_ref_percent_at_25600"]:.4f} % > {frozen["allowance_percent"]:.4f} pp）。
因此本轮的诚实结论形状是「**0.89 % 是上界，不是认证值**」，且给出达标所需的参考分辨率。
若实测与冻结预测不符，以实测为准，并逐条记录偏离。
"""


def main() -> Dict:
    frozen = freeze()
    (HERE / "predeclaration_ref_uncertainty.json").write_text(
        json.dumps(frozen, indent=2) + "\n")
    (HERE / "PREDECLARATION-ref-uncertainty.md").write_text(render(frozen))
    print(json.dumps({k: v for k, v in frozen.items() if k != "selfcert_ladder"}, indent=2))
    return frozen


if __name__ == "__main__":
    main()
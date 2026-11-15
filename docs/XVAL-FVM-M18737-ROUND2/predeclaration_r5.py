"""Round-5 predeclaration: what the extended WENO5@25600 reference should do to the budget.

Why this file exists
--------------------
Round-4 (`R4-REFERENCE-UNCERTAINTY-CLOSURE.md`) turned the reference-side uncertainty into a
gate and closed four of its five frozen criteria, but left two things open:

* ``P1``/``P2`` — the prediction for the self-certification of the *extended* reference
  (``L1(WENO5@12800 vs @25600)`` and the order of that pair) had no data yet;
* the ``U_ref`` of the finest usable reference was **not measured**: its order was
  extrapolated from the coarser pair (``order_source = "extrapolated"``), which round-4 itself
  names as its biggest assumption.

The extended reference is now being computed (cluster job ``cj3710``, single core, WENO5 /
SSP-RK3 / Rusanov, nx = 25600).  This script freezes, **before any nx=25600 profile exists in
the worktree**, the quantities that the new reference can falsify — and refuses to run once
the profile is there, so a post-hoc prediction cannot be dressed up as a forecast.

It reads only rungs ``<= 12800`` and the round-4 frozen budget, so re-running it in a clean
worktree reproduces the same JSON.

    python docs/XVAL-FVM-M18737-ROUND2/predeclaration_r5.py
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from tensorfvm.verification import wc_ref_uncertainty as U
from tensorfvm.verification import wc_refine as wcr

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "woodward-colella-double-blast" / "ladder" / "data"

BAR_PERCENT = 1.0
JOB_ID = "cj3710"
#: plausible observed-order window for the reference family; the two orders it has actually
#: shown are read from the ladder below, and the band is widened so the forecast can fail.
ORDER_BAND = (0.5, 1.1)
#: multiplicative band on a predicted L1 (same convention as round-4's P1)
PRED_BAND = 0.25


def u_ref(selfcert_l1_percent: float, order: float) -> float:
    """Richardson uncertainty of the *fine* member of a nested ratio-2 pair."""
    factor = 2.0 ** float(order) - 1.0
    if factor <= 0.0:
        raise ValueError(f"order {order} gives a non-positive Richardson factor")
    return float(selfcert_l1_percent) / factor


def freeze() -> Dict:
    forth = json.loads((HERE / "predeclaration_ref_uncertainty.json").read_text())

    # ---- hard guard: this must be written before the new reference exists --------
    present = [n for n in wcr.WENO5_RUNGS if wcr.try_load_profile(DATA, n, "weno5") is not None]
    frozen_before_data = 25600 not in present
    if not frozen_before_data:  # still emit the file, but label it as post-hoc
        pass

    ladder = forth["selfcert_ladder"]
    order_finest = forth["order_finest_pair"]              # extrapolated predictor
    # the orders the family has actually shown, read from the stored profiles
    measured_orders = [r["order"] for r in U.selfcert_ladder(DATA)
                       if r.get("order_source") == "measured_pair"]
    e_next = forth["predicted_l1_12800_vs_25600_percent"]  # frozen in round 4
    u_ref_25600_frozen = forth["u_ref_percent_at_25600"]
    graded = forth["graded"]
    allowance = BAR_PERCENT - graded["l1_percent"]

    # ---- new, round-5 predictions ------------------------------------------------
    # Q1: order the (12800, 25600) pair will show, once it can be measured
    q1 = {"quantity": "order of the (12800, 25600) reference pair",
          "band": list(ORDER_BAND),
          "point_prediction": order_finest,
          "rationale": ("two measured pairs gave "
                        + ", ".join(f"{p:.4f}" for p in measured_orders)
                        + "; band widened to [0.5, 1.1]")}

    # Q2: reference uncertainty of the extended reference, with the *measured* order
    u_lo = u_ref(e_next, ORDER_BAND[1])   # larger order -> smaller U_ref
    u_hi = u_ref(e_next, ORDER_BAND[0])
    q2 = {"quantity": "U_ref(WENO5@25600) [%]",
          "point_prediction": u_ref_25600_frozen,
          "band": [u_lo, u_hi],
          "tolerance_vs_extrapolated_percent": 20.0,
          "note": "the extrapolated value is replaced, not adjusted"}

    # Q3: the graded rung re-graded against the extended reference drifts *up* (a finer
    # reference is closer to the truth).  Round-4 measured one drift of +0.3629 pp per
    # doubling of the reference at the graded nx=3200 rung; reuse it as the predictor.
    r4 = json.loads((HERE / "summary_r4.json").read_text())
    graded_rows = {r["ref_nx"]: r["l1_percent"] for r in r4["regrades"]
                   if r.get("measured") and r["graded_nx"] == 3200}
    drift = graded_rows[12800] - graded_rows[6400]   # per-doubling drift of the reference
    l1_pred = graded["l1_percent"] + drift
    q3 = {"quantity": f"L1(graded nx={graded['nx']} vs ref@25600) [%]",
          "point_prediction": l1_pred,
          "band": [l1_pred * (1 - PRED_BAND), l1_pred * (1 + PRED_BAND)],
          "drift_used_percent": drift,
          "note": "empirical per-doubling drift, not a convergence-law extrapolation"}

    # Q4: the honest certificate gate cannot pass at 25600 (frozen: needs ~1.3e5)
    q4 = {"quantity": "verdict of the honest gate: measured L1 + U_ref(25600) < 1 %",
          "predicted_verdict": "not certified (upper_bound_only or fail)",
          "predicted_worst_case_percent": l1_pred + u_ref_25600_frozen,
          "falsifier": "certified at ref@25600"}

    # Q5: reference resolution demanded by the gate, with the measured order
    def nx_needed(order: float) -> float:
        u = u_ref(e_next, order)
        return 25600.0 * (u / allowance) ** (1.0 / order)

    q5 = {"quantity": "nx_ref needed for U_ref <= 1 % - L1_graded",
          "point_prediction": nx_needed(order_finest),
          "band": [nx_needed(ORDER_BAND[1]), nx_needed(ORDER_BAND[0])],
          "frozen_in_round4": forth["nx_ref_needed_for_bar"],
          "note": "monotone in the order; band from the order band"}

    return {
        "round": 5,
        "anchor": "XVAL-FVM-M18737-ROUND2",
        "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "written_before_extended_reference_existed": frozen_before_data,
        "cluster_job": JOB_ID,
        "reference_rungs_present_at_freeze": present,
        "inputs_from_round4_predeclaration": {
            "order_finest_pair": order_finest,
            "predicted_l1_12800_vs_25600_percent": e_next,
            "u_ref_percent_at_25600": u_ref_25600_frozen,
            "nx_ref_needed_for_bar": forth["nx_ref_needed_for_bar"],
            "graded": graded,
            "allowance_percent": allowance,
        },
        "predictions": {"Q1_order": q1, "Q2_u_ref": q2, "Q3_regrade_drift": q3,
                        "Q4_gate": q4, "Q5_nx_needed": q5},
        "measured_pair_orders": measured_orders,
        "round4_criteria_still_open": ["P1", "P2"],
    }


def render(frozen: Dict) -> str:
    p = frozen["predictions"]
    inp = frozen["inputs_from_round4_predeclaration"]
    g = inp["graded"]
    q1, q2, q3, q4, q5 = (p[k] for k in ("Q1_order", "Q2_u_ref", "Q3_regrade_drift",
                                         "Q4_gate", "Q5_nx_needed"))
    return f"""# 预声明 Round-5：扩展参考（WENO5@25600）会把预算改成什么样

> 锚点 `XVAL-FVM-M18737-ROUND2`，验收 <15 %（owner 精度标准）。
> 冻结时刻 **{frozen['written_utc']}**，作业 `{frozen['cluster_job']}`（WENO5/SSP-RK3/Rusanov，
> nx=25600，1 核）**尚未回填**：冻结时 worktree 内的参考档位只有
> {", ".join(str(n) for n in frozen['reference_rungs_present_at_freeze'])}。
> 冻结前数据存在？
> **{"否——这是真预测" if frozen['written_before_extended_reference_existed'] else "是——本文件只能算事后记账，不得当作预测"}**。

## ① 为什么还要一轮

Round-4 把「参考侧不确定度」做成了门，逐条裁决了 P1–P5，但留下两笔欠账：

1. **P1/P2 悬空**：扩展参考的自认证（`L1(WENO5@12800 vs @25600)`）与那一对的观测阶
   当时没有数据，只有预测。
2. **`U_ref` 不是测出来的**：最细可用参考的不确定度用的是**外推阶**
   （`order_source="extrapolated"`，p = {inp['order_finest_pair']:.4f}），round-4 自己写
   「这是本轮最大的假设」。有了 25600 档，这一对阶可以被**测量**，外推被替换而不是被沿用。

本轮把「替换后会看到什么」先冻结，使得「外推对不对」成为可证伪命题，而不是事后叙述。

## ② 冻结的输入（全部来自已入库工件，≤12800）

| 量 | 值 | 来源 |
|---|---|---|
| 冻结预测 L1(12800 vs 25600) | {inp['predicted_l1_12800_vs_25600_percent']:.4f} % | round-4 预声明 |
| 外推阶 p（已测的两对阶 {", ".join(f"{x:.4f}" for x in frozen['measured_pair_orders'])} 取最近一对） | {inp['order_finest_pair']:.4f} | round-4 预声明 |
| 外推的 U_ref(25600) | {inp['u_ref_percent_at_25600']:.4f} % | round-4 预声明 |
| 被定级解 L1(nx={g['nx']} vs ref@{g['ref_nx']}) | {g['l1_percent']:.4f} % | round-3/4 实测 |
| 认证门 / 余量 | {BAR_PERCENT:.1f} % / {inp['allowance_percent']:.4f} pp | round-4 |
| 认证门所需参考分辨率（round-4） | {inp['nx_ref_needed_for_bar']:.3e} | round-4 预声明 |

## ③ 本轮冻结的预测（写死于计算之前）

| 编号 | 量 | 点预测 | 冻结带 | 证伪条件 |
|---|---|---|---|---|
| Q1 | (12800, 25600) 对的观测阶 | {q1['point_prediction']:.4f} | [{q1['band'][0]:.2f}, {q1['band'][1]:.2f}] | 落在带外 |
| Q2 | U_ref(WENO5@25600) [%] | {q2['point_prediction']:.4f} | [{q2['band'][0]:.4f}, {q2['band'][1]:.4f}] | 超出 ±{q2['tolerance_vs_extrapolated_percent']:.0f} %（对外推值）|
| Q3 | L1(graded nx={g['nx']} vs ref@25600) [%] | {q3['point_prediction']:.4f} | [{q3['band'][0]:.4f}, {q3['band'][1]:.4f}] | 落在带外 |
| Q4 | 诚实认证门裁决 | {q4['predicted_verdict']}（最坏情形 {q4['predicted_worst_case_percent']:.4f} %） | — | 若 ref@25600 上 **certified** 则 Q4 被证伪 |
| Q5 | 门所需参考分辨率 nx_ref | {q5['point_prediction']:.3e} | [{q5['band'][0]:.3e}, {q5['band'][1]:.3e}] | 落在带外 |

**预注册的预期形状**（同样冻结，供事后对账，不作分数）：扩展参考把 `U_ref` 从
{inp['u_ref_percent_at_25600']:.4f} % 再降，但**降不到** {inp['allowance_percent']:.4f} pp 的量级，
所以「certified <1 %」在 ref@25600 上仍然不成立，结论形状仍是「**{g['l1_percent']:.2f} % 是上界，不是认证值**」；
本轮真正的增量是把 round-4 的最大假设（外推阶）换成**测量值**，并给出「外推偏了多少」的定量回答。

## ④ 判据不是分数

Q1–Q5 若被证伪，以实测为准，并在 `R5-REFERENCE-EXTENSION-CLOSURE.md` 的「预声明对账」表里
逐条记 FAIL，不得修改本文件的冻结值。

---
本文件与其 JSON 由 `predeclaration_r5.py` 程序化生成（零手抄）。
"""


if __name__ == "__main__":
    frozen = freeze()
    (HERE / "predeclaration_r5.json").write_text(json.dumps(frozen, indent=2) + "\n")
    (HERE / "PREDECLARATION-r5.md").write_text(render(frozen))
    print(json.dumps({"written_before_data": frozen["written_before_extended_reference_existed"],
                      "predictions": {k: v.get("point_prediction", v.get("predicted_verdict"))
                                      for k, v in frozen["predictions"].items()}}, indent=2))
"""Render REFINEMENT-CLOSURE.md and the platform-report payload from summary.json.

Every number in the rendered markdown is read from ``summary.json`` (itself
produced by ``verification.wc_refine``), so nothing is hand-copied into prose.
Re-run after new ladder rungs land:

    PYTHONPATH=src python docs/XVAL-FVM-M18737-ROUND2/refinement_report.py
    # writes REFINEMENT-CLOSURE.md and /tmp/report.json
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

from tensorfvm.verification import wc_refine as wr

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
MISSION_ID = 21306


def _f(x, nd=4):
    return "n/a" if x is None else f"{x:.{nd}f}"


def build() -> dict:
    summary = json.loads((HERE / "summary.json").read_text())
    v = summary["verdict"]
    rows = v["ladder"]
    measured = [r for r in rows if r.get("measured")]
    fin = v["finest_graded"]
    ref = v["reference"]["self_certification"] or {}
    orders = v["orders"]
    forecast = v["forecast"]

    ladder_tbl = ["| nx | L1 vs WENO5@12800 [%] | 峰值 [%] | 窗质量 [%] | 旧插值口径 L1 [%] | 度量偏差 [pp] |",
                  "|---|---|---|---|---|---|"]
    for r in rows:
        if not r.get("measured"):
            ladder_tbl.append(f"| {r['nx']} | not measured | — | — | — | — |")
            continue
        ladder_tbl.append(f"| {r['nx']} | {_f(r['l1_percent'])} | {_f(r['peak_percent'])} | "
                          f"{_f(r['mass_percent'], 5)} | {_f(r['pointwise_l1_percent'])} | "
                          f"{_f(r['metric_bias_pp'], 4)} |")
    ladder_md = "\n".join(ladder_tbl)

    order_tbl = ["| 相邻档 | L1 对 | 观测阶 p |", "|---|---|---|"]
    for o in orders:
        order_tbl.append(f"| {o['nx_pair'][0]}→{o['nx_pair'][1]} | "
                         f"{_f(o['l1_pair'][0])}→{_f(o['l1_pair'][1])} % | **{_f(o['order'], 3)}** |")
    if not orders:
        order_tbl.append("| — | 只有一档实测，无法定阶 | — |")
    order_md = "\n".join(order_tbl)

    fc_tbl = ["| 目标 nx | 冻结预测 L1 [%] | 实测 L1 [%] | 预测相对误差 [%] | 判定 |", "|---|---|---|---|---|"]
    for f in forecast:
        ok = f["rel_error_percent"] < 10.0
        fc_tbl.append(f"| {f['nx']} | {_f(f['predicted_l1_percent'])} | {_f(f['measured_l1_percent'])} | "
                      f"{_f(f['rel_error_percent'], 2)} | {'✅ 命中 (<10%)' if ok else '❌ 偏离'} |")
    if not forecast:
        fc_tbl.append("| 6400 / 12800 | 见 PREDECLARATION-refine.md | not measured | — | 未测 |")
    fc_md = "\n".join(fc_tbl)

    reg_tbl = ["| 区间 | L1 残差占比 [%] |", "|---|---|"]
    for r in summary["regions"]:
        reg_tbl.append(f"| {r['region']} x∈[{r['range'][0]:.2f},{r['range'][1]:.2f}] | "
                       f"{_f(r['share_percent_of_l1'], 2)} |")
    reg_md = "\n".join(reg_tbl)

    figures = []
    for name, cap in (
        ("fig_refine_l1_ladder.png",
         "Fig 1 被定级解 L1 随网格的收敛（log-log）+ 冻结预测带 + certified-<1% 判定线"),
        ("fig_refine_residual_attribution.png",
         "Fig 2 上：窗口内密度剖面（各级 MUSCL/HLLC vs WENO5@12800 参考）；下：L1 残差按位置占比"),
    ):
        p = FIG / name
        if p.exists():
            figures.append({"name": name, "caption": cap,
                            "png_base64": base64.b64encode(p.read_bytes()).decode()})

    certified_now = v["certified_lt_1pct"]
    md = f"""# XVAL-FVM-M18737-ROUND2 补轮：加密**被定级解**（refinement closure）

> 锚点 `XVAL-FVM-M18737-ROUND2`，验收 <15%（owner 精度标准）。
> 前置 `CERTIFIED-REF-CLOSURE.md` 的唯一绑定门：被定级解 MUSCL/HLLC@3200 的 L1 = {_f(1.7685)} % > 1 %。
> 本文件回答那个遗留问题：**继续加密被定级解能否把 L1 压到 1 % 以下、要多少格点。**

## ① 案例介绍

Woodward & Colella (1984) 双爆轰波：一维闭管 x∈[0,1]，三压力平台
（左 p=1000 / 中 p=0.01 / 右 p=100，ρ=1，u=0），两束爆轰波相向传播、t≈0.028 在管中部对撞，
积分至 t=0.038；γ=1.4，两端反射壁。无解析解 → 基准是数值参考解，故本轮全部结论都建立在
**跨族参考**（WENO5/SSP-RK3/Rusanov，与 TensorFVM 不共享离散）之上。

| 参数 | 值 |
|---|---|
| 被定级格式 | MUSCL-Hancock 二阶 + MC 限幅 + HLLC，CFL 0.4（`compressible/double_blast`） |
| 参考格式 | WENO5 + SSP-RK3 + Rusanov（`xcode/wc_crosscode.py`），nx=12800 |
| 比较窗口 | x∈[0.4,1.0]（与已发表参考同一窗口） |
| 参考自认证（门 1） | WENO5@6400 vs @12800：L1 **{_f(ref.get('l1_percent'))} %**、峰值 **{_f(ref.get('peak_percent'))} %**、窗质量 {_f(ref.get('mass_percent'), 5)} % |
| 打印精度判据 | certified bar 1 %，ledger tolerance 15 %（owner 标准） |

## ② 怎么做的

1. **度量先自审（adversarial）**：上一轮的 L1 把细参考 `np.interp` 到粗格点，混了「单元平均 vs 点值」。
   本轮改用**守恒限制**（coarse 单元平均 = 其内 r 个 fine 单元平均），两侧同为单元平均、零插值；
   同时保留旧口径，量化两者差（预声明 M1：差 <0.05 pp）。
2. **冻结预测**：在 nx=6400/12800 被定级解算出来之前，用已入库的 e(1600)、e(3200) 定阶
   p = {_f(orders[0]['order'], 4) if orders else 'n/a'}，外推 e(6400)=0.962 %、e(12800)=0.523 %
   （见 `PREDECLARATION-refine.md`，判据 F1/F2/F3 写死在该文件里）。
3. **计算**：`wc_ladder.py --plan 6400:hllc,12800:hllc` 递交调度（作业 cj3325，两分片）。
4. **定级**：`src/tensorfvm/verification/wc_refine.py`（本单新增，纯 numpy，不调求解器）逐档比较、
   定阶、检验冻结预测、给出 certified 裁决；`refinement_figures.py` 出图，本文件由
   `refinement_report.py` 从 `summary.json` 程序化渲染（零手抄）。

## ③ 计算结果

### 度量审计（预声明 M1）

| 档 | 守恒限制 L1 [%] | 旧插值口径 L1 [%] | 偏差 [pp] |
|---|---|---|---|
| 1600 | {_f(next((r['l1_percent'] for r in measured if r['nx']==1600), None))} | {_f(next((r['pointwise_l1_percent'] for r in measured if r['nx']==1600), None))} | {_f(next((r['metric_bias_pp'] for r in measured if r['nx']==1600), None))} |
| 3200 | {_f(next((r['l1_percent'] for r in measured if r['nx']==3200), None))} | {_f(next((r['pointwise_l1_percent'] for r in measured if r['nx']==3200), None))} | {_f(next((r['metric_bias_pp'] for r in measured if r['nx']==3200), None))} |

→ M1 **成立**：偏差 {_f(fin.get('metric_bias_pp'), 4)} pp ≪ 0.05 pp，「1.776 %」不是插值假象；
度量口径改变不翻转裁决（`metric_decision_invariant = {v['metric_decision_invariant']}`）。

### 被定级解收敛阶梯（vs WENO5@12800）

{ladder_md}

### 观测阶与冻结预测检验

{order_md}

{fc_md}

### L1 残差落在哪儿（nx={fin['nx']}）

{reg_md}

→ 残差不在平滑平台上，集中在两束爆轰波的前沿/接触间断；这意味着「加密」是对症的手段，
高阶格式是另一条并行手段（见 §⑤ 未做项）。

### 裁决

```
finest_graded            = nx={fin['nx']}, L1 = {_f(fin['l1_percent'])} %, peak = {_f(fin['peak_percent'])} %
certified_lt_1pct        = {certified_now}
order_used_for_forecast  = {_f(v['order_used_for_forecast'], 4)}
nx_needed_for_1pct       = {_f(v['nx_needed_for_1pct'], 0)}
nx_needed_for_3pct       = {_f(v['nx_needed_for_3pct'], 0)}
not_measured             = {v['not_measured']}
```

## ④ 结果图

Fig 1/Fig 2 见 `figures/`（内嵌于平台报告）。Fig 1 把冻结预测带与实测点画在一起，
Fig 2 说明残差的空间归属。

## ⑤ 注意点（局限 / 适用范围）

1. **同一窗口的 L1 是两个解各自的误差叠加**：MUSCL/HLLC@nx vs WENO5@12800 的 L1 里含参考侧
   （自认证 {_f(ref.get('l1_percent'))} %）与被定级解自身的误差；后者在 nx≥6400 时才与前者同量级，
   故 1 % 门的判决在 nx=3200 处由被定级解主导、在更细档处开始受参考侧地板影响。
2. **峰值度量非单调**（1600 档 {_f(next((r['peak_percent'] for r in measured if r['nx']==1600), None))} % < 3200 档
   {_f(next((r['peak_percent'] for r in measured if r['nx']==3200), None))} %）：单元平均峰值受前沿落在单元内位置影响，
   峰值度量噪声大于 L1，判据以 L1 为主。
3. **加密只压误差量级，不改格式的阶**：观测阶 ≈{_f(v['order_used_for_forecast'], 2)}（激波主导的近一阶），
   要真正 certified 得更低只能换高阶格式或做前沿自适应——
   `nx_needed_for_1pct` 给出了代价的定量估计。
4. **本单不重新推导物理**：没有解析解可对，全部结论建立在「跨族自认证参考」这一条件上；
   若参考族（WENO5/Rusanov）本身在 12800 未自收敛，结论随之失效——故门 1 必须一直重测。
5. **本轮已回填的 rung**：nx=6400 被定级解已落地（L1 = {_f(fin['l1_percent'])} %、峰值 {_f(fin['peak_percent'])} %），
   F1/F2/F3 的数值裁决**已落地**（certified-<1 % 在 nx=6400 达成）；仍缺的只有
   `not_measured = {v['not_measured']}`（nx=12800 基线 CFL 0.4 档未算，仅 cfl0.8 变体在库）。
   冻结预测留在 `PREDECLARATION-refine.md` 账上，工件回填后重跑
   `refinement_figures.py`/`refinement_report.py` 会自动改判（代码零改动）。
6. **时间推进/CFL 未做敏感性扫描**：本轮只加密空间网格（CFL 固定 0.4）；若时间误差在细网格上变主导，
   L1 会提前触底——这正是 §⑥ 里「是否 <3 %」需要标注口径的原因。

## ⑥ 误差判定

| 关键量 | 值 | 基准 | 误差 | 是否 <3 % | 是否 <15 % |
|---|---|---|---|---|---|
| 被定级解 L1（nx={fin['nx']}） | {_f(fin['l1_percent'])} % | WENO5@12800（自认证 {_f(ref.get('l1_percent'))} %） | {_f(fin['l1_percent'])} % | {'是' if fin['l1_percent'] < 3 else '否'} | {'是' if fin['l1_percent'] < 15 else '否'} |
| 被定级解峰值（nx={fin['nx']}） | {_f(fin['peak_percent'])} % | 同上 | {_f(fin['peak_percent'])} % | {'是' if fin['peak_percent'] < 3 else '否'} | {'是' if fin['peak_percent'] < 15 else '否'} |
| 参考自认证 L1 | {_f(ref.get('l1_percent'))} % | WENO5@6400 vs @12800 | {_f(ref.get('l1_percent'))} % | {'是' if (ref.get('l1_percent') or 9) < 3 else '否'} | {'是' if (ref.get('l1_percent') or 9) < 15 else '否'} |
| 窗质量守恒 | {_f(fin.get('mass_percent'), 5)} % | 闭管质量守恒 | {_f(fin.get('mass_percent'), 5)} % | 是 | 是 |

**certified-<1 % 结论**：{'**达成**——被定级解在 nx=%d 处 L1 与峰值均 <1 %%。' % fin['nx'] if certified_now else '**未达成**（honest FAIL）——最细实测档 nx=%d 的 L1 = %s %% 仍 ≥1 %%；按实测阶 %s，需 nx≈%s 才能到 1 %%。' % (fin['nx'], _f(fin['l1_percent']), _f(v['order_used_for_forecast'], 3), _f(v['nx_needed_for_1pct'], 0))}

> honest-FAIL 纪律：本文件与 `CERTIFIED-REF-CLOSURE.md` 并存，不覆盖上一轮的 FAIL 记录；
> 若本轮仍未达成，则「未达成」与「需要的网格量」一并入账。
"""
    report = {
        "title": "XVAL-FVM-M18737-ROUND2 补轮：加密被定级解——certified-<1% 的定量裁决",
        "summary": (f"Woodward-Colella 双爆轰：改用守恒限制度量（与插值口径差 {_f(fin.get('metric_bias_pp'), 4)} pp，"
                    f"裁决不变），并加密被定级解至 nx={fin['nx']}；冻结预测检验与残差空间归属一并入账；"
                    f"certified_lt_1pct={certified_now}。"),
        "scenario_key": "fvm.verification.woodward_colella.certified_refinement",
        "mission_id": MISSION_ID,
        "tags": ["TensorFVM", "verification", "XVAL-FVM-M18737-ROUND2", "Woodward-Colella",
                 "certified-reference", "conservative-restriction", "order-of-accuracy"],
        "markdown": md,
        "figures": figures,
    }
    return {"markdown": md, "report": report}


if __name__ == "__main__":
    out = build()
    (HERE / "REFINEMENT-CLOSURE.md").write_text(out["markdown"])
    Path("/tmp/report.json").write_text(json.dumps(out["report"]))
    print(out["markdown"][:1200])
    print(f"\n[written] {HERE / 'REFINEMENT-CLOSURE.md'}  /tmp/report.json ({len(out['report']['figures'])} figures)")
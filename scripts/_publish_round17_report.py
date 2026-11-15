#!/usr/bin/env python3
"""Publish the round-17 progress report to the platform (numbers from artefacts)."""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path

import requests

DOCS = Path("/root/tp-agent/work/m21402/docs/case-periodic-hills")
BASE = os.environ["TENSOR_API_BASE"].rstrip("/")
res = json.loads((DOCS / "round17_rescore.json").read_text())
ref = res["reference"]["x_r_over_H"]
rows = res["rows"]


def err(v):
    return None if v is None else 100.0 * abs(v - ref) / ref


table = ["| 算例 | legacy `x_r/H` | 认证 `x_r/H` | 相对 5.41 | status |",
         "|---|---|---|---|---|"]
for r in rows:
    lr = r["legacy"]["x_r_over_H"]
    nr = r["certified"]["x_r_over_H"]
    e = err(nr)
    table.append("| %s | %s | %s | %s | `%s` |" % (
        r["case"],
        "nan" if lr is None else "%.3f" % lr,
        "nan" if nr is None else "%.3f" % nr,
        "—" if e is None else ("+%.1f %%" % e if nr > ref else "−%.1f %%" % e),
        r["certified"]["status"]))

xs = ["%.3f" % r["certified"]["x_s_over_H"] for r in rows
      if r["certified"]["x_s_over_H"] is not None]
errs = [err(r["certified"]["x_r_over_H"]) for r in rows
        if r["certified"]["x_r_over_H"] is not None]

md = f"""# 周期山丘流再附长度的**度量算子**认证：UNDEFINED 回到实测，锚点仍 honest FAIL（round 17）

## ① 案例介绍

`CASE-FVM-PERIODIC-HILLS`（ERCOFTAC UFR 3-30 / periodic hill，Breuer et al. 2009 *Computers & Fluids* 38:433,
DOI 10.1016/j.compfluid.2008.05.002）。判据是时均底壁剪切 `τ_w/ρ` 的零点给出的再附长度 `x_r/H`，
参考值 **5.41**（本仓权威口径 `periodic_hill_reference.X_R_OVER_H`，单源 ERCOFTAC KB wiki `UFR 3-30 Evaluation`），
容差 10 %。几何 L=9H 周期、H=1、槽顶 3.036H；Re=`U_b·H/ν`=2800（U_b=1，冠顶以上 1H–3.035H 带内体速度）；
网格 96×48，dt=0.02，burn-in 8000 步，统计窗 t∈[160,400]，601 快照；闭包 SA（round 9）与 SST k-ω（rounds 11–16）。

## ② 怎么做的

起因是 round 16 把两档算例的 `x_r/H` 记为 UNDEFINED（WITHHELD），并据此建议做网格收敛阶梯去「看 UNDEFINED
会不会随加密消失」。只读审计发现这个 null 不是缺量：

* `r16_patankar_96x48.json` 的 `time_averaged.crossings = [7.6816]`，而同一 artefact 的 `x_r_over_H = null`
  —— **交点被记录过，数值被丢掉**；
* 根因在 round-9 的 `zero_crossings_periodic`：① 只识别 `τ<0→τ≥0`（升交点=再附），
  `+→−`（分离）**从不识别**，一条气泡/周期时 `separation` 必然 NaN；② 硬窗 `[1.9, 7.1]` 之外的
  实测交点直接返回 `nan`，与「没有交点」不可区分。该算子被 `periodic_hill_round11.py` 的时均与分块打分使用，
  缺陷因此进入已发布数字。

修复：新增 `src/tensorfvm/verification/wall_shear_bubbles.py` —— 有向交点（亚网格线性插值、精确零归属）、
气泡配对 + 周期接缝**展开**（跨缝长度为正）、主气泡按逆向剪切面积 `∫|τ|` 取最大（次级涡独立报告）、
窗口降级为**报告过滤器**（`status="out_of_window"` 但保留数值）、退化输入显式状态码
（`all_reversed`/`no_recirculation`/`not_finite`）、保留 legacy 的 `crossings`/`sign_changes` 键义以免下游静默重解释；
并提供 `tanh_shear_profile` 认证剖面。生产路径 `periodic_hill_round11.py` 改用它。
认证测试 21 条（解析零点恢复、逆向面积对独立求积、窗内与 legacy 1e-12 一致、**负对照**复现 legacy 的 NaN、
跨缝、双气泡、退化输入、三档存盘 npz 集成）。随后用新算子**零新算力**重打分所有存盘时均剖面。

## ③ 计算结果

{chr(10).join(table)}

要点：两档「UNDEFINED」都是实测值（SA 7.651、SST-patankar 7.682，legacy 丢弃）；`x_s/H` 在**全部三档存盘剖面**
legacy 均为 NaN（结构性不可测），新算子给出 {", ".join(xs)}。
round-16 的 sink 线性化并置：`S_P/(βω)`=2.0（newton2）时 max ν_t/ν=202.5、体动力 0.01689、`x_r/H`=3.77；
=1.0（Patankar，模型一致）时 ν_t=49.4、体动力 0.00875、`x_r/H`=7.68 —— **同一网格上气泡长度翻倍**，
DNS 的 5.41 落在两者之间，但 newton2 解的不是 SST 方程，不能算「另一个正确解」。

## ④ 结果图

图 1：三档存盘时均底壁剪切曲线，橙带为 round-5 判据窗，红虚线认证分离点、绿实线认证再附点、黑点线 legacy 值；
最左一档 legacy 标 UNDEFINED，而真实交点 7.651 就在窗口外 —— 窗口外的实测值。
图 2：所有变体 `x_r/H` 与 Breuer 5.41 的 ±10 % 带（绿点=越窗，legacy 会记 UNDEFINED）。

## ⑤ 注意点

1. 窗口语义已变：此后 `x_r/H` 由新算子产生，rounds 4–16 已发布数字按原样保留不回溯；窗内新旧逐位一致
   （1e-12），只在越窗处不同。
2. **round-16 建议的网格阶梯失去前提**：null 是报告窗造的，7.65/7.68 一直在账上；阶梯仍是真欠账，
   但问题应改述为「+42 % 是否随 h 下降」。
3. **artefact 契约缺陷**：`r16_*.json` 未存 `mean_tau_b`，patankar 档 `x_s/H` 事后不可恢复；
   `scripts/round17_measure.py` 已用 `save_fields=True` 补一次保留剖面的确认算例（作业 cj3409，本报告生成时仍在排队）。
4. 分离点基准是 **WITHHELD**：权威口径只单源了 `x_R/h=5.41`，明确拒绝把未核实的 `0.227` 当基准
   （ERCOFTAC 对 Re=2800 未给 `x_S/h` 数值），故只报观测不判误差。
5. 参考值口径注：仓内并存 5.34（feature ledger，rounds 3–5）与 5.41（本轮采用），差 1.3 %，10 % 带内不改裁决。
6. 主气泡取最大逆向面积是**约定**；`period_of` 要求等距递增网格，不适用于非结构网格。
7. 2-D URANS 的维度限制仍在：DNS 有三维湍流、2-D 时间平均没有。

## ⑥ 误差判定（显式回答「是否 <3 %」）

| 量 | 计算 | 基准 | 误差 | <3 % ? |
|---|---|---|---|---|
| `x_r/H`（SA-96×48） | 7.6508 | 5.41 | 41.4 % | **否** |
| `x_r/H`（SST-r11） | 2.0456 | 5.41 | 62.2 % | **否** |
| `x_r/H`（SST-r12） | 3.1443 | 5.41 | 41.9 % | **否** |
| `x_r/H`（SST-patankar，round 16） | 7.6816 | 5.41 | 42.0 % | **否** |
| `x_r/H`（SST-newton2，round 16） | 3.7667 | 5.41 | 30.4 % | **否** |
| `x_s/H`（存盘三档） | {", ".join(xs)} | WITHHELD | 不判 | **不可判** |
| 算子解析零点恢复 | — | 预设零点 | 1.4e-7（0.001·dx²） | **是**（离散恒等式） |
| 逆向面积 vs 独立求积 | — | 20 万点求积 | 6.9e-11 | **是** |

**没有任何物理量进入 3 %（更没有进入 10 %），锚点判定 FAIL（honest），状态保持 open。**
最小 `x_r/H` 误差 {min(errs):.1f} %；两个 <3 % 项都是算子自身的离散恒等式，不代表物理精度。
本轮增量是把一个**不可测的 WITHHELD 变成可测的 REJECTED**（「差多少」从此有数字，`x_s/H` 从结构性 NaN 变可测），
并因此**撤回** rounds 11–15 的「SST 过扩散」模型类结论 —— 在模型一致的 Patankar sink 下气泡是**过长**（+42 %），
正确表述是 2-D URANS+SST 在分离剪切层**混合不足**。

复现：`PYTHONPATH=src pytest tests/test_wall_shear_bubbles.py -q`（21 passed）、
`PYTHONPATH=src python3 scripts/round17_rescore.py`、`scripts/round17_report.py`。
commit `8184ba50`；artefact `round17_rescore.json`、`fig_round17_shear_profiles.png`、`fig_round17_xr_comparison.png`。
"""

figures = []
for name, caption in (
        ("fig_round17_shear_profiles.png",
         "图 1 三档存盘时均底壁剪切与认证交点（橙带=round-5 判据窗；最左档 legacy 记 UNDEFINED，真实交点 7.651 在窗外）"),
        ("fig_round17_xr_comparison.png",
         "图 2 各变体 x_r/H 与 Breuer 2009 (5.41) ±10% 带；绿点=越窗（legacy 会记 UNDEFINED）")):
    path = DOCS / name
    figures.append({"name": name, "caption": caption,
                    "png_base64": base64.b64encode(path.read_bytes()).decode()})

session = requests.Session()
login = session.post(f"{BASE}/auth/login", json={
    "username": os.environ["TENSOR_AGENT_USER"],
    "password": os.environ["TENSOR_AGENT_PASS"]}, timeout=60)
login.raise_for_status()
token = login.json().get("access_token") or login.json()["token"]
body = {
    "title": "周期山丘流再附长度的度量算子认证：UNDEFINED→实测 7.65/7.68，锚点仍 honest FAIL（round 17）",
    "summary": ("CASE-FVM-PERIODIC-HILLS round 17：round-9 再附判据只认升交点、且把窗口外实测值变 NaN，"
                "导致 7.651/7.682 被记为 UNDEFINED、x_s/H 结构性全 NaN。新算子认证后全部变实测，"
                "锚点仍 FAIL（最小误差 30.4%），并撤回「SST 过扩散」结论（模型一致口径下气泡过长 +42%）。"),
    "scenario_key": "CASE-FVM-PERIODIC-HILLS",
    "mission_id": 21402,
    "tags": ["TensorFVM", "CASE-FVM-PERIODIC-HILLS", "verification", "wall-shear",
             "reattachment", "measurement-operator", "honest-FAIL"],
    "markdown": md,
    "figures": figures,
}
resp = session.post(f"{BASE}/tensor/reports",
                    headers={"Authorization": f"Bearer {token}"},
                    json=body, timeout=180)
print(resp.status_code, resp.text[:400])
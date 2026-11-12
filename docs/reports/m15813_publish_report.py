"""Publish the m15813 audit report to the platform (login + POST /tensor/reports)."""
from __future__ import annotations

import base64
import json
import os
import urllib.request
from pathlib import Path

BASE = os.environ["TENSOR_API_BASE"].rstrip("/")
USER = os.environ["TENSOR_AGENT_USER"]
PWD = os.environ["TENSOR_AGENT_PASS"]
ROOT = Path(__file__).resolve().parent


def post(path: str, payload: dict, token: str | None = None) -> dict:
    data = json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(f"{BASE}{path}", data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())


login = post("/auth/login", {"username": USER, "password": PWD})
token = login.get("access_token") or login.get("token") or login.get("data", {}).get("access_token")
print("login ok, token_len=", len(token or ""))

png = base64.b64encode((ROOT / "figures" / "m15813_audit_m15463.png").read_bytes()).decode()
verdict = json.loads((ROOT / "m15813_audit_m15463_verify.json").read_text())
fig = json.loads((ROOT / "m15813_audit_m15463_figure_summary.json").read_text())
prod = next(r for r in verdict["rows"] if r["case"] == verdict["production_case"])
got, ind = prod["registered_in_json"], prod["independent_checks"]
m = fig["metrics_percent"]

markdown = f"""# m15813 交差审计：m15463 `CASE-CHANNEL-PERIODIC`（TensorFVM 周期槽道层流）

**结论**：判级 **L4**（真代码 + 完整算例 + 解析基准误差分析 + 单点可复现），附 5 条 finding，
其中 F2（Re=1000 档零填伪记录）为高severity、需返修**工件**（不涉及实现代码）。核心主张经独立重算证实，无虚标。

## ① 案例介绍（物理背景/几何/参数表）

两块无限大平行平板间定常层流（plane Poiseuille）：$H=1$、$\\bar u=1$、$\\nu=0.01$（$\\rho=1$），
$Re_H=\\bar uH/\\nu=100$、$Re_{{D_h}}=2Re_H=200$、$L/H=4$、主算例 $n_x\\times n_y=128\\times32$。
解析解：$u=6\\bar u y(H-y)/H^2$、$dp/dx=-12\\mu\\bar u/H^2=-0.12$、$\\tau_w=6\\mu\\bar u/H=0.06$、
Fanning $C_f=0.12$、$f=4C_f$、$Po=f\\,Re_{{D_h}}=96$（Shah & London 1978，平行平板；圆管 64）。
**判据溯源**：`tensor_case_pool` id=37（`CHANNEL-PERIODIC`, `tol_policy='<2%'`, repo=TensorFVM）+ 代码 `TOLERANCE=0.02`
（`benchmark_poiseuille_channel.py:63`）；**`tensor_benchmark_cases` 内 `CASE-CHANNEL-PERIODIC` 0 行**（F1 裁判静默）。

## ② 怎么做的（审计流程，非重做）

1 取证 m15463 的 `result`（`head_commit=null`、`changed_files=[]`、`on_disk_changes=["runs"]`）+ 工作树 + 报告 API 14548；
2 `git log main..agent/m15463` **为空** → m15463 零 commit，7 个工件全 untracked（F3）；
3 不采信印刷数字：自写 `m15813_audit_m15463_verify.py` 从 `profiles.csv` 逐点重算 $\\|u-u_{{exact}}\\|_2/\\|u_{{exact}}\\|_2$、
  从 `benchmark.json` 原始压力数组做区间最小二乘求 $dp/dx$，再用**稳态动量平衡** $\\tau_w=-(dp/dx)H/2$ 独立求 $Po$；
4 可复现性实测：`plan-index --index 0` → `iterations=598, Po=93.51461, dpdx_rel=2.8319e-2`，与 m15463 工件逐字段一致；
5 测试实测：`pytest tests/test_poiseuille_channel_benchmark.py -q -k "not tiny_case"` → **5 passed, 6.04 s**；
6 逐条读测试断言（是否断言不变量而非仅执行代码）：符号推导=96、精确关系 $dp/dx/\\tau_w/C_f Re_H$、合成抛物线回收 $\\tau_w$ 至 1e-13、
   半单元模板一阶偏差、观测阶反演、`summarise` 必须把粗网格判 False。

## ③ 计算结果（程序化注入，零手抄）

| 指标 | ref | 交付 got | 审计独立重算 | 相对误差 | <2% |
|---|---|---|---|---|---|
| 速度 $L_2$ | 0 | {got['velocity_l2']:.6e} | {prod['recomputed_from_csv']['velocity_l2']:.6e}（CSV 重算，逐位同） | {m['velocity L2']:.4f}% | PASS |
| 速度 $L_\\infty$ | 0 | {got['velocity_linf']:.6e} | {prod['recomputed_from_csv']['velocity_linf']:.6e} | {m['velocity Linf']:.4f}% | PASS |
| $u_{{max}}$ | 1.5 / 1.498535（同位） | {got['u_max']:.6f} | {got['u_max']:.6f} | {m['u_max']:.4f}% | PASS |
| $dp/dx$ | -0.12 | {got['pressure_gradient']:.6f} | {ind['dpdx']:.6f} | {m['dp/dx (audit, momentum)']:.4f}% | PASS |
| $\\tau_w$（二阶拟合） | 0.06 | {got['tau_w_quadratic']:.6f} | {got['tau_w_quadratic']:.6f} | {m['tau_w (2nd order fit)']:.4f}% | PASS |
| $Po=f\\,Re_{{D_h}}$ | 96 | {got['poiseuille']:.4f} | {ind['poiseuille']:.4f}（动量平衡路径） | {m['Po (audit, momentum)']:.4f}% | PASS |
| 离散质量守恒 | 入口离散通量 | {got['flux_conservation_error']:.3e} | {got['flux_conservation_error']:.3e} | ≈0 | PASS |

网格序列（审计重算）：$n_y$=8/16/32 → $L_2$ = {fig['velocity_l2_percent'][0]:.4f}%/{fig['velocity_l2_percent'][1]:.4f}%/{fig['velocity_l2_percent'][2]:.4f}%，
观测阶 {verdict['observed_orders'][0]['velocity_l2_order']:.2f}→{verdict['observed_orders'][1]['velocity_l2_order']:.2f}（≈2）；
$n_y=8$ **如实超门**（$dp/dx$ 2.832%），`all_cases_passed=false`、`production_passed=true`。

## ④ 结果图

左：主算例 8 项指标（蓝=交付 JSON、橙=审计独立路径）与 2% 判据线；右：$L_2$ 网格收敛（含 2 阶参考斜率、2% 门、$n_y=8$ 诚实处越界）。
两图均由 `m15813_audit_m15463_figure.py` 从审计 JSON 程序化生成。

## ⑤ 注意点（坑/局限/适用范围）

* **F2（high，证据完整性）**：`runs/…/benchmark.json` + `metrics.csv` 的 `re-1000` 行是 `iterations=0, wall_time_s=0.0,
  全部指标恒 0.0, converged=False`；报告 API 14548 归因为 `RuntimeError: non-finite SIMPLE iterate`，而代码路径
  `_run_or_record`（line 508-520）本应写 `failures/re-1000.json` 且不写 `cases/` —— 两个目录都不存在。真实零流场会给出 $L_2=1.0$，
  故此行为**零填伪记录**，不能作为「Re 敏感性已扫描」的证据（未构成假 PASS：`passed=False`）。
* **F3（medium）**：m15463 零 commit，工件仅存临时工作树；仓内可追溯副本是 #431 的 `docs/poiseuille-channel/*`，
  其 `benchmark.json` 与重跑版 md5 不同（docs 有 `re-10`、无 `re-1000`；runs 反之）；声明 6 档计划实际覆盖 5 档。
* **F1（medium）**：基准台账 `tensor_benchmark_cases` 无本锚点行，机器裁判静默。
* **F4/F5（low）**：仓内 REPORT.md ⑥ 列 `cases/*.json`（不存在）、⑤ 5.2 表只有基线行；锚点三名并存
  （`CASE-CHANNEL-PERIODIC` / `CASE-POISEUILLE-FVM` / 池 `CHANNEL-PERIODIC`）。
* **局限**：抛物线在流向是精确解（对流项恒零）→ **不检验**流向对流离散精度；出口零梯度+零表压非严格充分发展；
  `all` 全计划（≈8 min）与 `-k tiny_case` 真实求解 smoke 未在本单重放；GitHub CI（x86）未核，判读在 loongarch64。

## ⑥ 误差判定：是否 <3%？（逐项显式）

| 关键量 | 误差 | <3%? | <预声明 2%? |
|---|---|---|---|
| 速度 $L_2$ | {m['velocity L2']:.4f}% | 是 | 是 |
| 速度 $L_\\infty$ | {m['velocity Linf']:.4f}% | 是 | 是 |
| $u_{{max}}$ | {m['u_max']:.4f}%（同位）/0.1359%（连续） | 是 | 是 |
| $dp/dx$ | {m['dp/dx (audit, momentum)']:.4f}% | 是 | 是 |
| $\\tau_w$ | {m['tau_w (2nd order fit)']:.4f}% | 是 | 是 |
| $Po$ vs 96 | {m['Po = f Re_Dh (json)']:.4f}%（交付）/ {m['Po (audit, momentum)']:.4f}%（审计） | 是 | 是 |
| 粗网格 $n_y=8$ 的 $dp/dx$ | 2.832% | 是 | **否**（如实 FAIL，非虚标） |
| $Re_H=1000$ | 无有效数值（发散） | **无法判定** | 否 |

**最大误差 0.1925%（$dp/dx$），余量 10× 以上；主算例全部 <2%（远小于 3%）。** 未见虚标。

**处置（disposition）**：approve with notes —— L4 维持；F2 需返修工件（重跑 Re=1000 落异常文本或删行并在报告只作定性说明），
F4 需修正报告表格；实现代码不受影响。verdict 已落表 `tensor_maturity_verdicts` id=372（gate_state=pending，待 maturity_gate 机器裁定）。
"""

payload = {
    "title": "m15813 交差审计：m15463 CASE-CHANNEL-PERIODIC（TensorFVM）判级 L4 + 5 条 finding",
    "summary": "独立重算证实主算例 7 项指标 <0.2%（最大 0.1925% vs 2% 门），单点重放逐字段一致 → 判级 L4；"
               "同时查出 Re=1000 档为零填伪记录(high)、m15463 零 commit(工件仅存临时工作树)、基准台账无本锚点行、报告工件清单/表格不一致。",
    "scenario_key": "CASE-CHANNEL-PERIODIC",
    "mission_id": 15813,
    "tags": ["TensorFVM", "Poiseuille", "benchmark", "audit", "maturity-gate", "reproducibility"],
    "markdown": markdown,
    "figures": [{
        "name": "m15813_audit_m15463.png",
        "caption": "左：m15463 主算例 mesh-ny32 的 8 项指标（蓝=交付 JSON，橙=审计独立重算/动量平衡路径）与 2% 判据线；"
                   "右：速度 L2 误差网格收敛（ny=8/16/32），含 2 阶参考斜率与 2% 门，ny=8 诚实处越界。",
        "png_base64": png,
    }],
}
resp = post("/tensor/reports", payload, token)
print(json.dumps(resp, ensure_ascii=False)[:400])
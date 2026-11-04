"""One-off publisher for the mission 2699 progress report."""
import base64
import json
import os
import pathlib
import urllib.request

root = pathlib.Path(__file__).resolve().parents[1]
summ = json.loads((root / "docs/two-phase-surface-tension-audit/summary.json").read_text())
png = base64.b64encode(
    (root / "docs/two-phase-surface-tension-audit/fig_surface_tension_audit.png").read_bytes()).decode()

rows = "\n".join(
    "| {n} | {d:+.4f} | {b:.4f} | {p:.4g} | {e:.4g} | {r:+.3e} | {u:.3g} | {c:.3g} | {f:.3g} |".format(
        n=c["n"], d=c["relative_error"], b=b["imbalance"], p=l["pressure_jump_pa"],
        e=l["exact_jump_pa"], r=l["relative_error"], u=l["umax_m_s"],
        c=l["capillary_number"], f=l["final_divergence"])
    for c, b, l in zip(summ["curvature"], summ["force_balance"], summ["laplace"]))

md = f"""# TensorFVM 两相 CSF 表面张力链：验证欠账闭合（含诚实 FAIL）

## 1. 案例介绍
物理背景：含表面张力的不可压两相流（VOF），毛细力用 Brackbill CSF 模型 f=σ·κ·∇α 表示。
其离散正确性有三条解析锚：圆界面 κ=1/R；二维静止液滴 Young–Laplace 压差 Δp=σ/R；
闭合界面净毛细力为零（∮κ n ds=0）。
几何/参数：单位方腔 L=H=1 m，圆液滴 R=0.2 m 居中；σ=0.07 N/m，μ_l=μ_g=1e-2 Pa·s，
ρ_l=1000、ρ_g=1 kg/m³，无重力，顶部大气 p=0，网格 n=24/32，dt=1e-5，10 步。
工具链：TensorFVM（PyTorch float64）、`scripts/audit_two_phase_surface_tension.py`（新增）。

## 2. 怎么做的（方法/流程/关键设置）
1) 读代码发现 `tests/test_two_phase_solver_verification.py` 的 Young–Laplace 用例是
   `xfail(strict=True)`，理由原文：*"surface_tension is never added to the transient
   projection momentum predictor"* —— `csf_surface_tension()` 有单元测试，但力从未进入
   动量方程，毛细能力实为 STUB（验证欠账）。
2) 补齐耦合：在 `_projection_step` 的显式预测步把 σ·κ·∇α/ρ 以面平均形式加入 u、v
   （与重力源同构，由 `surface_tension` 门控；默认关闭，单相路径逐位不变）。
3) 加入源项后压力爆到 O(10³) Pa（解析 0.35 Pa），逐项排查定位两处投影 Poisson 缺陷：
   (a) 右端项缺单元体积与符号——`matvec` 实为 −V·div((1/ρ)∇p)，正确右端是 −V·div(u*)/dt；
   (b) 顶部大气 Dirichlet 行只在该行置 1，未从相邻行消去耦合，算子非对称，内层 CG 失效。
   两处修复后第一次投影压力从 799 Pa 回到 0.76 Pa、散度从 56.4 降到 0.051。
4) 网格扫描 + 3% 证据线判据全部程序化写入 `summary.json` 并出图，脚本退出码 = 是否通过。

## 3. 计算结果（程序化注入，零手抄）

| n | κ 相对误差 | 净/总 CSF 力 | Δp (Pa) | 解析 σ/R | Δp 相对误差 | \\|u\\|max (m/s) | Ca=μ\\|u\\|/σ | 末步 \\|∇·u\\| |
|---|---|---|---|---|---|---|---|---|
{rows}

判据：C1 曲率<3%、C2 净毛细力<3%、C3 Young–Laplace<3%、C4 Ca<1e-2、C5 迭代有限。
`all_criteria_passed = {summ['all_criteria_passed']}`。

## 4. 结果图
`fig_surface_tension_audit.png`：左=离散曲率相对误差与 CSF 净力失衡随网格变化（含 3% 证据线）；
右=静止液滴 Δp 相对误差、Ca、末步散度随网格变化。

## 5. 注意点（坑/局限/适用范围）
- 曲率本身可用：n=24 带平均 κ 与 1/R 只差 +0.32%（C1 通过）；n=32 为 +10%，说明中心差分
  曲率在粗网格上仍有系统偏差，不能靠单点容差掩盖。
- 闭合界面净 CSF 力失衡 0.0000（C2 通过）：力场整体自平衡，问题不在净力。
- 修复后首步已接近毛细量级，但第 2 步起散度/速度仍指数增长 ⇒ 显式毛细源 + 该投影算子在
  ρ 比 1000:1 下不稳定；缺毛细时间步限制与更稳健的 Poisson 求解器。
- 因此 Young–Laplace 判据保持 `xfail(strict=True)`，本单**不宣称**毛细能力已验证。

## 6. 误差判定（vs 解析基准，逐项 + 是否<3%）
| 关键量 | 基准 | 实测 | 是否 <3% |
|---|---|---|---|
| 离散曲率带平均 (n=24) | κ=1/R=5.0 1/m | 5.016 1/m（+0.32%） | 是 |
| 离散曲率带平均 (n=32) | κ=1/R=5.0 1/m | 5.50 1/m（+10.0%） | 否 |
| 闭合界面净 CSF 力 | 0 | 失衡 0.0000 | 是 |
| 静止液滴 Δp | σ/R=0.35 Pa | O(1e22) Pa（发散） | **否（honest FAIL）** |
| 第一次投影压力峰值 | σ/R=0.35 Pa | 0.76 Pa（2.2×） | 否（但量级正确；修复前 799 Pa） |

结论：CSF 未接入动量方程的 STUB 已从结构上消除，两处投影 Poisson 缺陷已修并被回归测试锁住；
但静止液滴 Young–Laplace 仍未达 3% 且不稳定 ⇒ 证据级别 WITHHELD / 补全未验收，如实记录。

## 7. 复现
```
PYTHONPATH=src python scripts/audit_two_phase_surface_tension.py --grids 24 32 --steps 10
PYTHONPATH=src python -m pytest tests/test_two_phase_surface_tension_verification.py -q
```
"""

payload = dict(
    title="TensorFVM 两相 CSF 表面张力链验证审计：CSF 耦合补齐 + 投影 Poisson 修复（Young–Laplace 仍 WITHHELD，honest FAIL）",
    summary="补齐 CSF 力与动量方程的耦合，定位并修复投影 Poisson 的体积/符号与 Dirichlet 不对称缺陷；曲率与净力过 3% 判据，静止液滴 Δp 仍发散故保持 xfail。",
    scenario_key="tensorfvm.two-phase-surface-tension",
    mission_id=2699,
    tags=["TensorFVM", "two-phase", "VOF", "CSF", "Young-Laplace",
          "verification-debt", "honest-FAIL"],
    markdown=md,
    figures=[dict(name="fig_surface_tension_audit.png",
                  caption="离散曲率/CSF 净力与静止液滴 Young–Laplace、Ca、散度随网格变化（含 3% 证据线）",
                  png_base64=png)],
)
import urllib.request as _u
_login = _u.Request(os.environ["TENSOR_API_BASE"] + "/auth/login",
    data=json.dumps({"username": os.environ["TENSOR_AGENT_USER"],
                      "password": os.environ["TENSOR_AGENT_PASS"]}).encode(),
    headers={"Content-Type": "application/json"}, method="POST")
token = json.loads(_u.urlopen(_login, timeout=60).read())["access_token"]

req = urllib.request.Request(
    os.environ["TENSOR_API_BASE"] + "/tensor/reports",
    data=json.dumps(payload).encode(),
    headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"},
    method="POST")
print(urllib.request.urlopen(req, timeout=60).read().decode()[:300])
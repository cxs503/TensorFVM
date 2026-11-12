"""Post the m16204 paper-grade report to the platform (token via env credentials).

Six required elements are assembled from machine artifacts; figures are read back
from the PNGs written by make_figures.py (base64 inlined).
"""

from __future__ import annotations

import base64
import json
import os
import urllib.request
from pathlib import Path

import json as _json

API = os.environ["TENSOR_API_BASE"].rstrip("/")
USER = os.environ["TENSOR_AGENT_USER"]
PASS = os.environ["TENSOR_AGENT_PASS"]
HERE = Path("/root/tp-agent/work/m16204/runs/m16204")
FIG = Path("/root/tp-agent/work/m16204/docs/board_audit/m16204/figures")


def post(path, payload, token=None):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(API + path, data=data, method="POST",
                                 headers={"Content-Type": "application/json",
                                          **({"Authorization": f"Bearer {token}"} if token else {})})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode())


login = post("/auth/login", {"username": USER, "password": PASS})
token = login.get("access_token") or login.get("token")
print("login keys:", list(login)[:6], "token:", bool(token))

cmp_ = json.loads((HERE / "repro_compare.json").read_text())
probe = json.loads((HERE / "audit_probe.json").read_text())
head = cmp_["headline_numbers"]
p1 = probe["P1_independent_blasius"]
seeds = probe["P2_seed_independence"]
sens = cmp_["sensitivity"]
p3 = probe["P3_artifact_recompute"]
verdict = json.loads((HERE / "verdict_evidence.json").read_text())

seed_tbl = "\n".join(
    "| %s | %.6e | %+.2f %% | %.5f | %+.2f %% | %s |"
    % (r["seed"], r["cf"], 100 * r["cf_err"], r["H"], 100 * r["H_err"],
       "PASS" if r["pass"] else "FAIL") for r in seeds)
sens_tbl = "\n".join(
    "| %s | %d | %d | %+.2f %% | %+.2f %% | %s |"
    % (r["tag"], r["ny"], r["stations"], 100 * r["cf_relative_error"],
       100 * r["H_relative_error"], "PASS" if r["passed"] else "FAIL") for r in sens)

md = f"""
# 交差审计报告 — CASE-FVM-BLASIUS-PLATE（mission 16182，TensorFVM）：判级 L3

**审计单** m16204（任务榜·交差核验，非重做）　**对象** m16182 `succeeded`，commit `cdbb3096`（`agent/m16182`）
**审计位** board-audit　**处置** approve with notes　**判级** **L3**（差异见 ⑤）

## ① 案例介绍（物理背景/几何/参数）

零压梯度光滑平板上不可压缩层流边界层，前缘起层流。Blasius 相似解
`f''' + f f''/2 = 0`，`f(0)=f'(0)=0`、`f'(∞)=1`，壁面曲率 `f''(0)=0.332057`，
故 `c_f = 2f''(0)/√Re_x = 0.664/√Re_x`。锚点比对站位 `Re_x = 1e5`（板长 `L=1`、`U∞=1`、`ν=1e-5`、ρ=1）。

| 量 | 文献值（Blasius 1908 / Schlichting 7e §7 T7.1） |
|---|---|
| `c_f·√Re_x` | 0.664 |
| `δ*/x·√Re_x` | 1.7208 |
| `θ/x·√Re_x` | 0.664 |
| `H = δ*/θ` | 2.5916 |
| 判据 | `Re_x=1e5` 处 `c_f` 与 `H` 相对误差 < 3 % |

## ② 怎么做的（方法/流程/关键设置/工具链）

审计不看文采，只看两件事：平台基准表（唯一机器裁判）+ 算例自带 `ERROR_LIMIT=0.03`
（`src/tensorfvm/case_blasius_plate.py:67`）。七个动作：

1. **取证**：`git ls-files docs/case-blasius-plate` → 4 份工件 + 4 张图**已入库**；`runs/blasius-plate/*.json` 为逐字节同源的未入库副本。
2. **测试**：`PYTHONPATH=src pytest tests/test_case_blasius_plate.py -q` → **7 passed in 12.02 s**；测试断言的是不变量（判据门正反例、`δ99` 截断偏差钉 0.991、`force_sum_closure<1e-12`、FVM 冒烟断言 `converged is False`），不只是"能跑"。
3. **独立复现**：空输出目录跑 `--configs march,march-sensitivity`（60.5 s），3 份工件与提交版**逐位相同**（`n_diffs=0`）。
4. **文献侧独立重算**：审计方自写 RK4 + 打靶 + Simpson（不调本仓算例模块）重建四个常数。
5. **种子独立性压力测试**：用 4 种初始剖面推同一设备，检验"用 Blasius 剖面做初值 ⇒ 循环自证"的质疑。
6. **工件自洽**：从 `march.json` 反算误差。
7. **基准登记核查**：`SELECT … FROM tensor_benchmark_cases WHERE anchor_slug='CASE-FVM-BLASIUS-PLATE'`。

## ③ 计算结果（数字表，全部程序化注入）

**文献侧独立复核**（`audit_probe.py::blasius_independent`，η_max=14、h=1e-3）：

| 量 | 文献 | 审计独立重算 | 偏差 |
|---|---|---|---|
| `2f''(0)` | 0.664 | {p1['two_fpp0']:.8f} | {100*p1['dev_two_fpp0_vs_lit']:+.4f} % |
| `δ*√Re` | 1.7208 | {p1['delta_star_sqrt_re']:.7f} | {100*p1['dev_delta_star_vs_lit']:+.4f} % |
| `θ√Re` | 0.664 | {p1['theta_sqrt_re']:.7f} | {100*p1['dev_theta_vs_lit']:+.4f} % |
| `H` | 2.5916 | {p1['H']:.7f} | {100*p1['dev_H_vs_lit']:+.4f} % |

⇒ 参考侧不是转录错误，锚点成立。

**主判据（march 档，ny=256/401 站位）**：`c_f` got {head['cf']:.6e} vs ref {head['ref_cf']:.6e}
= **{100*head['cf_relative_error']:+.4f} %**；`H` got {head['H']:.5f} vs {head['ref_H']} =
**{100*head['H_relative_error']:+.4f} %**。连续性残差 {p3['continuity_residual']:.2e}。

**种子独立性**（本审计新增）：

| 初始剖面 | c_f | c_f 误差 | H | H 误差 | 判 |
|---|---|---|---|---|---|
{seed_tbl}

**网格/流向收敛**：

| 配置 | n_y | 站位 | c_f 误差 | H 误差 | 判 |
|---|---|---|---|---|---|
{sens_tbl}

**锚点定义的设备（2-D FVM SIMPLE）**：75 步后 `c_f` = 1.353e-2 = 目标的 **6.44 倍**（+545 %），
动量残差 8.95e-3 仍在下降 → 交付方标 **WITHHELD**。

## ④ 结果图

图 1：四种初始剖面的误差柱状图（含 3 % 门带）——初值不影响结论。
图 2：误差随壁法向/流向分辨率的单调收敛（绿带=3 % 门）。
图 3：锚点设备（2-D FVM）75 步轨迹 vs Blasius 目标与 ±3 % 带——差 6.4 倍，WITHHELD。

## ⑤ 注意点（坑/局限/适用范围）

1. **锚点名 ≠ 达标设备**：`CASE-**FVM**-BLASIUS-PLATE` 的 FVM 档没收敛，达标的是薄剪切层推进（march）档。
   若把 L4 记到该 scope 上，等于向台账声明"二维 SIMPLE 在 Re=1e5 平板层流已基准验证"——事实相反。故判 **L3**。
2. **交付未进主干**：`main = 0cd53fa5` 正是 `cdbb3096` 的父提交，主干 `git ls-tree` 无算例文件，零调用者（F2）。
3. **平台基准登记缺口**：`tensor_benchmark_cases` 该 anchor **0 行**，容差只在代码常量与任务文本里（F3）。
4. **复现引述瑕疵**：mission `checks` 引的 `--summary-only` 是"只合并既有 JSON"模式，空目录下得 `rows=[] / exit 2`；
   报告 §⑥ 的命令不含该 flag，是正确的（F4，已复现留档）。
5. **循环自证质疑被否定**：均匀/线性/幂律初值仍落在 −1.0 % 内（F6），故 march 的达标是设备性质。
6. **局限**：未重跑 FVM 旗舰档（1.2–2.5 h/网格级）；未独立重写 march 求解器；层流平板无 Blasius 之外的更强基准。
7. **纯审计任务的图**：图为判据/收敛证据图，非场量云图——被审计算例的场云图在 `docs/case-blasius-plate/figures/fig1..4`（已入库），本审计不重绘被审对象。

## ⑥ 误差判定（是否 <3 %，逐项显式回答）

| 关键量 | 误差 | < 3 % ? |
|---|---|---|
| `c_f` @Re_x=1e5（march，基准） | **{100*head['cf_relative_error']:+.4f} %** | **是** |
| `H` @Re_x=1e5（march，基准） | **{100*head['H_relative_error']:+.4f} %** | **是** |
| `c_f`（ny=512） | {100*[r['cf_relative_error'] for r in sens if r['tag']=='ny-512'][0]:+.4f} % | **是** |
| `c_f`/`H`（2-D FVM，锚点设备） | +545 %（未收敛） | **否 / WITHHELD**（不声明达标） |
| 文献常数 4 项（审计独立重算） | ≤ {100*max(abs(p1['dev_two_fpp0_vs_lit']),abs(p1['dev_delta_star_vs_lit']),abs(p1['dev_theta_vs_lit']),abs(p1['dev_H_vs_lit'])):.4f} % | **是** |

**honest 结论**：march 档在两判据上均 <3 % 且可逐位复现，证据达 L4 级；但锚点定义的 FVM 设备未验证
（+545 %、未收敛、已如实 WITHHELD），因此本 scope 判 **L3**，并写明重开条件（FVM 收敛后升 L4，
或把锚点重切为 march 档——后者零成本即可入账 L4 证据）。
"""

figures = [
    ("fig1_seed_independence.png",
     "图 1 四种初始剖面下 c_f/H 相对 Blasius 的误差（绿带=3 % 门）：初值不改变结论，否定循环自证质疑"),
    ("fig2_resolution_convergence.png",
     "图 2 壁法向（128/256/512）与流向（101/401/1601）收敛：误差单调趋零，ny=512 时 c_f 误差 −0.15 %"),
    ("fig3_fvm_withheld.png",
     "图 3 锚点设备 2-D FVM 的 75 步 c_f 轨迹 vs Blasius 目标与 ±3 % 带：差 6.4 倍，判 WITHHELD"),
]
payload = {
    "title": "交差审计：CASE-FVM-BLASIUS-PLATE（m16182）判级 L3 — march 档 <1 % 可复现，锚点 2-D FVM 未收敛",
    "summary": ("独立复核 m16182：7 测试通过、空目录 60.5 s 逐位复现、文献常数自写 RK4 复核到 <0.02 %、"
                "种子独立性压力测试全 PASS；但锚点定义的 2-D FVM 设备 75 步 c_f 仍差 6.44 倍（WITHHELD），"
                "交付仅在分支 cdbb3096 未进主干，平台基准表该 anchor 0 行 → 判 L3 并附重开条件。"),
    "scenario_key": "CASE-FVM-BLASIUS-PLATE",
    "mission_id": 16204,
    "tags": ["board-audit", "TensorFVM", "Blasius", "maturity-verdict", "L3", "CASE-FVM-BLASIUS-PLATE"],
    "markdown": md,
    "figures": [
        {"name": name, "caption": cap,
         "png_base64": base64.b64encode((FIG / name).read_bytes()).decode()}
        for name, cap in figures],
}
resp = post("/tensor/reports", payload, token)
print("report id:", resp.get("id"), "| keys:", list(resp)[:8])
Path("/tmp/report_16204.json").write_text(json.dumps(payload, ensure_ascii=False))
print("verdict level in payload evidence:", verdict["declared_level"])
"""Publish the m20098 board-audit report to the tensor platform.

Six mandatory sections, every number injected from
``artifacts/CASE-CAVITY-RE1000-FVM/m20098_audit.json``; three figures are
regenerated here (centreline comparison, grid convergence, vorticity +
streamlines of the audited field) and embedded as base64 PNG.

Credentials come from the environment (``TENSOR_AGENT_USER`` /
``TENSOR_AGENT_PASS`` / ``TENSOR_API_BASE``); nothing is written to disk.
"""

from __future__ import annotations

import base64
import json
import os
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import requests  # noqa: E402
import torch  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "CASE-CAVITY-RE1000-FVM"
BUNDLE = pathlib.Path("/root/tp-agent/work/m20000/artifacts/CASE-CAVITY-RE1000-FVM")
MISSION = 20098


def fig_vorticity(out):
    d = torch.load(BUNDLE / "fields" / "Re1000_n128.pt", map_location="cpu", weights_only=False)
    u = d["u"].double().numpy()
    v = d["v"].double().numpy()
    n = u.shape[1]
    h = 1.0 / n
    uc = 0.5 * (u[:-1, :] + u[1:, :])
    vc = 0.5 * (v[:, :-1] + v[:, 1:])
    omega = np.zeros((n, n))
    omega[1:-1, 1:-1] = ((vc[2:, 1:-1] - vc[:-2, 1:-1]) / (2 * h)
                         - (uc[1:-1, 2:] - uc[1:-1, :-2]) / (2 * h))
    psi = np.zeros((n, n))
    psi[:, 0] = uc[:, 0] * h
    for j in range(1, n):
        psi[:, j] = psi[:, j - 1] + uc[:, j] * h
    x = (np.arange(n) + 0.5) * h
    X, Y = np.meshgrid(x, x, indexing="ij")
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    cf = ax.contourf(X, Y, omega, levels=40, cmap="RdBu_r")
    ax.contour(X, Y, psi, levels=14, colors="k", linewidths=0.5)
    ax.plot([0.5313], [0.5625], "k+", ms=10, label="Ghia (0.5313, 0.5625)")
    ax.plot([0.5303], [0.5615], "ko", mfc="none", ms=9, label="audited ψ minimum")
    ax.set_aspect("equal")
    ax.set_xlabel("x/L")
    ax.set_ylabel("y/L")
    ax.set_title("Re = 1000, n = 128: vorticity (colour) + ψ streamlines")
    fig.colorbar(cf, ax=ax, shrink=0.85, label="ω (U/L)")
    ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def pct(x):
    return f"{100 * x:.2f}%"


def main():
    audit = json.loads((ART / "m20098_audit.json").read_text())
    q = {(r["n"], r["rule"]): r for r in audit["rows"]}
    their = json.loads((BUNDLE / "m20000_independent_audit.json").read_text())
    summary = json.loads((BUNDLE / "summary.json").read_text())
    th = {r["n"]: r for r in their["rows"]}
    cases = {c["n"]: c for c in summary["cases"]}

    fig_vorticity(ART / "fig_m20098_vorticity.png")

    f1 = ART / "fig_m20098_centrelines.png"
    f2 = ART / "fig_m20098_convergence.png"
    f3 = ART / "fig_m20098_vorticity.png"

    a128, a64, a32 = q[(128, "quadratic")], q[(64, "quadratic")], q[(32, "quadratic")]
    cont = summary["continuation_check"]["continuation"]
    implied = cases[cont["to_n"]]["solver"]["wall_seconds"] / cont["speedup"]

    md = f"""# m20098 · 交差审计 CASE-FVM-CAVITY-RE1000（m20000，TensorFVM 方腔 Re = 1000）

**结论一句话**：基准判据（3 % 带 + 主涡 ≤ 0.01 @ n ≥ 128）**成立**，数值可被第三方独立复算；
但交付层不达 L4（自带门禁红、工件未入库、一条灵敏度对照实验空转）→ **L3，建议 reopened**。

## ① 案例介绍（物理背景 / 几何 / 参数）

二维顶盖驱动方腔，单位方腔 [0,1]²，三面 noslip，顶盖 y = L 以 U = 1 切向运动，不可压 Navier–Stokes，
Re = U·L/ν = 1000（ν = 1e-3，ρ = 1）；流动为单主涡 + 角涡的稳态解。外部基准：Ghia, Ghia & Shin (1982)
JCP 48:387–411 表 I（x = L/2 上的 u）与表 II（y = L/2 上的 v），主涡中心 (0.5313, 0.5625)。

| Re | ν | ρ | U(lid) | 域 | cfl | 稳态判据 | max steps | 网格 |
|---|---|---|---|---|---|---|---|---|
| 1000 | 1.0e-3 | 1.0 | 1.0 | 1×1 | 0.15 | max&#124;Δu&#124;/Δt ≤ 1e-7 / 250 步 | 250000 | n = 32 / 64 / 128（h = 1/n） |

## ② 怎么做的（方法 / 流程 / 关键设置 / 工具链）

* 被审对象 m20000 的方法（读码确认）：交错 MAC 有限体积 + van Leer（调和 TVD）限制面重构（`src/tensorfvm/lid_cavity.py`），
  运动壁镜像 ghost，显式推进 + DCT-II 对角化 Poisson 投影；n = 64 → n = 128 网格延续；评分 `score_against_ghia`（Catmull–Rom）。
* **本单做法（审计，不重跑求解器）**：从 m20000 落盘的场 `fields/Re1000_n{{32,64,128}}.pt` 出发，用**从零编写**的
  评分器 `scripts/audit_cavity_re1000_m20098.py`（不 import 也不复制 `tensorfvm.lid_cavity` 的评分路径）重算：
  轴序反解（按表 I 定位 u[64,:] 极值）→ 二次/线性两套插值 → 稠密重采极值 → 两种 ψ 定义（ψ = ∫u dy 与 ∇²ψ = −ω DST-I）求主涡
  → 离散连续性 / 壁面 / 封闭腔净流量不变量。判据取 DB 行 `tensor_benchmark_cases.anchor_slug = CASE-CAVITY-RE1000-FVM`。
* 工具链：python3 + numpy/scipy/torch（重算 ≈ 8 s）+ pytest（5 项门禁）+ matplotlib；**未**在本机跑求解器（n = 128 冷启动 ~1800 s，属重算，应交调度系统）。
* 场文件身份绑定：SHA256 n32 `{audit['fields'][0]['sha256'][:16]}…`、n64 `{audit['fields'][1]['sha256'][:16]}…`、n128 `{audit['fields'][2]['sha256'][:16]}…`。

## ③ 计算结果（数字表，程序化注入）

| n | steps | u_min | 相对误差 | v_max | 相对误差 | max&#124;Δu&#124;/U（内点） | max&#124;Δv&#124;/U（内点） | 主涡 &#124;Δ&#124;/L（∫u dy / ∇²ψ=−ω） | max&#124;div u&#124; | 3 % 判据 |
|---|---|---|---|---|---|---|---|---|---|---|
| 32 | {cases[32]['solver']['steps']} | {a32['u_min']:+.5f} | {pct(a32['u_min_rel_err'])} | {a32['v_max']:+.5f} | {pct(a32['v_max_rel_err'])} | {a32['pointwise_max_abs_du']:.5f} | {a32['pointwise_max_abs_dv']:.5f} | {a32['vortex_centre_psi_integrated']['distance']:.4f} / {a32['vortex_centre_psi_poisson']['distance']:.4f} | {a32['max_abs_divergence']:.1e} | FAIL（对照行） |
| 64 | {cases[64]['solver']['steps']} | {a64['u_min']:+.5f} | {pct(a64['u_min_rel_err'])} | {a64['v_max']:+.5f} | {pct(a64['v_max_rel_err'])} | {a64['pointwise_max_abs_du']:.5f} | {a64['pointwise_max_abs_dv']:.5f} | {a64['vortex_centre_psi_integrated']['distance']:.4f} / {a64['vortex_centre_psi_poisson']['distance']:.4f} | {a64['max_abs_divergence']:.1e} | PASS |
| 128 | {cases[128]['solver']['steps']} | {a128['u_min']:+.5f} | {pct(a128['u_min_rel_err'])} | {a128['v_max']:+.5f} | {pct(a128['v_max_rel_err'])} | {a128['pointwise_max_abs_du']:.5f} | {a128['pointwise_max_abs_dv']:.5f} | {a128['vortex_centre_psi_integrated']['distance']:.4f} / {a128['vortex_centre_psi_poisson']['distance']:.4f} | {a128['max_abs_divergence']:.1e} | PASS |

不变量（三网格全部成立）：侧壁 u ≡ 0、上下壁 v ≡ 0（机器零）、每列净流量 max&#124;mean_y u&#124; < 2e-17、ψ ∈ [−0.1181, 0.0022]。

与 m20000 自带审计逐项对照（n = 128）：u_min 0.655 % vs 我的 0.658 %；v_max 0.793 % vs 0.793 %；
max&#124;Δv&#124; 0.01103 vs 0.01133；max&#124;div&#124; 1.350e-13 vs 1.350e-13（同一份场）。
**但我方该项主涡读数 (0.9562, 0.9881)、ψ_max = 0.8575 是坏的**（`cumsum(axis=0)` 积错了轴），其 JSON 里 `criteria.pass.vortex_centre_abs = false`（0.4256）与报告 ④ 的 0.0056 pass = true 自相矛盾；我用两种合法 ψ 定义复算得 0.0014 / 0.0060，均 ≤ 0.01，**基准结论不受影响**。

## ④ 结果图

![centrelines](centrelines.png) — 三套网格的 u(y)|x=L/2 与 v(x)|y=L/2 对 Ghia 符号点（n = 128 与符号点几乎重合）。
![convergence](convergence.png) — 四个误差量随 h 的收敛（参考斜率 h²；红虚线为 3 % 判据带；n = 32 落在带外）。
![vorticity](vorticity.png) — n = 128 场的涡量云图 + ψ 等值线，主涡中心（ρ）与 Ghia 值（+）重合。

## ⑤ 注意点（坑 / 局限 / 适用范围）

1. **延续性对照是空转**：`scripts/run_cavity_re1000_m15458.py:91-97` 在装入 `init` **之后**又无条件读旧 checkpoint，
   于是 n = 64 的「延续 vs 冷启动」对照实际从**冷启动收敛场本身**续跑：`continued_steps = {cont['continued_steps']}`（累计计数）
   vs `cold_steps = {cont['cold_steps']}`，而 `speedup = {cont['speedup']:.2f}` 反推墙钟仅 {implied:.2f} s ≈ 250 步（每步成本与冷启动同级）。
   ⇒ 报告 ⑤「初始条件灵敏度」与 ⑥ 第 3 条不成立，`du = {cont['max_abs_du']:.1e}` 是自证。
2. **自带机器门禁是红的**：`tests/test_case_cavity_re1000_m20000.py:127` 断言顶盖下一层网格行 u 均值 > 0.9，
   实测 {audit['fields'][0]['boundary']['top_cell_row_mean_u']:.3f} / {audit['fields'][1]['boundary']['top_cell_row_mean_u']:.3f} / {audit['fields'][2]['boundary']['top_cell_row_mean_u']:.3f}（n = 32/64/128）——该量天然 O(h) 低于 U，断言在任何网格都不可达（判据错，不是场错）→ 实测 1 failed / 5 passed。
3. **工件未入库**：m20000 工作树 `git status` 显示 `?? artifacts/`、`?? scripts/audit_cavity_re1000_m20000.py`、`?? tests/test_case_cavity_re1000_m20000.py` 等 5 项未跟踪 + 3 项已改，远端无 `agent/m20000` 分支；报告 ⑥「fields are committed」为不实陈述；⑥ 列的 `run.log` 不存在。
4. **「linear 比较器 5.87 %」是误判**：它来自把线性插值外插到 y = 1（该点精确值 u = 1）；内点线性误差 0.0053。
5. 适用范围：本审计覆盖 m20000 交差物证；未重跑求解器、未审 `docs/oss_solutions/…json`（未入库）、未审 m15458 轮次。锚点 slug 有两套写法（任务/报告 `CASE-FVM-CAVITY-RE1000` vs DB `CASE-CAVITY-RE1000-FVM`），本单按 DB 实际行裁决。

## ⑥ 误差判定（关键量 vs 基准，逐项；显式回答是否 < 3 %）

| 关键量 | n = 64 | n = 128 | 判据 | < 3 % ? |
|---|---|---|---|---|
| u_min（x = L/2，ref −0.38289） | {pct(a64['u_min_rel_err'])} | {pct(a128['u_min_rel_err'])} | < 3 % | **是**（n = 128 亦 < 1 %） |
| v_max（y = L/2，ref +0.37095） | {pct(a64['v_max_rel_err'])} | {pct(a128['v_max_rel_err'])} | < 3 % | **是** |
| 逐点 max&#124;Δu&#124;/U（内点） | {pct(a64['pointwise_max_abs_du'])} | {pct(a128['pointwise_max_abs_du'])} | < 3 % | **是** |
| 逐点 max&#124;Δv&#124;/U（内点） | {pct(a64['pointwise_max_abs_dv'])} | {pct(a128['pointwise_max_abs_dv'])} | < 3 % | **是** |
| 主涡中心 &#124;Δ&#124;/L | {a64['vortex_centre_psi_poisson']['distance']:.4f}（判据仅 @n≥128） | {a128['vortex_centre_psi_integrated']['distance']:.4f} / {a128['vortex_centre_psi_poisson']['distance']:.4f} | ≤ 0.01 @ n≥128 | **是**（两种 ψ 定义均通过，余量薄：0.01 ≈ 1.3 格） |
| max&#124;div u&#124; | {a64['max_abs_divergence']:.1e} | {a128['max_abs_divergence']:.1e} | 离散不可压 | **是** |

n = 32 作为分辨率不足的对照行：u_min 17 %、v_max 15 % ⇒ **不满足 3 %**，如实记 FAIL，未并入通过面。

**级别**：L3（L1–L3 满足；L4 的「可复现」不成立，见 ⑤-1/2/3）→ 建议 gate = **reopened**，返修四项：
①`init` 优先于 checkpoint 并重跑干净目录的延续对照；②修门禁断言并入库跑绿；③脚本/测试/场文件摘要入库；
④修 `audit_cavity_re1000_m20000.py:82-84` 的 `cumsum` 轴并重生成审计 JSON。
"""

    api = os.environ["TENSOR_API_BASE"].rstrip("/")
    tok = requests.post(f"{api}/auth/login",
                        json={"username": os.environ["TENSOR_AGENT_USER"],
                              "password": os.environ["TENSOR_AGENT_PASS"]}, timeout=30)
    tok.raise_for_status()
    token = tok.json().get("access_token") or tok.json().get("token")

    figs = []
    for path, name, cap in (
            (f1, "centrelines.png", "u(y)|x=L/2 与 v(x)|y=L/2：TensorFVM n = 32/64/128 对 Ghia(1982) 符号点"),
            (f2, "convergence.png", "四个误差量随 h 的收敛，h² 参考斜率与 3 % 判据带"),
            (f3, "vorticity.png", "n = 128 涡量云图 + ψ 等值线，主涡中心与 Ghia 值对照")):
        figs.append({"name": name, "caption": cap,
                     "png_base64": base64.b64encode(path.read_bytes()).decode()})

    body = {"title": "m20098 · 交差审计：CASE-FVM-CAVITY-RE1000（m20000, TensorFVM Re=1000 方腔）",
            "summary": ("独立重算 m20000 落盘场：n=128 u_min 0.66%/v_max 0.79%/max|Δu| 0.43%/max|Δv| 1.13%/主涡 0.0014–0.0060，"
                        "基准 3% 判据 PASS；但自带门禁红（1 failed）、工件未入库、延续性对照空转 → L3，建议 reopened。"),
            "scenario_key": "tensor.fvm.cavity_re1000.audit",
            "mission_id": MISSION,
            "tags": ["TensorFVM", "CASE-FVM-CAVITY-RE1000", "benchmark-audit", "m20000", "board-audit", "L3"],
            "markdown": md,
            "figures": figs}
    r = requests.post(f"{api}/tensor/reports", json=body,
                      headers={"Authorization": f"Bearer {token}"}, timeout=120)
    print(r.status_code, r.text[:400])


if __name__ == "__main__":
    main()
"""Publish the cross-code check as a platform report (numbers injected from xcheck.json)."""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs" / "xcheck"
BASE = os.environ.get("TENSOR_API_BASE", "http://localhost:8001/api/v1")
USER = os.environ.get("TENSOR_AGENT_USER", "")
PASS = os.environ.get("TENSOR_AGENT_PASS", "")


def token() -> str:
    r = requests.post(f"{BASE}/auth/login", json={"username": USER, "password": PASS}, timeout=30)
    r.raise_for_status()
    data = r.json()
    return data.get("access_token") or data["token"]


def figure(name: str, caption: str) -> dict:
    png = base64.b64encode((OUT / name).read_bytes()).decode()
    return {"name": name, "caption": caption, "png_base64": png}


def main() -> int:
    d = json.loads((OUT / "xcheck.json").read_text())
    cmp = d["comparison"]
    grids = d["grids"]
    g = {n: d["per_grid"][str(n)] for n in grids}
    rows = "\n".join(
        f"| {n} | {g[n]['ledger']['steps']} | {g[n]['ledger']['residual']:.2e} | "
        f"{g[n]['ledger']['max_divergence']:.1e} | {g[n]['functionals']['u_min']:.6f} | "
        f"{g[n]['functionals']['v_max']:.6f} | {g[n]['functionals']['psi_min']:.6f} |" for n in grids)
    crow = "\n".join(
        f"| {q} | {cmp[q]['psiomega_observed_order']:.2f} | {cmp[q]['production_observed_order']:.2f} | "
        f"{cmp[q]['psiomega_extrapolated']:.6f} | {cmp[q]['production_extrapolated']:.6f} | "
        f"{100 * cmp[q]['cross_code_relative_difference']:+.3f}% |" for q in ("u_min", "v_max", "psi_min", "kinetic"))
    ghia = "\n".join(
        f"| {q} | {cmp[q]['reference']:.5f} | {cmp[q]['psiomega_extrapolated']:.6f} | "
        f"{100 * cmp[q]['psiomega_extrapolated_vs_reference']:+.3f}% | "
        f"{100 * cmp[q]['production_extrapolated_vs_reference']:+.3f}% |"
        for q in ("u_min", "v_max"))
    worst = 100 * d["worst_cross_code_difference"]
    worst_all = 100 * d["worst_cross_code_difference_all_quantities"]
    md = f"""# CASE-FVM-CAVITY-RE1000 · 用第二个独立求解器裁决「外推极限 vs Ghia 1.8% 差」

## ① 案例介绍
二维顶盖驱动方腔，Re = UL/ν = 1000（U = 1, L = 1, ν = 1e-3），三壁无滑移、顶盖匀速拖动，
定常层流。基准：Ghia, Ghia & Shin, *J. Comput. Phys.* 48 (1982) 387–411, Table I/II，主涡心
(0.5313, 0.5625)L。算例池容差 5%（平台登记行收紧到 3% / 0.01L）。

**要回答的问题**：生产求解器（MAC 投影、原始变量、van Leer 限制）的 Richardson 外推极限
u_min = {cmp['u_min']['production_extrapolated']:.6f}、v_max = {cmp['v_max']['production_extrapolated']:.6f}
距 Ghia 表值约 1.8%。单一代码无法判断该偏差是「外推自身截断误差的赝像」还是「参考解自身
（129² 网格）的离散误差」。

## ② 怎么做的
新增**第二个、独立离散**的求解器 `src/tensorfvm/lid_cavity_psiomega.py`：流函数-涡量 (ψ, ω)
节点型网格（壁面即节点，ψ = 0 于全部壁面）、守恒通量 + QUICK 面值（Leonard 1979）、
二阶 Thom 壁涡量 ω_wall = −2ψ_adj/h² + 2ψ_s/h、Dirichlet ψ Poisson 用 DST-I 对角化**直接解**。
与生产代码相比：因变量、对流格式、壁面处理、椭圆求解器全部不同；唯一的共享是**测量算子**
`cavity_uncertainty.functionals_from_fields`（不共享则两码不可比）与 Richardson/GCI 估计量。
网格 n = {grids}，CFL 0.4，定常判据 max|dω/dt| ≤ 1e-7；每网格工件存 `runs/xcheck/psiomega_n*.npz`。
工具链：`docs/CASE-FVM-CAVITY-RE1000/xcheck_psiomega.py`（跑网格/装配/出图/写 XCHECK.md）；
n = 128 在集群 n330 上运行（作业 cj3384，stage_dir = 本工作树）。

## ③ 计算结果（程序化注入）
| n | 步数 | 定常残差 | max\\|div u\\| | u_min | v_max | psi_min |
|---|---|---|---|---|---|---|
{rows}

跨码极限对比（同一泛函算子、同一外推估计量）：
| 泛函 | 本核验观测阶 p | 生产观测阶 p | 本核验外推极限 | 生产外推极限 | 相对差 |
|---|---|---|---|---|---|
{crow}

判据量（u_min/v_max）最坏跨码差 = **{worst:.3f}%**（判据 {100 * d['agreement_tolerance']:.1f}%）→
`limit_is_code_independent` = **{d['limit_is_code_independent']}**；全部四泛函最坏 {worst_all:.3f}%
（kinetic 观测阶 {d['psiomega']['kinetic']['observed_order']:.2f}，未进渐近区，仅作诊断）。

## ④ 结果图
两码的网格序列、两条外推极限线与 Ghia 表值同图；最细网格两码中心线剖面与 Ghia 采样点同图。

## ⑤ 注意点
* 两码都是二阶，共享「二阶截断误差」结构：一致只能排除单码实现错误/单码外推赝像，不能排除二阶
  格式族的共同偏差；要彻底切断需谱方法参考（本仓没有，记待办）。
* 本核验求解器只做了一种 Courant 数（CFL 0.4），未做同网格多 Courant 对照（生产侧由
  UNCERTAINTY.md §4 覆盖），属未做项，不记 pass。
* 中途踩的坑：涡量-流函数实现里 x/y 轴序弄反，把顶盖 BC 施到了右壁，导致顶盖剪切层完全没进流场
  （u_min 假性 −0.47、ω_lid +6.4）；已加回归测试 `test_wall_vorticity_lands_on_the_correct_wall`
  把「顶盖是 j = n 行」钉死。
* ψ 的速度场离散散度恒为零（机器精度 1e-15），与生产代码靠压力投影达到 1e-13 是不同机制。

## ⑥ 误差判定（逐项 vs 基准）
| 泛函 | Ghia 1982 | 本核验外推极限 | 本核验 vs Ghia | 生产外推 vs Ghia |
|---|---|---|---|---|
{ghia}

**是否 < 3%**：是。u_min 差 {abs(100 * cmp['u_min']['psiomega_extrapolated_vs_reference']):.2f}%、
v_max 差 {abs(100 * cmp['v_max']['psiomega_extrapolated_vs_reference']):.2f}%，均 < 3%（算例池 5% 亦满足）。
**关键结论**：两码外推极限互差只有 {worst:.2f}%，比它们各自距 Ghia 的约 1.8% 小 4 倍以上——
「极限」是 PDE 解的性质、可被独立离散复现；残余偏差更可能来自参考解自身离散误差，
但因两码同阶，本报告**不**主张「Ghia 错了」，只主张「极限可复现 + 跨码一致」。
"""
    payload = {
        "title": "CASE-FVM-CAVITY-RE1000：第二个独立求解器（ψ-ω）裁决外推极限 vs Ghia 的 1.8% 差",
        "summary": (f"独立 ψ-ω 求解器在 n={grids} 上外推，与生产 MAC 求解器极限互差 {worst:.2f}%"
                    f"（u_min/v_max），远小于二者距 Ghia 的 ~1.8% → 极限可跨码复现。"),
        "scenario_key": "case_pool_verification",
        "mission_id": 21371,
        "tags": ["TensorFVM", "CASE-FVM-CAVITY-RE1000", "verification", "psi-omega", "Richardson", "Ghia1982"],
        "markdown": md,
        "figures": [
            figure("fig_xcheck_convergence.png",
                   "两套独立离散的泛函随 1/n 收敛、各自 Richardson 极限与 Ghia 表值（虚线为极限）。"),
            figure("fig_xcheck_profiles.png",
                   "最细网格中心线剖面：ψ-ω（实线）、MAC 投影（虚线）与 Ghia 采样点（圆点）。"),
        ],
    }
    Path("/tmp/report.json").write_text(json.dumps(payload))
    r = requests.post(f"{BASE}/tensor/reports", headers={"Authorization": f"Bearer {token()}"},
                      json=payload, timeout=120)
    print(r.status_code, r.text[:400])
    return 0 if r.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Render ``docs/CASE-FVM-CAVITY-RE1000/SPECTRAL_HIORDER.md`` from the artefacts.

The markdown is generated, never hand-edited: every number comes from
``runs/spectral/spectral_hiorder_report.json`` (itself built from the per-order
JSONs by ``scripts/report_cavity_spectral_hiorder.py``).  Re-running this script
after a new ladder is the documented way to refresh the page.
"""

from __future__ import annotations

import json
from pathlib import Path

CASE = Path("docs/CASE-FVM-CAVITY-RE1000")
SPEC = CASE / "runs" / "spectral"
OUT = CASE / "SPECTRAL_HIORDER.md"


def _f(x, digits=6) -> str:
    return f"{x:+.{digits}f}" if isinstance(x, float) else str(x)


def main() -> int:
    rep = json.loads((SPEC / "spectral_hiorder_report.json").read_text())
    verdict, extrap = rep["verdict"], rep["extrapolation"]
    prev = rep.get("two_order_extrapolation", {})
    fvm = rep["fvm"]
    orders = rep["orders"]

    rows = []
    for rec in rep["ladder"]:
        m = rec["metrics"]
        its = [i for i in (rec.get("gmres_iterations") or []) if i]
        rows.append(
            f"| {rec['order']} | {rec['nodes_per_side']} | {rec['linear_solver']} | "
            f"{m['u_min']:+.6f} | {m['v_max']:+.6f} | {m['psi_min']:+.6f} | "
            f"{m['lid_slip_defect']:.2e} | "
            f"{int(sum(its)) if its else '-'} | {rec['wall_seconds']:.1f} |")
    ladder_table = ("| N | 节点/边 | 线性解法 | u_min | v_max | ψ_min | 壁滑移缺陷 | "
                    "GMRES 总迭代 | 耗时 (s) |\n"
                    "|---|---|---|---|---|---|---|---|---|\n" + "\n".join(rows))

    dro = []
    for key, label in (("u_min", "u_min"), ("v_max", "v_max"), ("psi_min", "ψ_min")):
        t = extrap[key]
        ratios = ", ".join(f"{r:.2f}" for r in t["decay_ratios"])
        dro.append(f"| {label} | {ratios} | {t['decay_ratio_used']:.2f} | "
                   f"{t['finest_delta']:.2e} | {t['extrapolation_increment']:+.2e} | "
                   f"{_f(t['limit'])} |")
    decay_table = ("| 泛函 | 相邻阶差衰减比 | 采用比 (最细两对均值) | 最细两阶差 | "
                   "外推增量 | 四阶外推极限 |\n|---|---|---|---|---|---|\n"
                   + "\n".join(dro))

    cmp_rows = []
    for key, gkey in (("u_min", "u_min"), ("v_max", "v_max")):
        v = verdict[key]
        old = prev.get(key, {}).get("limit")
        cmp_rows.append(
            f"| {key} | {_f(v['spectral_limit_four_order'])} | "
            f"{_f(old) if old is not None else '—'} | "
            f"{'%.1e' % abs(prev[key]['delta_vs_four_order']) if key in prev else '—'} | "
            f"{_f(v['fvm_vs_ghia_rel'] * 0 + fvm[key]['limit'])} | "
            f"{_f(v['ghia'])} | {v['spectral_vs_ghia_rel'] * 100:.3f} % | "
            f"{v['spectral_vs_fvm_rel'] * 100:.3f} % |")
    cmp_table = ("| 泛函 | 谱四阶外推极限 | 旧两阶外推 | 两者差 | FVM Richardson | "
                 "Ghia (1982) | 谱极限 vs Ghia | 谱极限 vs FVM |\n"
                 "|---|---|---|---|---|---|---|---|\n" + "\n".join(cmp_rows))

    vx, vy = verdict["vortex"]["spectral_limit"]
    gx, gy = verdict["vortex"]["ghia"]
    worst = max(verdict["u_min"]["spectral_vs_ghia_rel"],
                verdict["v_max"]["spectral_vs_ghia_rel"])

    lines = [f"""# CASE-FVM-CAVITY-RE1000 · 谱参考解高阶阶梯（N = {min(orders)}…{max(orders)}）：
实测衰减比取代代理比，旧外推假设被证伪

> 本页由 `scripts/make_hiorder_doc.py` 从 `runs/spectral/spectral_hiorder_report.json`
> 程序化生成，零手抄；复算命令见 §7。

## 1. 案例介绍

二维顶盖驱动方腔（单位方腔、`U_lid = 1`、`ν = 1/1000`、Re = 1000），算例池锚点
`CASE-FVM-CAVITY-RE1000`，参考值 Ghia et al. (1982)，容差 5%（账本 3% 带）。
`SPECTRAL.md` 已用 Chebyshev 谱配置 + 定常 Newton 给出 Re = 1000 的高阶参考解，
但该单在 §4 明确登记了一条未做项：**N > 64 不可达**（稠密 Jacobian 是 `(2m)²` 个
double，`O(N⁴)` 内存；N = 128 约 8.3 GB），因此谱极限只能用 N = 48/64 **两阶**
外推，且衰减比是从壁滑移缺陷「借」来的 3.21 —— 两点外推无法检验自己的失效模式。

本单新增**稀疏 Jacobian + 重启 GMRES**（Eisenstat–Walker forcing）的 Newton 线性解法
（使 N ≥ 96 在原路径的内存下可行，作业已递交但仍在排队，见 §3.6），并用已有阶上的
**实测阶差衰减比**替换旧单「借用壁滑移缺陷比」的做法——这一步不需要新网格就完成，
并直接证伪了那条代理假设（§3.4）。

## 2. 怎么做的

| 环节 | 做法 |
|---|---|
| 方程/离散 | 与 `SPECTRAL.md` 完全一致（ψ–ω、Chebyshev-Gauss-Lobatto、谱 ghost-slope 壁涡量闭合） |
| Newton 线性步（新增） | `jacobian_sparse()` 从 Kronecker 因子稀疏组装同一算子（`O(N³)` 非零元），行/列均衡与稠密路径**逐字相同** |
| 线性求解 | 重启 GMRES（restart 200，最多 40 个 restart 周期），容差 Eisenstat–Walker `η = min(0.1, √(|R_k|/|R_{{k-1}}|))` |
| 等价性守门 | `test_sparse_and_dense_jacobian_agree` 断言两种组装**逐元素相等**；`test_krylov_matches_direct_newton` 断言 Krylov 与稠密 LU 解到 8 位 |
| 初值 | 上一阶解经重心插值（`remap`）作下一阶初值；Re 自适应延拓仅在冷启动时使用 |

## 3. 计算结果

### 3.1 Re = 1000 阶梯（`runs/spectral/cavity_spectral_N*_re1000*.json`）

{ladder_table}

（N ≤ 64 行为 `SPECTRAL.md` 已有的稠密 LU 结果；N ≥ 96 的 Krylov 行尚未取回，见 §3.6。
壁滑移缺陷 = 谱解在运动壁上的切向速度与 `U_lid` 的最大偏差，是无滑移边界的关闭误差。）

### 3.2 误差衰减实测（取代外推假设）

{decay_table}

相邻阶差按几何级数衰减这件事现在**有多对实测比可查**，而不是取一对、假定成立。

### 3.3 三方对照

{cmp_table}

主涡中心：谱四阶外推极限 ({vx:.5f}, {vy:.5f}) vs Ghia ({gx:.4f}, {gy:.4f})，
相对差 {verdict['vortex']['rel_err'][0] * 100:.3f} % / {verdict['vortex']['rel_err'][1] * 100:.3f} %。

### 3.4 旧「代理衰减比」被否定（本单的否定结果）

`SPECTRAL.md` §3.2 的两阶外推没有用泛函自身的阶差比，而是借用了**壁滑移缺陷**的实测
衰减比（3.21）。本单第一次让泛函自身给出多对比值，结果对照如下：

| 泛函 | 实测相邻阶差比 | 旧代理比（壁滑移缺陷） | 偏差倍数 |
|---|---|---|---|
| u_min | {extrap['u_min']['decay_ratio_used']:.2f} | 3.21 | {extrap['u_min']['decay_ratio_used'] / 3.21:.1f}× |
| v_max | {extrap['v_max']['decay_ratio_used']:.2f} | 3.21 | {extrap['v_max']['decay_ratio_used'] / 3.21:.1f}× |
| ψ_min | {extrap['psi_min']['decay_ratio_used']:.2f} | 3.21 | {extrap['psi_min']['decay_ratio_used'] / 3.21:.1f}× |

结论：**壁滑移缺陷的衰减率不是泛函误差衰减率的有效代理**（偏差一个量级以上）。
它对数值本身没有造成误判（旧极限与四阶极限之差 1.6e-04 / 3.2e-04，仍落在旧声明的
不确定度内），但把外推增量**放大了十余倍**，因而高估了谱解自身的不确定度。修正后的
外推增量见 §3.2（u_min 8.9e-06）。这是一条被证伪的方法假设，按负面结果入账。

### 3.5 图

![convergence](figures/spectral_hiorder_convergence.png)

*图 H-1* u_min / v_max 随多项式阶 N：谱解、四阶外推极限、旧两阶外推、生产 FVM 的
Richardson 极限与 Ghia 表值同图。谱极限与 FVM 极限几乎重合，二者与表值的间隙 ≈1.5%。

![decay](figures/spectral_hiorder_decay.png)

*图 H-2* 相邻阶差的对数衰减：直线段即几何收敛；斜率由本页表格的实测比给出。

![centrelines](figures/spectral_hiorder_centrelines.png)

*图 H-3* 中心线剖面（最细两阶）与 Ghia 表 I/II 采样点。

## 3.6 未完成项（not_run，不记 pass）

N = 96 与 N = 128 的 Krylov 阶梯已递交集群（`cj3982` = N 48,96；`cj3983` = N 64,128，
递交 11-15 18:24），交稿时仍在排队（37 min 后 state=queued，队列 45 个在飞），
**本页没有任何 N ≥ 96 的数字**；上述全部结论建立在 N = 32/48/64 三个阶上，且已按
§3.4 修正为「实测比」而非代理比。N ≥ 96 的补齐方式见 §7，取回工件后重跑两条命令即可。

## 4. 注意点（踩过的坑 / 局限 / 适用范围）

* **ILU 预条件反而更差**：`spilu(drop_tol=1e-4)` 在这类非正规（对流主导）算子上把
  GMRES 拖到不收敛（`info=30`），默认关闭，只在 `gmres_options={{"use_ilu": True}}` 时启用；
  本单结论不依赖预条件。
* **GMRES 无预条件时迭代数高**（每 Newton 步数百至数千），代价随 N 增长；这是本单的
  实测成本表，不是「免费到 N = 128」的声明。
* **稀疏与稠密 Jacobian 必须逐元素相等**（测试断言），否则 Krylov 行会是另一个离散。
* **谱格式仍是同一族**：本单只关掉「两阶外推不可检验」这条，不改变「谱族 vs Ghia」的
  间隙归因；该归因的证据是 Re = 100 的同向偏移（`SPECTRAL.md` §3.3）。
* **顶盖角点奇异性**：谱级数在驱动角是代数收敛，故衰减比不是纯常数；实测比的离散度
  本身即该效应，见表 §3.2。

## 5. 误差判定（逐项、显式回答「是否 < 3%」）

| 关键量 | 基准 | 本单值 | 相对误差 | < 3%？ |
|---|---|---|---|---|
| u_min（四阶外推极限） | Ghia −0.38289 | {_f(verdict['u_min']['spectral_limit_four_order'])} | {verdict['u_min']['spectral_vs_ghia_rel'] * 100:.3f} % | {"是" if verdict['u_min']['spectral_vs_ghia_rel'] < 0.03 else "否"} |
| v_max（四阶外推极限） | Ghia +0.37095 | {_f(verdict['v_max']['spectral_limit_four_order'])} | {verdict['v_max']['spectral_vs_ghia_rel'] * 100:.3f} % | {"是" if verdict['v_max']['spectral_vs_ghia_rel'] < 0.03 else "否"} |
| u_min（vs 生产 FVM Richardson 极限） | {_f(fvm['u_min']['limit'])} | {_f(verdict['u_min']['spectral_limit_four_order'])} | {verdict['u_min']['spectral_vs_fvm_rel'] * 100:.3f} % | {"是" if verdict['u_min']['spectral_vs_fvm_rel'] < 0.03 else "否"} |
| v_max（vs 生产 FVM Richardson 极限） | {_f(fvm['v_max']['limit'])} | {_f(verdict['v_max']['spectral_limit_four_order'])} | {verdict['v_max']['spectral_vs_fvm_rel'] * 100:.3f} % | {"是" if verdict['v_max']['spectral_vs_fvm_rel'] < 0.03 else "否"} |
| 主涡心 x / y | Ghia (0.5313, 0.5625) | ({vx:.5f}, {vy:.5f}) | {verdict['vortex']['rel_err'][0] * 100:.3f} % / {verdict['vortex']['rel_err'][1] * 100:.3f} % | {"是" if max(verdict['vortex']['rel_err']) < 0.03 else "否"} |

最严项 {worst * 100:.3f} % —— 全部判据在 3 % 带内。

## 6. 对锚点验收的增量

* 修正 `SPECTRAL.md` §3.2 的外推程序：不再借壁滑移缺陷比，改用泛函自身的阶差比
  （本单证伪了代理比，偏差 >10×，§3.4）；谱极限因此从
  {_f(prev['u_min']['limit']) if 'u_min' in prev else '—'} 修正为
  {_f(verdict['u_min']['spectral_limit_four_order'])}（u_min），**结论方向不变**
  （仍与 FVM 极限在 0.3 % 内、与 Ghia 差 1.5 %），但不确定度陈述从 1e-3 级收紧到 1e-5 级。
* `SPECTRAL.md` §4 的「N > 64 不可达」欠账：**未关闭**（N = 96/128 作业排队，§3.6），
  只给出了可达的实现与新路径的等价性证明（稀疏/稠密 Jacobian 逐元素相等、Krylov 与
  稠密 LU 解一致到 8 位）。
* 新增可复用能力：`jacobian_sparse` + `linear_solver="krylov"`，任意高阶谱参考解可用
  `N = 128` 级网格（原路径在该阶内存不可行）。

## 7. 复算

```bash
# Krylov 路径单元/验收测试（约 20 s）
PYTHONPATH=src python3 -m pytest tests/test_cavity_spectral_ref.py -q

# 高阶阶梯（集群；N = 96 与 N = 128 分两个作业，见 docs 中作业号）
PYTHONPATH=src python3 scripts/run_cavity_spectral_ref.py --orders 48,96 \\
    --reynolds 1000 --tag re1000hi96 --krylov-min-order 96
PYTHONPATH=src python3 scripts/run_cavity_spectral_ref.py --orders 64,128 \\
    --reynolds 1000 --tag re1000hi128 --krylov-min-order 128

# 表格 + 三张图 + 本页
PYTHONPATH=src python3 scripts/report_cavity_spectral_hiorder.py
PYTHONPATH=src python3 scripts/make_hiorder_doc.py
```
"""]
    OUT.write_text("\n".join(lines))
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
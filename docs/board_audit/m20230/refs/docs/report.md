# CASE-CAVITY-RE1000-FVM — 方腔驱动流 Re = 1000 对标 Ghia, Ghia & Shin (1982)

> 本文件由 `scripts/render_case_cavity_re1000.py` 从 `docs/CASE-CAVITY-RE1000-FVM/artifacts/summary.json` 生成（零手抄）；参考值来自 `src/tensorfvm/lid_cavity_reference.py` 的逐字转录，机检门为 `tests/test_case_cavity_re1000.py`。

## ① 问题设定与基准出处

二维方腔驱动流：单位方腔 `[0,1]²`，三面固壁无滑移，顶盖 `y = L` 以 `u = U = 1` 切向运动（不可穿透）。以 `Re = U L / ν` 定义雷诺数，`Re = 1000` ⇒ `ν = 0.001`（ρ = 1，L = 1）。定常解。

- 基准：U. Ghia, K.N. Ghia, C.T. Shin, J. Comput. Phys. 48 (1982) 387-411, Table I (u at x = 0.5) and Table II (v at y = 0.5)；基准值单源转录在 `src/tensorfvm/lid_cavity_reference.py`（`table(Re, 'u'|'v')`、`VORTEX_CENTRE`），由 `self_check()` 与参考审计测试钉住。
- 验收判据（总账 `CASE-CAVITY-RE1000-FVM`）：关键量相对误差 < 5%。本模块把该判据展开成 7 条逐项机检（见 ④）。

| 量 | 基准值 | 出处 |
|---|---|---|
| u_min（x = L/2 反向流极小） | -0.38289 @ y = 0.1719 | Ghia 1982 Table I |
| v_max（y = L/2 极大） | 0.37095 @ x = 0.1563 | Ghia 1982 Table II |
| 主涡中心 | (0.5313, 0.5625) | Ghia 1982 §3（Table I/II 的极值位置） |

## ② 方法与实现

求解器：`tensorfvm.lid_cavity.LidCavitySolver` —— 交错（MAC）有限体积，显式对流（van Leer 受限面通量）+ 隐式扩散，投影法解耦压力，压力泊松用 DCT-II 在纯 Neumann 边界下的谱解（离散拉普拉斯精确求逆，无迭代误差）。时间推进到 `max|Δu|/Δt ≤ tolerance` 判为定常。

本单新增/复用的成本装置（纯成本，不改变被计分的离散定常解，且经检验）：

1. **网格延拓**：粗网格收敛 → 单元中心速度双线性延拓到细网格 → 继续推进；
2. **延拓校验**：基础网格同时冷启动跑一遍，两个收敛场逐点比对（`max|Δu|`、`max|Δv|`），延拓若引入偏差会在此暴露；
3. **逐算例检查点**（每 20 000 步 + 每算例结束），长作业被截断也留有可用行。

关键代码指针：

| 职责 | 位置 |
|---|---|
| 算例实现（运行/延拓/判据/工件/CLI 入口） | `src/tensorfvm/case_cavity_re1000.py` |
| 数值内核（交错 FV + 投影 + 谱压力解） | `src/tensorfvm/lid_cavity.py` |
| 基准转录（单源） | `src/tensorfvm/lid_cavity_reference.py` |
| 延迟兼容入口 | `scripts/run_cavity_re1000_m15458.py`（薄封装 → `case_cavity_re1000.main`） |
| 集成调用方（主验证梯） | `src/tensorfvm/verification/ladder.py` → `lid_cavity_verdict`（C2，Re = 1000 取最细网格） |
| 机检门 | `tests/test_case_cavity_re1000.py` |
| 报告渲染 | `scripts/render_case_cavity_re1000.py` |

## ③ 参数表

| 参数 | 值 | 说明 |
|---|---|---|
| Re | 1000 | `U L / ν`，L = 1，U = 1 |
| ν | 0.001 | `U L / Re`，ρ = 1 |
| CFL | 0.15 | `dt = CFL · h / U` |
| 收敛容差 | 1e-07 | `max|Δu|/Δt`（每 250 步测一次） |
| 步数上限 | 250000 | 达到即停止并如实标记未收敛 |
| 网格序列 | 32, 64, 128 | 含 1 个故意欠分辨的对照网格 |
| 精度 | float64 | 单进程，`torch.set_num_threads(1)` |
| 离散 | 交错 MAC + van Leer + DCT-II 压力解 | 见 `lid_cavity.py` 模块文档 |

## ④ 对照表（ref vs got，逐判据打勾/叉）

每个网格一行；`u_min`/`v_max` 为相对误差，`max|Δu|`/`max|Δv|` 为以顶盖速度 U 归一的无量纲偏差，`涡心` 为以 L 归一的绝对距离。

| n | 角色 | u_min got | u_min 相对误差 | v_max got | v_max 相对误差 | 涡心 got | 涡心距离 | max|Δu| | max|Δv| | 5% 判据 |
|---|---|---|---|---|---|---|---|---|---|---|
| 32 | control (under-resolved) | -0.31643 | 17.36% | 0.31544 | 14.96% | (0.5469, 0.6094) | 0.0469 | 0.2033 | 0.0754 | ❌ fail |
| 64 | production (cold start) | -0.37218 | 2.80% | 0.36228 | 2.34% | (0.5234, 0.5703) | 0.0079 | 0.0293 | 0.0197 | ✅ pass |
| 128 | production (continued from n=64) | -0.38537 | 0.65% | 0.37384 | 0.78% | (0.5273, 0.5664) | 0.0040 | 0.0043 | 0.0112 | ✅ pass |

最细生产网格 n = 128 的逐判据结论（判据定义见 `case_cavity_re1000.CRITERIA`）：

| 判据 | 说 | 实测 | 阈值 | 结论 |
|---|---|---|---|---|
| C1_u_extremum_rel | relative error of the reverse-flow minimum u (Table I) < 5 % | 0.00647 | 0.05 | ✅ |
| C2_v_extremum_rel | relative error of the maximum v (Table II) < 5 % | 0.00779 | 0.05 | ✅ |
| C3_vortex_centre_abs | primary-vortex centre within 0.05 L of (0.5313, 0.5625) | 0.00396 | 0.05 | ✅ |
| C4_u_profile_max_abs | max |u - u_Ghia| on x = 0.5 below 5 % of the lid speed | 0.00431 | 0.05 | ✅ |
| C5_v_profile_max_abs | max |v - v_Ghia| on y = 0.5 below 5 % of the lid speed | 0.01118 | 0.05 | ✅ |
| C6_converged | steady state reached: max|du|/dt <= tolerance | true | True | ✅ |
| C7_divergence | discrete divergence of the converged field below 1e-10 | 0.00000 | 1e-10 | ✅ |

逐点剖面（Ghia 采样点，最细网格）：

| y (u 剖面) | u got | u Ghia | Δ | x (v 剖面) | v got | v Ghia | Δ |
|---|---|---|---|---|---|---|---|
| 1.0000 | 0.99995 | 1.00000 | -0.00005 | 1.0000 | 0.00570 | 0.00000 | +0.00570 |
| 0.9766 | 0.66011 | 0.65928 | +0.00083 | 0.9688 | -0.22160 | -0.21388 | -0.00772 |
| 0.9688 | 0.57673 | 0.57492 | +0.00181 | 0.9609 | -0.28667 | -0.27669 | -0.00998 |
| 0.9609 | 0.51350 | 0.51117 | +0.00233 | 0.9531 | -0.34798 | -0.33714 | -0.01084 |
| 0.9531 | 0.46957 | 0.46604 | +0.00353 | 0.9453 | -0.40306 | -0.39188 | -0.01118 |
| 0.8516 | 0.33502 | 0.33304 | +0.00198 | 0.9063 | -0.52108 | -0.51550 | -0.00558 |
| 0.7344 | 0.18736 | 0.18719 | +0.00017 | 0.8594 | -0.42582 | -0.42665 | +0.00083 |
| 0.6172 | 0.05624 | 0.05702 | -0.00078 | 0.8047 | -0.31909 | -0.31966 | +0.00057 |
| 0.5000 | -0.06236 | -0.06080 | -0.00156 | 0.5000 | 0.02539 | 0.02526 | +0.00013 |
| 0.4531 | -0.10831 | -0.10648 | -0.00183 | 0.2344 | 0.32349 | 0.32235 | +0.00114 |
| 0.2813 | -0.28011 | -0.27805 | -0.00206 | 0.2266 | 0.33203 | 0.33075 | +0.00128 |
| 0.1719 | -0.38537 | -0.38289 | -0.00248 | 0.1563 | 0.37384 | 0.37095 | +0.00289 |
| 0.1016 | -0.29530 | -0.29730 | +0.00200 | 0.0938 | 0.33011 | 0.32627 | +0.00384 |
| 0.0703 | -0.21801 | -0.22220 | +0.00419 | 0.0781 | 0.30714 | 0.30353 | +0.00361 |
| 0.0625 | -0.19765 | -0.20196 | +0.00431 | 0.0703 | 0.29361 | 0.29012 | +0.00349 |
| 0.0547 | -0.17689 | -0.18109 | +0.00420 | 0.0625 | 0.27819 | 0.27485 | +0.00334 |
| 0.0000 | -0.00149 | 0.00000 | -0.00149 | 0.0000 | 0.00470 | 0.00000 | +0.00470 |

## ⑤ 收敛性与参数敏感性

| n | 步数 | 稳态判据 max|Δu|/Δt | 散度 max|∇·u| | 墙钟 (s) |
|---|---|---|---|---|
| 32 | 30500 | 9.76e-08 | 3.55e-15 | 123 |
| 64 | 58750 | 9.97e-08 | 2.13e-14 | 313 |
| 128 | 104250 | 9.89e-08 | 1.35e-13 | 1111 |

网格加密 64 → 128：`max|Δu|` 0.0293 → 0.0043，`max|Δv|` 0.0197 → 0.0112（比值 6.80× / 1.76×）。误差随加密单调下降且全程处在 5% 带宽内，说明 5% 的通过不是单网格侥幸。

延拓校验（同一 n = 64 网格，两种初值）：`max|Δu| = 3.45e-08`、`max|Δv| = 3.50e-08` ⇒ 同一离散定常解（阈值 1e-6）：✅；延拓 59750 步 vs 冷启动 58750 步，墙钟 324s vs 313s（省时 0.96×）。

欠分辨对照：n = 32 未通过 5% 判据（32: fail），如实保留在工件中，不用平均掩盖。

## ⑥ 失败模式、工件与复现

工件（全部由程序写入，路径相对仓库根）：

- `docs/CASE-CAVITY-RE1000-FVM/artifacts/summary.json` —— 全量度量、逐判据结论、参数表、延拓校验
- `docs/CASE-CAVITY-RE1000-FVM/artifacts/fields/Re1000_n32.pt` —— 收敛流场
- `docs/CASE-CAVITY-RE1000-FVM/artifacts/fields/Re1000_n64.pt` —— 收敛流场
- `docs/CASE-CAVITY-RE1000-FVM/artifacts/fields/Re1000_n128.pt` —— 收敛流场
- `docs/CASE-CAVITY-RE1000-FVM/figures/fig1_centrelines.png` —— centrelines
- `docs/CASE-CAVITY-RE1000-FVM/figures/fig3_convergence.png` —— convergence
- `docs/CASE-CAVITY-RE1000-FVM/figures/fig2_fields.png` —— fields
- `src/tensorfvm/verification/artifacts/verification_ladder/lid_cavity_Re1000_n*.json` —— 供主验证梯 C2 判据直接消费的行

复现命令（>10 min，按 owner 计算调度令递交队列）：

```bash
# 入口：跑整条网格序列（控制网格 32 → 生产 64 → 延拓 128 → 延拓校验）
PYTHONPATH=src python -m tensorfvm.case_cavity_re1000 \
    --out docs/CASE-CAVITY-RE1000-FVM/artifacts --grids 64,128 --cold-grids 32 \
    --ladder-dir src/tensorfvm/verification/artifacts/verification_ladder
# 机检门（从工件重算误差与判据，不重跑求解器）
PYTHONPATH=src python -m pytest tests/test_case_cavity_re1000.py -q
# 报告与图
PYTHONPATH=src python scripts/render_case_cavity_re1000.py
```

已知失败模式与局限：

- **欠分辨网格会假通过**：n = 32 的极值误差可能偶然落在 5% 内，因此对照行必须与生产行分开报告，且机检门要求对照行 `pass is False`；
- **显式推进的步数 ∝ n**：n = 128 约需 10⁵ 步，墙钟以小时计；本单用网格延拓与检查点把风险压到可接受，但收敛判据 `max|Δu|/Δt` 是必要非充分——它不能排除缓慢衰减的伪定常态，故要求两个独立初值给出同一解（延拓校验）；
- **只覆盖定常解**：Ghia 的 Re = 1000 方腔是定常的，本工件不对非定常/三维方腔作任何声明；
- **仅二分解法自证**：参考值是文献表，不是解析解；表格采样点之间的误差未评估。


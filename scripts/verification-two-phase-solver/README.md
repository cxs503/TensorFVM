# 两相投影求解器验证证据（mission m2564）

针对 `src/tensorfvm/two_phase_solver.py` 三个缺陷的修复证据与复现入口。

## 一键复现

```bash
# 1) 四个验收算例，写 results_<tag>.json + 图
python scripts/verification-two-phase-solver/verify_two_phase_solver.py --tag after

# 2) 修复前/后对照图（读取 results_before.json / results_after.json）
python scripts/verification-two-phase-solver/plot_before_after.py

# 3) 验收测试
python -m pytest tests/test_two_phase_solver_verification.py -q
#    回归
python -m pytest tests/test_two_phase_vof.py tests/test_solver.py -q

# 4) 生成（可选上传）论文级报告 /tmp/report_m2564.json
python scripts/verification-two-phase-solver/make_report.py [--publish]
```

## 缺陷 → 修复 → 证据

| 缺陷 | 根因 | 修复位置 | 证据（数值） |
|---|---|---|---|
| A. `surface_tension=True` 物理无效 | `_projection_step` 动量预测只含黏性扩散 + 重力，`csf_surface_tension` 从未被调用 | `_projection_step` Step 1：`f_csf/ρ` 单元→面算术插值加入 `u`/`v` 预测，受 `surface_tension`/σ 门控 | 液滴 Δp 由 0 变为 σ/R（误差 0.03%，R=0.15…0.30 m 全部 < 3%）；负控 `surface_tension=False` 时 Δp=0 |
| B. 压强 Poisson 与散度算子不自洽 | ① RHS 少体积因子 `V=dx·dy`（p 放大 1/V≈1600×）；② 非对角项取负号（左端成 `diag·q + Σc·q_nb`，非 SPD，CG 放大误差）；③ 顶行 Dirichlet 破坏对称性；④ 动量预测只更新 `u[1:-1,1:-1]`/`v[1:-1,1:-1]`，壁面条带无重力 → RHS 沿 x 不连续 | `_projection_pressure`（体积积分形式 + 正非对角 + 对称 Dirichlet 规范 + 共用 `_face_inv_density`/`_divergence_of`）；`_projection_step` 预测改为全内部面 + 零梯度鬼面 | 静止水柱 max\|u\| 6.4e55 → 5.2e-15 m/s；max\|div u\| 1.0e58 → 2.6e-13 1/s；注入散度 0.044 1/s 投影后 4.6e-14 1/s（修复前放大 5419×） |
| C. 全 Neumann 奇异系统 | `top_atmosphere=False` 无规范 | `TwoPhaseConfig.__post_init__` 抛 `ValueError` | `pytest.raises(ValueError, match="top_atmosphere")` |

## 文件

| 文件 | 内容 |
|---|---|
| `verify_two_phase_solver.py` | 四算例驱动：静止水柱 / 静液滴 / 封闭域拒绝 / 投影自洽 |
| `plot_before_after.py` | 修复前后对照图 `fig_compare.png` |
| `make_report.py` | 由 JSON 程序化生成报告（零手抄）+ 可选上传 |
| `results_before.json` | 修复前基线（指数发散 1e58、液滴 Δp=0、封闭域不拒绝、投影放大 5419×） |
| `results_after.json` | 修复后（全绿） |
| `fig_compare.png` | 修复前后 max\|div u\| 曲线 + Laplace 压差对照 |
| `fig_after_column.png` | 残差历史 + 静水压剖面 vs 解析 + 液滴中线 p/α |

## 结果摘要（修复后）

* 静止水柱（40×40，dt=1e-3 s，10 步）：max\|u\| = 5.2e-15 m/s，
  max\|div u\| = 2.6e-13 1/s，p 剖面 vs 分段解析相对误差 2.5e-5。
* 静液滴（64×64，R=0.25 m，σ=0.0728 N/m）：Δp = 0.291108 Pa vs σ/R = 0.291200 Pa，
  相对误差 0.031%（R=0.15/0.30 m：0.30%/0.19%；n=32：1.18%）。
* 投影自洽：注入 max\|div u*\| = 0.0443 1/s → 投影后 4.6e-14 1/s。
* 全封闭域：配置阶段 `ValueError`。
* pytest：`tests/test_two_phase_solver_verification.py` 10 passed；
  回归 `tests/test_two_phase_vof.py tests/test_solver.py` 33 passed + 39 subtests。

## 已知局限

* CSF 曲率取自光滑 α 的中心差分，无 PLIC 界面重构；液滴用 tanh 界面（宽 0.75dx）。
* CG 为 Jacobi 预条件；ρ 比 1000:1 时条件数 ~1e6，本算例 184 次迭代收敛到 1e-12，
  更大网格/更极端密度比建议换多重网格预条件（未实现）。
* 顶行为零梯度外流 + Dirichlet 压力规范，顶行自身不作连续性约束。
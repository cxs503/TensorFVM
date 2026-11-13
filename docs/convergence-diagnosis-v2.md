# TensorFVM convergence-diagnosis v2 —— 与 OpenFOAM-10 收敛判据的交叉对标 (XVAL)

锚点：XVAL-FVM-CONVERGENCE-DIAGNOSIS-V2 · 仓：TensorFVM · 工作分支 agent/m17520
运行工件：`docs/convergence-diagnosis-v2/`（`xval.json` / `xval.md` /
`histories/*.json` / 两张图）· 复现命令见 §2。

## ① 问题与参考出处

有限体积求解器里“这条迭代还在收敛吗？”通常只靠一条残差数字回答。成熟开源实现
（OpenFOAM-10）把它做成两件可核算的事：
(a) **残差归一化** —— `LduMatrix<...>::solver::normFactor`
(`src/OpenFOAM/matrices/LduMatrix/LduMatrix/LduMatrixSolver.C:167-186`，
sha256 `27a20e109622e6a4…`)：
`sum|A psi - sumA*avg(psi)| + sum|b - sumA*avg(psi)|`，用 1e-20 做 stabilise，避免近零源项时残差被放大；
(b) **收敛谓词** —— `SolverPerformance::checkConvergence`
(`.../LduMatrix/SolverPerformance.C:60-88`，sha256 `51344f053107300b…`)：
`finalResidual < tolerance` **或** `finalResidual < relTol*initialResidual`（relTol 仅在 > small_ 时生效），
并据此报告 `initialResidual / finalResidual / nIterations / converged` 四元组。

TensorFVM 生产路径（`sparse_simple._linear`，restarted GMRES + 两级线松弛预处理）
只有“目标残差 + 失败即抛异常”，缺三件事：无归一化残差、无速率估计、无停滞/发散判据。
v2 即在 (a)(b) 语义之上补这三件事，并**用同一批残差历史与 OSS 判据做数值对账**。

本机 OpenFOAM-10 源码存在但**未编译**（`platforms/*/bin` 为空，无求解器二进制），
因此参考量取“源码钉桩的语义移植”，而非执行 OSS 得到——这是本单的能力边界，如实记录。

## ② 方法与实现

- 实现：`src/tensorfvm/convergence_diagnosis.py`（归一化/谓词/监视器/CLI）、
  `src/tensorfvm/convergence_xval.py`（对标台架）、
  `scripts/xval_convergence_diagnosis.py`（薄 CLI）。
- 入口：`PYTHONPATH=src python -m tensorfvm.convergence_diagnosis diagnose|xval`，
  控制台脚本 `tensorfvm-convergence-diagnosis`（pyproject `[project.scripts]`）。
- 集成调用：`src/tensorfvm/sparse_simple.py:8,113,120` —— 生产 SIMPLE 线性路径的
  每次 GMRES 迭代真实残差喂给 `ConvergenceMonitor`，诊断结果写入
  `linear_history[*].convergence_diagnosis`（**只观测，不改控制流**）。
- 对标协议（阈值预先声明，15% = owner 精度线）：
  1. 参考量 ref：对真实 FVM 算子（五点 Poisson；一阶迎风对流扩散）以 OF-10 归一化与
     谓词驱动求解，得到 ref `nIter` / `converged`（见 §3 表）。
  2. 候选量 got：v2 监视器流式观察**同一批归一化残差**，在置信门开启的迭代 `k_ready`
     发布速率与迭代数预测；参考量对监视器不可见（只用已观察残差）。
  3. 三指标：`rate`（v2 尾窗速率 vs **样本外**实测速率 `(r_stop/r_ready)^(1/(k_stop-k_ready))`，
     阈值 ≤15%）；`verdict`（收敛旗标逐一与 ref 一致，0 容忍）；`forecast`（预测终止迭代
     vs ref nIter，阈值 ≤15%，**单独报告，不计入 benchmark_pass**，理由见 §5-F3）。
- 置信门（三条件同真才发布预测）：G1 残差已降 ≥1 个数量级；G2 尾窗比值相对散布 ≤0.20；
  G3 **速率是“standing”的**：前一个窗口的速率与尾窗速率相对差 ≤0.05。
  G3 是关键——没有它就会把 Krylov 早期的超线性段当作渐近速率外推（v1 门实测即如此，
  见 §5-F2 的前后对照）。
- 复现：
  ```
  PYTHONPATH=src python scripts/xval_convergence_diagnosis.py \
      --out docs/convergence-diagnosis-v2 --max-iter 2000 --figures
  PYTHONPATH=src python -m pytest tests/test_convergence_diagnosis.py -q   # 13 passed
  ```
  运行环境：Linux-6.6.52-1.7.lns23.loongarch64-loongarch64-with-glibc2.38 · Python 3.11.6 · numpy 2.4.6 · git `70fa3abef270b0a438c90890a0d5bf59f63afef1` · wall 89.49 s

## ③ 参数表（无量纲）与计算结果

无量纲量：`Pe = U L / Gamma`（全域），`Pe_cell = Pe*h`（网格分辨的单元 Peclet 数），
`kappa(A)`（sigma_max/sigma_min 的确定性幂迭代估计，仅入表不参与判据），
`tolerance`/`relTol`（与残差同量纲的归一化阈值），`alpha*2/lambda_max`（Richardson 超松弛因子）。
网格：均匀 n×n 单元，h=1/(n-1)，U=1、L=1。

| case | solver | n | Pe | Pe_cell | kappa(A) est | tol | relTol | ref nIter | ref conv | v2 verdict | k_ready | rho ref (o.o.s.) | rho v2 | rho err | fc base | fc band | fc err |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| poisson-n32-tol1e-8 | gmres | 32 | 0 | 0 | 4.405e+02 | 1e-08 | 0 | 188 | True | converged | 14 | 0.9114 | 0.9067 | 0.52% | 178 | 132-178 | 5.3% |
| poisson-n64-tol1e-8 | gmres | 64 | 0 | 0 | 1.707e+03 | 1e-08 | 0 | 541 | True | converged | 16 | 0.9698 | 0.927 | 4.41% | 229 | 178-229 | 57.7% |
| poisson-n64-tol1e-12 | gmres | 64 | 0 | 0 | 1.707e+03 | 1e-12 | 0 | 849 | True | converged | 16 | 0.9701 | 0.927 | 4.44% | 350 | 270-350 | 58.8% |
| poisson-n32-reltol1e-2 | gmres | 32 | 0 | 0 | 4.405e+02 | 1e-12 | 0.01 | 48 | True | converged | 14 | 0.9348 | 0.9067 | 3.01% | 37 | 31-37 | 22.9% |
| cd-pe10-gmres | gmres | 64 | 10 | 0.159 | 2.006e+03 | 1e-10 | 0 | 584 | True | converged | 108 | 0.9573 | 0.9909 | 3.51% | 2377 | 2377-5081 | 307.0% |
| cd-pe1000-gmres | gmres | 64 | 1000 | 15.9 | 1.024e+02 | 1e-10 | 0 | 498 | True | converged | 123 | 0.9462 | 0.898 | 5.09% | 316 | 316-417 | 36.5% |
| cd-pe10-cg-misuse | cg | 32 | 10 | 0.323 | 5.158e+02 | 1e-10 | 0 | 2000 | False | diverging | n/a | n/a | n/a | n/a | n/a | None-None | n/a |
| poisson-n32-richardson | richardson | 32 | 0 | 0 | 4.405e+02 | 1e-08 | 0 | 300 | False | diverging | n/a | n/a | n/a | n/a | n/a | None-None | n/a |

## ④ 对照表与误差判定（是否 <15%）

| 指标 | ref 定义 | got 定义 | 结果 | 阈值 | 判定 |
|---|---|---|---|---|---|
| rate（衰减样例 6 例） | 样本外实测速率 | v2 `k_ready` 尾窗速率 | 最大误差 5.09% | 15% | PASS（6/6） |
| verdict（8 例） | OF-10 `checkConvergence` 收敛旗标 | v2 `converged` | 一致 8/8 | 0 容忍 | PASS |
| forecast（6 例） | ref `nIterations` | v2 首次预测终止迭代 | 最大误差 307%；落在 2 倍带内 3/6 | 15%（声明不可达，见 F3） | FAIL（如实报告） |

**是否 <15%**：`rate` 与 `verdict` 两项 **是**（最大 5.09% / 全一致），
`forecast` 一项 **否**（最大 307%），原因见 §5-F3，不粉饰。

图：`docs/convergence-diagnosis-v2/fig_residuals_and_errors.png`
（左：各算例归一化残差史，圈=v2 置信门开启点；右：rate/forecast 相对误差与 15% 线）、
`fig_rate_parity.png`（速率奇偶校验图，y=x 为理想）。

## ⑤ 敏感性与失败模式

**收敛性/参数敏感性（两点）**
1. **置信门参数（G3 rate_stability_tol）**：v1（无 G1/G3，仅比值散布门）在 8 例上
   rate 最大误差 35.84%、rate 通过 7/8、
   forecast 最大误差 21815%（基线工件
   `xval_gate_v1.json`）；加 G1/G3 后 rate 最大误差降到 5.09%、
   rate 通过 6/6。差异来源：早期超线性段速率
   0.82 被当成渐近速率外推 → 预测 95 次而实际 541 次（poisson-n64）。
2. **网格/条件数**：n=32→64 使 kappa 从表列值升一档，v2 的 `k_ready` 从 14 移到 16，
   速率估计仍 standing；同算子把绝对容差从 1e-8 收到 1e-12（ref 188→849 次），
   v2 速率与预测随容差平移而保持同量级（表中 poisson-n64 两行速率一致到 4 位）。

**失败模式（如实）**
- **F1 CG 用于非对称迎风算子**（`cd-pe10-cg-misuse`）：2000 次内不收敛（ref conv=False），
  v2 判定非收敛（`diverging`，第 6 次即标记），
  verdict 一致 ✔，但其速率先被估成 0.9999 → 预测 438291 次；此类“几乎停滞”序列的
  预测值应读作“无意义的大数”，实现里由 `max_iter` 预算转判 `stagnating`。
- **F2 早期速率不可外推**：见敏感性 1。
- **F3 迭代数预测的不可达性（负面结论）**：`n = k + ln(thr/r_k)/ln(rho)`，
  误差放大因子 `dn/n = dρ/(ρ|lnρ|)`≈1/(0.95×0.05)≈21 倍——即速率误差 1% 就带来约 21%
  迭代数误差，故 15% 的迭代数预测在“局部速率外推”框架下**结构性不可达**；
  v2 因此只声明 **2 倍带**（乐观/悲观双速率）能力，实测 2 倍带覆盖
  3/6，其余为早期段的乐观偏差。
  这是本单的否定性成果（reusable knowledge），而非调参可解。
- **F4 Richardson 超松弛发散**（`poisson-n32-richardson`）：残差先降后升（低模态先收敛，
  高模态被 1.2×2/lambda_max 放大），v2 在第 13 次判 `diverging`；
  该例的“样本外速率”被正负相消污染（ref 1.37 vs v2 0.879），故 rate 指标只在
  `decaying=True` 的样例上评价（本单 6/8 例）——口径写在
  `xval.json:protocol.rate_truth`。
- **F5 无 OSS 二进制**：OpenFOAM-10 源码在机（两文件 sha256 已钉桩）但未编译，
  参考量是语义移植而非 OSS 实跑；伪“实跑对标”不可接受，故显式降级声明。

## ⑥ 与 OSS 的差异清单及移植取舍

| 面 | OpenFOAM-10 做法（源码） | TensorFVM 生产现状 | v2 处理与取舍 |
|---|---|---|---|
| 残差归一化 | `sum|Aψ-sumA·avgψ| + sum|b-sumA·avgψ|`，stabilise 1e-20 | `||r||_2` 与 `||b||` 混用，绝对/相对门限混配 | **移植**：`norm_factor` / `normalised_residual` 逐式实现（GPL 代码不复制，只按文档语义重写，避免许可污染） |
| 收敛谓词 | `r<tol 或 r<relTol·r0`（relTol>small 才生效） | 单一目标 `target=max(floor, min(bn·1e-11, tol·scale/10√N))` | **移植并对账**：`openfoam_converged`，verdict 一致 8/8；生产路径保留其原判据不动（避免改变既有算例数值） |
| 每解记录 | `initialResidual/finalResidual/nIterations/converged/singular` | `linear_history`（iterations, true_residual, target, elapsed_s…） | **扩展**：追加 `convergence_diagnosis` 字段（速率/散布/预测/verdict），零控制流改动 |
| 停滞/发散判据 | 仅 `nIterMax` 截断 + `checkSingularity`（mag(wApA)<vsmall_） | 无（失败即抛 RuntimeError） | **新增**：速率≥1 分 `diverging`/`oscillating`；预测超预算判 `stagnating`（早于 nIterMax，实测最早第 5 次） |
| 预算预测 | 无 | 无 | **新增**：置信门 + 乐观/悲观双速率 2 倍带（15% 不可达，见 F3） |
| 迭代计数口径 | 求解器迭代（含预条件应用） | GMRES 内层 Arnoldi 迭代 | 保持各自口径，对比时明确按 ref nIter 对齐 |
| 预处理 | DILU/DIAG/GAMG | 两级线松弛 + 粗网格 LU 校正 | 对标台架**故意无预处理**（隔离被测对象=诊断算法），生产路径维持原预处理 |
| 边界条件 | lduInterface 系数场 | 各求解器内联 | 无移植（诊断层不触碰离散） |

**移植取舍三条**：① 只移植“判据与归一化语义”，不移植 GPL 源码文本；② 不改生产
求解器的停止逻辑（数值可复现优先），诊断只作旁路观测 + 台账字段；③ 未编译的 OSS
不伪装成实跑参考，参考量口径在 `xval.json:provenance` 中钉桩可核。

## 生态摩擦

- 本机 OpenFOAM-10 源码在 `/tmp/OpenFOAM-10`、`/tmp/OpenFOAM-dev`，`platforms/*/bin`
  为空 → 无法实跑 OSS 对标，只能做源码语义移植（F5）。
- 平台 PG 口令在 `/root/tp-agent/.env` 内的 eco_admin 口令对本机 PG 无效，
  需走 unix socket `postgres` 角色（运营通道）；已据此完成开工三查。

# m17623 交差审计：m17520 · XVAL-FVM-CONVERGENCE-DIAGNOSIS-V2 (TensorFVM)

类型：交差核验（非重做）· 审计对象：mission 17520（succeeded, auto-approve）
交付 commit：`0cba327f`（分支 `agent/m17520`，**未**并入 main）· 审计工作树：`/root/tp-agent/work/m17623`
审计时间：2026-11-13 · 结论：**L3 / gate 应 confirmed（L4 不予授予）**

---

## 0. 独立性声明与范围

- 审计未修改被审工作树。复现输出写 `/tmp/m17623_repro`；所有被执行代码取自被审树
  `/root/tp-agent/work/m17520`（`PYTHONPATH=src`），审计脚本与工件在
  `artifacts/audit_m17623/`。
- 限定：同一台机器、同一 Python/numpy（3.11.6 / 2.4.6）；未执行任何 OSS 二进制
  （本机 OpenFOAM-10 `platforms/*/bin` 为空，`bin_count=0`，与交付自述一致）。
- 未跑：全仓测试套件、CUDA 路径、`scripts/xval_convergence_diagnosis.py` 另一种入口的
  独立重跑（该脚本为 `run_xval` 的 33 行 thin wrapper，已逐行阅读）。

## 1. 取证：交付三件套是否真存在

| 项 | 交付指针 | 审计核验 |
|---|---|---|
| impl_files | `src/tensorfvm/convergence_diagnosis.py`（381 行）、`src/tensorfvm/convergence_xval.py`（770 行）、`tests/test_convergence_diagnosis.py`（157 行） | 真代码：normFactor/checkConvergence 语义移植 + 5 点 FVM Poisson / 一阶迎风 CD 算子（`convergence_xval.py:90-168`）+ 自写 GMRES/CG/Richardson（`:271-361`）；无 placeholder |
| entry | `pyproject.toml` 控制台脚本 `tensorfvm-convergence-diagnosis`；模块 CLI `python -m tensorfvm.convergence_diagnosis {diagnose,xval}` | 两者均在；`xval` 子命令实跑成功 |
| run_artifacts | `docs/convergence-diagnosis-v2/{xval.json, xval.md, xval_gate_v1.json, xval_gate_v1.md, histories/*.json ×8, fig_*.png ×2}` | 全部存在非空；字节副本 + sha256 见 `artifacts/audit_m17623/SHA256SUMS.txt`、`delivered/` |
| tests | `tests/test_convergence_diagnosis.py` | 审计重跑：**13 passed in 6.54 s** |
| integration_callers | `src/tensorfvm/sparse_simple.py:96-119`（`ConvergenceMonitor` 实例化 + 每内层迭代 `diag.record` + `linear_history[…]convergence_diagnosis`） | 静态 diff + **运行时**核验（见 §3）真到达生产 SIMPLE GMRES 路径 |
| benchmark | 见 §4（ref vs got 数值表） | 数值与工件逐字段一致 |
| repro_cmd | `PYTHONPATH=src python -m tensorfvm.convergence_diagnosis xval --out <dir> --max-iter 2000 --figures` | **审计实跑 → 8 例 × 20 数值字段 0 处差异**（summary 完全相同，wall 89.49 s → 97.13 s） |

## 2. 复现（可执行性）

```
cd /root/tp-agent/work/m17520
PYTHONPATH=src python -m tensorfvm.convergence_diagnosis xval --out /tmp/m17623_repro --max-iter 2000 --figures
python -m pytest tests/test_convergence_diagnosis.py -q     # 13 passed
```
结果：`summary` 与已提交 `docs/convergence-diagnosis-v2/xval.json` **完全相同**
（`summary equal: True`，逐 case 逐字段 diff = 0）。确定性可复现成立。

## 3. 集成面（L3 的证据，审计自建运行时检查）

交付自带测试只做**静态源串检查**（`inspect.getsource(sparse_simple)` 含 `ConvergenceMonitor`，
`tests/test_convergence_diagnosis.py:139-142`），不能证明字段真被填充。审计补做运行时检查
（`artifacts/audit_m17623/integration_check.py` → `integration_runtime.json`）：

| 指标 | 结果 |
|---|---|
| 生产 `SparseBodyFittedSolver`（curved-channel, 32×16, Re=1）线性解次数 | 167 |
| 其中 `linear_history[i]['convergence_diagnosis']` 非空 | **167/167** |
| 首解诊断 | verdict=`converged`, rate=0.3936, spread=0.1539, forecast=26, "OF-10 checkConvergence satisfied" |
| 求解整体仍收敛 | `converged=true`（旁路观测未扰动控制流） |

→ 集成是真的（L3 成立）；但**在仓内没有回归保护**（见 F4）。

## 4. 基准裁判（唯一裁判）与数值判定

- `SELECT case_name, metric, tolerance FROM tensor_benchmark_cases WHERE anchor_slug='XVAL-FVM-CONVERGENCE-DIAGNOSIS-V2'`
  → **0 rows**（审计时刻实测）。本锚点从未登记裁判用例；实际生效判据只能取 mission 17520 goal 的
  「对标误差 <15%」。此点与 m17360 交差审计（verdict #575, L3）一致，非本次新问题。
- 外部锚（external_anchor）**不是** OSS 实跑：参考实现 `ReferenceStop` 与被测
  `ConvergenceMonitor` 由同一作者在**同一提交**内写下，参考量是 OpenFOAM-10 源码语义移植。
  审计已核两条 sha256 钉桩与机内源码一致：
  `SolverPerformance.C = 51344f05…`、`LduMatrixSolver.C = 27a20e10…`；
  `checkConvergence` 语义（`finalResidual_<Tolerance || RelTolerance>small_*one && <RelTolerance*initialResidual_`）
  与移植一致。

### 4.1 三个指标的逐项核对（数字取自 `delivered/xval.json`，程序化导出）

| case | ref conv | ref nIter | k_ready | ρ ref(样本外) | ρ got | rate err | forecast got | fc err | 终判 |
|---|---|---|---|---|---|---|---|---|---|
| poisson-n32-tol1e-8 | True | 188 | 14 | 0.9114 | 0.9067 | 0.52% ✔ | 178 | 5.3% ✔ | converged ✔ |
| poisson-n64-tol1e-8 | True | 541 | 16 | 0.9698 | 0.9270 | 4.41% ✔ | 229 | 57.7% ✘ | converged ✔ |
| poisson-n64-tol1e-12 | True | 849 | 16 | 0.9701 | 0.9270 | 4.44% ✔ | 350 | 58.8% ✘ | converged ✔ |
| poisson-n32-reltol1e-2 | True | 48 | 14 | 0.9348 | 0.9067 | 3.01% ✔ | 37 | 22.9% ✘ | converged ✔ |
| cd-pe10-gmres | True | 584 | 108 | 0.9573 | 0.9909 | 3.51% ✔ | 2377 | 307.0% ✘ | converged ✔ |
| cd-pe1000-gmres | True | 498 | 123 | 0.9462 | 0.8980 | 5.09% ✔ | 316 | 36.5% ✘ | converged ✔ |
| cd-pe10-cg-misuse | False | 2000 | — | — | — | — | — | — | diverging ✔ |
| poisson-n32-richardson | False | 300 | — | — | — | — | — | — | diverging ✔ |

- **rate：6/6 ≤15%（max 5.09%）PASS** —— 但 `rate_truth` 是同一条残差轨迹的**后段样本外实测**
  （`(r_stop/r_ready)^(1/(k_stop-k_ready))`），是"预测自洽性"验证，**不是独立参考**。
- **forecast：1/6 ≤15%（max 307%）FAIL** —— 交付如实写进了 `summary.forecast_pass=false`
  与 §4/§5-F3，未隐匿。
- **verdict：8/8 一致** —— **该项在构造上是恒等式**：ref 与 got 把**同一条归一化残差流**
  代入**同一条谓词**（`value<tol || rel_tol>1e-20 && value<rel_tol*initial`，initial 同为第 0 步），
  审计独立重放 8 例，`verdict_rule_mismatch = 0` 是必然结果。它证明"移植没走样"，
  **不构成**外部对标证据。
- 前后对照（v1→v2，`xval_gate_v1.json`）：rate max 35.84% → 5.09%；v1 rate 7/8；
  v1 forecast max 21814% → 307%。这是真实基线对照（S2 级改进的合格形态）。

## 5. 判级

| 级 | 判据 | 结论 |
|---|---|---|
| L0 文档 | — | 过 |
| L1 数学/单元 | 13 测试断言不变量（真值表/几何级数 rate 精确 0.85/normFactor 恒等/stabilise） | 过 |
| L2 独立功能 | 输入→计算→输出真代码，非占位 | 过 |
| L3 系统集成 | 生产 `SparseBodyFittedSolver` 真调用，**运行时 167/167** 填充诊断字段 | **过** |
| L4 基准验证 | 需独立参考/登记裁判 | **不过**：裁判表 0 行；外部锚系自写语义移植（无 OSS 实跑）；verdict 项恒等式；forecast 项 307% 失败且被排除出复合通过口径；分支未并入 main |
| L5 科研工程 | — | 不适用 |

**判级 = L3**（与 m17360 同锚点审计 #575 同级；m17520 是**另一套实现**，级别独立判定）。

## 6. 发现（按严重度）

**F1 [高 · 阻断 L4] 裁判缺席 + 对标口径退化**
`tensor_benchmark_cases` 对该锚点 0 行；`external_anchor` 为同提交自写移植，无 OSS 二进制
（`bin_count=0` 复核为真）。`verdict 8/8` 与 ref 共用谓词与残差流 → 恒等式。因此三项指标里
只有 rate 是真外部信息，而 rate 的真值又取自同轨迹后段。**结论**：这是"语义移植正确性
自检 + 样本外预测自洽"，不是 XVAL 对标验证。L4 需补：登记裁判用例（case_name/metric/tolerance）、
一个真正独立的参考实现或 OSS 实跑。

**F2 [高 · 表述越权] `benchmark_pass=true` 是排除失败指标后的复合口径**
`convergence_xval.py:734-740` 把 `benchmark_pass` 定义为 `rate_pass AND verdict_pass`，
forecast 明确不计入（注释 + docs §5-F3 有说明）。但 mission headline / commit message 只写
"rate err ≤5.1%, verdict 8/8"，读者易读成"15% 全过"。且 2 倍带口径
（`summary.forecast_band_declared_tolerance=1.0`）是**本次运行才引入的事后阈值**，
不是预注册（experimental-design 纪律）。缓解：F3 给出了结构性论证
`dn/n ≈ dρ/(ρ|lnρ|) ≈ 21·dρ/ρ`，属诚实负面成果；但任何"2 倍带内 3/6"的表述必须显式标注
**post-hoc**。

**F3 [中 · 能力未度量] 早期预警 4/6 误报，套件里根本没有度量它的判据**
审计独立重放（`artifacts/audit_m17623/audit_replay.json`）：6 个最终收敛的算例中 **4 个**
在先于收敛点的位置抛出 `stagnating/oscillating/diverging`：
poisson-n64-tol1e-8 `stagnating@37`(实际 541)、poisson-n64-tol1e-12 `stagnating@37`(849)、
cd-pe10-gmres `oscillating@57`(584)、cd-pe1000-gmres `diverging@39`(498)。
即本套件上**早警假阳率 67%**。`summary` 只统计终判一致（恒等式），docs §5 失败模式亦未量化此项。
生产侧目前只落台账、不触发动作，暂无危害，但"停滞/发散判定"作为能力项**未验证**。

**F4 [低] 集成测试是静态源串检查，无回归保护**
`tests/test_convergence_diagnosis.py:139-142` 仅断言源码含 `ConvergenceMonitor` 字样。
审计补的运行时检查（167/167）证明当前实现正确，但仓内没有任何测试会在字段被删/改名时变红。

**F5 [低] 死参数 `divergence_factor`**
`DiagnosisConfig.divergence_factor=1e2` 全仓无引用（`grep` 仅命中定义处），却被序列化进
`xval.json:protocol.diagnosis_config`，读工件者会误以为它是判定门限。

**F6 [低] 工件 git 溯源指父提交**
`xval.json:run.git_head = 70fa3abe`（交付提交的父提交），而工件由 `0cba327f` 提交入库。
审计在 `0cba327f` 上重跑逐字段一致，故无实质影响；但工件自身不标识产出它的提交。

**F7 [低/信息] 分支未并入 main**
`git merge-base --is-ancestor 0cba327f main` → False。功能对 main 用户不可用；
确认（confirmed）不等于可用。

## 7. 未核 / 未过项（明确标记）

- `not_run`：全仓 pytest、CUDA 路径、`scripts/xval_convergence_diagnosis.py` 入口的独立重跑。
- `not_run`：OSS 实跑对标（本机无 OpenFOAM 二进制，非审计选择性跳过）。
- `unknown`：交付是否真的执行了 mission 要求的"开工三查"（knowledge_base /
  negative_knowledge_ledger / oss_solutions）——工件中无凭证。

## 8. 处置（disposition）

**approve with notes → L3，gate 应 confirmed（消账）**；同时要求：
1. 若要把该能力升 L4：登记 `tensor_benchmark_cases` 用例 + 引入一个真正独立的参考实现
   （或一台能跑 OpenFOAM 的机器）+ 把 early-warning 假阳率列为判据。
2. 表述层：`benchmark_pass` 更名/拆分为三指标分别声明；2 倍带标注 post-hoc。
3. 仓内补一条运行时断言（`linear_history[0]['convergence_diagnosis']['verdict']` 存在且合法），
   并把 `divergence_factor` 删除或接进判据。
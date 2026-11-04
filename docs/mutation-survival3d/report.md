# 三维原生有限体积核的变异存活审计（3-D mutation coverage）

## 1. 案例介绍

本仓已有的变异审计 `src/tensorfvm/verification/mutation_catalogue.py` 自述只覆盖**二维**生产核（`body_fitted` / `solver`）；承担 SUBOFF、HLPW 类验收的三维原生路径（面/算子几何、相容面通量投影、同位网格 Rhie–Chow 型压力-速度耦合）从未做过故障注入。本审计对该缺口做一次可复现的测量，并回答一个问题：现有测试**是否约束了三维核的离散算子**，而不是它们算得准不准。

每个变异体只改一行语义（符号、删项、权重互换、漏除体积），`find` 串必须在被改文件中**恰好出现一次**，否则判 `invalid`；改后编译、跑定向测试、再从 git 恢复源码并以 sha256 校验。

## 2. 怎么做的

工具链：Python 3.11.6、numpy 2.4.6、torch 2.7.1、scipy 1.16.0；`scripts/audit_mutation_survival3d.py` 复用二维审计脚本的注入/恢复/判定函数，只换目录与输出，保证两次审计判据一致（exit=1 → killed，exit=0 → survived，其余 → invalid）。测试运行时 `OMP_NUM_THREADS=MKL_NUM_THREADS=1`，逐个变异体串行执行。

本次纳入 5 个变异体，覆盖 2 个模块：`src/tensorfvm/native_collocated3d.py`（压力-速度耦合、面偏斜修正）与 `src/tensorfvm/native_operators3d.py`（仿射精确重构、Gauss 梯度）。

## 3. 计算结果

| 变异体 | 文件 | 缺陷 | 预声明 | 实测 | 首个失败测试 | 用时/s |
|---|---|---|---|---|---|---:|
| `CV3-PFD-SIGN` | `src/tensorfvm/native_collocated3d.py` | Compatible face-flux defect gets the wrong sign (interpolated gradient flux plus compact flux) | killed | killed | tests/test_native_collocated3d.py::test_pressure_wall_extrapolation_does_not_impose_zero_normal_gradient | 8.3 |
| `CV3-PFD-DROP` | `src/tensorfvm/native_collocated3d.py` | Compatible face-flux defect deleted: the compact two-point gradient is used as if it were a face flux | killed | killed | tests/test_native_collocated3d.py::test_pressure_wall_extrapolation_does_not_impose_zero_normal_gradient | 8.8 |
| `CV3-SKEW-DROP` | `src/tensorfvm/native_collocated3d.py` | Face skewness correction silently disabled regardless of the caller's flag | killed | killed | tests/test_native_collocated3d.py::test_pressure_wall_extrapolation_does_not_impose_zero_normal_gradient | 11.2 |
| `OP3-RECON-SWAP` | `src/tensorfvm/native_operators3d.py` | Owner/neighbour interpolation weights swapped in the affine-exact face reconstruction | killed | killed | tests/test_native_mesh3d.py::test_face_reconstruction_reduces_to_geometric_projection_on_nonuniform_pair | 8.9 |
| `OP3-GAUSS-VOLUME` | `src/tensorfvm/native_operators3d.py` | Gauss gradient returned as an integrated flux (cell volume division dropped) | killed | killed | tests/test_native_mesh3d.py::test_linear_cells_have_real_volume_and_affine_response[tetra-points0-0.16666666666666666] | 7.5 |

汇总：killed 5，survived 0，invalid 0，共 5 个。
源码恢复校验：5 / 5 个变异体恢复了原始 sha256。

每个变异体的处置：

- **CV3-PFD-SIGN** (killed)：The defect operator is what keeps the collocated pressure and the face flux compatible; reversing it turns the damping into an anti-damping, so a steady solve must lose convergence or move the pressure field.
  - pytest 末行：`1 failed, 2 warnings in 5.86s`
- **CV3-PFD-DROP** (killed)：Deleting the interpolated-gradient part leaves an inconsistent face flux, i.e. exactly the checkerboard-prone operator this module exists to replace; the affine/consistency tests should notice.
  - pytest 末行：`1 failed, 2 warnings in 6.15s`
- **CV3-SKEW-DROP** (killed)：On skewed native meshes the skew term is the second-order part of the face reconstruction.  If no test uses a skewed mesh with the flag on, the defect survives and the suite only certifies orthogonal behaviour.
  - pytest 末行：`1 failed, 2 warnings in 5.86s`
- **OP3-RECON-SWAP** (killed)：With the weights swapped the geometric weight (which is not 1/2 on a graded mesh) inverts; any test that checks interpolation on a non-uniform mesh or affine reproduction must fail.
  - pytest 末行：`1 failed, 7 passed, 1 skipped, 2 warnings in 5.35s`
  - 补测记录：首次审计时该变异体 survived：被指向的 tests/test_native_mesh3d.py 只用单胞网格，没有内部面，权重路径从未被走到。已补 test_face_reconstruction_reduces_to_geometric_projection_on_nonuniform_pair 把存活项关闭。
- **OP3-GAUSS-VOLUME** (killed)：The gradient is a per-volume quantity; dropping the division scales it by the cell volume.  Only a test that compares against an analytic affine gradient on cells of different volume can see it.
  - pytest 末行：`1 failed, 2 warnings in 5.12s`

## 4. 结果图

`mutation-survival-3d.png`：逐变异体的 killed/survived 与用时（本审计程序化生成）。

## 5. 注意点

* 变异体是**人工挑选**的语义缺陷，不是随机缺陷分布，killed 比例不能当成缺陷检出概率的估计；
* 判定依赖被指向的测试文件本身是否够格：定向测试若只断言「有限值」，本审计只会把它记为 survived；
* 一个变异体 killed 只说明**某处**有约束，不说明这条断言正是为它写的；
* 结论只覆盖被改的两行语义与所列测试文件，不外推到全部三维核；
* 审计过程会临时改写工作树源码，必须串行、禁止与其它写源任务并发。

## 6. 误差判定

本审计没有连续量的参考值，判定是布尔量（测试红/绿），因此不存在「是否 <3%」的误差问题：判据是退出码，逐项列于上表「实测」列。若以「killed 比例」作为覆盖度量，本次为 5/5，其不确定性来源是变异体的选取，而不是数值误差。

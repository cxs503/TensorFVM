# 《torch 线性求解器调研》分析报告

> 分析对象：`torch线性求解器调研.md`（上传文档）
> 分析视角：结合 **TensorLBM / PyTorch-based CFD** 在 Wuxi 超算上的运行与性能分析
> 结论先讲：**文档主体技术准确、结构清晰，是一份合格的后端映射笔记；但在版本论断、GPU 后端性能背景、以及 CFD 实际选型三方面存在可纠正/可补充之处。**

---

## 一、文档概览与定位

原文档按"三层栈"组织，定位是 **CUDA 后端的算子—库映射表**，而非算法选型指南：

| 层次 | 内容 | 价值 |
|---|---|---|
| 一、底层接口 | `torch.linalg.*` → cuSOLVER 例程（potrf/getrf/geqrf/gesvd/syevd…） | 定位"谁调谁"，便于排查性能瓶颈 |
| 二、上层接口 | `solve/inv/det/pinv/…` → 由底层组合而成 | 明确调用链，避免无谓重复分解 |
| 三、稀疏求解 | `torch.sparse.spsolve` → **cuDSS**（非 cusolverSp） | 指出 torch 2.12 新增的稀疏直接法后端 |
| 四、三方库 | torch-sla / torchsparsegradutils / Pytorch-Sparse-Linalg / BiCGSTAB gist | 纯 PyTorch 实现的 Krylov 迭代法 |

整体判断：方向对、事实基本对，缺少的是**"在什么规模/什么矩阵下该用哪个"**的判据。

---

## 二、准确性核验（对照官方/社区证据）

### ✅ 已验证正确的核心论断

- **底层 cuSOLVER 映射**：cholesky→`potrf`、cholesky_solve→`potrs`、lu→`getrf`、lu_solve→`getrs`、qr→`geqrf`+`orgqr`、svd→`gesvd`/gesvdj(Batched)/gesvdaStridedBatched、eigh→`syevd`/`syevjBatched` —— 均与 PyTorch 官方 [GPU backend tracking (#47953)](https://github.com/pytorch/pytorch/issues/47953) 一致。
- **上层组合关系**：`solve = lu_factor + lu_solve`、`inv = LU`、`slogdet/det = lu_factor_ex 取对角元`、`pinv/matrix_rank/cond = svd`、`cholesky_inverse = cholesky + potrs` 均正确。
- **cuDSS 后端与限制**：`torch.sparse.spsolve` 确实走 **cuDSS**（非 cusolverSp），且**标准 pip 二进制不含 cuDSS**，需 `USE_CUDSS=1` 源码编译；仅 `SparseCsrCUDA`、仅 float32/float64、仅 `left=True`——全部被 [官方文档](https://docs.pytorch.org/docs/main/generated/torch.sparse.spsolve.html) 与 [Runebook 实测](https://runebook.dev/zh/docs/pytorch/generated/torch.sparse.spsolve) 印证。
- **三方库清单**：torch-sla（CG/BiCGStab/GMRES/MINRES，多后端可微）、torchsparsegradutils、Pytorch-Sparse-Linalg、BiCGSTAB gist 的描述均属实。

### ⚠️ 需要纠正的论断：第 16 行 `torch.linalg.eig` 的 CUDA 门槛

> 原文档："`torch.linalg.eig` / `eigvals` 需 cuSOLVER ≥ 11.7.2（即 CUDA ≥ 12.8）"

证据显示该版本号**对不上**：

| 事实 | 来源 |
|---|---|
| 非对称特征值底层例程在 **cuSolver 12.6 update 2**（2025-04）才加入 | [data-apis/array-api#935](https://gitmemories.com/data-apis/array-api/issues/935) |
| torch 的 **cuSOLVER eig GPU 路径**在 **2.10 nightly**（2025-11）才可用；此前 GPU 走 **MAGMA**，更早版本 GPU 上直接 `NotImplementedError` | [PyTorch Forum 实测](https://discuss.pytorch.org/t/torch-linalg-eig-parallelisation/223386/16)（2.9 仍 MAGMA，2.10.dev20251104+cu130 才有 cusolver 路径） |

**建议修正为**：一般方阵（非对称）特征分解的 GPU 加速路径依赖 cuSolver 12.6 update 2+ 的底层例程，且 torch 在 **2.10+（nightly 起、稳定版随后）** 才把 `eig`/`eigvals` 的 GPU 后端从 MAGMA 切到 cuSOLVER；"CUDA ≥ 12.8 / cuSOLVER ≥ 11.7.2" 这一具体组合未见公开依据，需核实或删除。

### ⚠️ 被忽略的关键性能背景：MAGMA → cuSOLVER 是逐步迁移的

原文档把 cuSOLVER 当成"默认后端"，但实际历史是：**GPU 上多数线性代数算子长期默认走 MAGMA，cuSOLVER 是逐个算子迁移的**。

- `eigh/eigvalsh` 的 cuSOLVER 路径早在 CUDA ≥ 10.1.243 即启用（[#53040](https://github.com/pytorch/pytorch/pull/53040)）；
- 但 `eig`（非对称）、部分 batch 算子迁移很晚；
- 可运行时强制切换后端：`torch.backends.cuda.preferred_linalg_library("cusolver")`，实测 `eig` 在 RTX 上 **cuSOLVER 比 MAGMA 快约 9×**（fp32 4×、fp64 1.3×）。

> 对 CFD 调优的直接含义：**如果 profiling 发现 `eig`/`solve` 慢，先确认走的是哪个后端**，而不是盲目换算法。

---

## 三、与 TensorLBM / CFD 的相关性分析（最关键）

原文档是"映射表"，但 **LBM/CFD 选型真正要回答的是另一组问题**，文档基本没触及：

### 1. LBM 里"线性求解"到底发生在哪？

- 标准显式 LBM（D2Q9/D3Q19 + BGK/MRT）**本身不求解线性系统**，靠碰撞+流转移显式推进；
- 真正消耗线性求解器的场景通常是：
  - **不可压压力投影（Poisson 方程）**——典型大型稀疏 **SPD** 系统；
  - 隐式/半隐式时间格式、多重网格预条件、 adjoint/可微 LBM 中的梯度回传。

> 结论：TensorLBM 若涉及压力投影或隐式项，主战场是**大稀疏 SPD 系统**，而非文档重点罗列的稠密算子。

### 2. 选型判据（文档缺失的核心）

| 场景 | 推荐路径 | 理由 |
|---|---|---|
| 小/中规模稠密 SPD | `cholesky` + `cholesky_solve` | 比 LU 快约 2×，torch-sla 也印证 |
| 稠密非对称/通用 | `torch.linalg.solve`（LU 路径） | 简单直接 |
| **大稀疏 SPD（CFD 主战场）** | **Krylov + 预条件**（CG/AMG） | 直接法内存 O(n²) 不可承受 |
| 中等稀疏、需精确解、内存够 | `torch.sparse.spsolve`（cuDSS） | 原生直接法，但限制多 |
| 超大/分布式 | 三方库：torch-sla（strumpack/amgx/pyamg 后端） | 支持分布式、AMG 预条件 |

**关键缺口**：torch 原生**没有**面向大稀疏系统的迭代求解器，cuDSS 直接法在 `torch.sparse.spsolve` 上**内存代价高、限制多**（仅 CSR-CUDA/f32-f64/left-only）。CFD 大规模网格上基本必须上三方库（torch-sla 的 cuDSS/AMGX/strumpack 后端，或 Pytorch-Sparse-Linalg 的 AMGX 后端）。

### 3. 超算环境必须纳入的维度（文档完全没提）

- **FP64 吞吐**：Wuxi 超算若为 A100/H100，FP64 是 FP32 的 1/2；若为消费级/游戏卡则 FP64 仅 1/32~1/64。**多数 cuSOLVER 例程仅 f32/f64**，迭代法建议 f64 保收敛（torch-sla 实测提示）。直接影响在超算 vs 本地 GPU 上的可行性。
- **Batch 性能**：LBM 常并行多分辨率/多样本，`potrfBatched`/`getrfBatched` 等 batched 例程的特征（小矩阵才划算）未分析。
- **精度支持**：bfloat16/float16 在 torch 线性求解器中覆盖不全，混合精度 CFD 需逐算子确认。
- **`torch.compile`**：文档未提；部分求解器可 compile，但三方库的 `torch.compile` 开关（如 Pytorch-Sparse-Linalg 的 `_JIT_ENABLED=False`）需手动开。

---

## 四、对原文档的补充建议清单

1. **修正第 16 行** `eig` 的 CUDA 门槛（见第二节），删除无依据的版本组合。
2. **补充 GPU 后端背景**：MAGMA↔cuSOLVER 迁移史 + `preferred_linalg_library` 切库技巧。
3. **新增"稀疏迭代法"小节**：明确 torch 原生无大稀疏迭代器，CFD 必须依赖三方库；附选型表（第三节表）。
4. **补充超算/精度维度**：FP64 吞吐、batch、half/bf16 覆盖、`torch.compile`。
5. **核实并修正链接**：
   - 第三节相对路径 `../../torch2.12/...` 跨环境失效，需改为绝对/公开链接；
   - torch-sla 文档链接写 `sparsexlab/torch-sla`，实测 GitHub 上游为 `walkerchi/torch-sla`，建议核实。
6. **补充新库**：可加入 [cudass](https://moca-technicon.github.io/cudass/)（moca-technion）——统一稀疏求解 API，自动在 cuDSS / cuSOLVER Dense / cuSolverSp 间分派，支持奇异/矩形系统与 batch RHS。

---

## 五、下一步行动建议（针对 TensorLBM）

1. **先确认 TensorLBM 是否真的要解线性系统**：grep 排查 `solve/spsolve/eig/inv` 调用点，定位是压力投影还是其他。
2. **若是大稀疏 SPD**：在超算上 benchmark `torch.sparse.spsolve`(cuDSS) vs torch-sla 的 `cudss+cholesky` / `amgx` / `cg` 三档，按 DOF 与内存选。
3. **核对 torch 编译**：当前环境 `torch` 是否 `USE_CUDSS=1` 编译（否则 `spsolve` GPU 直接 RuntimeError）。
4. **后端确认**：对 `eig`/`solve` 慢的路径跑 `preferred_linalg_library("cusolver")` 对比。

---

### 一句话总结
文档是一份**准确的后端映射速查表**，但作为 CFD 选型依据还缺三块：**大稀疏迭代法生态、GPU 后端切换技巧、超算精度/吞吐约束**。先把第 16 行版本论断改对，再补第三章"稀疏迭代法"即可从"笔记"升级为"可落地的选型指南"。

# 开源 FVM 技术盘点：TensorFVM 推广波（第 1 轮 · 功能点全扫）

**scenario_key**: oss-inventory ｜ **feature_slug**: FEAT-OSS-INVENTORY-FVM ｜ **mission**: 14887 ｜ **仓**: TensorFVM
**本轮产物**：30 仓普查注册表（`registry.json` / `registry.md`）、8 条 `GAP-FVM-*` 锚点（`gap-proposals.md`）、
1 条外部锚基准实测（`artifacts/sod_literature_benchmark.json` + 2 图）、1 个门禁测试（`tests/test_oss_inventory_fvm.py`）。

---

## 一、案例介绍（为什么做、盘什么、边界在哪）

### 1.1 物理/工程背景

TensorFVM 是船舶总体性能链路里的**有限体积求解器**：不可压 RANS/稳态、两相 VOF 自由面、可压多材料、
CHT、势流耦合面都已建起验证面（`docs/benchmarks-index.md`）。真正限制它在「船 + 冰 + 高速/空化」工况上
往下走的，不是某个算例差 3%，而是**几类机制在仓里根本不存在**（自适应网格、隐式推进、低马赫预条件、
旋转/滑移交界面、并行域分解、通用 AMG）。这些机制在开源生态里已有成熟实现，本波的任务是
**先把生态盘清楚、把缺口锚点立起来**，并在动手前把「缺口将来用来比对的基线」先验一遍。

### 1.2 普查口径（可复算）

- **纳入标准**：以有限体积离散为主、且覆盖「OpenFOAM 系 / 可压缩密度基 / 多相」三条主线之一；补充少量
  网格、耦合、线代、非 FVM 对照项目（T3，只作状态参照）。
- **规模**：30 个项目 = T1 8 / T2 17 / T3 5（机器计数见 `registry.json:counts`）。
- **每个项目的字段**：技术族、离散方式、能力标签、Tier、分级理由、可吸收点、许可风险、缺口映射、
  以及**元数据实测**（stars / SPDX licence / 最近 push / 语言，GitHub REST API 逐仓抓取，抓取时间入库）。

### 1.3 几何/参数（本波唯一的数值算例）

| 项 | 取值 |
|---|---|
| 算例 | Sod 激波管（1-D，x∈[0,1]，膜片 x₀=0.5） |
| 初值 | 左 ρ=1, u=0, p=1；右 ρ=0.125, u=0, p=0.1；γ=1.4 |
| 终止时间 | t=0.2（激波 x≈0.8504，接触间断 x≈0.6855，稀疏波头 x≈0.2634） |
| 数值设置 | HLLC 通量 + MUSCL-MC 二阶重构，CFL=0.4，网格族 nx=100/200/400 |
| 外部锚 | Sod (1978) JCP 27:1–31 精确解星状态值（Toro 2009 Ch.4 表值） |

---

## 二、怎么做的（方法/流程/工具链）

### 2.1 开工三查（避免重复造轮子）

| 查 | 结果 | 含义 |
|---|---|---|
| `knowledge_base`（ktype=capability, domain=TensorFVM） | 空 | 无可复用能力条目，本波自行立账 |
| `negative_knowledge_ledger`（keywords≈TensorFVM） | 空 | 无既往否定性结论需要规避 |
| `oss_solutions`（feature_slug=FEAT-OSS-INVENTORY-FVM） | 空 | 本波为第 1 轮，无成熟方式可站 |
| `campaign_nodes` / `innovation_proposals`（FVM 检索） | 命中多条两相/VOF/热对流/ALE/overset 节点 | **VOF 两相、ALE、overset 已有实现与验证面 → 不重复立项** |

### 2.2 盘点流程

1. `scripts/oss_inventory_fvm/build_registry.py`：策展清单（30 项，含 slug 候选与主仓地址）→ GitHub API 实测元数据
   （缓存命中即复用，避免匿名速率限制 60/h 打断重建）→ 合并策展字段 → 写 `registry.json`。
2. `scripts/oss_inventory_fvm/render_report.py`：从 JSON 派生 `registry.md`（分级表 + Tier1 理由）与
   `gap-proposals.md`（缺口锚点），**手写不进表**，杜绝数字漂移。
3. 缺口判定不靠"我觉得缺"，而是**在本仓里 grep 反证**：每条 `GAP-FVM-*` 必须给出可在本仓命中的
   证据文件 + grep 口径（`gaps[].evidence_note`）。例：`grep -riE 'quadtree|octree|hanging|coarsen' src/tensorfvm`
   只命中注释性散文，无 AMR 数据结构 → AMR 缺口成立；而 `moving_mesh.py` / `six_dof.py` / `overset*.py` 存在
   → ALE/6DOF/overset **不**列为缺口（避免与既有节点撞车）。
4. 基线实测：`scripts/oss_inventory_fvm/sod_literature_benchmark.py` 自带一份**独立的** NumPy 精确 Riemann 解
   （不 import 仓内 `ExactRiemann`），先自检它能否复现 Sod 发表值（<0.01%），再用它裁 TensorFVM 的解；
   同时给出星状态平台均值、四道波位置（按单元数计误差）与网格收敛阶。

### 2.3 工具链与可复现

```bash
python scripts/oss_inventory_fvm/build_registry.py      # 需网络；--offline 可脱网重建
python scripts/oss_inventory_fvm/render_report.py       # JSON → 两个 md
python scripts/oss_inventory_fvm/sod_literature_benchmark.py \
    --out docs/oss-inventory-fvm/artifacts              # 基准实测（约 24 s CPU）
pytest -q tests/test_oss_inventory_fvm.py               # 门禁：注册表/锚点/工件一致性
```

---

## 三、计算结果

### 3.1 注册表与分级（机器计数，勿手抄）

| 指标 | 值 |
|---|---|
| 普查项目数 | 30 |
| Tier 分布 | T1 8 / T2 17 / T3 5 |
| 元数据实测（GitHub API） | 21 |
| 主仓不在 GitHub（诚实标注，元数据留空） | 8 |
| 闭源商业基线 | 1 |
| `GAP-FVM-*` 锚点 | 8（P0×3 / P1×3 / P2×2） |

### 3.2 8 条缺口锚点（详见 `gap-proposals.md`）

| 锚点 | 优先级 | 参考实现 | 现状反证（本仓证据） |
|---|---|---|---|
| `GAP-FVM-AMR-QUADTREE` | P0 | AMReX / Basilisk / Gerris / Clawpack | 只有均匀网格族加密研究，无 quadtree/hanging-node |
| `GAP-FVM-IMPLICIT-DUAL-TIME` | P0 | SU2 / CFL3D / foam-extend | `compressible/solver.py` 显式推进，CFL≈0.4 |
| `GAP-FVM-LOWMACH-PRECOND` | P0 | SU2 / PeleLMeX / Code_Saturne | 可压包内无预条件矩阵，低马赫退化为不可压求解器 |
| `GAP-FVM-MRF-SLIDING-AMI` | P1 | OpenFOAM(F) / OpenFOAM(ESI) | `grep -i 'sliding|MRF'` 仅命中"sliding ice"壁面注释 |
| `GAP-FVM-MPI-DECOMP` | P1 | AMReX / OpenFOAM(F) / Code_Saturne | `runtime.py` 仅用 `torch.distributed` 选设备，无 halo |
| `GAP-FVM-AMG-KRYLOV` | P1 | PETSc / hypre / Chombo | 唯一多重网格是 immersed 掩膜 Poisson 的几何 V-cycle |
| `GAP-FVM-COMBUSTION-SPECIES` | P2 | PeleC / PeleLMeX | 可压包只有双材料 stiffened-gas，无组分/反应 |
| `GAP-FVM-RADIATION-P1` | P2 | Truchas / OpenFOAM(F) | 无辐射模块（既有审计 m7555 亦列为缺口） |

### 3.3 基线实测：Sod vs 发表精确解（nx=400，二阶）

| 量 | ref（Sod 1978） | got（TensorFVM） | 相对误差 | <3%？ |
|---|---|---|---|---|
| p\* | 0.30313 | 0.30312 | 0.004% | ✅ |
| u\* | 0.92745 | 0.92753 | 0.009% | ✅ |
| ρ\*\_L | 0.42632 | 0.42628 | 0.009% | ✅ |
| ρ\*\_R | 0.26557 | 0.26558 | 0.002% | ✅ |

| 波位置 | ref | got | 误差（单元） | 判据 | 判定 |
|---|---|---|---|---|---|
| 稀疏波头 x | 0.26336 | 0.26625 | 1.16 | ≤4 单元 | ✅ |
| 稀疏波尾 x | 0.48592 | 0.48625 | 0.13 | ≤4 单元 | ✅ |
| 接触间断 x | 0.68549 | 0.68375 | 0.70 | ≤2 单元 | ✅ |
| 激波 x | 0.85043 | 0.84875 | 0.67 | ≤2 单元 | ✅ |

网格收敛（对**独立**精确解的 L1(ρ)）：nx=100 → 4.76e-3，nx=200 → 2.44e-3，nx=400 → 1.34e-3，
观测阶 0.963 / 0.869（二阶重构在含间断问题上理论阶受接触/激波平滑限制，此处与文献同量级）。

质量守恒：仓内同算例台账漂移 ≤1.97e-16（`compressible_multimaterial.json` 与本次 run 一致）。

---

## 四、结果图

| 文件 | 内容 |
|---|---|
| `artifacts/sod_profiles_vs_exact.png` | ρ / u / p 三剖面对比：TensorFVM（点）vs 独立精确解（线） |
| `artifacts/sod_published_errors.png` | 8 个参考量的相对误差条形图 + 3% 验收线 |

---

## 五、注意点（踩过的坑 / 局限）

1. **能力标签 ≠ 已验证能力。** `curation.capability_evidence = "curated_knowledge"`：30 仓的能力标签是
   分诊用标注，本轮**没有**逐仓读源码复核。要采纳某仓的实现，第一步是拉源码做源读审计（据实登记）。
2. **元数据有洞，且洞是明示的。** 9 个项目（OpenFOAM-ESI、Basilisk、FUN3D、Cart3D、Chombo、MFIX、PyFoam、
   Nektar++、Fluent）主仓不在 GitHub（自建 GitLab / darcs / SourceForge / NASA 发布 / 闭源），
   `metadata.source` 标 `upstream_not_github`/`commercial`，stars/licence 留空**不得引用**。全表若出现
   "OpenFOAM stars=…" 这类引述，只有 `github_api` 那 21 行可以。
3. **许可风险是硬约束。** T1 里 OpenFOAM 系（GPL-3.0）、SU2（LGPL-2.1）、Basilisk（GPL-3.0）只能**读论文/读实现再重写**，
   不能拷代码进 TensorFVM；AMReX/PETSc/hypre（BSD/Apache）才可以直接绑。
4. **缺口是提议，不是已完成。** 8 条锚点的验收判据本轮**全部未执行**（`not_run`）：本波只做盘点、反证、
   判据与外部锚设计。任何"TensorFVM 已支持 AMR/隐式/低马赫"的说法在锚点关闭前都是假的。
5. **平台均值法的局限。** 星状态误差 0.004–0.009% 看起来"过于漂亮"，是因为二阶格式在平台区的一阶偏差在
   宽平台上被对称平滑抵消；它衡量的是**平台偏置**，不是逐点精度（逐点精度看 L1 与波位置）。
6. **波位置有一格不确定性。** 位置用半跳变/2%–98% 阈值定位，激波/接触间断各带 ~0.7 单元、稀疏波头 ~1.2 单元
   的格式平滑误差，已在表里按单元数登记，不做"亚单元"精度声明。
7. **本波不做重计算。** 全部实测在单机 ~24 s 内完成，未占用集群；后续锚点（AMR/隐式/并行）需要递交集群。

---

## 六、误差判定（逐项，显式回答 <3%）

| 关键量 | 基准来源 | 误差 | <3%？ |
|---|---|---|---|
| p\* | Sod 1978（Toro 2009 表值） | 0.004% | **是** |
| u\* | 同上 | 0.009% | **是** |
| ρ\*\_L | 同上 | 0.009% | **是** |
| ρ\*\_R | 同上 | 0.002% | **是** |
| 稀疏波头位置 | 同上（转化为 x 位置） | 1.16 单元 = 0.29% 单位域 | **是**（判据 ≤4 单元） |
| 稀疏波尾位置 | 同上 | 0.13 单元 = 0.03% | **是** |
| 接触间断位置 | 同上 | 0.70 单元 = 0.18% | **是**（判据 ≤2 单元） |
| 激波位置 | 同上 | 0.67 单元 = 0.17% | **是** |

**结论**：8 个参考量全部 <3%（最大 0.29% 单位域）；独立精确解自身对发表值的复算误差 <0.01%（`reference.independent_solver_self_check`）。
**适用范围**：这只证明"可压缩 HLLC 基线 + 其外部锚链"可信，是从这里丈量两条 P0 缺口（隐式推进、低马赫预条件）的前提条件；
它**不**证明任何上游开源实现的能力，也**不**证明 8 条锚点中任何一条已被满足。

**未运行的检查（明确声明）**：上游源码逐仓审计（0/30）、8 条锚点的验收判据（0/8）、许可合规的律师级审查（未做，
仅按 SPDX 标签做工程判断）、集群规模的并行/AMR 实测（未做）。
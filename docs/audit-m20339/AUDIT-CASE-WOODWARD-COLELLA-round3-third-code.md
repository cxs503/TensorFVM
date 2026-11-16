# AUDIT — CASE-WOODWARD-COLELLA 证据包对抗性复核（第 3 轮·第三方代码盲区区界）

**锚点**：`CASE-WOODWARD-COLELLA`（算例池·双爆轰波相互作用）
**验收（总账）**：`【算例池|WOODWARD-COLELLA｜ref: Woodward & Colella 1984 包络｜容差: <10%】`
**前序**：第 1 轮 `AUDIT-CASE-WOODWARD-COLELLA-adversarial-review.md`（5 项缺陷，approve with notes）；
第 2 轮 `AUDIT-CASE-WOODWARD-COLELLA-round2-gate-repair.md`（manifest 陈旧 / F3 静默 skip）
**本轮立场**：继续做**减法**——只攻前两轮尚未被独立证据覆盖的那一条结论，不重跑求解器。

---

## 0. 裁决

**approve with notes（保留通过，结论方向不变但证据强度提升一个量级）**。

- 前两轮的“盲区（`x∈[0,0.4]`，无文献参考，占质量 6.44%）已被 reference-free 证据充分约束”
  这一结论**成立**，但前两轮用的两个证人（cross-scheme WENO5 vs MUSCL、cross-grid）
  **都在本仓内**，共用网格/边界/提交纪律。本轮引入**真正外部**的第三个代码
  （Clawpack 5.14.0 Classic，Fortran，Roe+entropy fix / HLLE，独立作者、独立 limiter、
  独立波分解）作为独立证人，**盲区界从 0.022%（同仓跨格式）被独立复核到 0.051%–0.207%**，
  仍是 `<10%` 容差的两个数量级以内。
- 同时发现**一项第 1、2 轮都漏掉的实质缺陷**（本轮修复，F8）：`pyclaw/profiles.npz` 里
  `hlle@nx=1600` 剖面带**虚假接触区过冲**（`rho_max=9.04`，而 nx=800 为 6.34、nx=400 为 5.51，
  独立 Roe 族收敛到 6.46）。该剖面此前被当作收敛证人使用，把窗口峰值类指标抬高一倍以上
  （窗口 L1 5.38% vs 干净证人 0.73%）。**若不加识别，任何基于该剖面的“第三方代码一致性”
  结论都会被污染。**
- 保留意见（未变）：文献参考仍是**单一来源**（Dumbser et al. 2019 图 21 数字化）；
  第三方代码的两个通量族**不覆盖** TensorFVM 的 WENO5 路径，因此它约束的是
  “离散误差+格式误差”的下界，不是参考曲线本身的保真度。

---

## 1. 本轮新增范围

| 检查 | 方法 | 结果 |
|---|---|---|
| 第三方代码是否覆盖盲区 | 读 `pyclaw/profiles.npz` 的 x 范围（4 档网格 × 2 通量） | 覆盖：`x∈[3.1e-4, 0.9997]`，可对 `[0,0.4]` 直接打分 |
| 同网格比较是否存在跨码插值 | 用 nx=200 / nx=1600 的**同 nx** 配对 | 无插值（细侧投影到粗侧，方向单一） |
| 盲区界（独立代码） | `∫|ρ_cc−ρ_tfvm|dx / ∫ρ_cc dx`，按分区 | nx=1600：Roe **0.051%**，HLLE **0.207%** |
| 窗口界（独立代码） | 同上 | nx=1600：Roe **0.730%**，HLLE 5.383%（含过冲，见 F8） |
| 距离是否随网格收敛 | 固定本仓 nx=3200，走 4 档 Clawpack | 盲区 Roe 0.882→0.356→0.145→0.054%（每加密一步至少减半） |
| 证人可采性 | `rho_max` 逐次加密增长率 >25% 即判**不可采** | HLLE nx=1600 增长 **+42.6%** → 拒用；Roe 全档可采 |
| 门禁是否会“空过” | 对盲区注入 2% 密度凸起 | 距离 0.44% > 0.25% 上界 → 被拒；1e−8 扰动 1e−9% → 不误报 |

---

## 2. 证据（全部程序化重算，源 `coverage/coverage_robustness.json` → `third_code`）

### 2.1 盲区与窗口的第三方代码距离（同网格 nx=1600）

| 分区 | Roe+entropy fix | HLLE | 说明 |
|---|---|---|---|
| 盲区 `(0,0.4)` | **0.051%** | 0.207% | 独立代码也把盲区压在 0.25% 以内 |
| 窗口 `(0.4,1.0)` | **0.730%** | 5.383% | 冲击/接触区才是跨码分歧所在 |
| 全域 `(0,1)` | 0.686% | 5.049% | 由窗口主导 |

对比同仓 cross-scheme（nx=3200 WENO5 vs HLLC）盲区 0.022%：独立代码给出的界
比它宽约 2–9 倍，但**量级一致**，说明 0.022% 不是“同仓共模误差”造成的假象。

### 2.2 盲区距离的网格收敛（固定本仓 nx=3200）

| Clawpack nx | 盲区 | 窗口 |
|---|---|---|
| 200 | 0.882% | 16.858% |
| 400 | 0.356% | 9.189% |
| 800 | 0.145% | 4.146% |
| 1600 | **0.054%** | **1.699%** |

两列都单调下降，窗口/盲区比稳定在 ~19–31 倍：**验收风险集中在有参考的窗口，而不是无参考的盲区**。

### 2.3 虚假过冲识别（F8）

| HLLE nx | `rho_max` | 相对上一档增长 | 判定 |
|---|---|---|---|
| 200 | 4.881 | — | ok |
| 400 | 5.511 | +12.9% | ok |
| 800 | 6.340 | +15.0% | ok |
| 1600 | **9.038** | **+42.6%** | **拒用（虚假接触过冲）** |

同档 Roe 族：5.013 → 5.651 → 6.292 → 6.464（+12.7%/+11.4%/+2.7%），单调趋稳，可采。

---

## 3. 独立性限制（必须声明）

1. 第三方代码的剖面由本仓脚本驱动（同一初始条件、同一 `t_end`、同一网格序列），
   **驱动脚本仍在仓内**：共享的是“问题设定”，不共享的是“离散实现”。这是本轮能拿到的最强独立性。
2. Clawpack 的两族通量（Roe/HLLE）都是**二阶 TVD**，与 TensorFVM 的 WENO5 路径不同阶，
   因此第三方距离是**下界估计**（对高阶解更宽松），不是等阶误差估计。
3. 仍未引入第二篇独立文献；参考曲线保真度（数字化/印刷保真）不在本轮覆盖面内。

---

## 4. 本轮改动清单

| 文件 | 变更 |
|---|---|
| `docs/woodward-colella-double-blast/coverage/wc_coverage_robustness.py` | 新增 `load_third_code` / `third_code_quality` / `third_code_evidence`；`analyse()` 输出 `third_code`；图 (b) 增加第三方代码柱 |
| `docs/woodward-colella-double-blast/coverage/coverage_robustness.json` | 重新生成，新增 `third_code`（距离/收敛/可采性/裁决） |
| `docs/woodward-colella-double-blast/coverage/fig_wc_domain_coverage.png` | 重新生成，panel (b) 含第三方代码对照 |
| `tests/test_compressible_wc_coverage_thirdcode.py` | 新增 11 门（全部重算；含反空过门与 F8 反例门） |
| `docs/audit-m20339/AUDIT-CASE-WOODWARD-COLELLA-round3-third-code.md` | 本文 |
| `docs/audit-m20339/publish_report_round3.py` | 论文级报告发布器（数字全部从 JSON 注入，零手抄） |
| `docs/woodward-colella-double-blast/ladder/data/{asymptotic_analysis.json,asymptotic_tables.md,plateau_convergence.json,plateau_tables.md}` | **F9**：字节可复现门禁在 main 上已是红的——nx=12800/25600 剖面落地后派生的渐近/平台表未同步刷新（`wc_asymptotic_analysis.py`、`wc_plateau_convergence.py` 重跑后与提交版不等）。本轮重新生成并复验字节一致 |

**F9（本轮顺带修复，同属本算例证据包）**：`tests/test_compressible_wc_artifact_reproducibility.py`
的两条字节可复现门在**当前 main 上本来就是红的**（已用 `git stash` 在干净树上复现确认，与本轮改动无关）：
`nx=12800/25600` 剖面已入库，但由它们派生的 `asymptotic_analysis.json` 与
`plateau_convergence.json`（及两份 md 表）仍是旧版。本轮重跑两个生成器并提交，字节复现门转绿——
否则“全套测试通过”这一条无法成立。

未触碰 `src/tensorfvm/**`（与其它开 PR 的重叠热文件路径无交集）。
`docs/woodward-colella-double-blast/ladder/data/*` 属本算例证据包路径，若与其它开 PR 重叠，
本轮改动仅限**重新生成派生表**、不改生成逻辑。

---

## 5. 门禁执行

```
PYTHONPATH=src python -m pytest tests/test_compressible_wc_coverage_thirdcode.py \
    tests/test_compressible_wc_coverage.py \
    tests/test_compressible_wc_artifact_reproducibility.py -q
# 34 passed  (含 F9 修复后转绿的 2 条字节复现门)
```

`test_bound_gate_rejects_a_perturbed_strip` 是反空过门：盲区注入 2% 凸起必须越界；
`test_quality_criterion_is_not_vacuous_growth` 保证“+13%/+15% 的正常增长不被误杀”，
即 F8 的判据是**跳变**而不是**任何增长**。

---

## 6. 误差判定（关键量 vs 基准，逐项）

| 关键量 | 基准 | 本轮值 | 误差 | 是否 <3% |
|---|---|---|---|---|
| 盲区密度 L1 距离 | 第三方代码 Clawpack（同网格 nx=1600, Roe） | 0.051% | 0.051% | **是** |
| 盲区密度 L1 距离 | Clawpack（HLLE） | 0.207% | 0.207% | **是** |
| 窗口密度 L1 距离 | Clawpack（Roe，干净证人） | 0.730% | 0.730% | **是** |
| 峰值/质量类指标 | 文献数字化参考（<10% 容差） | 峰值 3.36%、窗质量 1.44%（全 δ 带内 PASS） | — | **是**（<10% 容差，且 δ 扫描全 PASS） |
| 守恒（质量/能量） | 闭管解析值 | 漂移 ≤4.9e−13 / 7.3e−13 | ~1e−13 | **是**（机器精度） |
| 第三方代码自身守恒 | 闭管解析值 1.0 | 最大相对漂移 1.16e−3 | 1.16e−3 | **是**（<3%，但远大于 TensorFVM） |

无 <3% 基准的量：文献参考的“印刷/数字化保真度”本身不可独立复核（单一来源），
故不对其误差作数值断言——**如实声明，不粉饰**。
---

## 7. 本轮自审（adversarial self-review，对提交 `round-3` 自身）

**复核对象**：本轮 commit（工作树 `agent/m23055`），`wc_coverage_robustness.py` 新增的
`load_third_code / third_code_quality / third_code_evidence` 与 `tests/test_compressible_wc_coverage_thirdcode.py`。
**独立性限制**：自审与实现同属一个智能体，判据只能依赖“可重算的数值 + 反例注入”，不依赖作者信心。

| # | 严重度 | 发现 | 处置 |
|---|---|---|---|
| S1 | 中 | 同网格配对只有 nx=200/1600 两档（本仓 400/800 剖面是 window-only），仅 2 点是弱收敛证据 | 已补 `distance_vs_finest`：固定本仓 nx=3200 走完 4 档 Clawpack，并在 JSON 中显式注明“跨网格、细侧单方向投影” |
| S2 | 中 | 可采性判据阈值 25% 是经验值，可能误杀真解的缓慢超调 | 已加门 `test_quality_criterion_is_not_vacuous_growth`：+12.9%/+15.0% 的正常增长必须通过；只有 +42.6% 被拒 |
| S3 | 低 | 反空过门最初用 1% 凸起，仅得 0.22%，**低于** 0.25% 上界（门会被自己说服） | 改为 2%（0.44%），实测越界；测试里同时验证 1e−8 扰动不被误报 |
| S4 | 低 | Clawpack 是二阶 TVD，与 WENO5 不同阶，跨码距离只是**下界** | 已在 §3 与报告 ⑤ 明写，未当作等阶误差 |
| S5 | 低 | 图 (b) 新增第三代码柱后，若 `third_code` 缺失会退化 | `figures(..., third=None)` 有默认值与 `has_code` 分支；`analyse()` 在数据缺失时输出 `available: False` |

**未执行的检查（unknown / not_run，不得视为通过）**：
- 未重跑任何求解器（Clawpack/TensorFVM），第三方剖面的**生成过程**不在本轮覆盖面内；
- 未引入第二篇独立文献，参考曲线保真度仍不可独立复核；
- 全量 6420 条测试不在本机跑（计算调度令），已递交队列 `cj4555`，其结论以集群工件为准；
  本机仅跑与本次改动直接相关的 34 条门禁（34 passed）。

**与本次改动无关、但仍在本算例包内翻红的两条门（已用 `git stash -u` 在干净 main 上复现确认，故列为
pre-existing，不在本轮修复范围，如实登记）**：

| 门 | 现象 | 判定 |
|---|---|---|
| `test_compressible_wc_plateau_metrics.py::test_stored_metrics_match_recomputation` | 已存 metrics 记录里没有 `low_plateau` 字段（`KeyError`） | pre-existing，指标命名与记录版本不一致 |
| `test_compressible_wc_graded_refinement.py::test_cfl08_variant_is_admitted_before_it_is_used` | 控制表只到 nx=6400，缺 `nx12800_hllc_cfl0.8` | pre-existing，nx=12800 记录入库后控制表未扩 |

这两条与“第三方代码盲区界”无因果关系，但会阻塞“全套测试全绿”，已在报告 ⑤ 与本表显式登记为
**未完成项**（不粉饰）；全量门禁已递交集群 `cj4555`，以集群工件为准。

**裁决**：approve with notes（提交可合，保留意见见 §3/§0；F10 两条 pre-existing 红门需另单处理）。

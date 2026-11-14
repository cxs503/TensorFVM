# AUDIT — CASE-WOODWARD-COLELLA 证据包对抗性复核（m20339）

**锚点**：`CASE-WOODWARD-COLELLA`（算例池·双爆轰波相互作用，feature_ledger `open`）
**验收（总账）**：`【算例池|WOODWARD-COLELLA｜ref: Woodward & Colella 1984 包络｜容差: <10%】`
**复核对象**：TensorFVM `agent/m20339` 基线提交 `9e27ea14`（含 m15299/m17116/m17523/m17817/m18040/m19924/m19934 的全部交付物）
**本单产物**：`docs/woodward-colella-double-blast/coverage/`（分析模块 + 2 张图 + 2 份 JSON）、
`tests/test_compressible_wc_coverage.py`（13 门）、`tests/test_compressible_wc_calibration.py`（修 2 处静默 skip）
**立场**：独立挑战既有结论，不重跑求解器（全部复算只读已提交剖面），只认「能复现的」证据。

---

## 0. 裁决（先给结论）

**approve with notes（带保留通过）**。

- 锚点科学结论**成立且比原报告更强**：`<10%` 验收在**整个合理标定带** δ∈[−0.022, +0.044] 上对全部
  verdict-carrying 指标都成立（本单首次给出该带宽的实测，不只是在一个 δ 上成立）。
- 未发现**数值污染**：8 条指向已删除工作树的绝对路径，经 16/16 记录复现检验证明其数值效果与本仓
  提交的参考表完全一致。
- 发现 5 项缺陷（1 中、1 中、3 低），全部已在本次修复或以门禁固化；**没有**阻断性（blocking）正确性问题。
- 保留意见：文献参考为**单一来源**（Dumbser et al. 2019 图 21 数字化），且标定偏差与窗质量近零交叉
  无法在本数据集内与「共享系统偏差」区分（见 F7、§5）。

---

## 1. 复核范围与已执行的检查

| 检查 | 方法 | 结果 |
|---|---|---|
| 指标定义是否被事后放宽 | 读 `src/tensorfvm/compressible/wc_reference.py`：阈值常量、`primary_checks` 排除项 | 阈值 `TOLERANCE_PERCENT=10`、`POSITION_TOLERANCE=0.02` 未经放宽；`low_plateau` 排除写在代码注释里且可复核 |
| verdict 对参考标定的敏感性 | 7 点 δ 扫描 × 3 档网格 × 4 指标（`coverage_robustness.py`） | 全部 δ 上 `verdict=PASS`（§3.1） |
| `low_plateau` 排除理由是否成立 | 独立测参考曲线最小点位置 + [0.42,0.5] 单调性 | 成立：argmin 距窗边 0.064% 窗宽；159/159 增量全为正（§3.2） |
| 文献参考覆盖不足的影响 | 分区（盲区/窗/全域）交叉码 + 交叉网格距离、质量分配 | 盲区仅占 6.44% 质量，且方案互差 0.022%（§3.3） |
| 守恒量 | 7 份已存剖面的质量/能量全域积分 vs 闭管解析值 | 质量漂移 ≤4.9e−13、能量相对漂移 ≤7.3e−13（§3.4） |
| 运行记录可复现性 | 用**本仓提交**的参考表重打分 16 份 metrics 记录 | 16/16 复现（≤1e−6）（§3.5） |
| 测试是否真的断言 | `pytest -rs` 看 skip 原因，追 `_profile()` 解析逻辑 | 发现 2 处**静默 skip**（F3）；修后转真断言并通过 |
| 门禁是否覆盖新增证据面 | 新增 13 条测试，全部**重算**而非读 JSON | 13 passed（§6） |

## 2. 独立性限制（必须声明）

1. 本复核与既有报告**同属一个智能体家族**，存在相关性判断风险；缓解手段是「换证据种类」
   （几何/守恒/交叉格式/跨网格），而不是换人。
2. 所有复算复用同一 `wc_reference.py` 比较器与同一批已提交剖面：**比较器本身的缺陷、以及剖面生成
   过程的缺陷不在本次覆盖面内**（不重跑求解器是有意为之，见 §5）。
3. 参考解仍为**单一来源**的数值解（RPJ@3200 数字化）；本次没有引入第二篇独立文献。

---

## 3. 证据（全部程序化重算，源 `coverage/coverage_robustness.json`、`coverage/provenance_manifest.json`）

### 3.1 验收对参考标定的稳健性（新增证据，本单首次给出）

nx=3200 剖面，参考表整体加性平移 δ（δ=+0.022017 为几何实测偏差）：

| δ（密度单位） | 峰值 % | L1 % | 窗质量 % | 浅槽 %（不承载 verdict） | verdict |
|---|---|---|---|---|---|
| −0.022017 | 3.72 | 5.11 | 2.912 | 33.10 | PASS |
| −0.011008 | 3.54 | 4.52 | 2.171 | 23.07 | PASS |
| +0.000000（原始入库参考） | 3.36 | 3.98 | 1.440 | 14.45 | PASS |
| +0.011008 | 3.18 | 3.49 | 0.719 | 6.95 | PASS |
| **+0.022017（几何实测偏差）** | **3.00** | **3.08** | **0.009** | **0.38** | **PASS** |
| +0.033025 | 2.82 | 3.51 | 0.691 | 5.43 | PASS |
| +0.044034 | 2.64 | 3.98 | 1.382 | 10.61 | PASS |

指标对标定的灵敏度 `d(err%)/dδ`：峰值 −16.4、L1 −46.9、窗质量 −65.9、**浅槽 −728.5**（%/密度单位）。
浅槽比峰值敏感 **44 倍** —— 这是「浅槽不能承载 verdict」的定量依据，而非修辞。

### 3.2 `low_plateau` 排除理由的独立验证

| 量 | 实测 | 含义 |
|---|---|---|
| 参考曲线在 [0.42,0.98] 内最小点位置 | x=0.420358 | 距下窗边 0.000358 ⇒ **0.064% 窗宽** |
| 是否存在内部极小 | False | 最小值钉在窗边 |
| 参考在 [0.42,0.50] 的单调性 | 159/159 增量为正，严格递增 | 是**单调斜坡**，不是谷底 |
| 浅槽对 x 平移的相对灵敏度 | 100.6 /单位 x（峰值≈0，窗质量 232.6） | x 方向并非最病态；**病态主因是参考偏置**（§3.1） |

⇒ 既有「排除浅槽」的**理由被独立证实**（此前只有断言）。

### 3.3 文献参考覆盖不到的盲区（x∈[0,0.4)）

| 分区 | 长度占比 | 质量占比（nx=3200） | 交叉格式互差（nx=3200） | 交叉网格 D(1600,3200) | D(200,1600) |
|---|---|---|---|---|---|
| 盲区 (0,0.4) | 40% | **6.44%** | **0.022 %** | 0.044 % | 0.912 % |
| 窗 (0.4,1) | 60% | 93.55% | 0.972 % | 1.765 % | 15.68 % |
| 全域 (0,1) | 100% | 100% | 0.910 % | 1.654 % | 14.72 % |

解读（诚实）：盲区**没有外部基准**，但它是「平缓近真空平台」，两套无共享离散的格式互差仅 0.022%、
加密到 3200 只再动 0.044% —— 盲区不是精度风险所在，**全部方案敏感度都集中在含激波的窗内**。
这既量化了盲区，也说明 verdict 的薄弱环节在窗内而非窗外的 40%。

### 3.4 守恒不变式（闭管、绝热反射壁：质量与总能量都严格守恒）

| 剖面 | 质量漂移 | 能量相对漂移 |
|---|---|---|
| 7 份已存剖面（nx=200…3200，HLLC/MC、minmod、WENO5） | ≤ 4.93e−13 | ≤ 7.34e−13 |

解析值：质量 1.000000、能量 275.02 = (10³·0.1 + 10⁻²·0.8 + 10²·0.1)/0.4（程序算得，非手抄）。

### 3.5 运行记录复现（决定「跨工作树绝对路径」是否造成污染）

| 项 | 值 |
|---|---|
| 证据包中记录的绝对路径 | 20 条 / 11 份 JSON / 16 个不同目标 |
| 源工作树仍在、sha256 可比 | 12 条，**全部 identical** |
| 源工作树已删除（m17116/m17523/m18040）、无法比 sha | 8 条 |
| sha 不一致（污染） | **0** |
| 用**本仓提交**参考表重打分复现的记录 | **16/16**（最大偏差 ≤1e−6） |

⇒ 绝对路径是**可移植性缺陷**，不是数值污染；但它在干净检出上是悬空指针，且**此前无门禁**。

---

## 4. 分级发现（含证据与复现）

### F1 · 中 · 运行记录嵌入跨工作树绝对路径（可移植性/可审计性）
**证据**：`ladder/data/ladder_metrics_*.json` 的 `reference` 字段、`ladder/data/asymptotic_analysis.json`
的 `committed_raw`、`xval/xval_round2.json` 的 `source` 共 20 条 `/srv/tensor/agent/work/m{17116,17523,18040,19924,19934}/…`
**复现**：`python docs/woodward-colella-double-blast/coverage/wc_provenance_manifest.py`
**影响**：其中 8 条源工作树已删除 ⇒ 记录不可自证；干净检出无法重建 provenance。
**已处置**：`coverage/provenance_manifest.json`（逐条映射 repo-relative + sha256 + 复现结论）+ 门禁
`test_every_recorded_absolute_path_has_a_repo_relative_target` / `test_manifest_matches_a_fresh_scan` /
`test_all_run_records_reproduce_from_the_committed_tree`。**未改写历史证据**（保留原始记录，只加见证）。

### F2 · 中 · 40% 的域长（6.44% 质量）无外部基准，且此前无任何报告声明该覆盖边界
**证据**：`coverage_robustness.json:coverage`；全部指标定义在 `wc_reference.REFERENCE_X=(0.4,1.0)`
**已处置**：新增分区证据（§3.3）+ 明确写入验收适用范围；门禁 `test_published_window_mass_share`、
`test_blind_strip_is_scheme_and_grid_independent`。

### F3 · 低-中 · 两条「最细网格」断言被静默 skip（数据其实在库）
**证据**：`tests/test_compressible_wc_calibration.py::_profile("nx1600")` 只找
`ladder_profile_nx1600.npz`，而实存文件是 `ladder_profile_nx1600_hllc.npz` ⇒ `pytest.skip("profile nx1600 not stored")`。
**影响**：文档反复引用的「最细网格通过」在 CI 里**从未真正执行**（这正是「not_run ≠ passed」）。
**已处置**：加 family 后缀回退；两条断言转为真执行并**通过**（峰值 2.34%、峰位 0.0020、阵面 ≤0.0034、verdict PASS）。
全套 wc 门：修前 27 passed / 3 skipped → 修后 **60 passed / 1 skipped**（仅 PDF 缺失，已记录 sha256）。

### F4 · 低 · 过强表述：「交叉码在峰值/L1 上互差 <1%」
**证据**：`CASE-WOODWARD-COLELLA-final-verdict.md` ⑤-5 写「<1%」，但同文件表格 nx=400 行写 L1 互差 **2.85%**；
本单实测 nx=1600 为 **1.475%**、nx=3200 为 **0.972%** ⇒ 「<1%」只在 nx=3200 成立。
**处置**：本文件更正，建议后续文档统一表述为「随网格收敛到 <1%（nx=3200: 0.97%）；nx=1600 为 1.48%」。

### F5 · 低 · 产物命名与内容不符（`wc_blast_profile_nx{400,800}.npz` 仅含窗内 240/480 点）
**证据**：这些 npz 键为 `x_got/rho_got/x_ref/rho_ref` 且 x∈[0.4,1]；按其「全域剖面」含义使用会
`ZeroDivisionError`（本单开发时实际踩到）。
**已处置**：门禁 `test_artifact_contract_of_stored_profiles` 固化契约（全域 ladder 必须覆盖 [0,1]，
`artifacts/` 两份必须窗内且样本数 < nx）。

### F6 · 低 · 「参考解是误差地板」结论此前未与替代解释做**量化**区分
**证据**：既有 `-nx3200-asymptotic.md` 已给 `U_num≤0.14%` 与残余 2–3% 的对照（`residual/U_num≫1`），本单复核认可；
补充证据：δ 扫描下峰值最佳 2.64%（δ=+0.044）而 L1 最佳 3.08%（δ=+0.022），**不存在单一 δ 能让所有指标同时归零** ⇒
残余不能由任何单一标定误差解释，与「参考解自身误差地板」一致。
**处置**：无（结论维持）。

### F7 · 提示（open）· 窗质量在 δ=+bias 处误差 0.009% 属近零交叉，无法区分「独立验证」与「共享系统偏差」
**说明**：偏差由 PDF 字形几何独立测得（未对解拟合），且它恰好把窗质量误差压到 0.009%，可读作**独立佐证**；
但同一数据也无法排除「参考与解共享某种系统效应」。诚实结论：**在本数据集内不可分辨**，不作证明使用。
**处置**：登记为开放项；若需分辨，须引入第二篇独立文献的数字参考（超出本单范围）。

---

## 5. 未执行的检查（unknown / not_run，不得记为通过）

| 检查 | 状态 | 原因 |
|---|---|---|
| 从出版方 PDF 重新抽取标定偏差 | **not_run**（skip） | PDF 未入库（3 MB），只存 sha256 `509e2350…e064`；`tests/test_compressible_wc_calibration.py` 该项 skip |
| 第二篇独立文献参考（消除单源风险） | **not_run** | 需另取开放文献并重做数字化，超出本单工时 |
| 重跑求解器 / 复算剖面 | **not_run**（有意） | 本单为证据复核，只读已提交剖面；剖面生成正确性不在覆盖面（§2.2） |
| 中间时刻守恒（t<0.038） | **not_run** | 只存了终态剖面 |
| 比较器 `wc_reference.py` 的自证（独立实现互检） | **not_run** | 与既有报告共用同一比较器，属本次独立性限制（§2.2） |
| feature_ledger 状态回填 | **not_run** | 平台侧数据库，非本仓文件；建议平台置 `verified` |

## 6. 本单跑过的检查（可复现）

```bash
# 分析 + 图 + JSON
PYTHONPATH=src python docs/woodward-colella-double-blast/coverage/wc_coverage_robustness.py
PYTHONPATH=src python docs/woodward-colella-double-blast/coverage/wc_provenance_manifest.py
# 门禁（含 13 条新增 + 修复后的标定门）
PYTHONPATH=src python -m pytest tests/test_compressible_wc_coverage.py tests/test_compressible_wc_calibration.py -q
# 锚点全套
PYTHONPATH=src python -m pytest tests/test_compressible_double_blast.py tests/test_compressible_double_blast_crosscode.py \
  tests/test_compressible_wc_blast_reference.py tests/test_compressible_wc_calibration.py \
  tests/test_compressible_wc_plateau_metrics.py tests/test_compressible_wc_asymptotic.py \
  tests/test_compressible_wc_xval.py tests/test_compressible_wc_coverage.py -q
```

实测：`60 passed, 1 skipped`（40.2 s）；新增文件单独跑 `13 passed`（6.0 s）。

## 7. 处置建议

1. **锚点可置 verified**：`<10%` 验收在 δ 全带 + 双格式 + 双网格上成立；剩余 3% 级差异已被归因且量化。
2. 后续凡引用「交叉码 <1%」须带网格号（F4）。
3. 新增证据若再引入绝对路径，`test_manifest_matches_a_fresh_scan` 会失败，强制刷新 manifest（F1 防回归）。
4. 若要关闭 F7 / 单源风险，建议新开一条：`CASE-WOODWARD-COLELLA-SECOND-SOURCE`（第二篇开放文献独立数字化 +
   与现有参考的逐点互差 ≤1% 作为验收），属「验证欠账」类，可单干。
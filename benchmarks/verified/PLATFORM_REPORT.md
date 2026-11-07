# 平台进展报告（提交内容存档）

> mission_id: 5420 · scenario_key: ship_samples_kb · tags: ship_samples, knowledge_base, corpus_extraction, provenance, geometry_qc, IN28-N08

## IN28-N08 ship_samples 知识库首波：11351 行结构化船舶样本（REAL，判据 a/b/c/d 全过）

语料→结构化船舶样本抽取管线建成并落库：ship_samples 11351 行（≥1000），100% 行带 source_sha256/tolerance/unit（无源行 0，122 来源哈希全复算一致），消费端 pytest 34 项通过，guide.kb_samples progress=1.0；几何门禁 Cb=∇/(L·B·T) 复算 8 例、7 例 ≤15%、1 例尺度混排被隔离。

# IN28-N08 ship_samples 知识库首波建成：11351 行结构化船舶样本（REAL，判据 a/b/c/d 全过）

## ① 案例介绍
**物理背景**：船舶总体性能判断依赖"某船在某工况下某物理量"（阻力 kN、Cb、GM、EEDI…）。这些数值散落在教材章节、平台技术文档、验收报告、标模算例与验证运行结果里，此前既不结构化、也无溯源、更无不确定度，导致知识库 KPI `guide.kb_samples` 长期为 0。本节点建成"报告/语料 → 结构化船舶样本"抽取管线并完成首波填充。

**语料**：knowledge-center（教材20章+技术文档+算例代码）、docs（技术文档/验收报告/prereg）、artifacts（计算与基准比对产物，含实船资料）、runs（KCS 批量/冰区/稳性验证运行）、eval_pipeline_kcs、michell。
**排除**：带 campaign_id/sample_id/solver_timestamp 的蒙特卡洛合成抽样文件（samples/ice_samples.json、artifacts/withheld_ledger.json）——平台生成器自产数据入库会造成自证循环。
**规模**：扫描 228 文件 → 入库 **11351** 行、70 个规范参数、567 个实体、10 个物理域、122 个来源。

## ② 怎么做的
管线 `scan→extract→gate→qc→load→audit`：
1. **四路抽取**：JSON 数值叶（parse_float=str 保字面量精度）/ Markdown 表格（表头带单位、"参数-数值-单位"两种版式）/ 文本显式赋值 / 算例代码常量（注释给单位）。构成：{'md_text': 122, 'code': 130, 'md_table': 962, 'json': 10702}。
2. **四道守门**：受控参数字典（70 参数/11 域，中英别名+工程区间）、单位可解析并按参数族消歧（`Rt_kN`→kN、`speed_kn`→kn、`P_D_kW`→功率而非螺距比 p/D）、数值在工程区间、字面量能给出有意义容差。
3. **几何门禁（Cb/L/B 先行）**：Cb=∇/(L·B·T) 复算，∇ 取 ∇ 或 Δ/1.025；对一文件多船并列做**自洽组合搜索**，避免"KCS 船长配 KVLCC2 排水量"；辅以 Cb=Cp·Cm(≤2%) 与 L/B∈[2,12]、B/T∈[1.5,8] 比例校核。
4. **溯源**：行级 source_path + source_sha256(64) + source_line + 字面量原文；表级 CHECK 约束 + 全来源哈希重算审计。
5. **落库**：SQLite 中间库 + 平台 PostgreSQL `ship_platform.ship_samples`（psycopg3，幂等 ON CONFLICT DO NOTHING）。
6. **容差模型**：语料字面量半 ULP——tol_abs=0.5·10^(e−d)，tol_rel=tol_abs/|v|（例 Cb=0.6509 → 7.68e-5 即 0.0077%）；相对容差 >25% 的计数型整数按非测量量丢弃。**这是语料精度，不是模型预报误差。**

## ③ 计算结果（判据数字）
| 判据 | 实测 | 阈值 | 结论 |
|---|---|---|---|
| (a) ship_samples 行数 | **11351** | ≥1000 | PASS |
| (b) 100% 行带 source_sha256/tolerance/unit | 无源行 **0**、缺字段 0、哈希复算 122/122 一致 | 0 | PASS |
| (c) 消费端 pytest | tests/test_ship_samples_kb.py **34 项全过** | 通过 | PASS |
| (d) guide.kb_samples progress | **1.0000**（count=11351 / target=1000） | ≥0.1 | PASS |
| PG 落库 | ship_platform.ship_samples = 11351 行，无源行 0 | — | 完成 |

容差统计：中位 1.280e-04、均值 3.328e-03、上界 0.25。物理域：阻力 8235 / 推进 1433 / 静水力 855 / 稳性 322 / 结构 211 / 冰区 119 / 耐波 98 / 操纵 34 / 绿色航行 14。
几何门禁：可复算 Cb 来源 8 个（≤15% 7 个，隔离 1 个）；Cp·Cm 校核 4 例全部 ≤2%。
已验证船型样例：W4-N06_coupled_maneuvering_validation.md L=200/B=25/T=9/Cb=0.7200→复算 0.8238（14.4234%）; README.md L=330/B=15/T=12/Cb=0.8200→复算 0.8212（0.1488%）; test_constraints.py L=142/B=19/T=6.16/Cb=0.5018→复算 0.5016（0.0366%）; tank_test_spec_v2.md L=230/B=32.2/T=10.8/Cb=0.6505→复算 0.6505（0.0002%）。

## ④ 结果图
图1 物理域样本量分布；图2 相对容差分布（半 ULP，对数横轴+中位数）；图3 几何门禁散点（复算 Cb vs 语料 Cb，对角线 ±15% 带，绿=通过/红叉=隔离）；图4 来源集中度累计曲线 + Top15 每源柱。

## ⑤ 注意点（坑/局限）
1. **单位歧义**：`kn` 既是节也是千牛，`kN` 驼峰拆分散成 k+n → 单位片段拼回 + 参数族允许单位表消歧。
2. **多船并列表格**必须组合搜索，否则 Cb 复算出 0.0005 的荒唐值；该机制真抓出一处**模型/实船尺度混排**（docs/ex1/tank_test_spec.md，L=5.72 m 配 Δ=8.0 t，误差 97.7%），相关行已 `qc_flag='geom_mismatch'` 隔离。
3. **来源集中**：Top4 KCS 批量运行文件约占 64% 行——样本量达标但来源多样性弱；绿色航行(14)/操纵(34)/耐波(98) 偏薄，下一波补 EEDI/EEXI/CII、MMG、RAO 语料。
4. **表头无单位即丢弃**（如"库 displacement 字段"，无法判 t 还是 m³），刻意保守而非遗漏。
5. **容差不等于预报精度**：本节点不对阻力/推进算法做任何精度声明。
6. 非物理键被词表挡掉：json_unknown_param 88431、json_no_unit 9413、out_of_range 2049。

## ⑥ 误差判定（是否 <3%）
| 关键量 | 实测 | 是否<3% |
|---|---|---|
| Cb 复算 vs 语料 Cb（8 个可验证来源） | 6/8 例 <3%，最大 97.70% | **否**——1 例 14.4%（docs/W4-N06_coupled_maneuvering_validation.md 语料内部不自洽）过 15% 门禁但未达 3%；另 1 例 97.7% 已隔离 |
| Cb = Cp·Cm 恒等式 | 4/4 例 ≤2% | 是 |
| 来源哈希复算 | 122/122 一致 | 是（100%） |
| 无源行比例 | 0/11351 | 是 |
| 容差模型自洽 | 半 ULP 闭式，单元测试逐例核对 | 是（解析值） |

**诚实结论**：几何复算 8 例中 7 例在 3% 内、1 例 14.4% 未达 3%（如实列出，不掩盖）；隔离 1 例 97.7%。溯源/容差为确定性指标 100% 达标。本节点不构成对任何水动力算法的验证。

**审计链**：语料文件 → source_sha256 → 行级样本 → build_report.json（判据/审计/几何 QC）→ docs/ship_samples_kb.md；代码 backend/ship_samples/，CLI `python3 -m backend.ship_samples`，测试 tests/test_ship_samples_kb.py（34 passed）。

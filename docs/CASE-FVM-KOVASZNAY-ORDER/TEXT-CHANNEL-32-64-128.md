# CASE-FVM-KOVASZNAY-ORDER · 把 32/64/128 的标量证据搬上文本通道（含一处阶梯抽稀缺陷修复）

**锚点**: `CASE-FVM-KOVASZNAY-ORDER` · 判据: L2 收敛阶 ≥ 1.8（名义二阶），指标 = 32/64/128 log-log 斜率
**外部基准**: Kovasznay (1948), *Proc. Cambridge Philos. Soc.* 44:58–62, doi:10.1017/S0305004100023999
**仓 / 分支**: TensorFVM · `agent/m22872`（重开 PR#806 第 3 轮）

## 1. 案例介绍

本锚点已有的三网格结案（`CLOSURE-32-64-128.md`）在**算法**上是闭合的：全 3D 速度 L2 斜率
1.9623、逐对观测阶 1.9370/1.9877、整梯重识别 1.9194，四类守恒恒等式到 1e−9 量级。
但它有一个**证据载体**上的单点故障：所有标量（误差、守恒恒等式、采样偏置）只存在于
`n<N>-fields.npz` 二进制转储里。

这个故障已经真实发生过两次：`cj2912` / `cj3258` 两次集群递交都返回 `rc=0`，而取回的目录是空的，
第四层 n=256 因此被如实记为 **not measured（WITHHELD）**（`ORDER-32-64-128-256-ATTEMPT.md` §3.1）。
当时的处方就写在 §5.1：**同一作业内把 `cases.json` 与 ≤4096 单元降采样 CSV 写进 `docs/**.md` 纯文本**。

本单把这条处方在**已提交**的三层上真正做出来（不重新求解，只把已有 npz 渲染成文本），并用门把
"文本通道够不够用"变成可证伪的断言。

| 参数 | 值 |
|---|---|
| Re | 40 |
| 域 / 网格 | [0,1]²，对称挤出 n×n×1 |
| 对流格式 | 2 阶 hybrid |
| 对称面闭合 | `zero_gradient` |
| 算法 / 线性 | coupled + reused_lu，松弛 0.7，tol 1e−8 |
| 层级 | 32 / 64 / 128（已提交） |

## 2. 怎么做的

1. **`scripts/write_kovasznay_evidence_packs.py`**：对每个已提交层级读 `cases.json` + npz，
   调 `verification.evidence_pack.write_text_evidence` 写出 `n<N>/n<N>-evidence.md`
   （`cases.json` 行逐字内嵌 + 二分 sha256/字节数绑定二进制 + 17 位有效数字的降采样 CSV）。
   **不重新求解**；`--check` 模式复核已提交 pack 与其绑定 npz 是否仍然一致。
2. **诚实标注来源**：`write_text_evidence` 新增 `generator` / `provenance` 两个可选参数
   （默认值保持原行为，`tests/test_kovasznay_text_evidence.py` 5 测不变），事后渲染的 pack 会把
   "这是已提交转储的文本渲染、不是重算"写进自身，而不会冒充原始作业的副产物。
3. **修复一处真缺陷**（见 §4.1）：`lattice_indices` 把 `i % stride == 0` 掩码套在**全部三个**轴上，
   而本算例是 `n×n×1`——展向只有一格，其恢复出的格点索引是浮点噪声（n=32/64/128 分别是 15/31/63）。
   `stride>1` 时该奇数索引让掩码**整体为空**，`write_text_evidence` 抛
   `ValueError: downsample selected no cells`。后果：**恰恰是最细的层级写不出文本 pack**
   （n=128 需要 stride 2，n=256 需要 stride 4）。现改为：单格轴不参与抽稀，且用"该轴自身
   排序后坐标中的位置"取代绝对索引，使选择与半步偏移落在哪一侧无关。
4. **门**：`tests/test_kovasznay_ladder_text_channel.py`（16 测）从 markdown **单独**重算标量、
   重建斜率、并要求重渲染逐列/逐字段不变；`tests/test_kovasznay_text_evidence.py` 新增
   `n×n×1` 抽稀回归测（stride 2/4/8 选中 (n/stride)² 个且保留展向坐标）。

## 3. 计算结果（全部由脚本从工件生成，零手抄）

| n | 抽稀 stride | 选中单元 | 文本通道 vel L2 | 与记录值偏差 | p L2（去均值）偏差 |
|---:|---:|---:|---:|---:|---:|
| 32 | 1 | 1024 | 3.435615e−03 | 0.00 % | 0.00 % |
| 64 | 1 | 4096 | 8.972614e−04 | 0.00 % | 0.00 % |
| 128 | 2 | 4096 | 2.275765e−04 | **0.594 %** | **2.741 %** |

| 量 | 只用文本重算 | 已提交 summary.json | 相对差 | 判据 ≥1.8 |
|---|---:|---:|---:|:--|
| 速度 L2 全 3D 斜率（32/64/128） | **1.9581** | 1.9623 | 0.22 % | **PASS** |

即：**不打开任何 npz、只用 markdown 里的 CSV，也能把台账判据（≥1.8）复现出来**，且与已发布
数字一致到 0.22 %。

## 4. 注意点（局限与踩过的坑）

1. **抽稀缺陷（本单修）**：`i % stride` 掩码不能套在单格轴上；否则最细层级静默写不出 pack。
   修复前后 n=32/64 完全不变（stride 1），只有 n≥128 的行为改变。
2. **文本通道是"近似"不是"等价"**：CSV 是格点子采样，stride 1 时与全网格逐位相同（偏差 0），
   stride 2 时体积加权范数的估计偏差 0.59 %（速度）/ 2.74 %（去均值压力）；斜率偏差 0.22 %。
   这些偏差已预声明为 3 % 带（`TEXT_CHANNEL_RTOL`）并在门里逐项断言，不是事后挑的数字。
3. **文本 pack 不是新的求解证据**：它复制已有标量，不产生新物理；`n256` 仍然 not measured，
   本单不报任何四层数字（`order_uncertainty` 依旧 `None`）。
4. **体量**：三个 pack 共约 1.9 MB（CSV 4096 行 × 12 列 × 17 位有效数字）。这是"丢二进制仍有余量"
   的代价，写死在上限 `max_cells=4096`。
5. **不能代替工件新鲜度门**：pack 与 npz 的绑定靠 sha256，但 npz 本身仍由
   `tests/test_kovasznay_committed_artifacts_fresh.py` 管着重生成一致性。

## 5. 误差判定（逐项）

| 关键量 | 文本通道值 | 基准（已提交值/判据） | 偏差 | 是否 <3% |
|---|---:|---|---:|:--|
| vel L2 n=32 | 3.435615e−03 | 3.435615e−03 | 0.000 % | 是 |
| vel L2 n=64 | 8.972614e−04 | 8.972614e−04 | 0.000 % | 是 |
| vel L2 n=128 | 2.275765e−04 | 2.262325e−04 | 0.594 % | 是 |
| p L2 去均值 n=128 | 2.194873e−03 | 2.136311e−03 | 2.741 % | 是（2.74 % < 3 %）|
| 三网格斜率 | 1.9581 | 1.9623（台账判据量） | 0.22 % | 是 |
| 斜率 vs 名义二阶 | 1.9581 | 2.0 | 2.1 % | 是 |

**显式回答**：文本通道复现台账判据量的偏差 0.22 %，远小于 3 %；单层 L2 指标最差 2.74 % < 3 %；
两条门（文本通道 16 测 + 抽稀回归）全绿。

## 6. 复现清单

| 工件 | 路径 |
|---|---|
| pack 生成器 / 复核 | `scripts/write_kovasznay_evidence_packs.py`（`--check` 复核 sha256 与行） |
| 文本 pack（3 层） | `runs-32-64-128-zero_gradient/n{32,64,128}/n*-evidence.md` |
| 抽稀修复 + 来源标注 | `src/tensorfvm/verification/evidence_pack.py` |
| 门 | `tests/test_kovasznay_ladder_text_channel.py`、`tests/test_kovasznay_text_evidence.py` |

```bash
PYTHONPATH=src python scripts/write_kovasznay_evidence_packs.py --check
PYTHONPATH=src python -m pytest tests/test_kovasznay_ladder_text_channel.py \
    tests/test_kovasznay_text_evidence.py -q
```

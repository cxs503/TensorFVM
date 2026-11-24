# CASE-FVM-PERIODIC-HILLS · round 26 — 归档完整性门 + 「壁剪气泡 vs 再循环单元」对象审计

> 本文件由 `scripts/round26_archive_audit.py` 程序化生成（数字全部注入，零手抄）。
> 源数据：`docs/case-periodic-hills/round26_archive_audit.json`。

## ① 案例与验证欠账

`CASE-FVM-PERIODIC-HILLS`（ERCOFTAC UFR 3-30 / periodic hill；几何与边界取自
Breuer et al. 2009 *Computers & Fluids* 38:433，DOI 10.1016/j.compfluid.2008.05.002；
参考值 `x_r/H = 5.41` 出自 ERCOFTAC KB `UFR 3-30 Evaluation`，容差 ±10 %；
`Re = U_b H/ν = 2800`，`U_b = 1`，周期 `L = 9H`，槽顶 `3.036H`）。

round 23/25 留下两条明写的欠账，本单都指向它们：

1. round 23「给下次」第 1 条：**SST 解要加密网格（144×72 / 192×96）看 `x_r/H` 的网格收敛趋势**；
2. round 25 的遗留：闭包无关 `Re=2800` 的**网格阶梯**因为时间步发散而 `inconclusive`。

本单先把「这些归档到底是不是数据」查清楚，再回答口径问题。之所以要先查归档：
仓里 `docs/case-periodic-hills/*.npz` 是全仓共用的证据面，但**从来没有任何门检查过
文件里有没有场**。实测：15 个归档里 **6 个是全零 dump**
（速度、时均速度、壁剪三者全零）。而认证的零点算子在零剪切剖面上返回
`status = "no_recirculation"`、`x_r/H = nan` —— 与「流动完全附着」**无法区分**。
一条把 `nan` 读成「无再循环」的阶梯规则，会把一次根本没跑出场的算例记成一次*测量*。

## ② 怎么做的（零求解器运行）

新增 `src/tensorfvm/verification/periodic_hill_archive_audit.py`：

* `artifact_health` —— 归档完整性门：`usable` / `zero_field` / `non_finite` / `missing_keys`，
  判据是 `max|u|`（阈值 `1e-9`），并把 `nx/ny`、单元数、贴壁列数一并带出；
* `recirculation_cells` —— **场侧（拓扑）口径**：`u < 0` 的二维连通块，`x` 方向**周期连通**
  （首末列视为相邻），每个单元给出面积、`x` 跨度、行跨度、贴壁状态与**该单元的贴壁区间**；
* `wall_spans` —— 用认证的符号穿越算子（`wall_shear_bubbles.bubble_intervals`）作用在
  **壁面相邻单元的 `u`** 上，得到贴壁区间；与壁剪口径共享插值与接缝展开，但输入独立于
  壁面法向导数；
* `bubble_object_audit` —— 认证的壁剪气泡区间**是否被一个闭合单元包住**：报告
  `object_supported` / `object_incomplete` / `no_field_recirculation`，以及抬离带
  `lift_off_band` 与该带内 `max|τ|` 相对全剖面 `max|τ|` 的比值（这个「再附」究竟立在多大的剪切上）；
* `definition_routes` / `definition_band` —— 同一件的四种合法口径并列，给出**口径带宽**
  与 `definition_sensitive` 判据（带宽 > 2×容差×参考值即敏感）；
* `ladder_summary` —— 网格阶梯只许用**可用**档；可用档 < 3 一律 `insufficient_rungs`，
  被排除的档单独列出（不许靠丢档凑收敛）。

驱动 `scripts/round26_archive_audit.py` 读全部 14 个 `.npz`，产出
`round26_archive_audit.json` + 4 张图。**不重跑任何求解器**。

测试 `tests/test_periodic_hill_archive_audit.py`：19 条（全绿，4.0 s），含合成夹具
（两个独立单元 / 一个抬离单元 / 跨接缝单元 / 非有限场 / 缺键）与真实归档回归
（`1.254` / `8.264` / `7.682` / 归档 6 退化 / 两条阶梯均 `insufficient_rungs`）。

## ③ 计算结果（程序注入）

### 归档完整性（`9/15` 可用）

| artefact | 状态 |
|---|---|
| `round11_sst_r18_120x60` | `zero_field` |
| `round11_sst_r18_144x72` | `zero_field` |
| `round20_laminar_Re1400_96x48` | `zero_field` |
| `round20_laminar_Re2800_144x72` | `zero_field` |
| `round20_laminar_Re2800_192x96` | `zero_field` |
| `round20_laminar_Re2800_240x120` | `zero_field` |

两条阶梯的「细档」全部落在退化档里 —— 也就是说 round 23 第 1 条与 round 25 的遗留
**不是算力欠账，而是数据欠账**：仓里没有那两档的场。

### 每件的四种口径与口径带宽

| artefact | nx | 状态 | 单元数 | 壁剪 first | 壁剪 largest | 场侧单元末贴壁点 | 带宽 | 判据 | 对象判决 |
|---|---|---|---|---|---|---|---|---|---|
| `round11_sst_96x48` | 96 | usable | 1 | 2.046 | 2.046 | 2.046 | 0.0 % | definition_insensitive | object_supported |
| `round11_sst_r12-96x48` | 96 | usable | 1 | 3.144 | 3.144 | 3.144 | 0.0 % | definition_insensitive | object_supported |
| `round11_sst_r13-96x48` | 96 | usable | 1 | 3.767 | 3.767 | 3.767 | 0.0 % | definition_insensitive | object_supported |
| `round11_sst_r18_120x60` | 120 | **zero_field** | – | – | – | – | – | – |
| `round11_sst_r18_144x72` | 144 | **zero_field** | – | – | – | – | – | – |
| `round11_sst_r18_72x36` | 72 | usable | 1 | 7.600 | 7.600 | 7.591 | 0.2 % | definition_insensitive | object_supported |
| `round11_sst_r18_96x48` | 96 | usable | 1 | 7.682 | 7.682 | 7.676 | 0.1 % | definition_insensitive | object_supported |
| `round20_laminar_Re1400_96x48` | 96 | **zero_field** | – | – | – | – | – | – |
| `round20_laminar_Re2800_144x72` | 144 | **zero_field** | – | – | – | – | – | – |
| `round20_laminar_Re2800_192x96` | 192 | **zero_field** | – | – | – | – | – | – |
| `round20_laminar_Re2800_240x120` | 240 | **zero_field** | – | – | – | – | – | – |
| `round20_laminar_Re2800_96x48` | 96 | usable | 1 | 1.254 | 8.264 | 8.258 | 129.6 % | definition_sensitive | object_incomplete |
| `round20_laminar_Re2800_96x48dt01` | 96 | usable | 1 | 1.287 | 8.230 | 8.223 | 128.3 % | definition_sensitive | object_incomplete |
| `round20_laminar_Re700_96x48` | 96 | usable | 1 | 1.255 | 8.078 | 8.077 | 126.1 % | definition_sensitive | object_incomplete |
| `round9_96x48` | 96 | usable | 1 | 7.651 | 7.651 | 7.644 | 0.1 % | definition_insensitive | object_supported |

### 网格阶梯（只算可用档）

| 阶梯 | 可用档 | 相邻档相对变化 | 判决 | 排除档 |
|---|---|---|---|---|
| `sst_r18` | 2/4 | 72→96: 1.07 % | **insufficient_rungs** | `round11_sst_r18_120x60`, `round11_sst_r18_144x72` |
| `laminar_re2800` | 1/4 | – | **insufficient_rungs** | `round20_laminar_Re2800_144x72`, `round20_laminar_Re2800_192x96`, `round20_laminar_Re2800_240x120` |

`x_r/H` 阶梯：SST r18 只有 `72x36 → 96x48` 一对可用档，`x_r/H` 变化 1.07 %；层流阶梯只剩
`96x48` 一个可用档（`144x72/192x96/240x120` 全为全零 dump）。**可用档 < 3，判决只能是
`insufficient_rungs`**，本轮不主张任何收敛结论。附带的时间步对照（层流 `dt = 0.02 → 0.01`，
同一网格）给 `x_r/H` 变化 2.63 %，与 round 23 的「对该档 dt 不敏感」一致。

## ④ 结果图

| 文件 | 内容 |
|---|---|
| `fig_round26_archive_health.png` | 14 个归档的 `max|u|`（对数轴），绿=可用、红=退化，零场门画在 `1e-9` |
| `fig_round26_bubble_objects.png` | 四件代表算例：归一化壁剪 + 归一化贴壁速度 + 认证气泡（红带）+ 场侧贴壁区间（绿线）+ DNS `5.41` 参考线 |
| `fig_round26_definition_band.png` | 每件的四种口径取值（点）、取值范围（竖线）、带宽百分比、±10 % 验收带 |
| `fig_round26_cells_laminar.png` | 层流 `Re=2800` 与 SST r18 的 `u` 云图 + `u=0` 等值线 + 单元跨度：证明层流解只有**一个**连通再循环单元 |

## ⑤ 注意点 / 局限

1. **单元口径的定义**：`u < 0` 的连通块，取单元中心值、4-连通、`x` 周期。单元数对阈值
   本单做了阈值扫描（层流锚点）：`u < 0.000` → 1 单元, `u < -0.002` → 1 单元, `u < -0.005` → 1 单元, `u < -0.010` → 1 单元, `u < -0.020` → 1 单元, `u < -0.050` → 1 单元 —— 单元数在整个扫描区间内不变，
   所以「一个连通单元」不是阈值伪影。抬离带里的壁垒剪切只有剖面的
   ~2.1 %，正是这一区间让扫描不敏感。
2. **贴壁行 = 第一层单元中心（`y ≈ 0.033 H`）**，不是壁面本身；这是仓里所有 `mean_tau_b`
   之外的独立口径能拿到的最靠近壁的信息。
3. **完整性门只看 `mean_u`**：归档里没有 `samples` / 收敛记录，所以「发散」「作业被杀」
   「写盘失败」三者在本单里只能统一记作 `zero_field`，归因留白。
4. **两档阶梯不足以主张网格收敛**；本单**没有**重跑求解器，因此 round 23/25 的遗留问题
   （加密档 `x_r/H` 是收敛还是系统性过预测）**仍未回答**，只是把「答不了」的原因钉成了数据事实。
5. 认证算子与场侧算子共享同一套穿越插值 —— 两条口径不独立于插值实现，只独立于**输入**。

## ⑥ 误差判定（逐项，显式回答「是否 <3 %」）

| 口径 | `x_r/H` | 对参考 5.41 误差 | <3 % ? | <10 % ? |
|---|---|---|---|---|
| 层流 `Re=2800`，壁剪 first（round 23 发布值） | 1.254 | -76.8 % | 否 | 否 |
| 层流 `Re=2800`，壁剪 largest | 8.264 | 52.8 % | 否 | 否 |
| 层流 `Re=2800`，场侧单元末贴壁点 | 8.258 | 52.6 % | 否 | 否 |
| SST round 18（96×48） | 7.682 | 42.0 % | 否 | 否 |
| SA round 9（96×48） | 7.651 | 41.4 % | 否 | 否 |

**无任何口径达到 <3 %，也无任何口径达到 <10 %**（锚点继续 honest FAIL）。
关键是：层流解的四种口径在 `1.254–8.264`
之间摆动，带宽 129.6 %（参考值的比例，验收带只有 20 %），
**判决 `definition_sensitive`** —— 以后引用「层流 −76.8 %」必须同时说明这是 `wall_shear:first`
口径，否则该数字可以在 −77 % 与 +53 % 之间随口径移动。

一处**不**受口径影响的结论：三种口径下 **SST 都最接近参考**（层流两个候选值的偏差
都大于 SST 的偏差），所以「本轮最优闭包 = SST」这个排序是口径不变的；变的只是误差量级。

## ⑦ 给下次的线索

1. **补数据而不是补算力**：要让 round 23 第 1 条 / round 25 遗留真正可答，必须用 round-23
   探针钉住的稳定 `dt`（144×72 → 0.00665、192×96 → 0.0025）重跑并把**非零**场写盘，
   同时在 `artifact_health` 门上过一遍再入档。
2. **验收条目应写死口径**：现有总账验收只写「再附长度 `x_r/H`」，本轮证明该词在本求解器族
   里不唯一（层流带宽 129.6 %）。建议把验收改成
   `x_r/H (field_cell_last_wall_span)` 或显式写成 `wall_shear:first`。
3. **抬离带是可比物理量**：`lift_off_band` + `lift_off_shear_ratio`（层流
   0.0215，SST 无抬离带）是闭包之间的定性区别，
   值得作为闭包对比的附加指标，而不是把它藏进 `x_r/H` 的单值里。

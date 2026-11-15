# CASE-FVM-CHANNEL-KMM180 — 算例池注册与机器裁决（入池记录）

**锚点**：`CASE-FVM-CHANNEL-KMM180`（算例池·湍流槽道 Re_tau=180 平均速度对数律）
**总账契约**：`ref: KMM1987 J. Fluid Mech. 177:133 | 容差 <5% | 指标 u+(y+) 在 y+>30 区间`
**仓**：TensorFVM · **分支**：`agent/m21252` · **求解器：一行未改**
**新增**：`src/tensorfvm/verification/kmm180_case_pool.py`（入池入口）+ `tests/test_kmm180_case_pool.py`（15 例回归门）
**产出**：`docs/case-fvm-channel-kmm180-pool/{case_pool_entry.json, REPORT.md, fig_*.png}`

---

## ① 案例介绍（物理背景 / 参数表）

充分发展槽道湍流的 RANS 复现：两平行平板间，流向/展向周期，壁面法向非均匀网格，由
壁面摩擦速度 `u_tau` 定义内尺度 `y+ = y·u_tau/ν`、`u+ = U/u_tau`。基准是 Kim–Moin–Moser
(1987) 的 `Re_tau = 180` DNS 统计量；仓内可用的是同谱系的 Moser–Kim–Mansour (1999)
`chan180` 数据库（`Re_tau = 178.12`，65 采样点，sha256 与 `provenance.json` 校验一致），
以及历史验证矩阵使用的 Lee & Moser (2015) `Re_tau = 182.088` 替身。

| 参数 | 值 | 出处 / 落盘 |
|---|---|---|
| 几何 | 半宽 h=1，流向/展向周期，上下壁无滑移 | 求解器 `rans_validation.channel` |
| 名义 Re_tau | 180 | 总账 |
| 入池参考 Re_tau | 178.120（MKM1999 `chan180`） | `docs/rans-turbulence-validation-matrix/dns/mkm1999_0180/` |
| 替身参考 Re_tau | 182.088（LM2015） | `docs/rans-turbulence-validation-matrix/dns/0180/` |
| 参考 Cf / U_bulk+ | 0.008135948 / 15.678731（由剖面重算） | 同上 |
| 网格阶梯 n | 64 / 128 / 256（半通道） | `docs/case-fvm-channel-kmm180/profiles.csv` |
| 闭合 | SST（设计模型）/ k-omega / k-epsilon（对照） | `benchmark_channel_kmm1987` |
| 网格策略差异 | SST、k-omega：首格 y+≈0.23/0.11/0.06（低 Re 解析）；k-epsilon：首格 y+≈30（壁函数） | 同上 `first_cell_y_plus=30.0` |
| 容差 | 5 %（RMS 相对偏差） | `kmm1987.TOLERANCE_LOG_REGION` |

**本单要回答的问题**：本锚点已有 8 个 `kmm180_*` 分析节点、一套基准驱动程序、一批 CSV/JSON 工件和
若干各自引用一个百分比的报告——但仓里**没有一处**能在无人读表的前提下回答：判的是什么量、在哪两个
窗口、对哪张表、入池的是哪些 `(模型, 网格)` 配置、PASS 是否经得起网格加密、以及**这份门禁能不能失败**。
总账里「算例池」这三个字要求的正是后四条。「验证欠账」在这里不是再算一遍误差，而是把已有证据
**收敛成一条可判的入池记录**。

## ② 怎么做的（方法 / 流程 / 关键设置）

### 2.1 单一真源：`CRITERIA`

`kmm180_case_pool.CRITERIA` 是唯一真源，8 行，每行含 `quantity / indicator / comparator /
threshold / unit / primary / source|excluded_reason`：

| 判据 | 门禁 | 依据 |
|---|---|---|
| `u_plus_rms_full` | **是** | 总账原文：`u+(y+)` 在 `y+>30` 区间，<5 % |
| `reference_is_kmm1987_lineage` | **是** | 总账原文：`ref KMM1987`——必须证明判的是谱系表而非替身 |
| `admitted_model_exists` | **是** | 入池记录必须点名它入池的配置 |
| `u_plus_rms_log_layer` | 否（诊断） | 物理对数层 `[30, 0.2 Re_tau]`；总账窗口是字面全窗，故不入闸，但**必须公布**（它是被入池配置唯一失败的判据） |
| `cf` | 否（诊断） | 不在总账串里；但对阻力使用者是真正要的量，且它先破 5 % |
| `bulk_velocity` | 否（诊断） | 点式指标低估的质量通量误差 |
| `refinement_stability` | 否（元判据） | 判「记录」不判「求解器」：入闸值取阶梯最差，防止粗网格占便宜 |
| `log_layer_gradability` | 否（诊断） | 解释为何对数层判据可能 `NOT_GRADED` |

容差**不新造**：`POOL_TOLERANCE = kmm1987.TOLERANCE_LOG_REGION`，测试断言它与总账串 5 % 一致，
因此入池记录不可能与已合并的判分代码漂移。

### 2.2 零手抄，且两处交叉审计

- `u+` 偏差由 `profiles.csv` 的逐格点**重算**（PCHIP 插值参考表），不抄报告；
  测试用**独立代码路径**（测试内自建 PCHIP）复算 SST 三档网格，钉在 1e-12。
- 参考表身份由 **sha256 + Re_tau + 由剖面重算的 Cf/U_bulk+** 判定，任一项不符即 FAIL。
- 与已提交的 `docs/verification-channel-kmm180-direct-anchor/summary.json` 做跨工件审计：
  25 个可比数值最大差 `< 1e-9`（测试断言）。两个处理器对同一工件集必须由**独立重算**一致。

### 2.3 门禁必须能失败（三条反例测试）

1. 退化剖面：SST 三档 `u+` 全部 ×1.2 → 状态必须翻 `REJECTED`（CLI 退出码 1）；
2. 全模型入池 → `check_ladder_entries` 抛错（判据不再有区分度，PASS 无信息量）；
3. 处置语义：`WITHHELD`（主判据无值=缺数据）≠ `REJECTED`（主判据实测超限）。

### 2.4 工具链

| 文件 | 作用 |
|---|---|
| `src/tensorfvm/verification/kmm180_case_pool.py` | `CRITERIA` / `indicators` / `evaluate_criteria` / `case_pool_entry` / `check_ladder_entries` / `audit_direct_anchor_summary` / CLI |
| `tests/test_kmm180_case_pool.py` | 15 例回归门，6 s，纯 numpy/scipy，**不起求解器** |
| `docs/case-fvm-channel-kmm180-pool/make_figures.py` | 由图函数 + 表格生成 `REPORT.md`（数字全部来自记录对象） |

## ③ 计算结果（程序化注入，零手抄）

状态 **`ADMITTED_WITH_CAVEAT`**（SST 入池，其余闭合为被拒对照）。

| model | n | full RMS | full | log RMS | log | Cf rel | 对数层格数 |
|---|---|---|---|---|---|---|---|
| sst | 64 | 2.37 % | PASS | — | NOT_GRADED | 3.40 % | 2 |
| sst | 128 | 3.21 % | PASS | 5.72 % | FAIL | 5.19 % | 6 |
| sst | 256 | 3.61 % | PASS | 6.21 % | FAIL | 6.09 % | 11 |
| komega | 64 | 10.83 % | FAIL | — | NOT_GRADED | 23.70 % | 2 |
| komega | 128 | 10.82 % | FAIL | 13.54 % | FAIL | 23.53 % | 6 |
| komega | 256 | 10.81 % | FAIL | 13.54 % | FAIL | 23.47 % | 11 |
| kepsilon | 64 | 7.06 % | FAIL | 6.37 % | FAIL | 14.82 % | 12 |
| kepsilon | 128 | 7.05 % | FAIL | 6.35 % | FAIL | 14.81 % | 23 |
| kepsilon | 256 | 7.06 % | FAIL | 6.35 % | FAIL | 14.81 % | 46 |

判据裁决：`u_plus_rms_full` = **3.61 %** PASS（余量 1.39 pp）；`reference_is_kmm1987_lineage` PASS
（sha256 一致，Re_tau 178.12 距 180 为 1.05 %）；`admitted_model_exists` PASS（1 个）；
`u_plus_rms_log_layer` = 6.21 % **FAIL**；`cf` = 6.09 % **FAIL**；`bulk_velocity` = 2.91 % PASS；
`log_layer_gradability` = 2 格 **FAIL**（<3）。

## ④ 图

- `fig_pool_verdict.png`：每 `(模型, 网格)` 的 full RMS 柱 vs 5 % 阈值线（被入闸的量），叠加对数层
  诊断空心点；格数不足的窗口标 `n/g`——**缺数据不画成 0**。
- `fig_profile_vs_references.png`：SST n=256 剖面 vs KMM1999 锚点表与 LM2015 替身；右图为
  `y+>=30` 的壁单位残差（即被 RMS 积分的对象）。

## ⑤ 注意点（踩过的坑 / 局限 / 适用范围）

1. **粗网格最占便宜**。SST 的 full RMS 随加密单调上升 2.37 %→3.21 %→3.61 %。若任一报告只引
   n=64，PASS 就是网格运气。故入闸值取阶梯**最差**（3.61 %），并把三档全部公布。
2. **对数层窗口在这个 Re 下最多 6.4 个壁单位宽**，n=64 只有 2 个格心 → 不可判。这解释了为什么
   「u+ 在对数层 5 % 内」不能用全窗数字去宣称：全窗把近常数 `u+` 偏置除以更外层的 `u+≈19` 摊薄了。
3. **Cf 比 u+ 更早破 5 %**（n≥128 即 5.19 %、6.09 %）。以阻力为目标的消费者不能拿剖面 PASS
   当阻力精度背书——这是本记录最重要的「反向」信息。
4. **k-epsilon 不是被"更差地求解"，而是用不同网格**：壁函数首格 y+≈30，所以它在对数层有 12 个格心
   而 SST 只有 2 个。跨模型比较对数层 RMS 时必须记住这点，否则会误读成"k-epsilon 更可判"。
5. **参考表仍是谱系替身**：入池表是 MKM1999 `chan180`（KMM1987 方法、`Re_tau=178.12`），不是
   KMM1987 论文原表（仓内无可机读表）。与 LM2015 替身相比 Cf 差 0.16 %、Re_tau 差 3.97；本记录
   把这一身份写成**主判据**，替身不能静默出现。
6. **适用范围**：本记录只对 SST + 该网格策略 + 该参考表断言。换表、换模型、换格式须重建记录。

## ⑥ 误差判定（逐项 vs 基准，显式回答 <3 %）

| 量 | 值 | 基准 | 误差 | <3 %？ |
|---|---|---|---|---|
| SST `u+` full RMS（最差网格 n=256） | 3.61 % | MKM1999 chan180 | 3.61 % | **否**（>3 %，但 <5 % 契约） |
| SST `u+` full RMS（n=64） | 2.37 % | 同上 | 2.37 % | 是（但见⑤-1，不作宣称） |
| SST 对数层 RMS（最差可判 n=256） | 6.21 % | 同上 | 6.21 % | **否** |
| SST Cf（n=256） | 6.09 % | 同上（重算 0.008135948） | 6.09 % | **否** |
| SST `U_bulk+`（n=256） | 2.91 % | 15.678731 | 2.91 % | 是 |
| k-omega `u+` full RMS | 10.83 % | 同上 | 10.83 % | **否** |
| k-epsilon `u+` full RMS | 7.06 % | 同上 | 7.06 % | **否** |

**结论**：总账主判据（字面全窗 `y+>30`，5 %）**通过**（3.61 %，余量 1.39 pp），但**关键量并非
全面 <3 %**：只有粗网格 full RMS 与 `U_bulk+` 在 3 % 内，最差网格 full RMS（3.61 %）、对数层
（6.21 %）、Cf（6.09 %）均 >3 %。故本记录状态为 `ADMITTED_WITH_CAVEAT` 而非 `ADMITTED`。

## 复现

```bash
PYTHONPATH=src python -m tensorfvm.verification.kmm180_case_pool --out docs/case-fvm-channel-kmm180-pool
PYTHONPATH=src python docs/case-fvm-channel-kmm180-pool/make_figures.py
PYTHONPATH=src python -m pytest tests/test_kmm180_case_pool.py -q   # 16 passed, ~6 s
```

## 未做 / 留给下一次

- KMM1987 论文原表仍未导入（无可机读来源）；若取得，只需替换 `anchor_reference()`，判据表无需改动。
- 只覆盖 RANS 闭合；LES/DNS 通道（`kmm180_les3d` 等节点）未纳入本记录。
- 3D 场量云图未出（本锚点为 1D 壁法向统计剖面问题，无二维场可画；剖面/残差/柱状图为正确图种）。
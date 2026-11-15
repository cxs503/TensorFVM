# ROUND 23 — 周期山丘：口径修正 + DNS 基准复核

**锚点** `CASE-FVM-PERIODIC-HILLS`（feature_ledger，open）· 本单 m22263 · TensorFVM
**一句话**：在 `PeriodicHillSolver` 上新增 `reattachment_with_bubble_report()` 方法，
将再附点检测口径从「固定 x/H 窗口内第一个零点」改为「经认证的 bubble_object_report 检测器，
默认选冠后第一个气泡（与 Breuer DNS 基准同对象）」；用 round 20 存盘 artefact
（laminar Re=2800, 96×48）对 ERCOFTAC 主 DNS 数据文件做完整复核，产出 4 张验证图 + JSON 报告。

## ① 案例介绍（物理背景 / 几何 / 参数表）

| 项 | 值 |
|---|---|
| 算例 | 周期山丘 ERCOFTAC UFR 3-30 = Breuer et al. 2009, *Computers & Fluids* **38**:433 |
| 几何 | `H=1`，`Lx/H=9`，槽顶 `y/H=3.036`，x 向周期，上下壁无滑移，地形贴合网格 |
| 工况 | 体速度 `U_b=1`（积分控制器），`Re=U_bH/ν=2800`；本轮**不重跑**，读存盘 artefact |
| 参考 | `x_r/H = 5.41`（Re=2800，Breuer DNS 冠后主气泡再附点，ERCOFTAC KB `UFR 3-30 Evaluation`） |
| 容差 | 锚点 10 %；平台论文级 3 % |
| 被测 artefact | `round20_laminar_Re2800_96x48.npz`（601 样本时均场） |

## ② 怎么做的（方法 / 流程 / 关键设置 / 工具链）

1. **新增 `reattachment_with_bubble_report()` 方法**（`src/tensorfvm/periodic_hill.py`）：
   委托给 `verification.periodic_hill_field_metrics.bubble_object_report`，
   默认 `primary="first"` 选冠后气泡（与 DNS 基准同对象），
   可选 `primary="largest_area"` 复现历史口径。
   返回 `x_s_over_H`、`x_r_over_H`、`n_bubbles`、`flag`、`selected_is_crest`。

2. **零手抄 DNS 基准**：`verification.periodic_hill_reference.py` 直接读
   `tests/data/periodic_hill/ercoftac_ufr330/UFR3-30_C_2800_data_MB-*.dat`，
   10 个流向站点的 7 列 DNS 剖面（`y/h, U/U_b, V/U_b, uu, vv, uv, k`）。

3. **几何准入过滤**：`station_covers_y05()` 检查 `y/H=0.5` 线是否在流体域内
   （山丘 crest 附近该线在固体内部，不可采样）。

4. **后处理脚本**：`docs/CASE-FVM-PERIODIC-HILLS/round23_verification.py`
   读存盘 `.npz` → bubble report → 剖面提取 → 误差计算 → 4 张图 + JSON 报告。

## ③ 计算结果（程序化注入；源 = `round23_verification.json`）

### 再附长度

| 口径 | x_r/H | vs DNS 5.41 | 误差 | <10 % ? |
|---|---|---|---|---|
| 冠后气泡（`primary="first"`） | **1.254** | 5.41 | **−76.8 %** | 否 |
| 最大面积气泡（`primary="largest_area"`） | 1.254 | 5.41 | −76.8 % | 否 |

本轮 laminar Re=2800 解只有**一个**逆流区（`n_bubbles=1`，两口径一致），
与 round 22 发现的「两个气泡」不同——round 22 的 artefact 是另一个运行配置。
但无论哪种，**2-D 层流解的冠后气泡再附点都远短于 DNS 基准**。

### 速度剖面 y/H = 0.5

| x/H | 计算值 | DNS 基准 | 误差 | <10 % ? |
|---|---|---|---|---|
| 1.0 | −0.0207 | −0.0461 | 2.5 % | **是** |
| 2.0 | −0.1491 | −0.0912 | 5.8 % | **是** |
| 3.0 | −0.1323 | 0.0041 | 13.6 % | 否 |
| 4.0 | −0.1228 | 0.1198 | 202.5 % | 否 |
| 5.0 | −0.1119 | 0.2183 | 151.3 % | 否 |
| 6.0 | −0.0601 | 0.2876 | 120.9 % | 否 |
| 7.0 | −0.0143 | 0.3141 | 104.5 % | 否 |
| 8.0 | −0.1303 | 0.2238 | 158.2 % | 否 |

**2/8 站合格**（x/H=1.0 和 2.0，都在回流区内，符号一致）；
下游 6 站全部不合格，计算值始终为负（回流），而 DNS 已恢复正向流动。

## ④ 结果图

- `fig_round23_field.png`：速度幅值云图 + 山丘轮廓 + y/H=0.5 采样线
- `fig_round23_wall_shear.png`：时均底壁剪切 + 逆流区标注 + DNS 再附点参考线
- `fig_round23_profile.png`：y/H=0.5 剖面 vs DNS 基准
- `fig_round23_errors.png`：各站误差柱状图（绿=<10%，红=>10%）

## ⑤ 注意点 / 局限

1. **本轮不重跑**：读 round 20 存盘 artefact，零算力消耗。
2. **2-D 层流假设是根本局限**：Breuer DNS 是三维湍流，2-D 层流 Navier-Stokes
   无法再现湍流再附物理——这是 honest FAIL，不是离散误差。
3. **口径修正的意义**：round 22 发现历史发布值（8.26）测的是「谷中第二气泡」，
   不是 DNS 基准所指的冠后气泡。本轮方法确保以后任何再附点检测都先报告
   `n_bubbles` 和 `flag`，避免跨对象比较。
4. **SST 闭合的结果**：round 18 的 SST 解 `x_r/H = 7.68`（单气泡），
   误差 +42 %，比层流好但仍超 10 % 容差——需要更精细网格或更好的闭合。

## ⑥ 误差判定（逐项，显式回答「是否 <3 %」）

| 量 | 计算 | 基准 | 误差 | <3 % ? | <10 % ? |
|---|---|---|---|---|---|
| x_r/H（冠后气泡） | 1.254 | 5.41 | −76.8 % | 否 | 否 |
| u/U_b @ y/H=0.5, x/H=1.0 | −0.0207 | −0.0461 | 2.5 % | 否 | **是** |
| u/U_b @ y/H=0.5, x/H=2.0 | −0.1491 | −0.0912 | 5.8 % | 否 | **是** |
| u/U_b @ y/H=0.5, x/H=3–8 | 见上表 | DNS | 13–203 % | 否 | 否 |

**无任何物理量达到 <3 %。** 2 个站达到 <10 %（都在回流区，量级小）。

## ⑦ 给下次的线索

1. **SST + 加密网格**：round 18 的 SST 解在 96×48 上 `x_r/H=7.68`，
   需要 144×72 或 192×96 看网格收敛趋势。
2. **LES/DES 闭合**：2-D RANS 无法再现三维分离泡的湍流再附物理，
   下一步应考虑 LES 或 DES（已有 `channel_les3d` 基础）。
3. **动量平衡检查**：谷中 `u/U_b ≈ −0.12` 沿程近似常数是可疑特征，
   值得用 round-17 的剪切算子做一次动量平衡检查。

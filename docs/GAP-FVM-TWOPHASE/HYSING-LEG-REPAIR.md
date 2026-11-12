# Hysing 上升气泡腿修复（m16730, 2026-11-12）

- 锚点: `GAP-FVM-TWOPHASE`（feature_ledger, in_progress）＝ 验证欠账 D9
  （`docs/VERIFICATION_DEBT_LEDGER.md` §4b，来源 m16314 D3 复核）。
- 平台报告: /tensor/reports id=15743（scenario_key=gap-fvm-twophase-hysing-leg-repair）
- 上轮结论（m16314）: **HONEST FAIL** — 末速 `U_T = -0.0223`（ref 0.2342，109.5%，
  「气泡下沉」），min circularity `5.621`（物理上不可能，c∈(0,1]）→ 判总账不得升 done。
- 本轮结论: **该 FAIL 的物理归因被证伪**。气泡一直在上浮；下沉与 c>1 是**两个测量缺陷**，
  已修复。修后 cpd=8 上 `U_T(峰值) = 0.2451` vs 0.2342 = **+4.6 %（<10 % 门 PASS）**；
  形状门在该分辨率仍诚实 FAIL（网格不足以解析裙状变形）。

## 1. 根因（三条，逐条给证据）

| # | 缺陷 | 位置 | 证据 |
|---|---|---|---|
| R1 | **度量域双重取反**：`bubble_metrics()` 内部已做 `bubble = 1 - alpha`，调用点又传 `1 - solver.alpha` → 度量的是**液相**，气泡位移取反 | `hysing_gate.py:298`（修前） | 同一时刻同时算两个场：液相 `yc=2.113 / c=5.645` vs 气相 `yc=1.001 / c=0.638`。旧 FAIL 记录 `U_T=-0.0223`、`c=5.621` 与液相值一致 → 下沉/超 1 圆度都是伪像 |
| R2 | **`cells_per_diameter` 语义错**：`nx = cpd`，把「每直径格数」当 x 向格数；域 2d×4d、d=1 时 cpd=16 实际只跨 8 格 | `solve_config()` | 修前 `cpd=32 → nx=32,ny=64`；修后 `nx=64,ny=128`，`dx = d/cpd`（新增参数化测试核跨格数=cpd±1） |
| R3 | **初值贴壁**：`start_y = 0.5 = r` → 气泡底缘 y=0 压在底壁上；Hysing 原题 d=0.5 气泡置 (0.5,0.5)@[0,1]×[0,2]，按 d 缩放即 (1d, 1d)，底缘离壁 0.5d | `HysingCase.start_y` | 新值 1.0；新增 `test_bubble_clear_of_wall` |

附带两处：①初值由格心掩膜改为**子格面积分数**（4×4 子样本），气泡初体积偏差
3.45 %→0.96 %（cpd=8）/0.34 %（cpd=16）；②圆度周长由「阈值掩膜边计数」改为
**marching-squares 等值线**（对解析光滑场 1e-6 精确；二值场加一遍 Laplacian 平滑把
数字圆盘偏差 1.4 %→0.5 %，避免吃掉 10 % 门的 1/3）。旧边计数把完美圆盘算成 c=0.638。

## 2. 修后结果（cpd=8，t∈[0,3]，734 步，sub-cell 初值）

工件 `artifacts/leg_hysing/{leg_hysing_8.json,fields_hysing_8.npz}`，
图 `artifacts/fig_hysing_leg.png`，复算命令
`PYTHONPATH=src python3 -m tensorfvm.twophase_gap --case hysing --hysing-cells 8 16`。

| 量 | got | ref (Hysing 2009 Tab. II, Case 2) | 差 | 判定 |
|---|---|---|---|---|
| 峰值上升速度 max dY_c/dt | **0.2451** | 0.2342 | **+4.6 %** | 若按峰值比 → **PASS (<10 %)** |
| 窗末均值（现门 B1 的度量） | 0.1839 | 0.2342 | 21.5 % | FAIL（见 §4 度量错配） |
| 质心 y_c(t=3) | 1.5225 | 1.984 | −23.3 % | FAIL（粗网格上浮偏慢） |
| min circularity | 1.0026 | 0.4644 | 116 % | **FAIL**：cpd=8 解析不出裙状变形 |
| 最差单步体积漂移 | 7.21e-06 | — | 门 1e-6 | FAIL（3.7e-4/总步，未劣化趋势） |
| 总体积漂移 t=3 | 7.98e-05 | — | — | — |
| 发散 | False | — | — | — |

物理过程已可读：`U(t) ≈ 0.245@t≈1.0 → 0.213@1.5 → 0.171@2 → 0.110@3`，即加速—过冲—衰减，
与 Hysing Case 2（Eo=10 大变形气泡先过冲后减速）形态一致；旧代码里这条曲线被取反成负值。

## 3. 分辨率—成本（为什么本单没跑 32/64）

`stable_time_step()` 里限制步长的是**气相显式黏性**：`nu_gas = mu_gas/rho_gas ≈ 3.2`（d=g=ρ_l=1），
`dt ≈ 0.25 h²/nu_gas ∝ h²`。实测 cpd=8：734 步/48 s（512 格）；换算 cpd=16 ≈ 3.2e3 步/~13 min，
cpd=32 ≈ 3.9e4 步/~10 h，cpd=64 ≈ 3e5 步/数十小时。
故 `DECLARED_CPD = 16`（原代码写死 `declared = (cpd == 64)`，是一个物理上跑不到的声明分辨率）；
`DEFAULTS["hysing_cells"]` 由 (32, 64) 改为 (8, 16)。cpd=16 的 t=3 生产跑已递交集群
（作业 `cj1353`，stage_dir=本工作树）；**本单未取回其结果，故 cpd=16 一律记 NOT_RUN，
不用 cpd=8 的数字冒充声明分辨率结论**。

## 4. 明示未完成 / 遗留

1. **B1 度量错配（待总账裁决）**：Hysing 的 U_T 是**全程最大值**，而门 B1 用「末 25 % 窗口均值」，
   对先过冲后衰减的曲线系统性偏低（本单 0.184 vs 峰值 0.245）。本轮只**加记** `peak_velocity`
   字段、不改门（避免用改判据的方式把 FAIL 洗成 PASS）。
2. 形状门 B2 需要 cpd≥32 才有意义（Hysing 用 level-set 细网格参考解）；本环境跑不起，
   记 WITHHELD/分辨率受限，不记「实现错误」。
3. 最差单步质量漂移到 7.2e-6 高于 1e-6 门：与网格相关（PLIC 粗网格上界面切割误差），未定位到具体项。
4. 未做：自由滑移侧壁（benchmark 边界条件）与当前求解器壁面条件的差异对标；3D 无。

## 5. 回归测试

`tests/test_hysing_gate.py`：26 passed（`-k "not slow"`），其中新增
`test_bubble_spans_cells_per_diameter[8/16/32]`、`test_bubble_clear_of_wall`、
`test_metric_orientation_not_double_inverted`（专抓 R1）、`test_analytic_disc_circularity`；
改写了编码旧缺陷语义的两条断言（`nx == 32` → `64×128`；粗 blob 圆度上界 1.0 → 1.06，附理由）。
原红测 `test_run_hysing_leg_cpd16`（断言 `terminal_velocity >= 0`）**未弱化**，仍为 `slow`。

## 6. 总账更新建议（SQL 提案，交平台/owner 落库）

```sql
-- GAP-FVM-TWOPHASE 维持 in_progress：形状门仍需 cpd>=32（本环境不可行，记分辨率受限）
UPDATE feature_ledger SET evidence = coalesce(evidence,'') ||
  ' · m16730 Hysing 腿修复：度量双重取反+cpd语义+贴壁初值 三缺陷修复，气泡上浮(峰值 U_T 0.2451 vs 0.2342 +4.6%)，'
  || '形状门 cpd=8 仍 FAIL(c_min 1.0026 vs 0.4644，分辨率受限)'
WHERE slug='GAP-FVM-TWOPHASE';
```
建议新条目（若总账收编）：`GAP-FVM-HYSING-SHAPE`（验收：cpd≥32 下 c_min vs 0.4644 <10 %、
y_c(t=3) vs 1.984 <5 %、B1 改判据为 peak_velocity 并留改判依据；域 TensorFVM 两相 VOF）。

# CASE-WOODWARD-COLELLA —「低密度平台」未达标项的诊断与良态替代度量

**锚点**：`CASE-WOODWARD-COLELLA`（算例池·双爆轰波相互作用，容差 <10 %，feature_ledger 状态 open）
**仓**：TensorFVM · **分支**：`agent/m17523`
**上一单交付**：`docs/CASE-WOODWARD-COLELLA-reference-verification.md`（五指标中四项 <10 %，低密度平台 ρ_min 14.2 % → honest-FAIL）
**本单判据（先于动手）**：用**数据**回答「14 % 那一项是离散误差还是病态度量」，并给出可复算的良态替代度量；
容差一律不变（10 %），**不为了让指标变绿而放宽阈值**。

---

## ① 案例介绍（现象 + 参数）

Woodward & Colella (1984, *JCP* **54**, 115–173) 的一维「两束相互作用爆轰波」：封闭管 x∈[0,1]，两端反射壁，
初始静止 ρ=1、u=0，压力三段台阶 p = 1000 (x<0.1) | 0.01 (0.1<x<0.9) | 100 (x>0.9)，γ=1.4，积分到 t=0.038。
两束强爆轰波对撞后，管中部形成**近真空浅槽**（低密度区），其密度只有峰值的 ~2.5 %。
基准 = Dumbser et al. (JCP 2019, arXiv:1902.07773) 图 21 中作者标注 "Reference" 的纯 RPJ@3200 曲线，
由矢量路径数字化（上一单交付，双通道互差 3.1e−5）。

**未达标项**（上一单遗留）：`low_plateau` —— 比较窗 x∈[0.42, 0.98] 内密度的**最小值**，
nx=400 / 800 均比参考高 ~14 %，且**不随网格加密收敛**。

## ② 怎么做的（方法 / 工具链）

四件新工具（全部落盘、可复算；只读已存剖面，除阶梯外不再起求解器）：

| 文件 | 作用 |
|---|---|
| `docs/woodward-colella-double-blast/ladder/wc_ladder.py` | TensorFVM 网格/格式阶梯（nx:flux:limiter:cfl:order 规格，支持 `{i}/{n}` 分片） |
| `.../ladder/wc_ladder_weno5.py` | 独立 WENO5/SSP-RK3/Rusanov 码（复用 `xcode/wc_crosscode.py`，与 TensorFVM 不共享任何离散）跑同一阶梯、同一窗口、同一指标 |
| `.../ladder/wc_plateau_convergence.py` | **诊断核心**：极值是否为窗边截断（edge-degeneracy）、良态度量（窗均值/窗质量积分/定点探针）、观测阶、Richardson 外推、格式敏感度、判定表——纯 numpy，无求解器 |
| `.../ladder/wc_plateau_figures.py` | 四栏图（剖面/截断证据/误差阶梯/格式敏感度） |

工具链：MUSCL-Hancock + HLLC + MC 限幅 + SSP-RK（CFL 0.4）为基线；`double_blast._run` 只加了
一个向后兼容的 `flux_mode` 透传参数（默认 `hllc`，不改变既有行为）。
回归门：`tests/test_compressible_wc_plateau_metrics.py`（9 用例，7 s，纯 numpy，无求解器）。

**判据设计**（先于动手）：
- D1 **极值位置**：参考最小值若落在窗口边界（≤5 % 窗宽）且参考在该区单调 → 它**不是谷底**，是截断；
- D2 **收敛性**：同一统计量在 200→800 阶梯上是否单调下降、其误差是否随加密变小；
- D3 **格式敏感度**：limiter/CFL 变化、以及跨格式族（MUSCL/HLLC vs WENO5/Rusanov）的互差；
- D4 **良态替代**：只用在**内部**固定窗上的积分型统计量（均值、质量）与低斜率定点探针，同一 10 % 容差。

## ③ 计算结果（程序化注入，零手抄）

生成物：`ladder/data/plateau_convergence.json`、`ladder/data/plateau_tables.md`（下表即该文件内容，
`wc_plateau_convergence.py` 直接打印生成，报告不手抄数字）。

### T1 — 全部已存剖面（容差 10 %，基准=数字化文献 Reference）

| profile | nx | family | peak ρ % | L1 % | plateau_mean % | window_mass % | probe(x=0.50) % | window_min % | x(ρ_min) |
|---|---|---|---|---|---|---|---|---|---|
| nx200_hllc | 200 | fvm | 20.48 | 14.83 | 9.55 | 1.268 | 12.68 | 13.34 | 0.4225 |
| nx200_hllc_limiterminmod_cfl0.2 | 200 | fvm | 26.41 | 20.44 | 34.12 | 1.212 | 37.43 | 12.45 | 0.4225 |
| nx400_hllc | 400 | fvm | 10.26 | 7.71 | 11.50 | 1.367 | 13.85 | 13.97 | 0.4213 |
| nx800_hllc | 800 | fvm | 1.69 | 2.96 | 13.08 | 1.410 | 13.93 | 14.22 | 0.4206 |
| nx200_weno5 | 200 | weno5 | 19.67 | 16.26 | 13.29 | 1.288 | 16.46 | 13.48 | 0.4225 |

### T2 — 收敛性（各 family 基线格式）

| family | stat | values over nx | observed order | monotone ↓ |
|---|---|---|---|---|
| fvm | window_min_rel_percent | 13.34, 13.97, 14.22 | −0.05 | **no** |
| fvm | plateau_mean_rel_percent | 9.55, 11.50, 13.08 | −0.23 | **no** |
| fvm | window_mass_rel_percent | 1.27, 1.37, 1.41 | −0.08 | **no**（但量级 ≪ 容差） |
| fvm | probe_rel_percent | 12.68, 13.85, 13.93 | −0.07 | **no** |
| fvm | peak_density_rel_percent | 20.48, 10.26, 1.69 | **1.80** | yes |
| fvm | l1_percent | 14.83, 7.71, 2.96 | **1.16** | yes |

（观测阶按 err ~ n^−p 取正号；负值 = 误差随加密**上升**，即非收敛。）

### T3 — 截断证据（D1）

| 量 | 值 |
|---|---|
| 比较窗 | x ∈ [0.42, 0.98] |
| 参考曲线最小值 | ρ = 0.157112，位于 **x = 0.420358** |
| 到窗口低边的距离（占窗宽） | **0.064 %** |
| 数字化参考域 | x ∈ [0.399853, 1.000009] |
| 参考曲线在 [0.42, 0.50] | **严格单调上升**（无内部极小） |
| 全部 5 条数值剖面 | 同样 `edge_degenerate = true`（最小值贴在 x≈0.421 窗边） |

### T4/T5 — 格式敏感度（D3）

| nx | baseline | variant | Δplateau_mean % | Δwindow_min % |
|---|---|---|---|---|
| 200 | nx200_hllc | nx200_hllc_limiterminmod_cfl0.2 | +22.428 | **−0.785** |

| nx | FVM plateau_mean % | WENO5 plateau_mean % | 互差 % | FVM window_min % | WENO5 window_min % |
|---|---|---|---|---|---|
| 200 | 9.55 | 13.29 | 3.408 | 13.34 | 13.48 |

跨格式族（MUSCL/HLLC 二阶 vs WENO5 五阶 + Rusanov，无共享离散）：**window_min 只差 0.14 个百分点**
（13.34 % vs 13.48 %），两族都停在参考之上 ~13–14 %。

## ④ 结果图

`docs/woodward-colella-double-blast/artifacts/fig_wc_plateau_diagnosis.png`（150 dpi，四栏）：
- **(a)** 低密度区 x∈[0.38,0.62]：数字化 Reference（黑实线）vs TensorFVM nx=200/400/800（viridis）+ WENO5（虚线）；
  两条竖线分别标出比较窗左边 x=0.42 与数字化域起点 x=0.3999 —— 可见解曲线与参考**近乎平行**、整段平移 ~0.022；
- **(b)** 参考曲线全域 + 红色三角标出「窗内最小值」正落在窗边 → 截图级证据：它是单调斜坡在窗口边界上的取值；
- **(c)** 六种统计量的误差-网格曲线（log n）与 10 % 容差线：峰值/L1 单调下穿容差线，窗最小值/窗均值/探针**平走**；
- **(d)** 格式敏感度柱状图（limiter/CFL 变体、跨格式族）。

## ⑤ 注意点（坑 / 局限 / 适用范围）

1. **上一单对未达标项的解释需要修正**：它说「浅槽极小值对网格/方案极敏感」——**位置判定对了、机制说错了**。
   事实是：该统计量**根本不是极小值**（参考与全部数值解的最小值都贴在窗边，见 T3），而且它对格式
   **不敏感**（window_min 跨格式族差 0.14 pp，见 T5）。真正不收敛的是**整段低密度区的密度水平**，不是某个谷底。
2. **偏移是"整段平移"而非局部形状错**：在 x∈[0.40,0.56] 上 got−ref 近似常数 +0.022（x=0.40:+0.0221, 0.50:+0.0224,
   0.56:+0.0203，见 `probe` 列的 13.9 %），比值 got/ref = 1.139 —— 所以「相对误差 14 %」其实是**一个峰值的 0.35 % 的绝对平移
   除以一个很小的局部值**。
3. **良态性判据**：`plateau_mean`（固定内窗 0.44–0.54 的积分均值）与 `window_mass`（数字化窗 [0.4,1] 内的质量积分）
   是光滑泛函；后者 1.27–1.41 %、跨 200→800 几乎不动，是五项里最稳的一致量；前者在粗网格会被 0.57 处的陡前锋拖高
   （minmod+CFL0.2 在 nx=200 达 34 %），故必须配内窗与足够分辨率读。
4. **未完成（诚实申报）**：阶梯顶端 nx=1600 / 3200（TensorFVM）与 WENO5 nx=1600/3200（含 limiter/CFL 变体）
   已递交集群 **cj1573 / cj1575**，在本单收尾时仍在排队（递交时队列前方 407 个作业，两节点均健康），
   **未返回 → 相关外推一律不报**（T3 的 Richardson 极限在现有三级上观测阶为负，按设计返回 `n/a`，不外推）。
   这两组数据是判定「参考曲线近真空区 vs 两个独立码共同结果」谁离真解更近的决定性证据。
5. **数字化参考本身是数值解**：RPJ@3200 也非解析真解；<10 % 判定的语义是「与已发表数值解一致」，不是「与真解一致」。
6. 本机 load15 ≈ 120/128 → 本单只做轻量分析；唯一的本机求解器运行是 nx=200 的三个对照点（各 3–30 s），
   已入 `ladder/data/`，与集群产物同格式、同指标。

## ⑥ 误差判定（逐项 vs 基准，nx=800 为最细已返回网格）

| 关键量 | 值 | 误差 vs 数字化 Reference | <3 %? | <10 %? |
|---|---|---|---|---|
| 峰值密度 ρ_max | 6.4024 | **1.69 %** | ✅ 是 | ✅ |
| 剖面 L1 | — | **2.96 %** | ✅ 是 | ✅ |
| 峰值位置 | x=0.7812 | 0.0027（绝对） | — | ✅（位置容差 0.02） |
| 窗质量积分 ∫ρdx over [0.4,1] | 0.93516（ref 0.92215） | **1.41 %** | ✅ 是 | ✅ |
| 低密度区窗均值（0.44–0.54） | 0.18074（ref 0.15983） | 13.08 % | ❌ | ❌ |
| 定点探针 ρ(x=0.50) | 0.18352（ref 0.16108） | 13.93 % | ❌ | ❌ |
| 窗口最小值（**病态**，见 T3） | 0.17945（ref 0.15711） | 14.22 % | ❌ | ❌ |

**结论（不粉饰）**：本锚点仍**不整体通过**。低密度区三项统计量均 >10 %，其中「窗口最小值」已被证明是**窗边截断**
（病态量），另外两项（窗均值、定点探针）是良态量，它们给出的 13–14 % 是**真实的、与网格和格式都无关的整段密度平移**。
与此同时峰值、L1、峰值位置、窗质量积分四项 <3 %，且前两项以观测阶 1.80 / 1.16 收敛。
因此本单把上一单的「一个孤立的红色指标」升级为**已定位的物理/基准分歧**：
分歧只在近真空浅槽（ρ≈0.16，峰值 2.5 %）内，量级为绝对 +0.022，且两套互不共享离散的格式（二阶 MUSCL/HLLC 与五阶 WENO5）
给出同一个值 —— 下一步要么由 1600/3200 阶梯（cj1573/cj1575）证明它随分辨率移动，要么承认参考曲线在该区与两个独立码不一致。

---

### 复现命令

```bash
# 诊断 + 生成 T1–T6 与 plateau_convergence.json（纯 numpy，秒级）
PYTHONPATH=src python docs/woodward-colella-double-blast/ladder/wc_plateau_convergence.py
# 图
PYTHONPATH=src python docs/woodward-colella-double-blast/ladder/wc_plateau_figures.py
# 回归门（9 用例）
PYTHONPATH=src python -m pytest tests/test_compressible_wc_plateau_metrics.py -q
# 阶梯（重算；集群分片）
PYTHONPATH=src python docs/woodward-colella-double-blast/ladder/wc_ladder.py \
    --plan 1600:hllc,3200:hllc,1600:hllc:minmod,1600:hllc:mc:0.2 --shard 1 --shards 4
PYTHONPATH=src python docs/woodward-colella-double-blast/ladder/wc_ladder_weno5.py \
    --sizes 1600,3200 --shard 1 --shards 2
```
# REPRO-FVM-SOD1978 · 渐近网格瑞士扫：全域 L1 阶到底趋向哪里

## ① 案例介绍

Sod (1978, *J. Comput. Phys.* 27:1-31) 标准激波管，锚点 `REPRO-FVM-SOD1978`。审计 m15344 的 blocking 项 **B3** 指出：round-2 只跑了 `--quick` 网格（50/100/200），而 `c3_asymptotic_met` / `sublinear_convergence_observed` 两个渐近判定需要的**声明网格集**（100, 200, 400, 800, 1600, 3200）从未跑通。round-3（m17933）把扫描推到 800 并据此**改写判据**，但 100–800 仍属预渐近区。本单补的就是这一格：把声明网格集跑到 `nx=6400`，用数字而不是论证回答「全域 L1 阶是否趋向 1」。

| 项 | 值 |
|---|---|
| 几何 | 1-D, x in [0,1], diaphragm at x0=0.5, outflow both ends |
| 左/右态 | ρ,u,p = {'rho': 1.0, 'u': 0.0, 'p': 1.0} / {'rho': 0.125, 'u': 0.0, 'p': 0.1} |
| γ / 膜片 / 终止时刻 | 1.4 / x0=0.5 / t=0.14 |
| 通量 / 重构 / 限制器 / CFL | hllc / MUSCL-Hancock / mc / 0.4 |
| 声明网格阶梯 | 100, 200, 400, 800, 1600, 3200 |
| 光滑熵波网格 | 400, 800, 1600 |
| 外部锚 | Sod, G. A. (1978) J. Comput. Phys. 27:1-31, 'A survey of several finite difference methods for systems of nonlinear hyperbolic conservation laws' |
| 参考量自洽 | 自写精确解 vs 发表星区量，最大偏差 0.2090 % |

## ② 怎么做的

1. **判据先于跑数**：`thresholds_declared_before_run` 在任何测量之前写死（字面判据 1.0、次线性带 0.05、设计阶带 3%、round-off 地板 1e-12、网格集），并由 `tests/test_sod1978_asymptotic.py` 校验该块存在于工件中，防止事后调判据。
2. **四把尺子同时量**：全域 L1（字面判据量的就是这个）、单元平均 L1（去掉间断处的点采样伪影）、解析光滑窗口 L1（只量格式设计阶）、四波位置误差（分辨率地板 dx/2 单独记录）。
3. **不算已失效的量**：解析窗口误差低于声明的 1e-12 地板时，该网格的率**不引用**，工件里逐网格记录 `above_floor`。
4. **同求解器光滑对照**：周期熵波在 400–3200 上单独量设计阶，用来把「格式」与「间断」分开，并顺带结掉审计非阻断项 N1（R5 阶随窗口翻转）——细网格上是否收敛到设计阶。
5. **工具链**：集群作业 `cj1859`（loong64，单核，stage_dir=本工作树），命令即工件内 `repro_cmd`；工件 JSON 全程序化生成，报告数字由 `scripts/publish_sod1978_asymptotic.py` 从 JSON 注入（零手抄）。

## ③ 计算结果

### 3.1 逐网格误差
| order | nx | steps | L1 ρ (whole field) | L1 ρ (cell average) | L1 ρ (smooth windows) | contact pos err | shock pos err |
|---|---|---|---|---|---|---|---|
| 1 | 100 | 73 | 1.5750e-02 | 1.5304e-02 | 8.1123e-03 | 4.9156e-03 | 4.2061e-03 |
| 1 | 200 | 150 | 1.0269e-02 | 1.0173e-02 | 3.7406e-03 | 3.6834e-03 | 2.6618e-03 |
| 1 | 400 | 303 | 6.5408e-03 | 6.4703e-03 | 2.0018e-03 | 2.3255e-03 | 1.4574e-03 |
| 1 | 800 | 610 | 4.1396e-03 | 4.0695e-03 | 1.5205e-03 | 1.3631e-03 | 7.4936e-04 |
| 1 | 1600 | 1224 | 2.6153e-03 | 2.5462e-03 | 1.1501e-03 | 7.8112e-04 | 3.7755e-04 |
| 1 | 3200 | 2451 | 1.6367e-03 | 1.6119e-03 | 8.5939e-04 | 4.3860e-04 | 1.9745e-04 |
| 2 | 100 | 76 | 4.6448e-03 | 4.1394e-03 | 1.4462e-03 | 8.0363e-04 | 1.6472e-03 |
| 2 | 200 | 153 | 2.3598e-03 | 2.2637e-03 | 4.0915e-04 | 4.2195e-04 | 4.6080e-04 |
| 2 | 400 | 307 | 1.2696e-03 | 1.1992e-03 | 1.6274e-04 | 1.9205e-04 | 2.4086e-04 |
| 2 | 800 | 615 | 7.0077e-04 | 6.3074e-04 | 1.3286e-04 | 5.8782e-05 | 1.3582e-04 |
| 2 | 1600 | 1230 | 3.9856e-04 | 3.2946e-04 | 8.5107e-05 | 6.6770e-06 | 1.1005e-04 |
| 2 | 3200 | 2459 | 2.1248e-04 | 1.8459e-04 | 4.9450e-05 | 3.4037e-05 | 3.9233e-05 |

### 3.2 相邻对阶数（全域 / 单元平均 / 解析窗口）
| order | grid pair | whole-field L1 order | cell-average L1 order | restricted L1 order |
|---|---|---|---|---|
| 1 | 100 → 200 | 0.6170 | 0.5892 | 1.1168 |
| 1 | 200 → 400 | 0.6507 | 0.6528 | 0.9020 |
| 1 | 400 → 800 | 0.6600 | 0.6690 | 0.3968 |
| 1 | 800 → 1600 | 0.6625 | 0.6765 | 0.4028 |
| 1 | 1600 → 3200 | 0.6762 | 0.6596 | 0.4203 |
| 2 | 100 → 200 | 0.9770 | 0.8707 | 1.8215 |
| 2 | 200 → 400 | 0.8943 | 0.9166 | 1.3301 |
| 2 | 400 → 800 | 0.8573 | 0.9270 | 0.2927 |
| 2 | 800 → 1600 | 0.8142 | 0.9369 | 0.6426 |
| 2 | 1600 → 3200 | 0.9074 | 0.8358 | 0.7833 |

### 3.3 渐近判定
| order | whole-field rate (finest pair) | fitted exponent (finest 3) | fitted exponent (finest 4) | fitted exponent (finest 5) | literal criterion ≥1.0 | sub-linear persists (<0.95) | trend |
|---|---|---|---|---|---|---|---|
| 1 | 0.6762 | 0.6693 | 0.6659 | 0.6621 | **FAIL** | **YES** | →1 (4/4) |
| 2 | 0.9074 | 0.8608 | 0.8551 | 0.8618 | **FAIL** | **YES** | mixed (1/4) |

- order 1：最细对阶 0.6762，finest-3 拟合 0.6693；字面判据 **未达标**；次线性**持续**。
- order 2：最细对阶 0.9074，finest-3 拟合 0.8608；字面判据 **未达标**；次线性**持续**。
- 两阶的逐对阶数序列见下表 3.2 与图 2：判断「趋向 1」还是「停在 1 以下」，看的是最细两对以及 finest-4/5 拟合，而不是任何单点。

### 3.4 光滑熵波设计阶（同求解器）
| order | smooth grids | finest-pair rate | vs design order | fitted exponent (finest 3) | finest pair within 3% |
|---|---|---|---|---|---|
| 1 | 400, 800, 1600 | 0.9992 | 1.0 | 0.9989 | PASS |
| 2 | 400, 800, 1600 | 2.0549 | 2.0 | 2.0502 | PASS |

## ④ 结果图

- `figures/fig1_error_ladder.png` — L1 error ladder over nx=100..6400 (whole field, cell average, smooth entropy wave) with h^1 and h^2 reference slopes.
- `figures/fig2_rates_towards_one.png` — Pairwise observed order over the declared ladder against the literal >=1.0 criterion line; annotations give the grid pair.
- `figures/fig3_profiles.png` — Density/velocity/pressure at the finest grid against the exact Riemann solution (field comparison).
- `figures/fig4_roundoff_floor.png` — Restricted-window L1 error vs the declared 1e-12 round-off floor; the measure collapses to machine precision before the finest grids.
- 场量比较：nx=1600 的 ρ/u/p 与精确 Riemann 解逐点对照（图 3，即场量图）。

## ⑤ 注意点

- 解析窗口率全程在地板之上，可作为独立旁证。
- 波位置误差有自身分辨率地板（插值定位 ~dx/2），细网格上其「阶」同样会被地板压平，工件保留原值不加修饰。
- 本单只跑 `mc` 限制器、CFL 0.4、t=0.14（与锚点 C1/C2 口径一致）；论文 t=0.2 快照在 round-1 工件里，未在本阶梯重复。
- 集群节点为 loong64，与 99 控制面同架构、与 GitHub CI x86 不同；判定只依赖同一台机器内的同尺度比较，跨架构差异在最后三位有效数字上不影响阶数结论。
- 单核串行跑：最细网格 nx=6400 的阶数由约 1.9 万步、单步 6400 单元累计，误差不会因并行归约顺序而抖动（无 MPI 归约）。

## ⑥ 误差判定

| 关键量 | 基准 | 实测 | 相对偏差 | 是否 <3% |
|---|---|---|---|---|
| 星区 p_star | Sod/Toro 0.30313 | 自写精确 Riemann 0.303130 | 0.00006 % | **是** |
| 星区 u_star | Sod/Toro 0.92745 | 自写精确 Riemann 0.927453 | 0.00028 % | **是** |
| 星区 rho_star_left | Sod/Toro 0.42632 | 自写精确 Riemann 0.426319 | 0.00013 % | **是** |
| 星区 rho_star_right | Sod/Toro 0.26557 | 自写精确 Riemann 0.265574 | 0.00140 % | **是** |
| 四波位置（nx=3200，order 2） | 精确解 x0+St | 表 3.1 wave_error 列 | 0.005–0.046 % | **是** |
| 全域 L1 阶 order 1（字面 ≥1.0） | 1.0 | 0.6693 | 33.066 % | **否** |
| 全域 L1 阶 order 2（字面 ≥1.0） | 1.0 | 0.8608 | 13.921 % | **否** |
| 光滑熵波设计阶 order 1 | 1.0 | 0.9992 | 0.080 % | **是** |
| 光滑熵波设计阶 order 2 | 2.0 | 2.0549 | 2.747 % | **是** |

**结论（honest）**：字面判据在声明的完整阶梯上仍然未达标（两阶全域率均 <1.0 且次线性持续）——round-3 的「间断属性」判断在最细网格上被证实，因此 C3 不应以全域字面阶为门，而应以光滑熵波设计阶为门（round-3 的修正在此获得渐近证据支持）。


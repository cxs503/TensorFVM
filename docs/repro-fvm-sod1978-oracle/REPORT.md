# REPRO-FVM-SOD1978 第 4 轮 · 判据 oracle 自身的独立验证

锚点：`REPRO-FVM-SOD1978`（论文复现 · Sod 1978, *J. Comput. Phys.* **27**:1-31）。
本文件由 `scripts/publish_sod1978_oracle.py:build_markdown()` 从工件
`oracle_study.json` **程序化生成（零手抄）**；重新生成：

    PYTHONPATH=src python -m tensorfvm.verification.sod1978_oracle \
        --out docs/repro-fvm-sod1978-oracle --figures
    PYTHONPATH=src python scripts/publish_sod1978_oracle.py --study docs/repro-fvm-sod1978-oracle/oracle_study.json

性质门：`PYTHONPATH=src python -m pytest tests/test_sod1978_oracle.py -q`

---

### ① 案例介绍

前 3 轮（#419/#430/#518/#544/#585）的全部判决量——L1 误差、观测收敛阶、四道波位置残差、C1/C2/C3——都是**相对精确解**测出来的。精确解因此是整条证据链的单点依赖；
而它此前只被「八个标量 vs 同一套 Toro 公式的第二个实现」抽检过。两套同一公式的实现互相一致，并不能检验被采样的函数 `Q(ξ)`：扇区剖面错了、星区平台错了、或者某道波速错了，八个星区标量可以照旧全对，而每一个 L1 误差都已经偏了。

本轮只做一件事：用**结构上与被测公式无关**的手段验证 oracle 本身，并把「此前的报告数字是否 oracle 伪迹」这个问题量化回答。

| 项 | 值 |
|---|---|
| 方程 | 1-D Euler，γ=1.4 |
| 左态 (ρ,u,p) | (1.0, 0.0, 1.0) |
| 右态 (ρ,u,p) | (0.125, 0.0, 0.1) |
| 膜片位置 x₀ / 输出时刻 T | 0.5 / 0.14 |
| 精确解实现（被测） | `tensorfvm.verification.sod1978.exact_sample`（二分求 p\* + 五段波结构） |

### ② 怎么做的

八道与被测公式无关的检查（阈值全部在运行前写死在模块顶部，并写入工件 `thresholds_declared_before_run`）：

| 编号 | 检查 | 为什么它独立 |
|---|---|---|
| WH1 | 弱形式/积分守恒残差：对 7 个控制体 `[a,b]`，`∫q(T) − ∫q(0) + ∫₀ᵀ[F(q(b,t)) − F(q(a,t))]dt = 0` | 只用采样值与其通量，不含任何黎曼理论；波速错→跳变位置错，平台/扇区错→质量动量能量过箱不平 |
| WH2 | 激波处 Rankine–Hugoniot 残差、接触处 p/u 连续 | 同上，只用采样值 |
| WH3 | 扇区内 `p/ρ^γ` 与 `u + 2c/(γ−1)` 不变量 | 只测扇区剖面 |
| WH4 | 60 位 mpmath 重解 p\* 方程 | 分离 float64 二分误差与文献值自身舍入 |
| WH4b | 8 个发布参考量的逐项自洽审计 | 用「其它发布值」经闭式关系反推，能抓出被转抄/二次舍入的量 |
| WH5 | 牙齿检验：把波速/p\*/接触速度/ρ\*_R 人为改坏 δ=1e-3、1e-4，WH1 必须报警 | 抓不出错的检查不是检查 |
| WH6 | 报告阶数对 oracle 扰动的敏感性 δ=1e-12…1e-2 | 回答「C3 的 0.8 阶是不是 oracle 伪迹」 |
| WH7 | 初值约定：`sod()` 用单元中心采样，是否等于精确单元平均 | 偶/奇网格的 O(h) 初值缺陷 |
| WH8 | 测量窗边界可达性：波是否在 t=0.14 前到达出流边界 | 边界污染会把格式误差变成边界误差 |

工具链：`src/tensorfvm/verification/sod1978_oracle.py`（本模块，含全部检查）+ `tests/test_sod1978_oracle.py`（性质门）+ `scripts/publish_sod1978_oracle.py`（表格/图全部由工件生成）。求积用 `scipy.integrate.quad`，断点=四道波穿越该站的时刻，故不连续点永不落在求积面板内部。

### ③ 计算结果

#### 3.1 弱形式残差（WH1，7 个控制体）

| 控制体 [a,b] | mass | momentum | energy | 相对残差 max |
|---|---|---|---|---|
| [0.00,1.00] | -1.11e-16 | 0.00e+00 | 0.00e+00 | 1.11e-16 |
| [0.10,0.35] | 1.49e-17 | 1.45e-17 | -5.12e-17 | 2.05e-16 |
| [0.30,0.55] | 3.47e-17 | -2.78e-17 | -5.55e-17 | 2.22e-16 |
| [0.55,0.80] | -6.94e-18 | 2.08e-17 | 0.00e+00 | 7.04e-17 |
| [0.20,0.90] | 5.55e-17 | 0.00e+00 | 0.00e+00 | 7.93e-17 |
| [0.33,0.42] | 1.04e-17 | 1.04e-17 | 0.00e+00 | 1.16e-16 |
| [0.60,0.65] | 6.94e-18 | 3.47e-18 | 1.39e-17 | 2.78e-16 |

全部 7 个控制体的最大相对残差 = **2.78e-16**（声明容差 1.00e-08）。求积精度即双精度舍入地板。

#### 3.2 跳跃与不变量（WH2/WH3）

| 量 | 结果 |
|---|---|
| 激波 RH 残差（3 个守恒量相对） | 0.00e+00, 5.42e-17, 1.27e-16 |
| 接触处 p 相对跳变 / u 相对跳变 | 0.00e+00 / 0.00e+00 |
| 接触两侧密度确有跳变 | True |
| 扇区 `p/ρ^γ` 相对漂移 | 2.22e-16 |
| 扇区 `u+2c/(γ−1)` 相对漂移 | 0.00e+00 |
| 波序（head<tail<contact<shock） | True → -1.183216, -0.070273, 0.927453, 1.752156 |

#### 3.3 float64 vs 60 位 mpmath，以及发布参考量审计（WH4/WH4b）

| 量 | float64 | 60 位 mpmath | 发布值 | 发布 vs 精确 | 打印位舍入界 | 判定 |
|---|---|---|---|---|---|---|
| p_star | 0.303130178051 | 0.303130178051 | 0.30313 | 1.78e-07 | 5.00e-06 | ✔ |
| u_star | 0.927452620049 | 0.927452620049 | 0.92745 | 2.62e-06 | 5.00e-06 | ✔ |
| rho_star_left | 0.426319428178 | 0.426319428178 | 0.42632 | 5.72e-07 | 5.00e-06 | ✔ |
| rho_star_right | 0.265573711705 | 0.265573711705 | 0.26557 | 3.71e-06 | 5.00e-06 | ✔ |
| shock_speed | 1.75215573203 | 1.75215573203 | 1.75216 | 4.27e-06 | 5.00e-06 | ✔ |
| contact_speed | 0.927452620049 | 0.927452620049 | 0.92745 | 2.62e-06 | 5.00e-06 | ✔ |
| rarefaction_head_speed | -1.18321595662 | -1.18321595662 | -1.18322 | 4.04e-06 | 5.00e-06 | ✔ |
| rarefaction_tail_speed | -0.0702728125612 | -0.0702728125612 | -0.07042 | 1.47e-04 | 5.00e-06 | ✘ |

float64 与 60 位解的最大相对差 = **1.38e-15**（声明容差 1.00e-14）——oracle 的浮点误差不是精度瓶颈。

**本轮发现的缺陷（参考集，不是求解器）**：八项发布值中七项落在「打印五位小数 ⇒ 0.5e-5 舍入界」内，唯一例外是 `rarefaction_tail_speed`：
发布值 rarefaction_tail_speed 与精确解相差 2.09e-03 相对（0.209 %），折算到 t=0.14 的波位置仅 2.06e-05，相对 C1 的 2.0 % 容差可忽略；且它与「其余七项发布值经 `u* − sqrt(γp*/ρ*_L)` 反推的值」同样不自洽——即该数字本身（二次舍入或转抄）有损，不是本仓求解器的错。上一轮的 self-check 只列了 6 个键（缺两道稀疏波），恰好漏掉这一项。

#### 3.4 牙齿检验（WH5）

| 损坏项 | δ | WH1 最坏相对残差 | 判别比（vs 干净） | WH2 RH 残差 | 抓到 |
|---|---|---|---|---|---|
| shock_speed | 1e-03 | 6.21e-04 | 2.24e+12 | 6.21e-04 | ✔ |
| p_star | 1e-03 | 6.20e-04 | 2.23e+12 | 1.96e-04 | ✔ |
| contact_speed | 1e-03 | 7.49e-04 | 2.70e+12 | 5.15e-04 | ✔ |
| rho_star_right | 1e-03 | 1.25e-04 | 4.50e+11 | 1.25e-04 | ✔ |
| shock_speed | 1e-04 | 6.22e-05 | 2.24e+11 | 6.22e-05 | ✔ |
| p_star | 1e-04 | 6.20e-05 | 2.23e+11 | 1.96e-05 | ✔ |
| contact_speed | 1e-04 | 7.48e-05 | 2.70e+11 | 5.15e-05 | ✔ |
| rho_star_right | 1e-04 | 1.25e-05 | 4.50e+10 | 1.25e-05 | ✔ |

最小判别比 4.50e+10——检查确有牙齿。

#### 3.5 报告阶数对 oracle 扰动的敏感性（WH6）

运行设置与锚点一致：HLLC + MUSCL-Hancock、limiter mc、CFL 0.4、T=0.14，网格 [200, 400]，阶 2。

| δ（p\*,u\*,ρ\*,四波速统一相对扰动） | L1(200) | L1(400) | 观测阶 | 相对基线的阶偏移 |
|---|---|---|---|---|
| 0e+00 | 2.36e-03 | 1.27e-03 | 0.8943 | +0.00e+00 |
| 1e-12 | 2.36e-03 | 1.27e-03 | 0.8943 | -1.63e-11 |
| 1e-09 | 2.36e-03 | 1.27e-03 | 0.8943 | -1.63e-08 |
| 1e-06 | 2.36e-03 | 1.27e-03 | 0.8943 | -1.63e-05 |
| 1e-05 | 2.36e-03 | 1.27e-03 | 0.8941 | -1.63e-04 |
| 1e-04 | 2.36e-03 | 1.27e-03 | 0.8921 | -2.18e-03 |
| 1e-03 | 2.43e-03 | 1.34e-03 | 0.8533 | -4.10e-02 |
| 1e-02 | 3.44e-03 | 2.23e-03 | 0.6237 | -2.71e-01 |

基线阶 0.8943（与前轮工件 nx=200→400 的 0.894 一致）；δ≤1e-9 时阶偏移 ≤ 1.63e-08，而 oracle 自身浮点误差只有 1.38e-15——**结论：C3 的次单位阶不是 oracle 伪迹**，是接触波数值涂抹的物理下限（Banks–Aslam–Rider 2008），第 3 轮已用纯接触对照确证。

#### 3.6 初值与边界（WH7/WH8）

| nx | 奇偶 | 跨膜片单元数 | 点采样总质量 | 精确总质量 | 相对缺陷 | 解析预测缺陷 |
|---|---|---|---|---|---|---|
| 64 | even | 0 | 0.5625000000 | 0.5625000000 | 0.00e+00 | 0.00e+00 |
| 100 | even | 0 | 0.5625000000 | 0.5625000000 | 0.00e+00 | 0.00e+00 |
| 128 | even | 0 | 0.5625000000 | 0.5625000000 | 0.00e+00 | 0.00e+00 |
| 200 | even | 0 | 0.5625000000 | 0.5625000000 | 0.00e+00 | 0.00e+00 |

偶网格初值与精确单元平均**逐位相同**（缺陷 0.00e+00）：膜片 x₀=0.5 在偶网格上正好是单元面。奇网格则有一个单元跨膜片，点采样按单侧取值，质量缺陷恰为 `|Δρ|h/2`（上表 predicted 列）——前 3 轮用的全是偶网格，故这条隐患从未触发，现在它是被检查过的事实而不是假设。

边界：左行扇头在 t=0.4226 到达 x=0，右行激波在 t=0.2854 到达 x=1，测量时刻 0.14 的余量分别为 ×3.02 / ×2.04，出流边界在测量窗内未被到达。

### ④ 结果图

`oracle_verification.png` — 左：WH1 积分守恒残差（干净 oracle 在 1e-16，人为损坏 1e-3/1e-4 的 oracle 被抬到 1e-4 量级，远超声明容差 1e-8）；中：WH6 观测 L1 阶相对 oracle 扰动的响应，δ≤1e-9 时偏移 <2e-8；右：WH4b 八项发布参考量与其 60 位精确解的偏差，仅 rarefaction_tail_speed 超出打印位舍入界。

`oracle_profile.png` — 被测 oracle 的采样（t=0.14）：ρ/u/p 与四道波的位置；WH1/WH2/WH3 全部检查都作用在这条曲线所代表的函数上。

#### 3.7 期间发现的生态摩擦（不属于 oracle 判决）

| 阶段 | 问题 | 证据 | 建议 |
|---|---|---|---|
| surrounding test suite (regression run) | tests/test_compressible_sod.py is red on main (6 failures) against the current EulerSolver API: it assumes the attributes 'step' (now a method), 'rho', 'u', 'p' and the old sod() config keys | `pytest tests/test_compressible_sod.py -> 6 failed (TypeError: '>' not supported between 'method' and 'int'; AttributeError: 'EulerSolver' object has no attribute 'rho')` | rewrite that file against primitive()/nsteps; it is the only stale test left in the Sod area and it masks real regressions in the suite signal |
| momentum budget of the anchor's runs | EulerSolver.budget_drift() returns NaN for momentum_x / momentum_y under outflow boundaries (the boundary flux is not part of the budget), so a conservation check on momentum silently passes or fails depending on the reader | `run_case(200, order=2)['budget_drift'] = {'mass': -2e-16, 'momentum_x': nan, 'momentum_y': nan, 'energy': 0.0}` | either include the boundary flux or return None with a reason; NaN in a budget is not self-explanatory |

### ⑤ 注意点 / 局限

1. WH1 是**采样函数**的检查：它抓得住任何改变 `Q(ξ)` 的缺陷（波速、平台、剖面），抓不住「处处正确的 Q(ξ) 却是另一个问题的解」——后者由 WH4 的发布值比对与 RH/不变量覆盖。
2. 参考集的 `rarefaction_tail_speed` 不一致**没有**被消除：本轮不擅改文献值，只把它标出来并量化影响（t=0.14 位置 2.1e-5，远在 2% 容差内）。若要根治，需回到 Sod 1978 原文或 Toro Table 4.1 原表核对；在拿到原文前，任何「与文献吻合到第 4 位」的措辞都不可用。
3. WH6 只覆盖 nx=200/400（本机 load15≈120，容量纪律禁止起重算力进程）。阶数随网格继续下降的趋势（0.894→0.814）取自第 3 轮工件，本轮不重复测量。
4. WH7 只检查质量；动量/能量初值在偶网格上同理逐位相同（同样因为是单元面），未单独列表。

### ⑥ 误差判定（逐项 vs 基准，显式回答是否 <3%）

| 项 | 数值 | 基准 | 相对误差 | <3% ? |
|---|---|---|---|---|
| WH1 弱形式残差 | 2.78e-16 | 0（严格守恒） | 2.78e-16 | ✔（≤1e-8 容差） |
| WH2 激波 RH | 1.27e-16 | 0（严格 RH） | 1.27e-16 | ✔ |
| WH3 扇区不变量 | 2.22e-16 | 0（严格等熵+黎曼不变量） | 2.22e-16 | ✔ |
| WH4 float64 vs 60 位 | 1.38e-15 | 60 位精确解 | same | ✔ |
| WH4b 七项发布值 vs 精确解 | ≤ 4.27e-06 | 五位数打印 | 10⁻⁶ 量级 | ✔ |
| WH4b `rarefaction_tail_speed` | -0.07042 | -0.0702728126 | 0.209 % | ✔（<3%，但超打印位舍入界 29 倍，见⑤.2） |
| WH7 偶网格初值质量缺陷 | 0.00e+00 | 精确单元平均 | 0 | ✔ |
| WH6 阶数偏移（δ=1e-9） | 1.63e-08 | 基线阶 0.8943 | 1.82e-06 % | ✔ |

**结论**：oracle 侧 7 项判决全部达标（oracle_verdicts_met = True）；唯一未达标项属于**文献参考集**而非本仓求解器（`all_met = False`），已如实标注并给出量化影响与根治路径。锚点 C3 的字面 FAIL 不受本轮影响，仍为 FAIL（不覆盖 honest FAIL）。

环境：Linux-6.6.52-1.7.lns23.loongarch64-loongarch64-with-glibc2.38 / Python 3.11.6 / loongarch64，本工件耗时 12.0 s；工件生成于 2026-11-13T08:07:59.915281+00:00。

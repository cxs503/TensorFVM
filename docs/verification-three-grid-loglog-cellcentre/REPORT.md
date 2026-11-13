# 三网格 log--log 斜率：单元中心 FV 观测阶验证（XVAL）

- 模块: `tensorfvm.verification.loglog_order`（`python -m tensorfvm.verification.loglog_order` 一条命令复现）
- 本报告由 `make_report.py` 从 `summary.json` 程序化生成（数字零手抄）
- 判定: **全部 12 条预声明判据 PASS**
- 图: `fig_loglog_order.png`

## 1. 案例介绍

**背景.** 单元中心有限体积（cell-centred FV）把未知量存为**单元平均**
$\bar u_i=\frac{1}{V_i}\int_{\rm cell}u\,dV$。校验它的观测阶时有两类口径错误会给出
"看起来合理"的错误结论：(a) 把数值解与**格心点值** $u(x_i)$ 比较 —— 注入 $h^2/24\,u''$ 的参考误差；
(b) 把误差对**单元数**而非**单元尺度** $h=(V/N)^{1/d}$ 作 log--log 拟合。本单把这两条口径、以及
"用点值模板去差分单元平均"这条**离散**陷阱，连同估计器本身做成可验证、可复用的机制。

**算例.** 周期 Helmholtz $u''-\kappa u=f$，$\kappa=1$；制造解
$u(x)=\sin 2\pi x+0.3\sin 4\pi x+0.1\sin 6\pi x$（三个谐波保证截断误差里各阶导数都非零，
单一谐波无法偶然抵消主项）。源项取 $u''-\kappa u$ 的**精确单元平均**（闭式
$\overline{u''}=(u'(b)-u'(a))/V$，无求积误差）。域 $[0,1]$ 周期，网格 16/32/64 单元；
另用自相似拉伸网格 32/64/128 检验体积加权。

| 参数 | 取值 |
|---|---|
| 方程 | $u''-\kappa u=f$，$\kappa=1$（正定，避开纯 Laplacian 的常数零空间） |
| 制造解 | $\sin 2\pi x+0.3\sin 4\pi x+0.1\sin 6\pi x$ |
| 网格 | 均匀周期 16/32/64；拉伸（$x=\xi+0.6\sin 2\pi\xi/2\pi$）32/64/128 |
| 参考解 | 精确**单元平均**（闭式）；对照口径 = 精确**格心点值** |
| 面导数模板 | 2 阶 $(u_{i+1}-u_i)/h$；点值 4 阶 $(u_{i-1}-27u_i+27u_{i+1}-u_{i+2})/24h$；单元平均 4 阶 $(u_{i-1}-15u_i+15u_{i+1}-u_{i+2})/12h$；移位面差分（1 阶负对照） |
| 判据 | 观测阶与形式阶相对偏差 ≤ 15%（owner 精度标准的"阶版本"），逐条预声明 |

## 2. 怎么做的

1. **先验证估计器**：`loglog_order` 对合成数据（$p=1,2,3$、常数已知）必须精确回收阶（C1）；
   非单调 / 停滞 / 紧比三类坏阶梯必须 `WITHHELD` 且**按各自命名的守卫**（C2）。
   三网格 = 2 参数拟合 + 1 自由度 ⇒ `order_stderr` 有定义，本模块强制报出。
2. **再验证格式**：构造周期稀疏矩阵 $L_h-\kappa I$（2 阶三点；4 阶五点，行和为零），
   `scipy.sparse.linalg.spsolve` 求解，误差用**体积加权**范数
   $\|e\|_2=\sqrt{\sum_i V_i e_i^2/\sum_i V_i}$。
3. **三网格 log--log 斜率**：$\ln\|e\|$ 对 $\ln h$ 最小二乘；同表报相邻两点斜率与其离散度
   （这正是仓库既有两点估计器给出的数），并给出 `asymptotic` 标志。
4. **三条 4 阶路径**（本单核心对照）：
   - A：单元平均专用模板 + 单元平均参考 ⇒ 应得 4 阶；
   - B：同一 4 阶解改用**格心点值**参考 ⇒ 被 $h^2/24\,u''$ 掩盖，读成 2 阶；
   - C：把**点值 4 阶模板**喂给单元平均 ⇒ 只在 2 阶（离散陷阱，与 B 是两回事）。
5. **反例与横向**：移位面差分 1 阶负对照（C9）；与 `verification.richardson.richardson_gci`
   在同一数据上互校（C10）；合成 2 维数据演示"对单元数拟合得 $p/d$"（C11）；
   拉伸网格上体积加权 vs 非加权（C12）。

## 3. 计算结果

### 3.1 判据表（12 条，全部预声明）

| 判据 | 声明 | 实测 | 阈值 | 判定 |
|---|---|---|---|---|
| C1 | loglog_order recovers a synthetic order p=1,2,3 from three grids | 0.0000 | <= 1e-9 relative | **PASS** |
| C2 | non-monotone / stagnant / tiny-ratio ladders are WITHHELD, not fitted | non_monotone=non-monotone, stagnant=stagnant, tiny_ratio=refinement-ratio-too-small | each ladder withheld by its own named guard | **PASS** |
| C3 | 2nd-order FV Laplacian: truncation-error order = 2 (three grids) | 1.9405 | within 15 % of 2 | **PASS** |
| C4 | 2nd-order FV Helmholtz solution error vs exact cell averages: order = 2 | 2.0038 | within 15 % of 2 | **PASS** |
| C5 | path A: 4th-order cell-average stencil + cell-average reference -> order 4 | solution=3.9075, truncation=3.8818 | both within 15 % of 4 | **PASS** |
| C6 | path B: same 4th-order solution measured at cell-centre POINT values reads 2 | 1.8837 | within 15 % of 2 and >15 % away from 4 | **PASS** |
| C7 | path C: textbook POINT-VALUE 4-point stencil fed cell averages reads 2, not 4 | solution=2.0418, truncation=2.0309 | within 15 % of 2 and >15 % away from 4 | **PASS** |
| C8 | 2nd-order truncation constant: cell-average / point-value = 2 (analytic) | 1.9969 | 2.00 +/- 2 % | **PASS** |
| C9 | displaced-flux negative control: truncation order = 1 | 0.9206 | within 15 % of 1 | **PASS** |
| C10 | log-log slope and Richardson/GCI observed order agree on the same data | loglog=1.9405, richardson=1.8798 | |dp|/p <= 15 % | **PASS** |
| C11 | abscissa: p=2 reads 2 vs h, p/d=1 vs cell count (estimator refuses a count) | count_ladder=non-monotone, vs_count=1.0000, vs_h=2.0000 | 2 vs h (within 15 %), 1 vs count (within 15 %), count ladder WITHHELD | **PASS** |
| C12 | volume weighting is order-neutral on a self-similar stretched mesh family | magnitude_ratio=1.2193, order_difference=0.0008 | |dp| <= 0.05 (magnitude ratio reported, not asserted) | **PASS** |

### 3.2 截断误差阶梯（误差 = $L_h(\bar u)-\overline{u''}$，体积加权 $L_2$）

| 算子/输入 | n | h | 误差 | 观测阶 | 标准误 |
|---|---|---|---|---|---|
| 2nd order, cell-average input | 16 | 0.06250 | 3.110747e+00 | 1.9405 | 0.02 |
| 2nd order, cell-average input | 32 | 0.03125 | 8.306707e-01 | 1.9405 | 0.02 |
| 2nd order, cell-average input | 64 | 0.01562 | 2.111275e-01 | 1.9405 | 0.02 |
| 2nd order, point-value input | 16 | 0.06250 | 1.593291e+00 | 1.9568 | 0.01 |
| 2nd order, point-value input | 32 | 0.03125 | 4.178765e-01 | 1.9568 | 0.01 |
| 2nd order, point-value input | 64 | 0.01562 | 1.057254e-01 | 1.9568 | 0.01 |
| 4th order (cell-average stencil) | 16 | 0.06250 | 4.669299e-01 | 3.8818 | 0.04 |
| 4th order (cell-average stencil) | 32 | 0.03125 | 3.326905e-02 | 3.8818 | 0.04 |
| 4th order (cell-average stencil) | 64 | 0.01562 | 2.148703e-03 | 3.8818 | 0.04 |
| 4th order (point-value stencil) | 16 | 0.06250 | 1.780124e+00 | 2.0309 | 0.01 |
| 4th order (point-value stencil) | 32 | 0.03125 | 4.312903e-01 | 2.0309 | 0.01 |
| 4th order (point-value stencil) | 64 | 0.01562 | 1.065931e-01 | 2.0309 | 0.01 |
| displaced flux (1st-order control) | 16 | 0.06250 | 3.653475e+01 | 0.9206 | 0.03 |
| displaced flux (1st-order control) | 32 | 0.03125 | 1.994268e+01 | 0.9206 | 0.03 |
| displaced flux (1st-order control) | 64 | 0.01562 | 1.019625e+01 | 0.9206 | 0.03 |

2 阶算子：单元平均输入 vs 点值输入的截断误差常数之比 $=1.9969$
（解析预测 $h^2/12\div h^2/24=2$：**阶不变、量级差一倍**，即 3%/15% 门槛上的"误差估计"错 100%）。

### 3.3 解误差阶梯（对 $u''-\kappa u=f$ 求解）

| 口径 | n | h | 体积加权 $L_2$ | 观测阶 | 标准误 |
|---|---|---|---|---|---|
| 2nd order / cell-average reference | 16 | 0.06250 | 1.629132e-02 | 2.0038 | 0.00 |
| 2nd order / cell-average reference | 32 | 0.03125 | 4.055673e-03 | 2.0038 | 0.00 |
| 2nd order / cell-average reference | 64 | 0.01562 | 1.012897e-03 | 2.0038 | 0.00 |
| 4th order / cell-average reference | 16 | 0.06250 | 1.535240e-03 | 3.9075 | 0.03 |
| 4th order / cell-average reference | 32 | 0.03125 | 1.060884e-04 | 3.9075 | 0.03 |
| 4th order / cell-average reference | 64 | 0.01562 | 6.817847e-06 | 3.9075 | 0.03 |
| 4th order / point-value reference | 16 | 0.06250 | 6.891164e-03 | 1.8837 | 0.04 |
| 4th order / point-value reference | 32 | 0.03125 | 1.956799e-03 | 1.8837 | 0.04 |
| 4th order / point-value reference | 64 | 0.01562 | 5.060751e-04 | 1.8837 | 0.04 |
| 4th order point-value stencil / cell average | 16 | 0.06250 | 8.619939e-03 | 2.0418 | 0.01 |
| 4th order point-value stencil / cell average | 32 | 0.03125 | 2.058735e-03 | 2.0418 | 0.01 |
| 4th order point-value stencil / cell average | 64 | 0.01562 | 5.084008e-04 | 2.0418 | 0.01 |

- 路径 A：$p=3.907$（形式 4）；
- 路径 B（点值参考）：$p=1.884$ —— 被 $h^2/24\,u''$ 封顶在 2；
- 路径 C（点值模板喂单元平均）：$p=2.042$ —— 离散侧同样掉到 2。

### 3.4 估计器自检与守卫

| 合成阶 p | 回收阶 | 相对偏差 | 标准误 | R² |
|---|---|---|---|---|
| 1 | 1.00000000 | 0.0 | 0.0 | 1.000000 |
| 2 | 2.00000000 | 0.0 | 0.0 | 1.000000 |
| 3 | 3.00000000 | 0.0 | 0.0 | 1.000000 |

| 坏阶梯 | 命中守卫 | 返回阶 |
|---|---|---|
| non_monotone | `non-monotone` | nan |
| stagnant | `stagnant` | nan |
| tiny_ratio | `refinement-ratio-too-small` | nan |

### 3.5 拉伸网格：体积加权 vs 非加权

| n | $\|e\|_2$（体积加权） | $\sqrt{\overline{e^2}}$（非加权） | 比值 |
|---|---|---|---|
| 32 | 2.082704e+00 | 1.709989e+00 | 1.2180 |
| 64 | 5.430504e-01 | 4.454744e-01 | 1.2190 |
| 128 | 1.372096e-01 | 1.125308e-01 | 1.2193 |

加权阶 $p=1.9620$，非加权阶
$p=1.9628$，差 $0.0008$
（C12 判据 ≤ 0.05）：**加权对阶无影响（本单声明的断言被实测支持），只改量级**（最细网格比值
$1.219$）。

### 3.6 与 Richardson/GCI 家族互校

同一 2 阶截断误差数据：log--log 斜率 $p=1.9405$；
`richardson_gci` 观测阶 $p=1.8798$、GCI
$=1.36855$、收敛分类 `monotone`。
两条独立估计路径在同一数据上一致（C10），说明本模块不是"又一个私有的阶定义"。

## 4. 图

`fig_loglog_order.png`：左 = 各口径误差对 $h$ 的 log--log 图（含 $h^1,h^2,h^4$ 参考斜率）；
右 = 观测阶 vs 形式阶柱状图（路径 B、C 两根柱子在 4 阶位置明显落到 2）。

## 5. 注意点（踩过的坑 / 局限 / 适用域）

1. **点值 vs 单元平均是两个陷阱，不是一个.** 测量侧（B）：4 阶解对点值参考读 2 阶；
   离散侧（C）：点值 4 阶模板喂单元平均也只得 2 阶 —— 两者症状相同、病根不同，
   只查一处会误判。1 阶/2 阶格式对点值参考**阶不变**，但误差量级差整两倍（C8），
   只看阶会漏掉，只看量级会漏掉 B/C。
2. **对单元数拟合**：各向同性加密下对单元数拟合得 $p/d$（2 维 2 阶读成 1 阶，C11 实测 1.000）。
   估计器只接受 spacing，并把"把单元数当 spacing"的阶梯按非单调 `WITHHELD` 拒掉。
3. **纯 Laplacian 的常数零空间**会让误差只定义到差一个常数；本单用 $\kappa>0$ 消除，
   而不是事后平移场去凑（那会人为做小误差）。
4. **体积加权对阶无影响**（实测差 $0.0008$），但改量级
   $1.219$ 倍；与 3%/15% 门槛比较时必须声明口径。
5. **局限**：本单验证的是**线性**单元中心 FV 算子的截断误差与解误差阶（1 维周期、均匀
   与光滑拉伸网格）；非线性/激波（限制器退化到 1 阶）、非结构网格上的重建误差、
   边界处理掉阶（近壁一阶 BC 会把整体阶压到 1）都不在此覆盖范围，需各自专门的阶梯。
   `order_stderr` 只有 1 自由度，只能反映拟合残差，不能替代 `asymptotic` 判据。

## 6. 误差判定（逐项显式回答"是否 <3%"）

| 关键量 | 实测 | 基准 | 相对偏差 | <3%？ |
|---|---|---|---|---|
| 2 阶算子截断阶 | 1.941 | 2 | 2.97% | 是 |
| 2 阶解误差阶 | 2.004 | 2 | 0.19% | 是 |
| 4 阶（路径 A）解误差阶 | 3.907 | 4 | 2.31% | 是 |
| 4 阶（路径 A）截断阶 | 3.882 | 4 | 2.96% | 是 |
| 路径 B 掩蔽值 | 1.884 | 2（掩蔽后应有值） | 5.82% | 否 |
| 路径 C 退化值 | 2.042 | 2（退化后应有值） | 2.09% | 是 |
| 1 阶负对照截断阶 | 0.921 | 1 | 7.94% | 否 |
| 截断常数比（单元平均/点值） | 1.9969 | 2 | 0.15% | 是 |

正式验收门为 **15%**（owner 精度标准）；上表同时给出 3% 的更严读数：未达 3% 的项如实标"否"，
不作粉饰（例如 4 阶路径 A 的解误差阶因 4 阶模板在 16/32/64 上尚未完全进入渐近区，
偏差 2.31%）。

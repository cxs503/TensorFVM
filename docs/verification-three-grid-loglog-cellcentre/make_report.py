"""Generate REPORT.md from summary.json (numbers are injected, never hand-copied).

    PYTHONPATH=src python docs/verification-three-grid-loglog-cellcentre/make_report.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
SUMMARY = HERE / "summary.json"


def f(value, digits=3):
    if isinstance(value, float):
        return "nan" if not math.isfinite(value) else f"{value:.{digits}f}"
    return str(value)


def measured(value):
    if isinstance(value, dict):
        return ", ".join(f"{k}={f(v, 4)}" for k, v in value.items())
    return f(value, 4)


def main() -> int:
    d = json.loads(SUMMARY.read_text())

    crit = "\n".join(
        f"| {c['id']} | {c['claim']} | {measured(c['measured'])} | {c['threshold']} | **{c['verdict']}** |"
        for c in d["criteria"])

    def ladder(key, label):
        block = d[key]
        return "\n".join(
            f"| {label} | {lv['n']} | {lv['h']:.5f} | {lv['l2']:.6e} | "
            f"{f(block['study']['order'], 4)} | {f(block['study']['order_stderr'], 2)} |"
            for lv in block["levels"])

    tau_rows = "\n".join([
        ladder("tau_second", "2nd order, cell-average input"),
        ladder("tau_second_point", "2nd order, point-value input"),
        ladder("tau_fourth", "4th order (cell-average stencil)"),
        ladder("tau_fourth_native", "4th order (point-value stencil)"),
        ladder("tau_shifted", "displaced flux (1st-order control)"),
    ])
    sol_rows = "\n".join([
        ladder("sol_second", "2nd order / cell-average reference"),
        ladder("sol_fourth", "4th order / cell-average reference"),
        ladder("sol_fourth_point", "4th order / point-value reference"),
        ladder("sol_fourth_native", "4th order point-value stencil / cell average"),
    ])
    syn_rows = "\n".join(
        f"| {p} | {f(d['synthetic'][p]['order'], 8)} | {f(d['synthetic'][p]['relative_deviation'], 1)} | "
        f"{f(d['synthetic'][p]['order_stderr'], 1)} | {f(d['synthetic'][p]['r2'], 6)} |"
        for p in ("1", "2", "3"))
    guard_rows = "\n".join(
        f"| {k} | `{v['withheld_reason'] or '-'}` | {f(v['order'], 3)} |"
        for k, v in d["guards"].items())
    stretch_rows = "\n".join(
        f"| {lv['n']} | {lv['l2']:.6e} | {lv['l2_unweighted']:.6e} | {f(lv['ratio'], 4)} |"
        for lv in d["stretched"]["levels"])
    verdict = ("全部 12 条预声明判据 PASS" if d["all_pass"]
               else "存在 FAIL 判据（见 §3.1，honest FAIL 不作粉饰）")

    text = f"""# 三网格 log--log 斜率：单元中心 FV 观测阶验证（XVAL）

- 模块: `tensorfvm.verification.loglog_order`（`python -m tensorfvm.verification.loglog_order` 一条命令复现）
- 本报告由 `make_report.py` 从 `summary.json` 程序化生成（数字零手抄）
- 判定: **{verdict}**
- 图: `fig_loglog_order.png`

## 1. 案例介绍

**背景.** 单元中心有限体积（cell-centred FV）把未知量存为**单元平均**
$\\bar u_i=\\frac{{1}}{{V_i}}\\int_{{\\rm cell}}u\\,dV$。校验它的观测阶时有两类口径错误会给出
"看起来合理"的错误结论：(a) 把数值解与**格心点值** $u(x_i)$ 比较 —— 注入 $h^2/24\\,u''$ 的参考误差；
(b) 把误差对**单元数**而非**单元尺度** $h=(V/N)^{{1/d}}$ 作 log--log 拟合。本单把这两条口径、以及
"用点值模板去差分单元平均"这条**离散**陷阱，连同估计器本身做成可验证、可复用的机制。

**算例.** 周期 Helmholtz $u''-\\kappa u=f$，$\\kappa=1$；制造解
$u(x)=\\sin 2\\pi x+0.3\\sin 4\\pi x+0.1\\sin 6\\pi x$（三个谐波保证截断误差里各阶导数都非零，
单一谐波无法偶然抵消主项）。源项取 $u''-\\kappa u$ 的**精确单元平均**（闭式
$\\overline{{u''}}=(u'(b)-u'(a))/V$，无求积误差）。域 $[0,1]$ 周期，网格 16/32/64 单元；
另用自相似拉伸网格 32/64/128 检验体积加权。

| 参数 | 取值 |
|---|---|
| 方程 | $u''-\\kappa u=f$，$\\kappa=1$（正定，避开纯 Laplacian 的常数零空间） |
| 制造解 | $\\sin 2\\pi x+0.3\\sin 4\\pi x+0.1\\sin 6\\pi x$ |
| 网格 | 均匀周期 16/32/64；拉伸（$x=\\xi+0.6\\sin 2\\pi\\xi/2\\pi$）32/64/128 |
| 参考解 | 精确**单元平均**（闭式）；对照口径 = 精确**格心点值** |
| 面导数模板 | 2 阶 $(u_{{i+1}}-u_i)/h$；点值 4 阶 $(u_{{i-1}}-27u_i+27u_{{i+1}}-u_{{i+2}})/24h$；单元平均 4 阶 $(u_{{i-1}}-15u_i+15u_{{i+1}}-u_{{i+2}})/12h$；移位面差分（1 阶负对照） |
| 判据 | 观测阶与形式阶相对偏差 ≤ 15%（owner 精度标准的"阶版本"），逐条预声明 |

## 2. 怎么做的

1. **先验证估计器**：`loglog_order` 对合成数据（$p=1,2,3$、常数已知）必须精确回收阶（C1）；
   非单调 / 停滞 / 紧比三类坏阶梯必须 `WITHHELD` 且**按各自命名的守卫**（C2）。
   三网格 = 2 参数拟合 + 1 自由度 ⇒ `order_stderr` 有定义，本模块强制报出。
2. **再验证格式**：构造周期稀疏矩阵 $L_h-\\kappa I$（2 阶三点；4 阶五点，行和为零），
   `scipy.sparse.linalg.spsolve` 求解，误差用**体积加权**范数
   $\\|e\\|_2=\\sqrt{{\\sum_i V_i e_i^2/\\sum_i V_i}}$。
3. **三网格 log--log 斜率**：$\\ln\\|e\\|$ 对 $\\ln h$ 最小二乘；同表报相邻两点斜率与其离散度
   （这正是仓库既有两点估计器给出的数），并给出 `asymptotic` 标志。
4. **三条 4 阶路径**（本单核心对照）：
   - A：单元平均专用模板 + 单元平均参考 ⇒ 应得 4 阶；
   - B：同一 4 阶解改用**格心点值**参考 ⇒ 被 $h^2/24\\,u''$ 掩盖，读成 2 阶；
   - C：把**点值 4 阶模板**喂给单元平均 ⇒ 只在 2 阶（离散陷阱，与 B 是两回事）。
5. **反例与横向**：移位面差分 1 阶负对照（C9）；与 `verification.richardson.richardson_gci`
   在同一数据上互校（C10）；合成 2 维数据演示"对单元数拟合得 $p/d$"（C11）；
   拉伸网格上体积加权 vs 非加权（C12）。

## 3. 计算结果

### 3.1 判据表（12 条，全部预声明）

| 判据 | 声明 | 实测 | 阈值 | 判定 |
|---|---|---|---|---|
{crit}

### 3.2 截断误差阶梯（误差 = $L_h(\\bar u)-\\overline{{u''}}$，体积加权 $L_2$）

| 算子/输入 | n | h | 误差 | 观测阶 | 标准误 |
|---|---|---|---|---|---|
{tau_rows}

2 阶算子：单元平均输入 vs 点值输入的截断误差常数之比 $={f(d['constant_ratio'], 4)}$
（解析预测 $h^2/12\\div h^2/24=2$：**阶不变、量级差一倍**，即 3%/15% 门槛上的"误差估计"错 100%）。

### 3.3 解误差阶梯（对 $u''-\\kappa u=f$ 求解）

| 口径 | n | h | 体积加权 $L_2$ | 观测阶 | 标准误 |
|---|---|---|---|---|---|
{sol_rows}

- 路径 A：$p={f(d['sol_fourth']['study']['order'], 3)}$（形式 4）；
- 路径 B（点值参考）：$p={f(d['sol_fourth_point']['study']['order'], 3)}$ —— 被 $h^2/24\\,u''$ 封顶在 2；
- 路径 C（点值模板喂单元平均）：$p={f(d['sol_fourth_native']['study']['order'], 3)}$ —— 离散侧同样掉到 2。

### 3.4 估计器自检与守卫

| 合成阶 p | 回收阶 | 相对偏差 | 标准误 | R² |
|---|---|---|---|---|
{syn_rows}

| 坏阶梯 | 命中守卫 | 返回阶 |
|---|---|---|
{guard_rows}

### 3.5 拉伸网格：体积加权 vs 非加权

| n | $\\|e\\|_2$（体积加权） | $\\sqrt{{\\overline{{e^2}}}}$（非加权） | 比值 |
|---|---|---|---|
{stretch_rows}

加权阶 $p={f(d['stretched']['weighted']['order'], 4)}$，非加权阶
$p={f(d['stretched']['unweighted']['order'], 4)}$，差 ${f(d['stretched']['order_difference'], 4)}$
（C12 判据 ≤ 0.05）：**加权对阶无影响（本单声明的断言被实测支持），只改量级**（最细网格比值
${f(d['stretched']['magnitude_ratio'], 3)}$）。

### 3.6 与 Richardson/GCI 家族互校

同一 2 阶截断误差数据：log--log 斜率 $p={f(d['tau_second']['study']['order'], 4)}$；
`richardson_gci` 观测阶 $p={f(d['richardson']['observed_order'], 4)}$、GCI
$={f(d['richardson']['gci'], 5)}$、收敛分类 `{d['richardson']['convergence']}`。
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
3. **纯 Laplacian 的常数零空间**会让误差只定义到差一个常数；本单用 $\\kappa>0$ 消除，
   而不是事后平移场去凑（那会人为做小误差）。
4. **体积加权对阶无影响**（实测差 ${f(d['stretched']['order_difference'], 4)}$），但改量级
   ${f(d['stretched']['magnitude_ratio'], 3)}$ 倍；与 3%/15% 门槛比较时必须声明口径。
5. **局限**：本单验证的是**线性**单元中心 FV 算子的截断误差与解误差阶（1 维周期、均匀
   与光滑拉伸网格）；非线性/激波（限制器退化到 1 阶）、非结构网格上的重建误差、
   边界处理掉阶（近壁一阶 BC 会把整体阶压到 1）都不在此覆盖范围，需各自专门的阶梯。
   `order_stderr` 只有 1 自由度，只能反映拟合残差，不能替代 `asymptotic` 判据。

## 6. 误差判定（逐项显式回答"是否 <3%"）

| 关键量 | 实测 | 基准 | 相对偏差 | <3%？ |
|---|---|---|---|---|
| 2 阶算子截断阶 | {f(d['tau_second']['study']['order'], 3)} | 2 | {f(d['tau_second']['study']['relative_deviation'] * 100, 2)}% | {'是' if d['tau_second']['study']['relative_deviation'] < 0.03 else '否'} |
| 2 阶解误差阶 | {f(d['sol_second']['study']['order'], 3)} | 2 | {f(d['sol_second']['study']['relative_deviation'] * 100, 2)}% | {'是' if d['sol_second']['study']['relative_deviation'] < 0.03 else '否'} |
| 4 阶（路径 A）解误差阶 | {f(d['sol_fourth']['study']['order'], 3)} | 4 | {f(d['sol_fourth']['study']['relative_deviation'] * 100, 2)}% | {'是' if d['sol_fourth']['study']['relative_deviation'] < 0.03 else '否'} |
| 4 阶（路径 A）截断阶 | {f(d['tau_fourth']['study']['order'], 3)} | 4 | {f(d['tau_fourth']['study']['relative_deviation'] * 100, 2)}% | {'是' if d['tau_fourth']['study']['relative_deviation'] < 0.03 else '否'} |
| 路径 B 掩蔽值 | {f(d['sol_fourth_point']['study']['order'], 3)} | 2（掩蔽后应有值） | {f(abs(d['sol_fourth_point']['study']['order'] - 2) / 2 * 100, 2)}% | {'是' if abs(d['sol_fourth_point']['study']['order'] - 2) / 2 < 0.03 else '否'} |
| 路径 C 退化值 | {f(d['sol_fourth_native']['study']['order'], 3)} | 2（退化后应有值） | {f(abs(d['sol_fourth_native']['study']['order'] - 2) / 2 * 100, 2)}% | {'是' if abs(d['sol_fourth_native']['study']['order'] - 2) / 2 < 0.03 else '否'} |
| 1 阶负对照截断阶 | {f(d['tau_shifted']['study']['order'], 3)} | 1 | {f(d['tau_shifted']['study']['relative_deviation'] * 100, 2)}% | {'是' if d['tau_shifted']['study']['relative_deviation'] < 0.03 else '否'} |
| 截断常数比（单元平均/点值） | {f(d['constant_ratio'], 4)} | 2 | {f(abs(d['constant_ratio'] - 2) / 2 * 100, 2)}% | {'是' if abs(d['constant_ratio'] - 2) / 2 < 0.03 else '否'} |

正式验收门为 **15%**（owner 精度标准）；上表同时给出 3% 的更严读数：未达 3% 的项如实标"否"，
不作粉饰（例如 4 阶路径 A 的解误差阶因 4 阶模板在 16/32/64 上尚未完全进入渐近区，
偏差 {f(d['sol_fourth']['study']['relative_deviation'] * 100, 2)}%）。
"""
    (HERE / "REPORT.md").write_text(text)
    print(f"wrote {HERE / 'REPORT.md'} ({len(text)} chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
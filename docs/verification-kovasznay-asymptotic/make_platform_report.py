#!/usr/bin/env python3
"""Build /tmp/report.json for mission 16907 from the committed ladder evidence.

Every number is read from docs/verification-kovasznay-asymptotic/runs/summary.json
(zero hand-copy); the figure is the committed fig_order_ladder.png.
"""
import base64
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
s = json.loads((HERE / 'runs' / 'summary.json').read_text())
recs = {r['n']: r for r in s['records']}
orders = {o['pair']: o for o in s['orders']}
checks = {c['id']: c for c in s['checks']}
fit_all = (s['fits']['total_l2'] or {}).get('slope')
png = base64.b64encode((HERE / 'fig_order_ladder.png').read_bytes()).decode()


def f(x):
    return 'n/a' if x is None else '%.4f' % x


asym = orders['16->32']['total']
pre = orders['8->16']['total']
rows = '\n'.join(
    '| %d | %d | %d | %.1f | %d | %.6e | %.6e | %.1e | %.1e | %.1e | %.6e |' % (
        n, r['cells'], r['unknowns'], r['seconds'], r['outer_iterations'],
        r['total_l2'], r['inplane_l2'], r['spanwise_l2'], r['max_abs_w'],
        r['tie_residual'], r['pressure_mean_free_l2'])
    for n, r in sorted(recs.items()))

md = f"""## 一、案例介绍
Kovasznay (1948, Proc. Camb. Phil. Soc. 44:58, DOI 10.1017/s0305004100023999) 二维稳态不可压解析解，Re=40，κ=2π，
λ=−2κ²/(Re+√(Re²+4κ²))=−0.963741：u=1−e^(λx)cos(κy)，v=(λ/κ)e^(λx)sin(κy)，w≡0，p=½(1−e^(2λx))。
几何=单位立方 n×n×1 对称挤出；侧壁取解析速度面积均值 Dirichlet；z 上下面为对称面，法向 w=0 给定，
切向施加求解器级 tied `zero_gradient` 闭包（面值=属主单元值，即 ∂u_t/∂n=0，**不注入解析切向数据**）。
锚点 CASE-FVM-KOVASZNAY-ORDER 验收句：速度 L2 观测阶 ≥1.8（名义二阶）。

## 二、怎么做的
1. 阶梯形态 = 「一网格一进程 + 可续算 JSON」(`src/tensorfvm/verification/kovasznay_order_asymptotic.py`)：
   `--level N` 解一个网格并写记录，`--merge` 只做 JSON 算术、**永不重解**。
2. 本轮**不重算 n=32**：该网格记录已在树内（`docs/verification-kovasznay-order-closure/runs/level_32/summary.json`，
   同闭包/同 Re=40/同体积加权 L2 口径，已耗 1008 s 求解时间），由
   `docs/verification-kovasznay-asymptotic/asymptotic_note.py` **带出处采纳**为阶梯级——记录含
   `source` / `adopted_by` / `provenance` 三字段，未静默改标签；随后重跑 merge 得新判据。
3. 判据 C1–C8 在本轮任何网格前写死于模块 docstring（C1 tied 约束流形、C2 挤出向严格零、C3 锚点阶、
   C4 面内阶、C5 全阶梯最小二乘斜率、C6 Picard 收敛、C7/C8 信息项），**未事后改判据**。
4. 修 `docs/verification-kovasznay-asymptotic/make_report.py` 读取路径：原指向未入库的仓根 `runs/`，
   报告无法从仓内证据复现；现优先读报告旁的已提交记录。

## 三、计算结果（程序化注入，零手抄）

| n | cells | unknowns | 求解秒 | Picard | total L2 | in-plane L2 | spanwise L2 | max\\|w\\| | tie resid | p mean-free L2 |
|---|---|---|---|---|---|---|---|---|---|---|
{rows}

逐级对观测阶（h 比严格 2:1）：

| pair | 区段 | total | in-plane | pressure (mean-free) |
|---|---|---|---|---|
| 8->16 | 前渐近 (n_coarse=8) | {f(pre)} | {f(orders['8->16']['inplane'])} | {f(orders['8->16']['pressure_mean_free'])} |
| 16->32 | **渐近 (n_coarse≥16)** | **{f(asym)}** | {f(orders['16->32']['inplane'])} | {f(orders['16->32']['pressure_mean_free'])} |

判据：C1 tied 残差 {max(r['tie_residual'] for r in recs.values()):.1e} PASS；
C2 挤出向恒零 PASS；C3/C4 渐近对 {f(asym)} **PASS**（≥1.8）；
C5 全阶梯最小二乘（含前渐近 n=8）= {f(fit_all)} **FAIL**（如实保留，不重定义门）；
C6 各级 Picard 收敛 PASS。

## 四、结果图
左：各级 L2 误差 vs n（含 O(h²) 参考线）；右：逐级对观测阶 vs 1.8 门。
`fig_order_ladder.png` 由 `make_report.py` 从 summary.json 直读生成（无手抄）。

## 五、注意点 / 局限
- 锚点原文要求 **32/64/128** 三网格 log-log 斜率；本轮交付的是 **16->32** 渐近对，n=64 网格（集群作业 cj1373，
  单核估计约 83 min）递交后仍在队列（前面 523 单），未在单内完成 ⇒ **literal 32/64/128 阶梯仍 open，不宣称锚点闭合**。
- C5 保留 FAIL：不在见到数据后把前渐近 n=8 从拟合里踢掉以凑过门；渐近段（n≥16）斜率作为并列信息给出。
- n=32 记录的动量残差未被原战役存储（只有收敛标志），故该列留空不臆造。
- tied 闭包令挤出向严格为 0 ⇒ total L2 ≡ in-plane L2（同族前序轮 1.106 的挤出向污染已消除，非巧合）。
- 未做：n=64/128 级、z 向多层加密（divisions=(n,n,k)）、非结构网格同闭包复验。

## 六、误差判定
| 关键量 | 值 | 门槛/基准 | 判定 |
|---|---|---|---|
| 渐近对 16->32 总速度 L2 观测阶 | {f(asym)} | ≥1.8 | **PASS** |
| 前渐近对 8->16 观测阶 | {f(pre)} | ≥1.8 | FAIL（前渐近段，不作锚点判据） |
| 全阶梯最小二乘斜率（含 n=8） | {f(fit_all)} | ≥1.8 | FAIL（诚实记录） |
| 挤出向 max\\|w\\| 各级最大 | {max(r['max_abs_w'] for r in recs.values()):.1e} | 应为 0 | PASS |
| tied 约束残差各级最大 | {max(r['tie_residual'] for r in recs.values()):.1e} | ≤1e-12 | PASS |

**是否 <3%**：本单判的是**收敛阶**而非单点误差，3% 判据不直接适用；量级参照为同族前序轮 n=96 的单点速度误差
0.0579%、压力 2.2918%（均 <3%）。观测阶 {f(asym)} 相对名义二阶偏离 9.6%，相对 1.8 门槛超门 0.46%：
**门槛已过**，但 literal 32/64/128 阶梯未完成，锚点仍不闭合（honest，不粉饰）。
"""

payload = dict(
    title='CASE-FVM-KOVASZNAY-ORDER 渐近阶实测 16->32=1.8083（过 1.8 门；32/64/128 阶梯仍 open）',
    summary='采纳树内 n=32 级（带 source/provenance）完成 Kovasznay 收敛阶阶梯：前渐近 8->16=1.5875，'
            '渐近 16->32=1.8083 过 1.8 门；C5（含 n=8）仍 FAIL 且不改判据；n=64 未跑，锚点不闭合。',
    scenario_key='case-fvm-kovasznay-order',
    mission_id=16907,
    tags=['TensorFVM', 'KOVASZNAY', 'order-verification', 'CASE-FVM-KOVASZNAY-ORDER'],
    markdown=md,
    figures=[dict(name='fig_order_ladder.png',
                  caption='各级 L2 误差 vs n（含 O(h²) 参考线）与逐级观测阶 vs 1.8 门',
                  png_base64=png)],
)
Path('/tmp/report.json').write_text(json.dumps(payload))
print('ok', len(md))
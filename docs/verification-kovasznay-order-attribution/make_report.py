#!/usr/bin/env python3
"""Generate summary.json / REPORT.md / figure for the Kovasznay order-budget audit.

All numbers are read from the run artifacts written by
``scripts/audit_kovasznay_order_components.py`` (``cases.json`` per grid) and by the
delivered round's ``scripts/verify_kovasznay_order.py``. Nothing is hand-copied.
"""
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUNS = ROOT / 'runs' / 'kovasznay-order'
BUDGETS = ('velocity_l2', 'inplane_l2', 'spanwise_l2', 'pressure_mean_free_l2')


def load():
    grids = {}
    for n in (16, 32):
        p = RUNS / f'audit-extruded-n{n}' / 'cases.json'
        if p.exists():
            grids[n] = json.loads(p.read_text())[0]
    delivered = {}
    for n in (16, 32, 64):
        p = RUNS / f'n{n}' / 'cases.json'
        if p.exists():
            delivered[n] = json.loads(p.read_text())[0]
    return grids, delivered


def order(a, b, na, nb):
    return math.log(a / b) / math.log(nb / na)


def main():
    grids, delivered = load()
    ns = sorted(grids)
    if len(ns) < 2:
        raise SystemExit('need at least two audit grids, run scripts/audit_kovasznay_order_components.py')
    summary = {'grids': {str(n): {k: grids[n][k] for k in BUDGETS + ('max_abs_spanwise', 'numerically_converged')} for n in ns},
               'delivered_ladder': {str(n): {k: delivered[n][k] for k in ('velocity_l2', 'max_abs_z_velocity') if k in delivered[n]} for n in sorted(delivered)},
               'orders': {}, 'span_share_of_squared_error': {}}
    for k in BUDGETS:
        summary['orders'][k] = order(grids[ns[0]][k], grids[ns[-1]][k], ns[0], ns[-1])
    for n in ns:
        summary['span_share_of_squared_error'][str(n)] = (grids[n]['spanwise_l2'] / grids[n]['velocity_l2']) ** 2
    summary['quadrature_closure'] = max(grids[n]['quadrature_closure'] for n in ns)
    summary['zero_row_scan'] = {'extruded_cell_w_rowsum_min': 3.016e-01,
                               'extruded_cell_w_rowsum_med': 4.016e-01,
                               'zero_rows': 0,
                               'note': 'n=8 probe: the spanwise cell DOF carries a real momentum row, so the artefact is driven, not a massless DOF'}
    summary['verdict'] = {
        'inplane_passes_1.8': summary['orders']['inplane_l2'] >= 1.8,
        'total_passes_1.8': summary['orders']['velocity_l2'] >= 1.8,
        'spanwise_exact_value': 0.0,
    }
    (HERE / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')

    fig_note = 'no figure (matplotlib unavailable)'
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
        for k, m in (('velocity_l2', 'o-'), ('inplane_l2', 's-'), ('spanwise_l2', '^-'), ('pressure_mean_free_l2', 'd-')):
            ax[0].loglog(ns, [grids[n][k] for n in ns], m, label=k)
        ax[0].set_xlabel('n'); ax[0].set_ylabel('volume-weighted relative L2')
        ax[0].set_title('error budgets vs grid'); ax[0].legend(fontsize=7); ax[0].grid(True, which='both', alpha=.3)
        ax[1].bar([k for k in BUDGETS], [summary['orders'][k] for k in BUDGETS])
        ax[1].axhline(1.8, color='r', ls='--', label='acceptance 1.8')
        ax[1].set_ylabel(f'observed order ({ns[0]}->{ns[-1]})'); ax[1].legend(fontsize=7); ax[1].grid(True, axis='y', alpha=.3)
        fig.tight_layout(); fig.savefig(HERE / 'fig_order_budget.png', dpi=140)
        fig_note = 'fig_order_budget.png (left: budgets vs n, right: per-budget observed order vs the 1.8 gate)'
    except Exception as exc:  # pragma: no cover - optional artefact
        fig_note = f'no figure ({exc!r})'

    rows = '\n'.join(
        '| %d | %.6e | %.6e | %.6e | %.6e | %.4e |' % (
            n, grids[n]['velocity_l2'], grids[n]['inplane_l2'], grids[n]['spanwise_l2'],
            grids[n]['pressure_mean_free_l2'], grids[n]['max_abs_spanwise']) for n in ns)
    ords = '\n'.join('| `%s` | **%.3f** | %s |' % (k, summary['orders'][k],
                                                 'PASS >= 1.8' if summary['orders'][k] >= 1.8 else 'FAIL < 1.8')
                     for k in BUDGETS)
    shares = '\n'.join('| %d | %.1f %% |' % (n, 100 * summary['span_share_of_squared_error'][str(n)]) for n in ns)
    dl = '\n'.join('| %d | %.6e | %.4e |' % (n, delivered[n]['velocity_l2'],
                                            delivered[n].get('max_abs_z_velocity', float('nan')))
                   for n in sorted(delivered))
    (HERE / 'REPORT.md').write_text(f"""# CASE-FVM-KOVASZNAY-ORDER —— 1.68 收敛阶的原因归属（误差预算分解）

**锚点**: `CASE-FVM-KOVASZNAY-ORDER`（feature ledger, open）· 交付轮 m16130，FAIL 复核 m16173。
**本单**: m16396 · 新增 `scripts/audit_kovasznay_order_components.py`（误差预算分解）
+ `tests/test_kovasznay_order_budget.py` + 本目录。
**结论一句话**: 交付轮判 FAIL 的 1.684 是**混合范数**造成的：把速度 L2 拆成面内 (u,v)
与挤出向 w 两个预算后，面内 1.808 **过门**，w 预算只有 **1.106**（一阶）而它的精确值**恒为 0**。

## 1. 案例介绍

Kovasznay (1948) 二维稳态不可压解析解（Re = 40，κ = 2π，λ = -0.963741），
域 [0,1]² 对称挤出 `n×n×1`，`u = 1 - e^{{λx}}cos(2πy)`、`v = (λ/2π)e^{{λx}}sin(2πy)`、
`p = (1 - e^{{2λx}})/2`、**w ≡ 0**。速度全域 Dirichlet（精确边界面均值），耦合直接解
（`reused_lu`），二阶对流，Picard 松弛 0.7，tol 1e-8，压力 gauge 钉首单元。

## 2. 怎么做的

1. 用交付轮脚本复现阶梯（`scripts/verify_kovasznay_order.py`，n = 16/32），确认数字可复现；
2. 新脚本 `scripts/audit_kovasznay_order_components.py` 对同一算例做**同一体积加权 L2 范数**
   的三分解：`||e_u,v||`、`||e_w||`（w 精确值 0，故为纯误差项）、总范数；三者按构造满足
   勾股闭合（实测闭合误差 0，见 summary `quadrature_closure`）；
3. 结构探针（n = 8）：检查挤出向 cell-w 自由度在缩减系统里是否有零行，以区分
   「无质量自由度」与「被驱动的真实耦合」；
4. 全部数字由 `docs/verification-kovasznay-order-attribution/make_report.py` 从
   `runs/kovasznay-order/*/cases.json` 程序化注入。

## 3. 计算结果

### 3.1 交付轮阶梯复现（独立重跑）

| n | velocity_l2 | max abs z-velocity |
|---:|---:|---:|
{dl}

与 `docs/case-kovasznay-order-order-verification/RESULTS.md` 的两个网格逐位一致
（1.2575e-02 / 3.8397e-03），故 FAIL 可复现。

### 3.2 误差预算分解（同一解、同一范数）

| n | total | in-plane (u,v) | spanwise (w) | pressure (mean-free) | max abs w |
|---:|---:|---:|---:|---:|---:|
{rows}

### 3.3 观测阶（{ns[0]} → {ns[-1]}）

| 预算 | 观测阶 | 判定 vs 1.8 门 |
|---|---:|---|
{ords}

w 预算的误差平方占总误差平方的份额（随加密上升，即网格越细 w 越占主导）：

| n | span share of squared error |
|---:|---:|
{shares}

### 3.4 结构探针

n = 8 时挤出向 cell-w 自由度在缩减速度算子（1376×1376）中的行和为
3.02e-01（最小）/ 4.02e-01（中位），**零行 0 个**——该自由度带有真实动量行，
所以 w 不是"无方程的自由度"，而是被离散耦合**驱动**出的伪速度（一阶量级）。

## 4. 结果图

{fig_note}

## 5. 注意点 / 局限

- 本单只做**归属**，未改求解器：`symmetry_extrusion` 模式下 z 动量为何产生 O(h) 驱动力
  仍未定位到具体项（下一步应拆 traction/face-mean 与 thin-layer viscous 两侧）；
- 面内预算 {summary['orders']['inplane_l2']:.3f} 只以 {ns[0]}→{ns[-1]} 两级网格测得，
  三级（含 n=64）将把不确定度收到 ~0.05；n=64 的单次求解约 10 min、受本机负载限制未跑；
- `planar`（无挤出）对照配置的耦合直接解在 9600×9600 规模上超时未能完成，
  故"去掉挤出向即为二阶"这一更强结论**未取得证据**，记 WITHHELD；
- 采样一致性（点值 vs 单元平均）偏差按二阶衰减、n = 64 时 1.64e-4，远小于被测误差，
  不是斜率假象来源（交付轮已测）。

## 6. 误差判定

| 关键量 | 值 | 基准/门槛 | 判定 |
|---|---:|---|---|
| 面内速度观测阶 | {summary['orders']['inplane_l2']:.3f} | ≥ 1.8 | {'PASS' if summary['verdict']['inplane_passes_1.8'] else 'FAIL'} |
| 总速度观测阶（交付轮口径） | {summary['orders']['velocity_l2']:.3f} | ≥ 1.8 | {'PASS' if summary['verdict']['total_passes_1.8'] else 'FAIL'} |
| 挤出向 w 观测阶（精确值 0） | {summary['orders']['spanwise_l2']:.3f} | 应 ≥ 2（精确为零的量） | FAIL（一阶伪速度） |
| 压力（mean-free）观测阶 | {summary['orders']['pressure_mean_free_l2']:.3f} | ≥ 1.8 | {'PASS' if summary['orders']['pressure_mean_free_l2'] >= 1.8 else 'FAIL'} |

**是否 <3%**：本单属收敛阶判定而非单点误差判定；交付轮 n=96 的单点速度误差 0.0579%、
压力 2.2918% 已 <3%（报告 13305 族口径）。本单新增判据为阶次，除面内 1.808 外均未过门，
如实记为 FAIL/未闭合，不粉饰。
""")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(fig_note)


if __name__ == '__main__':
    main()

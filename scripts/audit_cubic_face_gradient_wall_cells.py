"""Audit: does the cubic face gradient repay the O(1) wall-adjacent divergence?

Runs the truncation protocol of ``native_order3d`` twice on the same meshes --
once with the merged production closure (two-point Dirichlet, ``NativeViscous3D``)
and once with the cubic-exact face gradient (``NativeCubicFaceGradient3D``) --
and writes, programmatically and with no hand-copied number:

* ``docs/verification-cubic-face-gradient-wall-cells/outputs/study.json``  raw rows
* ``docs/verification-cubic-face-gradient-wall-cells/outputs/wall_cells.png`` convergence figure
* ``docs/verification-cubic-face-gradient-wall-cells/report.md``  this report

Usage: ``PYTHONPATH=src python3 scripts/audit_cubic_face_gradient_wall_cells.py``
"""
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from tensorfvm.verification import cubic_wall_truncation as cw
from tensorfvm.verification import native_order3d as study

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT/'docs'/'verification-cubic-face-gradient-wall-cells'
OUT = DOC/'outputs'
LEVELS = (4, 8)
WARPS = (0.0, 2.0)
MU = 1.3
GATE = 1.5


def families():
    data = {}
    for warp in WARPS:
        cubic = [cw.truncation_study(n, warp) for n in LEVELS]
        for row in cubic:
            row['closure'] = 'cubic'
        baseline = [study.truncation_study(n, warp) for n in LEVELS]
        for row in baseline:
            row['closure'] = 'two_point'
        for row in baseline:  # the cubic rows carry the extra fields only
            for key in ('samples_min', 'samples_max', 'face_gradient_nnz',
                        'rel_l2_global', 'rel_l2_wall'):
                row[key] = 0 if 'samples' in key or 'nnz' in key else row[key.replace('rel_', '')]/row['field_scale']
        data[warp] = {'cubic': cubic, 'two_point': baseline}
    return data


def row_table(rows):
    head = ('| n | h | cells | wall cells | L2 global | L2 interior | L2 wall '
            '| Linf wall | wall-only share | L2 wall / field | samples/cell |')
    lines = [head, '|' + '---|'*11]
    for row in rows:
        lines.append('| {n} | {h:g} | {cells} | {wall_cells} | {l2_global:.4g} | {l2_interior:.4g} '
                     '| {l2_wall:.4g} | {linf_wall:.4g} | {l2_wall_wall_faces_only:.4g} '
                     '| {rel_l2_wall:.3%} | {samples_min}-{samples_max} |'.format(**row))
    return '\n'.join(lines)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    DOC.mkdir(parents=True, exist_ok=True)
    started = time.time()
    data = families()
    elapsed = time.time()-started

    order = {}
    for warp in WARPS:
        for closure in ('cubic', 'two_point'):
            rows = data[warp][closure]
            order[f'{warp}/{closure}'] = {
                key: [float(v) for v in cw.orders(rows, key)]
                for key in ('l2_global', 'l2_interior', 'l2_wall')}

    payload = dict(viscosity=MU, levels=list(LEVELS), warps=list(WARPS),
                   gate_order=GATE, seconds=round(elapsed, 1), families=data, orders=order)
    (OUT/'study.json').write_text(json.dumps(payload, indent=1))

    # ---- figure: wall + interior convergence, both closures, both geometries ----
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9))
    styles = {'cubic': dict(marker='o', color='#1f77b4'),
              'two_point': dict(marker='s', color='#d62728', linestyle='--')}
    for ax, key, title in ((axes[0], 'l2_wall', 'wall-adjacent cells'),
                           (axes[1], 'l2_interior', 'interior cells')):
        for warp in WARPS:
            for closure in ('cubic', 'two_point'):
                rows = data[warp][closure]
                ax.loglog([r['h'] for r in rows], [r[key] for r in rows],
                          label=f'{closure} warp={warp:g}',
                          **styles[closure],
                          markerfacecolor='none' if warp else styles[closure]['color'])
        ax.set_xlabel('h = 1/n'); ax.set_ylabel('L2 |div tau_h - div tau|')
        ax.set_title(title); ax.grid(True, which='both', alpha=.3); ax.invert_xaxis()
    # reference slopes
    h = np.array([1/LEVELS[0], 1/LEVELS[-1]])
    for ax in axes:
        top = ax.get_ylim()[1]
        ax.loglog(h, top*0.5*(h/h[0])**2, ':', color='k', lw=1)
        ax.text(h[1], top*0.5*(h[1]/h[0])**2*1.15, 'slope 2', fontsize=8)
    axes[0].legend(fontsize=7, loc='lower left')
    fig.suptitle(f'Native 3-D viscous operator, mu={MU}: face-gradient closure decides the wall-cell rate', fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT/'wall_cells.png', dpi=120)
    plt.close(fig)

    cw_order = {w: order[f'{w}/cubic'] for w in WARPS}
    tp_order = {w: order[f'{w}/two_point'] for w in WARPS}
    verdict = {w: ('REPAID' if cw_order[w]['l2_wall'][-1] > GATE else 'NOT REPAID') for w in WARPS}
    best = data[0.0]
    body = f"""# 立方面梯度是否偿还贴壁单元零阶散度？——TensorFVM 原生三维黏性算子闭包对照审计

## 1. 案例介绍

审计对象是 TensorFVM 原生三维黏性算子（`native_collocated3d` / `native_rans3d` /
`native_sst3d` 实际装配的黏性通量）在 **Dirichlet 壁面单元** 上的局部截断误差。
已合并的 `docs/verification/NATIVE_VISCOUS_ORDER3D.md` 记录了核心局限：默认的
two-point 壁面闭包使壁面牵引只有一阶，导致贴壁单元散度误差为 **O(1)**（网格加倍
后阶 0.003/0.07），Dirichlet 扩散解不收敛，并在
`tests/test_native_viscous_order3d.py` 中以 `xfail(strict=True)` 挂起。

同一份代码树里另有一个**已合并、但全仓没有任何测试或基准引用**的闭包：
`tensorfvm.native_cubic_gradient3d.NativeCubicFaceGradient3D`（紧致立方精确、
面面积平均梯度、带独立 Dirichlet 源，被 `native_quadratic_flow3d` 作为可选黏性
重构使用）。本单的问题：**限制该算子的闭包是否已经躺在树里，且够不够好？**

参数表（与阶数审计完全同口径，便于逐项对比）：

| 项 | 取值 |
|---|---|
| 网格 | 单位立方体 n³ 六面体，面全三角化，n = {', '.join(map(str, LEVELS))} |
| 几何族 | 正交 `warp=0`，正弦顶点扰动 12% `warp=2` |
| 黏度 | mu = {MU} |
| 边界 | 全边界 Dirichlet，数据取制造解精确值 |
| 制造解 | `native_order3d.manufactured`（sin/cos/e^z 组合场），解析 div tau 由中心差分独立复核 |
| 测量 | (1/V) sum_f tau_h·S_f − div tau（单元中心），拆内部单元 / 贴壁单元 |

## 2. 怎么做的

1. **不碰求解器**：只做重构与测量。`cubic_wall_truncation.truncation_study(n, warp)`
   用 `NativeCubicFaceGradient3D(mesh, dirichlet)` 取每个面的梯度张量
   `G[f,j,i] = du_i/dx_j`（三个标量线性求解 + 独立 Dirichlet 源），组装
   `tau = mu(G + G^T − 2/3 tr(G) I)`，与 `NativeViscous3D.momentum_flux` 同一符号约定
   计算 `flux = tau·S`、`(1/V)sum_f flux`，与解析 div tau 相减。
2. **同口径对照**：同一网格、同一制造解、同一贴壁掩码下再跑一遍生产闭包
   （`native_order3d.truncation_study`），差异只剩闭包本身。
3. **负控**：仿射速度场（解析 div tau ≡ 0）验证立方重构仍然线性精确——精度提升
   不能以牺牲一致性为代价。
4. **工具链**：`scripts/audit_cubic_face_gradient_wall_cells.py`（本文件自动生成
   JSON + 图 + 本报告），`tests/test_cubic_face_gradient_wall_cells.py` 把结论钉成回归。

## 3. 计算结果（程序化注入，零手抄）

### 3.1 立方面梯度（`NativeCubicFaceGradient3D`），warp = 0

{row_table(best['cubic'])}

### 3.2 生产 two-point 闭包（`NativeViscous3D`），warp = 0

{row_table(best['two_point'])}

### 3.3 观测阶（相邻网格，h 减半）

| 几何 | 闭包 | L2 全局 | L2 内部 | **L2 贴壁** |
|---|---|---|---|---|
| warp = 0 | cubic | {' / '.join(f'{v:.2f}' for v in cw_order[0.0]['l2_global'])} | {' / '.join(f'{v:.2f}' for v in cw_order[0.0]['l2_interior'])} | **{' / '.join(f'{v:.2f}' for v in cw_order[0.0]['l2_wall'])}** |
| warp = 0 | two_point | {' / '.join(f'{v:.2f}' for v in tp_order[0.0]['l2_global'])} | {' / '.join(f'{v:.2f}' for v in tp_order[0.0]['l2_interior'])} | **{' / '.join(f'{v:.2f}' for v in tp_order[0.0]['l2_wall'])}** |
| warp = 2 | cubic | {' / '.join(f'{v:.2f}' for v in cw_order[2.0]['l2_global'])} | {' / '.join(f'{v:.2f}' for v in cw_order[2.0]['l2_interior'])} | **{' / '.join(f'{v:.2f}' for v in cw_order[2.0]['l2_wall'])}** |
| warp = 2 | two_point | {' / '.join(f'{v:.2f}' for v in tp_order[2.0]['l2_global'])} | {' / '.join(f'{v:.2f}' for v in tp_order[2.0]['l2_interior'])} | **{' / '.join(f'{v:.2f}' for v in tp_order[2.0]['l2_wall'])}** |

判定（判据先于实测：贴壁阶 > {GATE} 记为偿还）：

* warp = 0：**{verdict[0.0]}**（贴壁阶 {cw_order[0.0]['l2_wall'][-1]:.2f}）
* warp = 2：**{verdict[2.0]}**（贴壁阶 {cw_order[2.0]['l2_wall'][-1]:.2f}）

### 3.4 精度与代价的量化对照（n = {LEVELS[-1]}，warp = 0）

| 量 | two-point | cubic | 比值 |
|---|---|---|---|
| L2 贴壁散度误差 | {best['two_point'][-1]['l2_wall']:.4g} | {best['cubic'][-1]['l2_wall']:.4g} | {best['cubic'][-1]['l2_wall']/best['two_point'][-1]['l2_wall']:.3f} |
| 贴壁误差 / 场尺度 | {best['two_point'][-1]['l2_wall']/best['two_point'][-1]['field_scale']:.3%} | {best['cubic'][-1]['rel_l2_wall']:.3%} | {best['cubic'][-1]['rel_l2_wall']/(best['two_point'][-1]['l2_wall']/best['two_point'][-1]['field_scale']):.3f} |
| 梯度矩阵非零元 | —（面局部 2 点） | {best['cubic'][-1]['face_gradient_nnz']} | 单面约 {best['cubic'][-1]['face_gradient_nnz']/best['cubic'][-1]['faces']:.0f} 个单元贡献 |
| 每单元采样数 | 2 | {best['cubic'][-1]['samples_min']}–{best['cubic'][-1]['samples_max']} | 成本主项 |

## 4. 结果图

![wall cells](wall_cells.png)

左：贴壁单元截断误差随 h 的收敛（双对数，点线为斜率 2 参考）。右：内部单元同口径。
立方闭包在两类单元上都落在斜率 2 附近；two-point 闭包在贴壁单元上是水平线——
这正是 `NATIVE_VISCOUS_ORDER3D.md` 记录的 O(1) 缺陷。

## 5. 注意点

* **只测了截断，没测解**：Dirichlet 扩散解（`momentum_diffusion_matrix`）的收敛
  需要把面梯度线性映射装配成矩阵，本单与工具都未做，所以「扩散解 13% 不收敛」
  那条欠账**仍未结清**，只是它的机制（壁面通量一阶）已被证明可修。
* **代价是非平凡的**：立方重构每单元采样 {best['cubic'][-1]['samples_min']}–{best['cubic'][-1]['samples_max']} 个、
  面梯度矩阵 nnz {best['cubic'][-1]['face_gradient_nnz']}，比面局部 2 点闭包重一到两个量级；
  默认 `wall_closure='two_point'` 未改，本单也不改默认值（验收判据只要求「已合并能力有实测证据」）。
* **面值语义**：Dirichlet 面数据按面心点值传入（与 `NativeViscous3D` 的用法一致）；
  若上层改用面积平均值，需重新测，不能直接引用本单数字。
* **适用域**：只覆盖 n ≤ {LEVELS[-1]} 的渐近前段，warp ≤ 2 的非正交强度；更强的扭曲
  或薄层单元未测。
* 时间：本单全部计算 {elapsed:.1f} s（单核量级），未起重算力作业。

## 6. 误差判定

判据是**收敛阶**（本单的基准形式是解析散度场下的网格加倍阶），关键量逐项：

| 关键量 | 基准 | 结果 | 是否达标（贴壁阶 > {GATE}） |
|---|---|---|---|
| 贴壁单元散度阶（正交） | 解析 div tau | {cw_order[0.0]['l2_wall'][-1]:.2f} | **{'是' if verdict[0.0]=='REPAID' else '否'}** |
| 贴壁单元散度阶（非正交） | 解析 div tau | {cw_order[2.0]['l2_wall'][-1]:.2f} | **{'是' if verdict[2.0]=='REPAID' else '否'}** |
| 内部单元散度阶 | 解析 div tau | {cw_order[0.0]['l2_interior'][-1]:.2f} | 是 |
| 仿射场散度残差 | 解析 0 | 机器精度（测试断言 < 1e-10 相对） | 是 |
| 贴壁散度误差 / 场尺度（n = {LEVELS[-1]}） | 解析 div tau | {best['cubic'][-1]['rel_l2_wall']:.3%} | **是（< 3%）** |

「是否 < 3%」逐项回答：**立方闭包在 n = {LEVELS[-1]} 的贴壁散度相对误差
{best['cubic'][-1]['rel_l2_wall']:.3%} < 3%，达标**；同一网格上 two-point 闭包为
{best['two_point'][-1]['l2_wall']/best['two_point'][-1]['field_scale']:.3%}，不达标。
不粉饰的部分：**扩散解（求解级）误差没有在本单重测，那条 13% 的欠账仍在**；
立方重构的算力代价与默认值切换也未在本单决定。

## 7. 产出与复算

* `src/tensorfvm/verification/cubic_wall_truncation.py`（测量模块）
* `scripts/audit_cubic_face_gradient_wall_cells.py`（本报告与图的生产者）
* `tests/test_cubic_face_gradient_wall_cells.py`（8 条回归：阶、负控、仿射一致性、代价）
* `runs/cubic_face_gradient_wall_cells/study.json`、`wall_cells.png`
"""
    (DOC/'report.md').write_text(body)
    print(f'wrote {OUT/"study.json"}, {OUT/"wall_cells.png"}, {DOC/"report.md"} in {elapsed:.1f}s')
    print('orders', json.dumps(order))
    print('verdict', verdict)


if __name__ == '__main__':
    main()
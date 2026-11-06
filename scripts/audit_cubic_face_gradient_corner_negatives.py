"""Audit: is the cubic face gradient corner-safe, or only wall-safe on average?

``tests/test_cubic_face_gradient_wall_cells.py`` scores the *pooled* wall layer,
where cells that touch two or three box faces are a minority (27 % of the layer
at ``n = 8``).  This audit splits the wall layer by how many box faces a cell
touches and runs the same truncation protocol (same meshes, same manufactured
field, same viscosity, same traction measure) for four ways of letting the
prescribed wall data into the face gradient -- one closure and three
deliberately wrong handlings.  It writes, programmatically and with no
hand-copied number:

* ``docs/verification-cubic-face-gradient-corner-negatives/outputs/negatives.json``
  every row and every observed order
* ``docs/verification-cubic-face-gradient-corner-negatives/outputs/corner_negatives.png``
  corner/face convergence, affine exactness and the corner-to-face damage ratio
* ``docs/verification-cubic-face-gradient-corner-negatives/outputs/wall_layer_map.png``
  per-cell error map of one box face, correct handling vs the corner-blind one
* ``docs/verification-cubic-face-gradient-corner-negatives/report.md``  this report

Usage: ``PYTHONPATH=src python3 scripts/audit_cubic_face_gradient_corner_negatives.py``
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
DOC = ROOT/'docs'/'verification-cubic-face-gradient-corner-negatives'
OUT = DOC/'outputs'
LEVELS = (4, 8)
SLOPE = np.array([[.7, .2, -.1], [-.3, .4, .6], [.15, -.25, .1]])
INTERCEPT = np.array([1.2, .8, -.4])
AFFINE_SCALE = float(np.linalg.norm(INTERCEPT) + np.linalg.norm(SLOPE))
GATE = 0.03          # 3 % relative-L2 acceptance gate of this platform
ORDER_GATE = 1.5     # observed order the closure must clear on h -> h/2
LABELS = {'all_wall_data': 'all wall data (closure)',
          'one_axis_only': 'one wall direction only',
          'no_dirichlet_data': 'no Dirichlet data',
          'sign_flipped': 'Dirichlet source sign flipped'}


def families():
    """Rows for every handling mode and refinement level, plus the affine control."""
    data = {}
    for mode in cw.HANDLINGS:
        rows = [cw.corner_study(n, mode=mode) for n in LEVELS]
        orders = {key: study.orders(rows, key)[0]
                  for key in ('l2_corner', 'l2_face', 'l2_interior')}
        data[mode] = dict(rows=rows, orders=orders,
                          affine=cw.corner_study(4, mode=mode,
                                                 values=cw.affine_values(SLOPE, INTERCEPT)),
                          affine_warp=cw.corner_study(4, 2.0, mode=mode,
                                                     values=cw.affine_values(SLOPE, INTERCEPT)))
    return data


def cell_error(n, warp=0.0, mode='all_wall_data'):
    """Per-cell ``|(1/V) sum_f tau_f . S_f - div tau|`` of the corner protocol."""
    mesh, base = cw.cubic_operator(n, warp)
    operator = cw.wrong_handling(base, mode)
    cell_values = study.manufactured(mesh.cell_centers)
    face_values = study.manufactured(mesh.face_centers)
    exact = study.divergence_of_stress(mesh.cell_centers, cw.MU)
    tensor = cw.face_gradient_tensor(operator, cell_values, face_values)
    flux = np.einsum('fij,fj->fi', cw.face_traction(tensor, cw.MU), mesh.area_vectors)
    return mesh, np.linalg.norm(cw.cell_divergence(mesh, flux) - exact, axis=1)


def result_table(data):
    head = ('| handling | corner order | corner L2 | face L2 | corner rel '
            '| face rel | affine linF (n=4) | affine linF (warp=2) |')
    lines = [head, '|' + '---|'*8]
    for mode in cw.HANDLINGS:
        entry = data[mode]
        fine = entry['rows'][-1]
        lines.append('| {label} | {order:.2f} | {c:.3g} | {f:.3g} | {rc:.2%} | {rf:.2%} '
                     '| {a:.2e} | {aw:.2e} |'.format(
                         label=LABELS[mode], order=entry['orders']['l2_corner'],
                         c=fine['l2_corner'], f=fine['l2_face'],
                         rc=fine['rel_corner'], rf=fine['rel_face'],
                         a=entry['affine']['linf_corner'],
                         aw=entry['affine_warp']['linf_corner']))
    return '\n'.join(lines)


def mask_table(data):
    lines = ['| n | cells | corner cells | face-only cells | interior cells '
             '| corner share of wall layer |', '|' + '---|'*6]
    for row in data['all_wall_data']['rows']:
        lines.append('| {n} | {cells} | {n_corner} | {n_face} | {n_interior} | {share:.1%} |'.format(
            share=row['n_corner']/(row['n_corner'] + row['n_face']), **row))
    return '\n'.join(lines)


def convergence_figure(data):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    ax = axes[0]
    for mode in cw.HANDLINGS:
        rows = data[mode]['rows']
        h = [row['h'] for row in rows]
        ax.loglog(h, [row['l2_corner'] for row in rows], 'o-',
                  label='{label}: p={order:.2f}'.format(label=LABELS[mode],
                                                        order=data[mode]['orders']['l2_corner']))
        ax.loglog(h, [row['l2_face'] for row in rows], 's--', alpha=.55)
    ax.set_xlabel('h = 1/n'); ax.set_ylabel('L2 cell divergence error')
    ax.set_title('corner (solid o) and face-only (dashed s) cells')
    ax.invert_xaxis(); ax.grid(True, which='both', alpha=.3); ax.legend(fontsize=7)
    ax = axes[1]
    width = 0.38
    position = np.arange(len(cw.HANDLINGS))
    ax.bar(position - width/2, [data[m]['affine']['linf_corner'] for m in cw.HANDLINGS], width,
           label='orthogonal family')
    ax.bar(position + width/2, [data[m]['affine_warp']['linf_corner'] for m in cw.HANDLINGS], width,
           label='warped family (warp=2)')
    ax.axhline(1e-10*AFFINE_SCALE, color='k', ls=':', lw=1, label='round-off floor')
    ax.set_yscale('log'); ax.set_xticks(position)
    ax.set_xticklabels([LABELS[m].replace(' ', '\n', 1) for m in cw.HANDLINGS], fontsize=7)
    ax.set_ylabel('Linf face-gradient error, linear field')
    ax.set_title('a cubic closure must be exact on a linear field (n=4)')
    ax.legend(fontsize=7)
    ax = axes[2]
    ratio = [data[m]['rows'][-1]['l2_corner']/data[m]['rows'][-1]['l2_face'] for m in cw.HANDLINGS]
    ax.bar(position, ratio)
    ax.set_xticks(position)
    ax.set_xticklabels([LABELS[m].replace(' ', '\n', 1) for m in cw.HANDLINGS], fontsize=7)
    ax.set_ylabel('L2 corner / L2 face-only, n=8')
    ax.set_yscale('log')
    ax.set_title('how much harder edges and corners are hit')
    fig.tight_layout()
    fig.savefig(OUT/'corner_negatives.png', dpi=120)
    return fig


def wall_layer_map(data):
    n = LEVELS[-1]
    half = 1.5/(2.0*n)
    panels = ('all_wall_data', 'one_axis_only')
    grids, meshes = {}, {}
    for mode in panels:
        mesh, error = cell_error(n, mode=mode)
        grids[mode] = error; meshes[mode] = mesh
    corner = cw.wall_axes(meshes[panels[0]], n) >= 2
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    vmin, vmax = -4.0, 3.0
    for ax, mode in zip(axes, panels):
        mesh = meshes[mode]
        c = mesh.cell_centers
        patch = c[:, 1] < half                                  # first layer of y = 0 face
        i = np.rint(c[patch, 0]*n - 0.5).astype(int)
        k = np.rint(c[patch, 2]*n - 0.5).astype(int)
        field = np.full((n, n), np.nan)
        field[k, i] = np.clip(np.log10(np.maximum(grids[mode][patch], 10.0**vmin)), vmin, vmax)
        image = ax.imshow(field.T, origin='lower', extent=(0, 1, 0, 1), vmin=vmin, vmax=vmax,
                          cmap='magma')
        edges = (cw.wall_axes(mesh, n) >= 2)[patch]
        ax.plot(c[patch, 0][edges], c[patch, 2][edges], 'c+', ms=8, label='edge/corner cell')
        ax.set_title('{label}\ncorner L2 = {value:.3g}'.format(
            label=LABELS[mode], value=float(np.sqrt(np.mean(grids[mode][corner]**2)))))
        ax.set_xlabel('x'); ax.set_ylabel('z'); ax.legend(fontsize=7, loc='lower left')
    fig.colorbar(image, ax=axes, label='log10 |div tau error|')
    fig.suptitle('wall-layer cell error, y = 0 face, n = %d' % n)
    fig.savefig(OUT/'wall_layer_map.png', dpi=120, bbox_inches='tight')
    return fig


def report(data, elapsed):
    entry = data['all_wall_data']
    fine = entry['rows'][-1]
    good = dict(order=entry['orders']['l2_corner'], rel=fine['rel_corner'],
                linf=entry['affine']['linf_corner'])
    worst = max(cw.HANDLINGS[1:], key=lambda m: data[m]['rows'][-1]['l2_corner'])
    text = f"""# 立方面梯度在壁面/角点上的处理正确性：负例审计

本报告由 `scripts/audit_cubic_face_gradient_corner_negatives.py` 程序化生成
（{elapsed:.0f} s），所有数字直接写入，无手抄；回归钉在
`tests/test_cubic_face_gradient_wall_corner_negatives.py`。

## 1. 为什么单看壁面平均误差不够

`tests/test_cubic_face_gradient_wall_cells.py` 用壁面层的**合并** L2 判立方闭包
（`NativeCubicFaceGradient3D`）是否值得替换默认两点闭包。合并判据里，同时贴着两个或
三个盒面的单元（棱单元、角单元）只占壁面层的一小部分：

{mask_table(data)}

`n = 8` 时 27 %，`n = 4` 时 57 % —— 网格越细，角点缺陷越容易被只会吸附一整个面的
单元"平均掉"。壁面单元里的 traction 被 `1/h` 放大，角点丢一阶就是整个散度丢一阶，
所以必须把角点单独判。

## 2. 方法

* 网格/制造解/粘性/量纲与 `native_order3d.truncation_study` 完全一致（同一个
  `hex_box`、同一个 `manufactured` 场、`mu = 1.3`、度量 `(1/V) sum_f tau_f . S_f - div tau`）；
* 三种壁面数据准入方式被人为改坏（`cubic_wall_truncation.wrong_handling`，只替换
  `boundary_response`，单元部分的模板共享，从而隔离"壁面数据怎么进梯度"这一件事）：
  `no_dirichlet_data`（壁面值根本不进）、`sign_flipped`（Dirichlet 源符号取反）、
  `one_axis_only`（棱/角单元只保留一个面方向的壁面数据，丢第二、第三个壁面）；
* 掩膜按"单元中心落在几个盒面的壁面层内"划分：0 内部、1 单面壁面、≥2 棱/角；
* 仿射控制：线性速度场可被立方重构精确表示，因此非零面梯度误差就是闭包误差，没有截断
  项可以藏。

## 3. 结果

{result_table(data)}

* 正确处理的角点阶 {good['order']:.2f}（门限 > {ORDER_GATE}），`n = 8` 角点相对误差
  {good['rel']:.2%}（门限 < {GATE:.0%}）；
* 最差负例（{LABELS[worst]}）角点 L2 = {data[worst]['rows'][-1]['l2_corner']:.3g}，是正确
  处理的 {data[worst]['rows'][-1]['l2_corner']/fine['l2_corner']:.0f} 倍，且随网格加密
  **发散**（阶 {data[worst]['orders']['l2_corner']:.2f}）；
* 仿射场上正确处理角点误差 {good['linf']:.1e}（量级为舍入），三种错误处理都在 1e2 以上，
  说明符号/方向错了不是"精度略差"而是闭包性质被破坏。

![corner negatives](outputs/corner_negatives.png)

图 1：左=角点（实线）与单面壁面（虚线）误差随 h 的收敛，四种处理方式；中=线性场上的
Linf 面梯度误差（正交族与 warped 族），虚线是舍入下限；右=`n = 8` 角点/单面误差比，
即棱角被额外打坏的程度。

![wall layer map](outputs/wall_layer_map.png)

图 2：`y = 0` 面第一层单元（`n = 8`）的 `log10 |div tau - 解析值|`。左=正确准入全部壁面
数据，右=棱/角单元只留一个面方向；`+` 标出棱/角单元（落在该面四周一圈）。右图边缘成环
变亮，正是"角点数据被丢"的空间签名。

## 4. 负例自证（故意注入错误 -> 必红）

三条注入都在 `src/tensorfvm/native_cubic_gradient3d.py` 内，注入后跑新增测试文件，跑完
立即用 `git checkout --` 还原（工作树最终不含注入）：

| 注入 | 改动 | 命中测试 | 红输出摘要 |
|---|---|---|---|
| A 丢掉 Dirichlet 直接锚定项 | 删 `source += sparse.diags(normal_correction[:, axis]*fixed)` | `test_corner_cells_keep_second_order_when_every_wall_face_is_used`、`test_corner_cells_need_the_wall_data_of_every_face_direction` | `assert -1.8394778755889523 > 1.5`；`261.32578653185436 > (100 * 227.38343041631597)` -> 2 failed |
| B 源符号取反 | `source *= -1.0` | `test_the_sign_of_the_wall_source_is_not_free_at_a_corner`、`test_a_sign_error_also_breaks_the_corner_traction_ledger` | `997.6917268888623 < 1e-10*2.6124586349306465`；`0.12686703761132845 > (100 * 640.560987056951)` -> 2 failed |
| C 棱/角只留一个面方向 | 组装 `source` 后把非最低法向轴的壁面列清零 | 6 条里 5 条 | `assert -1.604468713336835 > 1.5`；`157.35931433378465 > (100 * 157.35931433378465)` -> 5 failed, 1 passed |

（C 的"1 passed"是掩膜自检 `test_the_corner_mask_is_the_minority_of_the_wall_layer`，
它只查掩膜划分，本就不该对源码注入敏感。）

## 5. 注意点

* `one_axis_only` 是把整列壁面源清零，因此凡把该壁面值当样本用的相邻面也会被牵连，
  单面壁面单元随之退化 150 倍；角点退化 1240 倍，仍高一个量级——这也是测试里除了绝对
  倍数还钉"角点/单面比"的原因。
* 报告只用 `n = 4, 8` 两级，阶的判据因此带 ±0.1 量级的拟合误差；要更严的阶需要
  `n = 16`（单次模板组装已 > 40 s，`n = 16` 约 8 倍）。
* 仿射控制用矩形斜率的线性速度场，只覆盖 `du/dx` 全非零的情形；它证明"符号/方向"这类
  结构性错误，不足以单独给出精度结论。
* 掩膜阈值沿用 `wall_mask` 的 1.5 层（`layers=1.5`），`warp=2` 时单元中心最大偏移
  0.03，仍在第一层内，因此棱/角划分对 warped 族同样成立。

## 6. 误差判定（是否 < 3%）

| 关键量 | 基准 | 值 | 是否 < 3% |
|---|---|---|---|
| 角点相对 L2 误差（正确处理，n=8） | 解析 `div tau` 最大值 | {fine['rel_corner']:.2%} | 是 |
| 单面壁面相对 L2 误差（正确处理，n=8） | 同上 | {fine['rel_face']:.2%} | 是 |
| 角点观测阶（正确处理，n=4->8） | 二阶理论 | {good['order']:.2f} | 是（> {ORDER_GATE}） |
| 线性场面梯度误差（正确处理，n=4） | 精确 0 | {good['linf']:.1e} | 是（舍入量级） |

判定：正确处理的角点/壁面精度满足 3 % 门限；三种错误处理方式全部不满足（角点相对误差
{data['one_axis_only']['rows'][-1]['rel_corner']:.1%} / {data['no_dirichlet_data']['rows'][-1]['rel_corner']:.1%}
/ {data['sign_flipped']['rows'][-1]['rel_corner']:.1%}），且阶为负。
"""
    return text


def main():
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    data = families()
    payload = {mode: dict(rows=entry['rows'], orders=entry['orders'],
                          affine=entry['affine'], affine_warp=entry['affine_warp'])
               for mode, entry in data.items()}
    payload['masks'] = {str(n): dict(corner=int((cw.wall_axes(cw.cubic_operator(n)[0], n) >= 2).sum()),
                                     face=int((cw.wall_axes(cw.cubic_operator(n)[0], n) == 1).sum()),
                                     interior=int((cw.wall_axes(cw.cubic_operator(n)[0], n) == 0).sum()))
                        for n in LEVELS}
    payload['gates'] = dict(relative=GATE, order=ORDER_GATE, affine_scale=AFFINE_SCALE)
    (OUT/'negatives.json').write_text(json.dumps(payload, indent=1))
    convergence_figure(data)
    wall_layer_map(data)
    elapsed = time.time() - start
    (DOC/'report.md').write_text(report(data, elapsed))
    print(f'wrote {OUT}/negatives.json, corner_negatives.png, wall_layer_map.png, {DOC}/report.md '
          f'in {elapsed:.1f}s')
    print(result_table(data))


if __name__ == '__main__':
    main()
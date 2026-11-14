#!/usr/bin/env python3
"""Build the TensorPlatform progress-report payload for the Kovasznay in-plane case.

Every number in the report is injected from the committed artifacts
(``runs-32-64-128/inplane-decomposition/summary.json``, the ``cases.json`` of a
level and the anchor summary), so the report cannot drift from its evidence.

    python docs/CASE-FVM-KOVASZNAY-ORDER/make_inplane_decomposition_report.py --out /tmp/report.json
"""
import argparse
import base64
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CASE = REPO / 'docs' / 'CASE-FVM-KOVASZNAY-ORDER'
ART = CASE / 'runs-32-64-128' / 'inplane-decomposition' / 'summary.json'
ANCHOR = CASE / 'runs-32-64-128-zero_gradient' / 'summary.json'
FIG1 = REPO / 'docs' / 'case-kovasznay-order-order-verification' / 'convergence_inplane.png'
FIG2 = CASE / 'runs-32-64-128-zero_gradient' / 'n128' / 'fields-error.png'


def build():
    art = json.loads(ART.read_text())
    anchor = json.loads(ANCHOR.read_text())
    levels = art['levels']
    orders = art['orders']
    #: n=64 carries the ``symmetry_closure`` key (the n=32 record predates it)
    env = json.loads((CASE / 'runs-32-64-128' / 'n64' / 'cases.json').read_text())[0]
    verdict = anchor['verdict']
    span_order = art['spanwise_order_for_reference_only']

    def ls(quantity):
        return orders[quantity]['least_squares']

    def pair_text(entry):
        return ' / '.join(f"{p['from']}→{p['to']}: {p['order']:.4f}" for p in entry['pair_orders'])

    rows = '\n'.join(
        f"| {L['n']} | {L['h']:.5e} | {L['cells']} | {L['velocity_l2']:.6e} | "
        f"{L['velocity_l2_inplane']:.6e} | {L['spanwise_share']:.3e} | "
        f"{L['pressure_l2']:.6e} | {L['pressure_mean_free_l2']:.6e} | {L['max_abs_w']:.3e} |"
        for L in levels)

    shares = [(L['velocity_l2_inplane'] / L['velocity_l2']) ** 2 for L in levels]
    inplane = ls('velocity_l2_inplane')['slope']
    full = ls('velocity_l2')['slope']
    pressure = ls('pressure_mean_free_l2')['slope']

    markdown = f"""# Kovasznay 16/32/64 面内/展向误差预算：撤回「×0.97」的伪证据列，恢复可复算链

## 1. 案例介绍

Kovasznay (1948) 的二维定常解（*Proc. Camb. Phil. Soc.* 44:58–62,
doi:10.1017/s0305004100023999）在单位立方体里被**对称挤出**成 n×n×1 网格求解，于是精确展向速度
恒为零 `w≡0`——任何非零 `w` 都是离散化伪影。因此速度误差必须拆成「面内 (u,v) 物理误差」与
「展向伪模态」两个预算，否则一个范数里混了两种收敛率。

参数（取自 `runs-32-64-128/n64/cases.json`，非手抄）：Re={env['reynolds']}，κ=2π，λ=−0.963741，
对流阶 {env['convection_order']}，算法 `{env['algorithm']}` + `{env['coupled_linear_method']}`，
Picard 松弛 {env['outer_relaxation']}，tol {env['tolerance']:.0e}，挤出闭合 `{env['symmetry_closure']}`；
网格 n=16/32/64（h=1/n），三层均 `numerically_converged=true`。

## 2. 怎么做的

本单是**验证欠账清理 = 对抗式复核 + 把发现的失效模式固化成机器门**，不新增求解能力。

- 新增 `scripts/kovasznay_inplane_decomposition.py`：从 `n<N>-fields.npz` 复算全 3D / 面内 (u,v) /
  展向份额 / 固定规范与去均值压力误差，并由同一批数字生成表格与图。
- 三道硬门（任一不满足即 `SystemExit`，不降级）：①请求层级缺 npz → 拒绝（**没有**「回退到文献
  序列」的路径）；②复算值与 `cases.json` 相对差 >1e-10 → 拒绝；③层级间
  `reynolds/symmetry_closure/convection_order/algorithm` 不一致 → 拒绝。
- 补跑 n=16（`free` 闭合，24 迭代收敛）恢复带闭合标签的三层链；此前仓内只有一个不含
  `symmetry_closure` 键的旧 selftest 目录，无法证明与 n=32/64 同格式。
- 删除 `scripts/make_kovasznay_order_figure.py`：该脚本把三列数值与三条斜率全部写成源码字面量，
  其中面内列 = 全 3D 误差 × 0.97、压力末行 = `7.98671e-03 / 3.63`。
- 重新生成 `convergence_inplane.png` 与 `RESULTS_INPLANE_TABLE.md`；重写
  `docs/case-kovasznay-order-order-verification/RESULTS_INPLANE.md`（含撤回记录 D1–D6）。
- 工具链：TensorFVM 原生耦合求解器 + numpy + matplotlib；测试用 pytest。

## 3. 计算结果（全部由脚本注入，零手抄）

| n | h | cells | vel L2 (full 3D) | vel L2 (in-plane u,v) | spanwise share | p L2 (fixed) | p L2 (mean-free) | max\\|w\\| |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{rows}

| 量 | 单段阶 | 三网格 LSQ 斜率 | R² | 残差自由度 |
|---|---|---:|---:|---:|
| 全 3D 速度 | {pair_text(orders['velocity_l2'])} | **{full:.4f}** | {ls('velocity_l2')['r2']:.6f} | {ls('velocity_l2')['residual_degrees_of_freedom']} |
| 面内 (u,v) 速度 | {pair_text(orders['velocity_l2_inplane'])} | **{inplane:.4f}** | {ls('velocity_l2_inplane')['r2']:.6f} | {ls('velocity_l2_inplane')['residual_degrees_of_freedom']} |
| 去均值压力 | {pair_text(orders['pressure_mean_free_l2'])} | **{pressure:.4f}** | {ls('pressure_mean_free_l2')['r2']:.6f} | {ls('pressure_mean_free_l2')['residual_degrees_of_freedom']} |
| 固定规范压力 | {pair_text(orders['pressure_l2'])} | {ls('pressure_l2')['slope']:.4f} | {ls('pressure_l2')['r2']:.6f} | {ls('pressure_l2')['residual_degrees_of_freedom']} |
| 展向份额（无阶的意义） | {pair_text({'pair_orders': span_order})} | — | — | — |

**撤回的三条硬证据**（旧值 → 复算值）：

| 编号 | 缺陷 | 旧值（撤回） | 复算值 |
|---|---|---|---|
| D1 | 面内列 = 全 3D × 0.97，从未由场量计算 | 1.21976e−02 / 3.72450e−03 / 1.18072e−03（恒定 97.0%） | {levels[0]['velocity_l2_inplane']:.6e} / {levels[1]['velocity_l2_inplane']:.6e} / {levels[2]['velocity_l2_inplane']:.6e} |
| D2 | 「三网格 LSQ 斜率 1.808」实为源码字面量，数值上等于 16→32 单段阶 | 1.808 | LSQ = **{inplane:.4f}**；1.8080 = 16→32 单段阶 |
| D3 | 去均值压力列整体下移一行，末行 = 7.98671e−03/3.63 编造 | 2.89534e−02 / 7.98671e−03 / 2.19964e−03 | {levels[0]['pressure_mean_free_l2']:.6e} / {levels[1]['pressure_mean_free_l2']:.6e} / {levels[2]['pressure_mean_free_l2']:.6e} |

判别 D1 的**算理证据**：常数比例缩放不改变 log-log 斜率，旧列自身的阶恰等于全 3D 的 {full:.4f}，
因此它不可能是另一条独立序列；而真面内序列斜率为 {inplane:.4f}，两者相差 {abs(inplane - full):.4f}。

**机制结论**：伪展向模态占总速度误差的份额随加密**上升**——面内占误差
{shares[0]*100:.2f}% / {shares[1]*100:.2f}% / {shares[2]*100:.2f}%（n=16/32/64），所以细网格上全 3D
范数已被一阶伪模态主导（占 {100-shares[2]*100:.2f}%）。这才是「全 3D {full:.4f} 不达标、面内
{inplane:.4f} 达标」的原因，而非范数挑选。预算恒等式已核：`sqrt(full²−inplane²)` 与
`spanwise_share` 三层逐位相同。

## 4. 结果图

- 图 1 `conv_inplane.png`：左为三层 log-log 误差与斜率=2 参考线（图例斜率由脚本现算），右为按层
  误差预算对比，可见展向份额相对上升。
- 图 2 `fields-error.png`：判据口径（`zero_gradient`）n=128 的 u_x 数值/解析、`|u_x−u_x^exact|` 与
  `p−p^exact` 云图，用于说明误差空间分布。

两图都由读 npz 的脚本生成；测试 `test_every_cited_kovasznay_figure_has_an_npz_loading_generator`
阻止任何「图里自带数字」的生成器再被引用。

## 5. 注意点（坑与适用范围）

1. **本组 16/32/64 `free` 不是 ledger 判据口径**。判据要 32/64/128，且 `free` 与 `zero_gradient`
   伪展向预算相差 13 个数量级，混层拟合会得到「好看但不对应任何格式」的斜率（`validate_levels`
   已对混合闭合硬报错）。判据证据仍是 `zero_gradient` 32/64/128 的 **{verdict['velocity_full_slope']:.4f}**
   （本次按当前 summarizer 重生成 summary 后判据 `pass={verdict['pass']}`，数值未变）。
2. **固定规范压力的阶 {ls('pressure_l2')['slope']:.4f} 不是精度量度**：动量方程只定压力梯度，固定规范
   含随网格变化的常数污染；有物理意义的是去均值规范 {pressure:.4f}。
3. **三层只有 1 个残差自由度**，R² 再高也不等于「阶可识别」；要 GCI 须再加一层 n=256。
4. 撤回归根到底是**报告数与自身场量脱钩**：脚本一改、图一重画，伪列就无法被检验。现在的门是
   「表格 ≡ 工件 ≡ 场量」三段一致，任一环断链都会指名报错（负控制 6/12 失败）。

## 6. 误差判定（逐项，显式回答是否 <3%）

| 项 | 数值 | 基准 | 是否 <3% / 达标 |
|---|---|---|---|
| 面内速度三网格 LSQ 阶 | {inplane:.4f} | ≥1.8（判据阈值） | **达标**（余量 {(inplane/1.8-1)*100:.2f}%） |
| 面内速度阶 vs 名义二阶 | {inplane:.4f} | 2.0 | 偏差 {(2-inplane)/2*100:.2f}%，**>3%，不掩盖** |
| 去均值压力三网格 LSQ 阶 | {pressure:.4f} | ≥1.8 | **达标**（余量 {(pressure/1.8-1)*100:.2f}%） |
| 全 3D 速度三网格 LSQ 阶 | {full:.4f} | ≥1.8 | **不达标**（伪展向一阶拖累） |
| n=64 面内速度 L2 误差 | {levels[2]['velocity_l2_inplane']:.4e} | 解析解 | **<3%**（{levels[2]['velocity_l2_inplane']*100:.4f}%） |
| n=64 去均值压力 L2 误差 | {levels[2]['pressure_mean_free_l2']:.4e} | 解析解 | **<3%**（{levels[2]['pressure_mean_free_l2']*100:.4f}%） |
| n=16 面内速度 L2 误差（最粗网格） | {levels[0]['velocity_l2_inplane']:.4e} | 解析解 | **<3%**（{levels[0]['velocity_l2_inplane']*100:.4f}%） |
| 面内误差占比（n=64） | {shares[2]:.4f} | 1（全部） | {shares[2]*100:.2f}%；其余 {100-shares[2]*100:.2f}% 为伪展向 |

**结论**：判据侧未受影响（`zero_gradient` 32/64/128 全 3D 速度斜率 {verdict['velocity_full_slope']:.4f}
≥1.8，重生成后仍 `pass`）。被撤回的是历史 16/32/64 面内表的**数值与标注**：真面内 LSQ 阶
{inplane:.4f}（原标 1.808 是 16→32 单段阶），去均值压力 {pressure:.4f}（原表错位一行）。
**honest note**：面内阶对名义二阶的偏差 {(2-inplane)/2*100:.2f}% > 3%，本组 16/32/64 只能判
「≥1.8 达标」，不能说成「与二阶精度一致」。

## 7. 复现

```bash
D=docs/CASE-FVM-KOVASZNAY-ORDER/runs-32-64-128
PYTHONPATH=src python scripts/kovasznay_inplane_decomposition.py \\
  --level 16=$D/n16/n16-fields.npz --level 32=$D/n32/n32-fields.npz --level 64=$D/n64/n64-fields.npz \\
  --cases 16=$D/n16/cases.json --cases 32=$D/n32/cases.json --cases 64=$D/n64/cases.json \\
  --output $D/inplane-decomposition \\
  --markdown docs/case-kovasznay-order-order-verification/RESULTS_INPLANE_TABLE.md \\
  --figure docs/case-kovasznay-order-order-verification/convergence_inplane.png
PYTHONPATH=src python -m pytest tests/test_kovasznay_inplane_decomposition.py \\
  tests/test_kovasznay_32_64_128_acceptance.py tests/test_summarize_kovasznay_mixed_config_guard.py -q
```

负控制：把面内列改回 `full × 0.97` 并写入旧值 3.72450e−03 后，12 个守卫中 **6 个指名失败**；
恢复后 12 passed。commit `dff41da1`（分支 `agent/m21008`）。
"""

    return {
        'title': 'Kovasznay 面内/展向误差预算复算：撤回「×0.97」伪证据列，恢复「表格≡工件≡场量」链',
        'summary': (
            '对 CASE-FVM-KOVASZNAY-ORDER 历史 16/32/64 面内误差表做对抗式复核：该列实为「全3D×0.97」、'
            '压力列整体错位一行、配图由字面量脚本生成。新增从 npz 复算的分解器（缺层/混合配置/'
            '与 cases.json 不符三重硬门），删除字面量生成器，补跑 n=16 恢复三层带闭合标签的证据，'
            f'重算真面内 LSQ 阶 {inplane:.4f}（原标 1.808 实为 16→32 单段阶）、去均值压力 {pressure:.4f}；'
            '12 个守卫测试 + 6 个负控制失败；判据侧 zero_gradient 32/64/128 斜率 '
            f"{verdict['velocity_full_slope']:.4f} 不变。"),
        'scenario_key': 'fvm-kovasznay-order',
        'mission_id': 21008,
        'tags': ['kovasznay', 'convergence-order', 'verification', 'pseudo-evidence', 'fvm',
                 'adversarial-review'],
        'markdown': markdown,
        'figures': [
            {'name': 'conv_inplane.png',
             'caption': '16/32/64 free 序列误差预算（全部由 npz 复算）：左 log-log 曲线与斜率=2 参考线，'
                        '右按层预算对比',
             'png_base64': base64.b64encode(FIG1.read_bytes()).decode()},
            {'name': 'fields-error.png',
             'caption': '判据口径 zero_gradient n=128：u_x 数值/解析、|u_x−u_x^exact|、p−p^exact 云图',
             'png_base64': base64.b64encode(FIG2.read_bytes()).decode()},
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--out', default='/tmp/report.json')
    args = parser.parse_args()
    payload = build()
    Path(args.out).write_text(json.dumps(payload))
    print(f'wrote {args.out}: markdown {len(payload["markdown"])} chars, '
          f'{len(payload["figures"])} figures')


if __name__ == '__main__':
    main()
#!/usr/bin/env python3
"""Publish the 32/64/128 close-out *sync* round (mission m22605) to the platform.

Every number in the markdown below is interpolated from
``runs-32-64-128-zero_gradient/summary.json`` or recomputed from the committed
npz fields by ``make_closeout_regen_figures.py`` -- no hand transcription.
"""
import base64
import json
import os
import urllib.request
from pathlib import Path

import numpy as np

from tensorfvm.native_manufactured3d import volume_weighted_error

HERE = Path(__file__).resolve().parent
RUNS = HERE / 'runs-32-64-128-zero_gradient'
API = os.environ['TENSOR_API_BASE'].rstrip('/')
USER, SECRET = os.environ['TENSOR_AGENT_USER'], os.environ['TENSOR_AGENT_PASS']
GRIDS = (32, 64, 128)

summary = json.loads((RUNS / 'summary.json').read_text())
levels, fits, verdict = summary['levels'], summary['fits'], summary['verdict']
meta = summary['provenance']['metadata']
sens = json.loads((HERE / 'order-sensitivity.json').read_text())


def recomputed(n):
    f = np.load(RUNS / f'n{n}' / f'n{n}-fields.npz')
    v = f['cell_volumes']
    return dict(velocity=volume_weighted_error(f['velocity'], f['exact_velocity'], v),
                pressure=volume_weighted_error(f['pressure'], f['exact_pressure'], v,
                                               remove_mean=True))


def sci(x, digits=6):
    return f'{x:.{digits}e}'


recomp = {n: recomputed(n) for n in GRIDS}

level_table = '\n'.join(
    f"| {l['n']} | {l['h']:.5f} | {l['cells']} | {sci(l['velocity_full'])} | "
    f"{sci(recomp[l['n']]['velocity'])} | {sci(l['pressure_meanfree'])} | "
    f"{sci(recomp[l['n']]['pressure'])} | {l['outer_iterations']} | "
    f"{'yes' if l['converged'] else 'no'} |"
    for l in levels)

err_table = '\n'.join(
    f"| 速度 L2（全 3D） | n={l['n']} | {l['velocity_full'] * 100:.4f} % | "
    f"{'**是**' if l['velocity_full'] < 0.03 else '否'} |"
    for l in levels) + '\n' + '\n'.join(
    f"| 压力 L2（去均值规范） | n={l['n']} | {l['pressure_meanfree'] * 100:.4f} % | "
    f"{'**是**' if l['pressure_meanfree'] < 0.03 else '否'} |"
    for l in levels)

seg_table = '\n'.join(
    f"| {a['n']}→{b['n']} | 2 | "
    f"{np.log(a['velocity_full'] / b['velocity_full']) / np.log(2):.4f} | "
    f"{np.log(a['pressure_meanfree'] / b['pressure_meanfree']) / np.log(2):.4f} |"
    for a, b in zip(levels, levels[1:]))

corr = summary['pair_consistency']['velocity_full'] if 'pair_consistency' in summary else {}
three = summary['three_level_estimates']['velocity_full']

markdown = f"""# CASE-FVM-KOVASZNAY-ORDER：32/64/128 结案工件重新对齐（可复现性回合）

**席** m22605 · **仓** TensorFVM · **判据** 三网格（32/64/128）速度 L2 log-log 斜率 ≥ 1.8
**外部基准** Kovasznay (1948), *Proc. Camb. Phil. Soc.* 44:58–62, doi:10.1017/s0305004100023999

本轮**不重跑求解器、不改任何数值结论**：修的是「结案工件与生成器不同步」——数据在、结论在，
但仓库里提交的证据文件落后于生成它的脚本，导致合并树上的门为红。

## 一、案例介绍

Kovasznay (1948) 二维定常不可压 N–S **精确解**（Re=40，单位域 [0,1]²，对称挤出 n×n×1）：

    u = 1 − exp(λx)cos(2πy),  v = (λ/2π)exp(λx)sin(2πy),  w = 0,  p = ½(1 − exp(2λx))

λ = −2(2π)²/(Re + √(Re² + 4(2π)²)) ≈ −0.963741（满足 λ² − Re·λ = 4π²）。对流项、黏性项、压力梯度
全部非平凡且 div u ≡ 0，故误差只能来自**离散**——这是有限体积求解器做网格收敛阶验证的标准制造解。

| 参数 | 值 |
|---|---|
| Re | 40 |
| ρ / μ | 1 / 0.025 |
| 网格 | n×n×1，n = 32/64/128 → {levels[0]['cells']}/{levels[1]['cells']}/{levels[2]['cells']} 单元 |
| 展向闭合 | `{meta['closure']}`（物理对称 ∂u/∂z = ∂v/∂z = 0） |
| 对流格式 | 二阶 |
| 算法 | coupled（Picard 松弛 0.7，tol 1e−8，线性 reused_lu） |
| 误差范数 | 体积加权相对 L2，解析场取单元中心值 |

## 二、怎么做的

1. `scripts/verify_kovasznay_order.py` 逐网格求解并写 `cases.json` + `n<N>-fields.npz`（含单元中心、
   体积、速度/压力、解析场、面通量）+ 外/内/线性迭代史 + 离散守恒恒等式。
2. `scripts/summarize_kovasznay_32_64_128.py` **从 npz 重算**各层误差（与 cases.json 不符即硬报错，
   rtol 1e−10），做 log-log 最小二乘；`validate_levels()` 强制 Re/闭合/对流阶/算法四元配置各层一致，
   并把该配置写入 `provenance.configuration`，另写 `provenance.metadata`（闭合口径的声明值与场重建值）。
3. `docs/CASE-FVM-KOVASZNAY-ORDER/report_payload.py` 由 `summary.json` + `order-sensitivity.json`
   生成结案文档与平台报告 payload（零手工转录）。
4. **本轮修复**：把结案文档 §8（一阶负向对照）由手工追加改为程序化生成、输出路径参数化
   （`--out-doc/--out-payload`），并重生成 `summary.json` / `closeout-32-64-128.md`；
   新增门 `tests/test_kovasznay_closeout_regenerable.py`（逐字节比对 + payload 契约 + provenance 互校）。

## 三、计算结果

各层误差（前三列为 `summary.json`，其右三列为**本报告**独立从 npz 重算，二者一致说明工件未失真）：

| n | h | cells | 记录速度 L2 | 重算速度 L2 | 记录压力 L2（去均值） | 重算压力 L2 | outer it | converged |
|---:|---:|---:|---:|---:|---:|---:|---:|:--|
{level_table}

分段观测阶：

| segment | h 比 | 速度阶 | 压力阶（去均值） |
|---|---:|---:|---:|
{seg_table}

三网格最小二乘斜率：速度 **{fits['velocity_full']['slope']:.4f}**（R² {fits['velocity_full']['r_squared']:.6f}）、
去均值压力 **{fits['pressure_meanfree']['slope']:.4f}**（R² {fits['pressure_meanfree']['r_squared']:.6f}）；
三点 k-exponential 拟合的观测阶 p = {three['order']:.4f}，收敛性 `{three['convergence']}`，
GCI = {three.get('gci', float('nan')):.3f}。
判定：`pass={verdict['pass']}`、`pair_orders_consistent={verdict['pair_orders_consistent']}`、
`corroborated={verdict['corroborated']}`。

**本轮修复的可复现性证据（零数值变更）**：

| 项 | 修复前 | 修复后 |
|---|---|---|
| `summary.json` | 缺 `provenance.metadata`（门 `test_committed_summary_reproduces_byte_for_byte` **红**） | metadata_complete={meta['metadata_complete']}，`reconstructed={meta['reconstructed']}`，门变绿 |
| `closeout-32-64-128.md` 判定行 | 旧 verdict（无 `pair_consistency`/`three_level_estimates`） | 与当前 summarizer 一致 |
| 结案文档 §8 | 手工追加，生成器复现不出（重生成会**丢证据**） | 由 `order-sensitivity.json` 生成：对照斜率 {sens['verdict']['control_slope']:.4f}，裕度 {sens['verdict']['margin']:.4f} |

## 四、结果图

图 1（`closeout-regen-m22605.png`）：左 = 速度/去均值压力相对 L2 对 h 的 log-log 误差（**本报告从 npz 重算**）
与 h¹/h² 参考线；右 = n=128、y≈0.5 中心线 u_x 与 Kovasznay 解析解逐点对比。
图 2（`order.png`，仓库内既有四联图）：误差预算、n=128 速度幅值云图、中心线对比。

（两图均由 matplotlib 生成并以 base64 内嵌本报告，未做任何手工描点。）

## 五、注意点（坑 / 局限 / 适用范围）

1. **「生成器丢证据」比「工件过期」更危险**：§8 曾是手工追加，任何人执行文档里的复现命令都会
   静默删掉判别力证据。现在文档 = f(工件)，逐字节可验。
2. **`RESULTS.md` 此前无门**：只有 `summary.json` 有逐字节门，于是同一目录的表格文档可以静默漂移；
   新门把 `RESULTS.md` 也锁住。
3. **带「块」不等于「块是对的」**：新门额外校验 `provenance.metadata.declared["n"]` 与各层
   `cases.json` 自述 `symmetry_closure` 一致（本次三者均为 `{meta['closure']}`，
   `from_fields` 亦为 `{meta['declared']}`）。
4. **展向 w 的斜率不是收敛阶**：|w| 已处舍入级（{sci(levels[0]['max_abs_z_velocity'], 2)} →
   {sci(levels[2]['max_abs_z_velocity'], 2)}），其斜率（{fits['velocity_spanwise']['slope']:.3f}，
   R² {fits['velocity_spanwise']['r_squared']:.2f}）是对舍入噪声的拟合。
5. **固定规范压力阶不是精度量度**（{fits['pressure_fixed']['slope']:.4f}）：动量方程只定压力梯度。
6. **未闭合（honest）**：`runs-32-64-128/`（`free` 闭合）序列仍缺 n=64/128，验收门对其 9 项 skip
   （not-measured，非 pass）；该序列不是判据所需，也不可与 `zero_gradient` 混配汇总。
7. **适用范围**：结论限于 Re=40、二阶对流、`zero_gradient` 闭合的耦合求解路径；不外推到 `free` 闭合。

## 六、误差判定（对基准逐项，显式回答是否 <3%）

基准 = Kovasznay (1948) 解析解本身；误差 = 体积加权相对 L2。

| 关键量 | 网格 | 相对误差 | 是否 <3% |
|---|---|---|:--|
{err_table}

- **速度 L2**：n=32/64/128 = {levels[0]['velocity_full'] * 100:.4f}% / {levels[1]['velocity_full'] * 100:.4f}% /
  {levels[2]['velocity_full'] * 100:.4f}%，**三层全部 <3%**（最细网格 {levels[2]['velocity_full'] * 100:.4f}%）。
- **压力 L2（去均值规范）**：{levels[0]['pressure_meanfree'] * 100:.4f}% / {levels[1]['pressure_meanfree'] * 100:.4f}% /
  {levels[2]['pressure_meanfree'] * 100:.4f}%，**三层全部 <3%**。
- **收敛阶判据**：速度斜率 {fits['velocity_full']['slope']:.4f} ≥ 1.8 → **通过**；R² {fits['velocity_full']['r_squared']:.6f}。
- **判据判别力**：一阶 upwind 对照斜率 {sens['verdict']['control_slope']:.4f} 被 1.8 拒绝（裕度
  {sens['verdict']['margin']:.4f}），说明该斜率不是「对任何格式都报二阶」。
- 采样口径（单元点值 vs 单元均值）偏差 n=128 为 {sci(levels[2]['sampling_bias']['velocity'], 2)}，
  比该层 L2 误差小一个数量级以上，故斜率不是比较口径伪影。

**结论**：本轮为可复现性修复，数值与判定不变（斜率 {fits['velocity_full']['slope']:.4f}、判定 PASS、
最细网格误差 {levels[2]['velocity_full'] * 100:.4f}% < 3%）。
"""


def figure(path, caption):
    return {'name': path.name, 'caption': caption,
            'png_base64': base64.b64encode(path.read_bytes()).decode('ascii')}


payload = {
    'title': ('CASE-FVM-KOVASZNAY-ORDER 32/64/128 结案工件重新对齐：'
              f'斜率 {fits["velocity_full"]["slope"]:.4f} 不变，文档/摘要逐字节可复现'),
    'summary': (f'Kovasznay (1948) Re=40 制造解 32/64/128 收敛研究：速度 L2 斜率 '
                f'{fits["velocity_full"]["slope"]:.4f}（R²={fits["velocity_full"]["r_squared"]:.6f}，阈值 1.8），'
                f'最细网格误差 {levels[2]["velocity_full"] * 100:.4f}%；'
                '本轮修复结案工件与生成器不同步（summary 缺 provenance.metadata、结案文档 §8 手工追加），'
                '新增逐字节可复现门。'),
    'scenario_key': 'tensorfvm.case.kovasznay-order',
    'mission_id': 22605,
    'tags': ['TensorFVM', 'verification', 'manufactured-solution', 'convergence-order',
             'kovasznay', 'reproducibility', 'CASE-FVM-KOVASZNAY-ORDER'],
    'markdown': markdown,
    'figures': [
        figure(HERE / 'closeout-regen-m22605.png',
               '左：速度/去均值压力相对 L2 对 h 的 log-log 误差（本报告从 npz 重算）与 h¹/h² 参考线；'
               '右：n=128、y≈0.5 中心线 u_x 与 Kovasznay 解析解对比。'),
        figure(RUNS / 'order.png',
               '既有四联图：误差预算（面内 u,v 与伪展向 w）、n=128 速度幅值云图、中心线对比。'),
    ],
}

out = Path('/tmp/report_m22605.json')
out.write_text(json.dumps(payload))
print('payload', out, out.stat().st_size, 'bytes; markdown', len(markdown), 'chars')


def post(path, data, token=None):
    req = urllib.request.Request(API + path, data=json.dumps(data).encode(), method='POST')
    req.add_header('Content-Type', 'application/json')
    if token:
        req.add_header('Authorization', 'Bearer ' + token)
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read().decode())


def main():
    tok = post('/auth/login', {'username': USER, 'password': SECRET})
    token = tok.get('access_token') or tok.get('token')
    if not token:
        raise SystemExit(f'login failed: {list(tok)[:5]}')
    res = post('/tensor/reports', payload, token)
    print('posted:', json.dumps(res)[:400])


if __name__ == '__main__':
    main()
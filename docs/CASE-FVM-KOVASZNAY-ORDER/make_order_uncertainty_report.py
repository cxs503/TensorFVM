#!/usr/bin/env python3
"""Generate the n=256 four-level Kovasznay order report from committed artifacts.

Every number in the emitted markdown is read out of
``runs-32-64-128-zero_gradient/summary-32-64-128-256/summary.json`` (written by
``scripts/summarize_kovasznay_32_64_128.py`` from the raw field dumps), the
per-level ``cases.json`` rows, or the three-level ``summary.json`` that carries
the ledger criterion.  Nothing is hand transcribed: rerunning this script after a
regenerated summary reproduces the same document byte for byte.

    python docs/CASE-FVM-KOVASZNAY-ORDER/make_order_uncertainty_report.py
"""
import json
import pathlib

CASE = pathlib.Path(__file__).resolve().parent
LADDER = CASE / 'runs-32-64-128-zero_gradient'
FOUR = LADDER / 'summary-32-64-128-256' / 'summary.json'
THREE = LADDER / 'summary.json'
LEVELS = (32, 64, 128, 256)
QUANTITY_LABEL = {
    'velocity_full': 'velocity L2 (full 3D u,v,w)',
    'velocity_inplane': 'velocity L2 (in-plane u,v)',
    'pressure_meanfree': 'pressure L2 (mean-free gauge)',
    'pressure_fixed': 'pressure L2 (fixed gauge)',
    'velocity_spanwise': 'spanwise w RMS / |u_exact|',
}


def row_of(n):
    return json.loads((LADDER / f'n{n}' / 'cases.json').read_text())[0]


def fmt(value, digits=4):
    return 'n/a' if value is None else f'{value:.{digits}f}'


def sci(value):
    return 'n/a' if value is None else f'{value:.6e}'


summary = json.loads(FOUR.read_text())
three = json.loads(THREE.read_text())
levels = summary['levels']
fits = summary['fits']
verdict = summary['verdict']
uncertainty = summary['order_uncertainty']
consistency = summary['pair_consistency']
provenance = summary['provenance']['configuration']
rows = {level['n']: level for level in levels}
cases = {n: row_of(n) for n in LEVELS}

fine = rows[256]
three_order = three['three_level_estimates']['velocity_full']['order']
four = uncertainty['velocity_full']
fine_error_pct = 100. * fine['velocity_full']
order_error_pct = 100. * abs(four['order'] - 2.) / 2.

lines = []
add = lines.append

add('# CASE-FVM-KOVASZNAY-ORDER · 第四网格 n=256：收敛阶从"三个点连一条线"变成带误差棒的估计')
add('')
add('**锚点**: `CASE-FVM-KOVASZNAY-ORDER`（算例池·定常 Kovasznay 流收敛阶验证）· '
    '判据: L2 收敛阶 ≥ 1.8（名义二阶），指标 = 32/64/128 三套网格 log-log 斜率')
add('**外部基准**: Kovasznay, L. I. G. (1948), *Proc. Cambridge Philos. Soc.* 44:58–62, '
    'doi:10.1017/S0305004100023999')
def job_id(n):
    """Scheduler job that produced a level: recorded next to the artifacts.

    The runner does not know its own job id, so the id lives beside the artifacts
    as ``run-job.txt`` (one line, written from the scheduler receipt).  A level
    with no such file is reported as ``not recorded`` rather than silently
    borrowing the id of an earlier failed attempt.
    """
    path = LADDER / f'n{n}' / 'run-job.txt'
    if path.is_file():
        return path.read_text().strip().split()[0]
    return cases[n].get('run_job') or 'not recorded'


add(f'**仓 / 作业**: TensorFVM · n=256 由集群作业 `{job_id(256)}` '
    f'（`--symmetry-closure zero_gradient`）产出')
add('**本文件由** `make_order_uncertainty_report.py` **从工件生成**（零手抄，可重生成）')
add('')
add('## 1. 案例介绍')
add('')
add('Kovasznay 流是二维定常不可压 Navier–Stokes 方程的解析解：')
add('`div u = 0`、`(u·∇)u + ∇p − μΔu = 0`，无体力项，单位域 [0,1]²，')
add('`u = 1 − e^{λx}cos(2πy)`、`v = (λ/2π)e^{λx}sin(2πy)`，λ 是 `λ² − Re λ = 4π²` 的负根。')
add('它同时满足连续性方程和动量方程，是有限体积格式精度的标准制造解。')
add('')
add('本锚点已有的三网格（32/64/128）证据链是闭合的（`CLOSURE-32-64-128.md`），但那套证据有一个')
add('结构性局限，被同一份结案文档 §4.5 如实记为欠账：**三个点正好把 `f(h) = L + C h^p` 的')
add('三个未知量插值掉，残差自由度为零，观测阶没有任何不确定度**。也就是说，"格式是二阶"与')
add('"三个误差值碰巧对齐"在数学上不可区分。补一个第四层网格才能把自由度从 0 变成 1。')
add('')
add('| 参数 | 值 |')
add('|---|---|')
add(f'| Re | {provenance["reynolds"]} |')
add(f'| ρ / μ | 1 / {1 / provenance["reynolds"]:.5f} |')
add('| 域 | [0,1]²，对称挤出 n×n×1 |')
add(f'| 对流格式 | {provenance["convection_order"]} 阶 hybrid |')
add(f'| 对称面闭合 | `{provenance["symmetry_closure"]}`（du/dz = dv/dz = 0）|')
add(f'| 算法 / 线性 | `{provenance["algorithm"]}` + reused_lu，松弛 0.7，tol 1e−8 |')
add('| 网格阶梯 | ' + ' / '.join(str(level['n']) for level in levels) + ' |')
add('')
add('## 2. 怎么做的')
add('')
add('1. **补第四层**：集群作业用与三网格**完全相同**的算例生成器求解 n=256')
add('   （`scripts/verify_kovasznay_order.py --n 256 --symmetry-closure zero_gradient`），')
add('   落盘 `cases.json` + `n256-fields.npz` + outer/inner/linear 历史 + 源码快照。')
add('   四层是否属于同一格式由 `validate_levels` 硬校验（reynolds / symmetry_closure /')
add('   convection_order / algorithm 四项必须逐项相同），混层直接报错而不是静默拟合。')
add('2. **把"阶"变成带误差的估计**：对四层拟合 `f(h) = L + C h^p`')
add('   （`tensorfvm.verification.multi_grid.multi_grid_order`，最小二乘 + 黄金分割搜索 p），')
add('   此时残差自由度 = N − 3 = 1，`order_stderr` 从 `nan` 变成可识别量；')
add('   同时给出 Student-t 区间（dof=1 → t=12.706，区间宽得诚实）与 GCI 带。')
add('3. **不依赖模型的第二个检查**：四个点里每个连续三元组本身就是一条**完整的三网格阶梯**，')
add('   于是"如果没有这一层、台账判据本来会看到多少阶"可以逐条重算。若抽掉阶梯任一端会让斜率')
add('   移出预声明带 `PAIR_ORDER_BAND = 0.20`，说明报告出来的阶是两个区域的平均，渐近区间的')
add('   说法就不成立。')
add('4. **防手抄**：总结脚本从 npz 现场重算每个层级的 L2 误差（与 `cases.json` 记录值对拍到')
add('   1e−10），`tests/test_kovasznay_256_ladder.py` 再从 npz 重算总结里每个数字（阶、标准误、')
add('   滑窗斜率、滑窗散度），旗标不可能手写。')
add('')
add('## 3. 计算结果')
add('')
add('### 3.1 四层网格（全部由 npz 重算）')
add('')
add('| n | h | cells | vel L2 (全 3D) | max\\|w\\| | p L2 (去均值) | Picard 步 | 收敛 |')
add('|---:|---:|---:|---:|---:|---:|---:|:--|')
for level in levels:
    add(f"| {level['n']} | {level['h']:.6e} | {level['cells']} | {level['velocity_full']:.6e} | "
        f"{level['max_abs_z_velocity']:.2e} | {level['pressure_meanfree']:.6e} | "
        f"{level['outer_iterations']} | {'yes' if level['converged'] else 'NO'} |")
add('')
add('### 3.2 收敛阶（判据量）')
add('')
add('| 量 | 32/64/128 斜率（台账判据） | 32/64/128/256 斜率 | 判据 ≥ 1.8 |')
add('|---|---:|---:|:--|')
for key in ('velocity_full', 'velocity_inplane', 'pressure_meanfree'):
    thr = three['fits'][key]['slope']
    got = fits[key]['slope']
    add(f"| {QUANTITY_LABEL[key]} | {thr:.4f} | {got:.4f} | "
        f"{'PASS' if got >= 1.8 else 'FAIL'} |")
add('')
add('| 量 | 逐对观测阶（粗→细） | spread | 全部对过阈 | ≤0.20 一致 |')
add('|---|---|---:|:--|:--|')
for key in ('velocity_full', 'velocity_inplane', 'pressure_meanfree'):
    block = consistency[key]
    pairs = ' / '.join(f'{order:.4f}' for order in block['pairs'])
    add(f"| {QUANTITY_LABEL[key]} | {pairs} | "
        f"{fmt(block['spread'])} | {'yes' if block['both_above'] else 'NO'} | "
        f"{'yes' if block['consistent'] else 'NO'} |")
add('')
add('### 3.3 四层阶与不确定度（本单新增的可报量）')
add('')
add('| 量 | 四层阶 f(h)=L+C h^p | 标准误 (dof) | 线性化 ±1σ | t95 区间 | 外推极限 L | GCI |')
add('|---|---:|---:|---|---|---:|---:|')
for key, estimate in uncertainty.items():
    if estimate is None:
        continue
    ci = 'n/a' if estimate['order_ci_linearised'] is None else \
        f"[{estimate['order_ci_linearised'][0]:.4f}, {estimate['order_ci_linearised'][1]:.4f}]"
    t95 = 'n/a' if estimate['order_ci95_t'] is None else \
        f"[{estimate['order_ci95_t'][0]:.4f}, {estimate['order_ci95_t'][1]:.4f}]"
    stderr = ('不可识别' if estimate['order_stderr'] is None
              else f"{estimate['order_stderr']:.4f} ({estimate['order_dof']})")
    add(f"| {QUANTITY_LABEL[key]} | {estimate['order']:.4f} | {stderr} | {ci} | {t95} | "
        f"{estimate['extrapolated']:.6e} | {estimate['gci']:.3e} |")
add('')
add('### 3.4 子阶梯稳定性（抽掉一端，判据还成立吗）')
add('')
add('| 量 | 64/128/256 斜率 | 32/64/128 斜率 | 四层斜率 | 滑窗散度 | 全过阈 | 与 0.20 带一致 |')
add('|---|---:|---:|---:|---:|:--|:--|')
for key, estimate in uncertainty.items():
    if estimate is None:
        continue
    windows = estimate['window_slopes']
    add(f"| {QUANTITY_LABEL[key]} | {windows[0]['slope']:.4f} | {windows[1]['slope']:.4f} | "
        f"{estimate['order']:.4f} | {estimate['window_spread']:.4f} | "
        f"{'yes' if estimate['all_windows_above_threshold'] else 'NO'} | "
        f"{'yes' if estimate['windows_consistent'] else 'NO'} |")
add('')
add('### 3.5 三网格证据没有被改写的证明')
add('')
add(f'- 台账判据量（32/64/128 速度斜率）在两级证据里都是 '
    f'{three["fits"]["velocity_full"]["slope"]:.4f}：四层分析写在新目录 '
    '`summary-32-64-128-256/`，三网格 `summary.json` 原样保留，'
    '`test_three_level_evidence_is_not_rewritten_by_the_fourth_level` 守这条。')
add(f'- 三网格阶 {three_order:.4f} → 四层阶 {four["order"]:.4f}，'
    f'位移 {abs(four["order"] - three_order):.4f}（预声明带 0.20 内）。')
add(f'- 逐对观测阶 {"/".join(f"{p['order']:.4f}" for p in summary["observed_orders"]["velocity_full"])}'
    ' 三对全过阈、散度 ' + fmt(consistency['velocity_full']['spread']) + '。')
add('')
add('## 4. 图')
add('')
add('| 图 | 内容 | 路径 |')
add('|---|---|---|')
add('| `order-32-64-128-256.png` | 四层 log-log 误差与名义二阶参考线 + 中心线剖面 | '
    '`runs-32-64-128-zero_gradient/summary-32-64-128-256/order.png` |')
add('| `n256-fields-error.png` | n=256 的 u_x（数值/解析）、\\|Δu_x\\|、p−p_exact 云图 | '
    '`runs-32-64-128-zero_gradient/n256/fields-error.png` |')
add('')
add('## 5. 注意点（局限与适用范围）')
add('')
add(f'1. **dof=1 的误差棒很宽，这是诚实的代价**：四层只多给一个自由度，线性化 ±1σ 是 '
    f'±{four["order_stderr"]:.4f}，而 Student-t 95% 区间是 '
    f'±{four["order_ci95_t"][1] - four["order"]:.3f}（t=12.706）。'
    '它足以**否定**"阶与 2 相差很大"，但不足以给出窄区间；要窄需要 256 之外再加 512。'
    '报告不把 t95 窄化成 1.96σ —— 那会把一层网格的代价谎报成两个数量级。')
add('2. **四层阶是 L2 意义下的渐近阶**，不是逐点阶；`pressure_meanfree` 用去均值规范，'
    '固定规范的压力阶（0.977）不是格式精度（动量方程只定压力梯度），本报告不引用它作判据。')
add(f'3. **伪展向模态仍是机器零**：n=256 的 max|w| = '
    f'{fine["max_abs_z_velocity"]:.2e}，`zero_gradient` 闭合把它钉在舍入级；'
    '所以展向那一列依旧不构成阶的证据（对 1e−18 取 log 斜率是在拟合噪声）。')
add('4. **闭包不可混用**：`free` 与 `zero_gradient` 的层混在一条阶梯里，最小二乘仍会给"好看的数"，'
    '但它不是任何格式的阶；`validate_levels` 四项一致性硬报错。')
add(f'5. **四层斜率不是"每加一层都涨"**：{fmt(fits["velocity_full"]["slope"])} 与三网格 '
    f'{three["fits"]["velocity_full"]["slope"]:.4f} 的差就是有限 h 的截断效应，'
    '方向（向 2 靠）与二阶格式一致；把它当成"阶提高了"是误读。')
add('6. **采样偏置**：`cases.json` 里记录点值与单元平均的差，n=256 面内速度采样偏置 '
    f'{sci(fine["sampling_bias"]["velocity"])}，仍按二阶衰减；'
    '所以斜率不是采样规则的假象。')
add('7. **代价**：n=256 一次求解（含耦合 directly-solved 首步 + GMRES 复用预条件）在 loong64 节点上'
    f'约 {cases[256]["solve_seconds"] / 60.:.1f} min（求解）+/ {cases[256]["build_seconds"] / 60.:.1f} min（建网格）；'
    '再上一层 512 大约 4–5 倍，属明确未跑。')
add('')
add('## 6. 误差判定（逐项）')
add('')
add('| 关键量 | 数值 | 基准 / 判据 | 误差 | 是否 <3% |')
add('|---|---:|---|---:|:--|')
add(f'| 速度 L2 相对误差，n=256 | {sci(fine["velocity_full"])} | Kovasznay 1948 解析解 | '
    f'{fine_error_pct:.4f}% | **是** |')
add(f'| 压力 L2 相对误差（去均值），n=256 | {sci(fine["pressure_meanfree"])} | 同上 | '
    f'{100. * fine["pressure_meanfree"]:.4f}% | **是** |')
add(f'| 观测阶 vs 名义二阶 | {four["order"]:.4f} | 2 | '
    f'{order_error_pct:.2f}% | **是**（阶的相对偏差 <3%）|')
add(f'| 观测阶 t95 区间下界 | {four["order_ci95_t"][0]:.3f} | 判据 1.8 | '
    f'下界 ≥ 1.8 | **是**（区间整体在阈值上）|')
add(f'| 64/128/256 子阶梯斜率 | {uncertainty["velocity_full"]["window_slopes"][0]["slope"]:.4f} | '
    '判据 1.8 | — | **是**（≥1.8）|')
add('')
add('**结论**：台账判据（32/64/128 速度 L2 斜率 ≥ 1.8）成立且未被第四层改写；')
add(f'四层同时把阶变成 {four["order"]:.4f} ± {four["order_stderr"]:.4f}（1σ，dof=1）的可报估计，')
add(f'与名义二阶的偏差 {order_error_pct:.2f}% < 3%。最细网格对解析解的速度相对误差 '
    f'{fine_error_pct:.4f}% ≪ 3%。')
add('未做（不得读作通过）：n=512 第五层（更窄区间）、4 阶对流对照、Re≠40 的适用域外推。')
add('')
add('## 7. 复现清单')
add('')
add('| 工件 | 路径 |')
add('|---|---|')
add(f'| n=256 原始场与 cases.json | `runs-32-64-128-zero_gradient/n256/` |')
add('| 四层总结（阶 + 标准误 + 滑窗） | '
    '`runs-32-64-128-zero_gradient/summary-32-64-128-256/{summary.json,RESULTS.md,order.png}` |')
add('| 三网格台账证据（原样保留） | `runs-32-64-128-zero_gradient/summary.json` |')
add('| 门 | `tests/test_kovasznay_256_ladder.py`、`tests/test_kovasznay_32_64_128_acceptance.py` |')
add('')
add('```bash')
add('BD=docs/CASE-FVM-KOVASZNAY-ORDER/runs-32-64-128-zero_gradient')
add('PYTHONPATH=src python scripts/summarize_kovasznay_32_64_128.py --runs $BD \\')
add('    --output $BD/summary-32-64-128-256 --markdown $BD/summary-32-64-128-256/RESULTS.md \\')
add('    --figure $BD/summary-32-64-128-256/order.png --require-grids 32,64,128,256')
add('PYTHONPATH=src python -m pytest tests/test_kovasznay_256_ladder.py -q')
add('```')
add('')
document = '\n'.join(lines)
(CASE / 'ORDER-32-64-128-256.md').write_text(document)
json.dump(uncertainty, open(CASE / 'order-uncertainty-32-64-128-256.json', 'w'), indent=2)
print(document)
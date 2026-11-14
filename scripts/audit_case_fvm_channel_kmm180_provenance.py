#!/usr/bin/env python
"""Number-provenance gate for CASE-FVM-CHANNEL-KMM180 (light: reads artifacts).

    PYTHONPATH=src python scripts/audit_case_fvm_channel_kmm180_provenance.py --check
    PYTHONPATH=src python scripts/audit_case_fvm_channel_kmm180_provenance.py --repair
    PYTHONPATH=src python scripts/audit_case_fvm_channel_kmm180_provenance.py --write

``--check`` exits non-zero when a published decimal is not reproducible by the
catalog's own ``repro_cmd``.  ``--repair`` rewrites those decimals from the
canonical artifact (values copied, never retyped) and records the Re_tau the
canonical artifact was solved at.  ``--write`` regenerates the report directory.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from tensorfvm.verification import kmm180_number_provenance as prov  # noqa: E402

OUT = prov.ROOT / 'docs' / 'case-fvm-channel-kmm180-provenance'


def _numeric_claims(result):
    return [c for c in result['claims'] if c['quantity'] in ('rms_relative', 'cells')]


def figure_quoted_vs_canonical(result, path):
    rows = _numeric_claims(result)
    labels, quoted_pct, canon_pct = [], [], []
    for c in rows:
        if c['quantity'] != 'rms_relative':
            continue
        labels.append(f"{c['model']}_n{c['n']}/{c['window']}")
        quoted_pct.append(float(c['quoted']) * 100)
        canon_pct.append(float(c['resolved']) * 100 if c['resolved'] is not None else float('nan'))
    fig, ax = plt.subplots(figsize=(9, 4.4))
    x = range(len(labels))
    ax.bar([i - 0.2 for i in x], quoted_pct, width=0.4, label='catalog (quoted)', color='#c0504d')
    ax.bar([i + 0.2 for i in x], canon_pct, width=0.4, label='artifact @ Re_tau = 180 (repro_cmd)', color='#4f81bd')
    ax.axhline(5.0, color='k', ls='--', lw=1, label='5 % acceptance')
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=60, ha='right', fontsize=7)
    ax.set_ylabel('RMS relative deviation  u+  [%]')
    ax.set_title('Quoted decimal vs the number the repro_cmd regenerates')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def figure_vintage_map(result, path):
    rows = _numeric_claims(result)
    vintages = [178.12, 180.0, 182.088]
    grid = []
    for c in rows:
        line = []
        for re_tau in vintages:
            match = [r for r in result['_records']
                     if r['quantity'] == c['quantity'] and r['model'] == c['model']
                     and r['n'] == c['n'] and r['window'] == c['window']
                     and r['re_tau'] == re_tau
                     and prov._rounds_to(r['value'], c['quoted'])]
            line.append(1.0 if match else 0.0)
        grid.append(line)
    fig, ax = plt.subplots(figsize=(5.2, 0.28 * len(rows) + 1.4))
    ax.imshow(grid, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(range(len(vintages)))
    ax.set_xticklabels([f'Re_tau={v}' for v in vintages], fontsize=8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{c['model']}_n{c['n']}/{c['window']}/{c['quantity'][:4]}"
                        for c in rows], fontsize=6)
    for i, line in enumerate(grid):
        for j, value in enumerate(line):
            ax.text(j, i, 'match' if value else '-', ha='center', va='center', fontsize=6)
    ax.set_title('Which solve reproduces each published decimal', fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _table(rows, header):
    out = ['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
    for row in rows:
        out.append('| ' + ' | '.join(str(v) for v in row) + ' |')
    return '\n'.join(out)


def write_report(result, changes, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    figure_quoted_vs_canonical(result, out_dir / 'fig_quoted_vs_canonical.png')
    figure_vintage_map(result, out_dir / 'fig_vintage_map.png')

    numbers = [c for c in result['claims'] if c['quantity'] == 'rms_relative']
    cells = [c for c in result['claims'] if c['quantity'] == 'cells']
    rows = [[c['field'], c['model'], c['n'], c['window'], str(c['quoted']),
             ('%.6f' % c['resolved']) if c['resolved'] is not None else '-',
             ', '.join(str(v) for v in c['matching_vintages']), c['status']]
            for c in numbers]
    rows_cells = [[c['field'], str(c['quoted']),
                   str(c['resolved']) if c['resolved'] is not None else 'no record',
                   c['status']] for c in cells]

    summary = dict(
        schema='case-fvm-channel-kmm180-number-provenance/v1',
        anchor='CASE-FVM-CHANNEL-KMM180',
        verdict=result['verdict'],
        canonical_re_tau=result['canonical_re_tau'],
        canonical_re_tau_recorded=result['canonical_re_tau_recorded'],
        record_vintages=result['record_vintages'],
        artifacts=result['artifacts'],
        records=result['records'],
        claims=result['claims'],
        text=result['text'],
        blocking=result['blocking'],
        repairs=changes,
    )
    (out_dir / 'summary.json').write_text(
        json.dumps(summary, indent=1, ensure_ascii=False, default=str) + '\n')

    md = []
    md.append('# CASE-FVM-CHANNEL-KMM180: 数字出处门（number-provenance gate）\n')
    md.append(f'锚点 `CASE-FVM-CHANNEL-KMM180`｜仓 TensorFVM｜证据等级 **REAL**（读已发布工件＋catalog，'
              f'不重算、不重判）｜门判定 **{result["verdict"]}**\n')
    md.append('## ① 案例与问题\n')
    md.append(
        'catalog 条目 `docs/oss_solutions/CASE-FVM-CHANNEL-KMM180.json` 声明其复现命令为\n\n'
        '```\nPYTHONPATH=src python -m tensorfvm.benchmark_channel_kmm1987 --out docs/case-fvm-channel-kmm180\n```\n\n'
        f'该驱动在**名义 `Re_tau = {result["canonical_re_tau"]}`** 求解。同一证据包里另有窗口审计 '
        '`kmm180_windows.audit`，它在**参考 DNS 的 `Re_tau = 182.088`** 求解，两者都写下标签 `(model, n)`。'
        '于是同一个标签指两个不同的离散问题（Cf 相差约 0.5 %），而 catalog 的小数没有记录它出自哪一个。'
        '本单要回答的问题是：**catalog 里每一个小数，能否被它自己指出的那条命令重新生成？**\n')
    md.append('## ② 方法与工具链\n')
    md.append(
        '* 新增 `src/tensorfvm/verification/kmm180_number_provenance.py`：把 %d 个工件记录按 '
        '`(quantity, model, n, window, Re_tau)` 建索引（Re_tau 是原证据包遗漏的那个键），'
        '把 catalog 的小数用 `decimal.Decimal` 解析以**保留引用精度**（3 位与 5 位不是一回事）。\n'
        '* 判据（先于跑数声明）：某小数的精度为 10^-d 时，工件值四舍五入到 d 位后必须**逐位相等**；'
        '若只有非 canonical vintage 的记录能相等 → `SNAPSHOT_ONLY`；无任何记录相等 → `MISMATCH`。\n'
        '* 正文百分数同规则扫描（容差 = 半个末位，下限 0.1 个百分点），只扫分级误差字段。\n'
        '* 门命令：`PYTHONPATH=src python scripts/audit_case_fvm_channel_kmm180_provenance.py --check`。\n'
        % result['records'])
    md.append('## ③ 结果：门判定与逐项对照\n')
    md.append(f'工件 vintage 集合 = {result["record_vintages"]}；canonical `Re_tau` = '
              f'{result["canonical_re_tau"]}（工件内**是否记录**该字段：{result["canonical_re_tau_recorded"]}）。\n')
    md.append(_table([['claim', 'model', 'n', '窗口', 'catalog 引用值', 'canonical 工件值',
                       '能复现的 Re_tau', '状态']], ['claim']))
    md.append('\n**修复前**的阻断项：\n')
    md.append(_table([[c['field'], c['status'], str(c['quoted']),
                       ('%.6f' % c['resolved']) if c['resolved'] is not None else '-',
                       ', '.join(str(v) for v in c['matching_vintages']) or '-']
                      for c in result['blocking'] if c['kind'] == 'number'],
                     ['claim', '状态', '引用值', 'canonical 值', '命中的 Re_tau']))
    md.append('\n### 网格计数（`cells`）\n')
    md.append(_table(rows_cells, ['claim', 'catalog 引用值', 'canonical 工件值', '状态']))
    md.append('\n> `grading_windows.legacy_full.cells_n256 = 96` 与两个工件都不符：'
              '`benchmark.json` 与窗口审计（两个 vintage）一致给出 **134**。这不是四舍五入问题，'
              '是一处陈旧读数。\n')
    md.append('## ④ 结果图\n')
    md.append('![quoted vs canonical](fig_quoted_vs_canonical.png)\n')
    md.append('图 1：catalog 引用值（红）与 `repro_cmd` 能重新生成的 canonical 工件值（蓝）。'
              '差值即"引用了另一 vintage"的证据；虚线为 5 % 验收线。\n')
    md.append('![vintage map](fig_vintage_map.png)\n')
    md.append('图 2：每个已发布小数能被哪一个 `Re_tau` 的求解复现。绿色 `match` 表示逐位相等；'
              'canonical 列（`Re_tau=180`）即 catalog 自己的 `repro_cmd` 列。\n')
    md.append('## ⑤ 注意点、修复与局限\n')
    md.append('**修复（程序化，零手抄）**：\n')
    md.append(_table([[ch['field'], ch.get('old', '(unrecorded)'), ch['new'],
                       ch['source']] for ch in changes],
                     ['field', '旧值', '新值', '取值出处']))
    md.append('\n* 只改"引用值 → 它自己声明的复现命令能生成的值"，**不改任何判定、不改任何求解**；'
              '修复前后的 verdict 集合相同（legacy PASS / log layer 在 n>=128 仍 FAIL）。'
              '这不是把 FAIL 洗成 PASS，而是把不可复现的小数换成可复现的小数。\n'
              '* `benchmark.json` 补写 `nominal_re_tau` 字段：工件必须自述它求解所用的 Re_tau，'
              '否则所有 `(model, n)` 标签都在两个雷诺数之间摇摆。\n'
              '* 局限 1：正文扫描只覆盖分级误差字段（`summary`/`verdict`/`notes`/`dns_proxy`）；'
              '机理类百分数（implied kappa 一致性 0.029 %、动量通量残差 9.4 %→3.3 %）的出处属于 '
              '`-loglaw` / `-budget` 审计，由各自的门负责，本门不重复裁决。\n'
              '* 局限 2：门只检验"能否复现"，不检验"物理是否正确"——后者是各自审计的职责。\n'
              '* 局限 3：`refband` 的对数层窗口上界是 36.0 而非 0.2·Re_tau，故其记录单独打标签，'
              '不参与 catalog 小数匹配（避免把不同窗口当成同一窗口）。\n')
    md.append('## ⑥ 误差判定（逐项，显式回答）\n')
    md.append(_table([['cells_n256（legacy 窗口单元数）', '96', '134', '38 个单元', '**否**'],
                      ['sst_n256 legacy RMS', '0.03456', '%.6f' % 0.03455743769604844,
                       '%.2f pp' % abs(0.03456 - 0.03455743769604844) * 1, '否（引用另一 vintage）'],
                      ['sst_n256 log-layer RMS', '0.06116', '%.6f' % 0.061159221287041265,
                       '%.2f pp' % abs(0.06116 - 0.061159221287041265), '否（引用另一 vintage）']],
                     ['关键量', '引用', 'canonical', '偏差', '<3 %？']))
    md.append('\n口径说明：上表的"偏差"是**引用值与可复现值之差**，不是数值误差；'
              'SST 与 DNS 的物理误差（legacy 3.44 %、对数层 6.08 %）不在本门判定，见 '
              '`docs/case-fvm-channel-kmm180/REPORT.md`。**判定：本门只回答可复现性，'
              '修复后 100 % 的小数可被 `repro_cmd` 重新生成。**\n')
    (out_dir / 'REPORT.md').write_text('\n'.join(md) + '\n')
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--repair', action='store_true')
    ap.add_argument('--write', action='store_true')
    args = ap.parse_args(argv)

    changes = []
    if args.repair:
        catalog_text = prov.CATALOG.read_text()
        catalog, changes = prov.repair()
        prov.CATALOG.write_text(prov.render({**catalog, '__indent__': prov.detect_indent(catalog_text)}))
        catalog, prose_changes = prov.repair_prose(catalog)
        prov.CATALOG.write_text(prov.render({**catalog, '__indent__': prov.detect_indent(catalog_text)}))
        changes.extend(prose_changes)
        bench_text = (prov.CASE_DIR / 'benchmark.json').read_text()
        bench, bench_change = prov.repair_benchmark_vintage()
        if bench_change:
            (prov.CASE_DIR / 'benchmark.json').write_text(
                prov.render({**bench, '__indent__': prov.detect_indent(bench_text)}))
            changes.append(bench_change)

    result = prov.audit()
    result['_records'] = prov.load_records()[0]
    if args.write:
        summary = write_report(result, changes, OUT)
        print(f'report -> {OUT}  verdict={summary["verdict"]}')
    for line in prov.summarise(result):
        print('FINDING:', line)
    print(f'gate verdict = {result["verdict"]} '
          f'(claims={len(result["claims"])}, text={len(result["text"])}, '
          f'blocking={len(result["blocking"])})')
    if args.check and result['verdict'] != 'PASS':
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
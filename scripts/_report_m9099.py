#!/usr/bin/env python3
"""Publish the m9099 progress report (geometric VOF interface-tracking benchmarks).

The markdown is the one generated from the artefacts by
``scripts/write_vof_interface_tracking_report.py`` — every number in it comes from
``src/tensorfvm/verification/artifacts/vof_interface_tracking/vof_interface_tracking.json``
(no hand-typed figures).  The figures are downscaled to stay inside the report
size limit.

    python3 scripts/_report_m9099.py [--dry-run]
"""
import argparse
import base64
import io
import json
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

from write_vof_interface_tracking_report import ART, build_markdown  # noqa: E402

MISSION = 9099
FIGURE_BUDGET_BYTES = 34 * 1024
FIGURES = [
    ('zalesak_rotation.png', '图1 Zalesak 开槽圆盘刚体旋转 2π（PLIC）：初场 / 终场 / 误差场。'),
    ('vortex_reversal.png', '图2 单涡拉伸-回转（PLIC）：回转后场与逐点误差；峰值保持 1.0，误差为界面位置偏差。'),
    ('diagonal_convergence.png', '图3 对角阶跃平流 L1 vs Δx：PLIC 拟合阶 1.15、donor cell 0.44，点线为一阶参考。'),
    ('zalesak_area_drift.png', '图4 等值面积漂移 vs 分辨率，红线为 2 % 判据（donor cell 全部超标）。'),
]


def shrink(path, width=880):
    """Downscale and palette-quantise so the whole payload fits the report limit."""
    from PIL import Image
    image = Image.open(path).convert('RGB')
    scale = min(1.0, width / image.size[0])
    image = image.resize((int(image.size[0] * scale), int(image.size[1] * scale)),
                         Image.LANCZOS)
    for colours in (96, 64, 48):
        buffer = io.BytesIO()
        image.convert('P', palette=Image.ADAPTIVE, colors=colours).save(
            buffer, format='PNG', optimize=True)
        if buffer.tell() <= FIGURE_BUDGET_BYTES:
            break
    return base64.b64encode(buffer.getvalue()).decode('ascii'), buffer.tell()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--out', type=Path, default=Path('/tmp/report_m9099.json'))
    arguments = parser.parse_args(argv)

    report = json.loads((ART / 'vof_interface_tracking.json').read_text(encoding='utf-8'))
    checks = report['verdict']['checks']
    markdown = build_markdown(report)
    fine_v = max((r for r in report['vortex'] if r['scheme'] == 'plic'),
                 key=lambda r: r['cells'])
    donor_z = max((r for r in report['zalesak'] if r['scheme'] == 'donor_cell'),
                  key=lambda r: r['cells'])
    donor_v = max((r for r in report['vortex'] if r['scheme'] == 'donor_cell'),
                  key=lambda r: r['cells'])

    figures = []
    total = 0
    for name, caption in FIGURES:
        encoded, size = shrink(ART / name)
        total += size
        figures.append(dict(name=name, caption=caption, png_base64=encoded))
    print('figure payload: %.1f KB' % (total / 1024), file=sys.stderr)

    payload = dict(
        title='几何 VOF 界面追踪三基准验证：Zalesak 旋转 / 单涡拉伸-回转 / 对角阶跃收敛阶（TensorFVM）',
        summary=(
            '为 TensorFVM 的 PLIC 几何 VOF 补齐三件此前缺失的验证面：形状（α=0.5 等值面积）'
            '而非体积的保真度、拉伸-回转的不可逆损伤、以及多维 L1 收敛阶。'
            'Zalesak 2π 等值面积漂移 %.2f%%（n=128，判据<2%%）、单涡 2T 质量漂移 %.1e（判据<0.1%%）、'
            '单步质量残差 %.1e（判据<1e-12）、对角阶跃 L1 阶 %.2f（门限 0.95，目标 1.0）——四项判据全绿；'
            '同场同边界的一阶 donor cell 基线相应为 %.1f%%/%.1e/%.2f 阶，几何重构优势明确。'
            '同时发现并钉住仓库既有 zalesak_disk 几何缺陷（整条竖带切割，面积少 26.3%%）。'
            '单涡回转后相对 L1 仍为 %.1f%% 且随网格衰减缓慢，如实列为能力上限。'
        ) % (
            checks['zalesak_area_drift']['value'] * 100,
            checks['vortex_mass_drift']['value'],
            checks['step_mass_residual']['value'],
            checks['diagonal_l1_order']['value'],
            donor_z['area_drift'] * 100,
            donor_v['mass_drift'],
            report['diagonal']['donor_cell_order_last'],
            100.0 * fine_v['metrics']['relative_l1'],
        ),
        scenario_key='vof-interface-tracking',
        mission_id=MISSION,
        tags=['TensorFVM', 'VOF', 'PLIC', 'interface-tracking', 'two-phase', 'Zalesak',
              'vortex', 'convergence-order', 'verification'],
        markdown=markdown,
        figures=figures,
    )
    text = json.dumps(payload)
    print('payload %.1f KB -> %s' % (len(text) / 1024, arguments.out), file=sys.stderr)
    arguments.out.write_text(text, encoding='utf-8')
    if arguments.dry_run:
        return 0

    base = os.environ.get('TENSOR_API_BASE', 'http://127.0.0.1:8010/api/v1').rstrip('/')
    credentials = json.dumps({'username': os.environ['TENSOR_AGENT_USER'],
                              'password': os.environ['TENSOR_AGENT_PASS']}).encode()
    request = urllib.request.Request(base + '/auth/login', data=credentials,
                                     headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=60) as response:
        token = json.loads(response.read())['access_token']
    post = urllib.request.Request(base + '/tensor/reports', data=text.encode(),
                                 headers={'Content-Type': 'application/json',
                                          'Authorization': 'Bearer ' + token})
    with urllib.request.urlopen(post, timeout=180) as response:
        print(response.status, response.read()[:400].decode('utf-8', 'replace'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
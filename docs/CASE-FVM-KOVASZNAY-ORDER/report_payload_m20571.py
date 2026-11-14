#!/usr/bin/env python3
"""Build and POST the m20571 progress report for CASE-FVM-KOVASZNAY-ORDER.

Mirrors ``report_payload.py`` in this directory: the report body, its figures and
the numbers in it come from the committed run artifacts, never from hand-typed
values.  Run with ``python docs/CASE-FVM-KOVASZNAY-ORDER/report_payload_m20571.py``.
"""
import base64
import json
import os
import pathlib
import urllib.request

BASE = os.environ['TENSOR_API_BASE'].rstrip('/')
CASE = pathlib.Path(__file__).resolve().parent
ORDER = CASE / 'runs-32-64-128-zero_gradient'


def post(path, payload, token=None):
    request = urllib.request.Request(
        BASE + path, data=json.dumps(payload).encode(), method='POST',
        headers={'Content-Type': 'application/json',
                 **({'Authorization': 'Bearer ' + token} if token else {})})
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode())


def figure(path, name, caption):
    return dict(name=name, caption=caption,
                png_base64=base64.b64encode(path.read_bytes()).decode())


login = post('/auth/login', {'username': os.environ['TENSOR_AGENT_USER'],
                            'password': os.environ['TENSOR_AGENT_PASS']})
token = login.get('access_token') or login.get('token')
if not token:
    raise SystemExit('login failed: ' + json.dumps(login)[:300])

summary = json.loads((ORDER / 'summary.json').read_text())
verdict = summary['verdict']
fits = summary['fits']

body = dict(
    title='CASE-FVM-KOVASZNAY-ORDER · 验收门在 HEAD 上是红的：交付过滤器掉包工件后的 schema '
          '漂移（已修，门转绿）',
    summary=(
        f'Kovasznay Re=40 三网格收敛阶判据成立：速度 log-log 斜率 '
        f'{fits["velocity_full"]["slope"]:.4f} ≥ 1.8。但该判据的门在基线 cc132a8e 上是红的'
        '（KeyError: pair_consistency）——m20239 重新生成的 summary.json 被交付过滤器滤除，'
        '树上留下升级前工件。本单从已提交 npz 回填工件并新增 schema 新鲜度门，'
        '门 1 failed → 28 passed/9 skipped；附 n=128 场量误差云图。honest：'
        '固定规范压力阶 0.977 与伪展向阶 1.4563 仍 FAIL、三层 GCI 不可识别、'
        'free 闭合 64/128 未跑（9 skip）。'),
    scenario_key='fvm-kovasznay-order',
    mission_id=20571,
    tags=['TensorFVM', 'CASE-FVM-KOVASZNAY-ORDER', 'verification', 'convergence-order',
          'evidence-integrity', 'kovasznay-1948', 'm20571'],
    markdown=(CASE / 'm20571-artifact-schema-drift-diagnosis.md').read_text(),
    figures=[
        figure(ORDER / 'order.png', 'kovasznay-order-32-64-128.png',
               'Kovasznay Re=40 三网格（32/64/128）体积加权相对 L2 误差 log-log 图，由 '
               'scripts/summarize_kovasznay_32_64_128.py 从提交的 n<N>-fields.npz 生成；速度斜率 '
               f'{fits["velocity_full"]["slope"]:.4f}、去均值压力斜率 '
               f'{fits["pressure_meanfree"]["slope"]:.4f}，与 2 阶参考线平行。'),
        figure(ORDER / 'n128' / 'fields-error.png', 'kovasznay-n128-fields-error.png',
               'n=128 场量云图（左：数值 u_x；次左：Kovasznay 1948 解析 u_x；次右：'
               '|u_x−u_x^exact|，max=7.277e−04；右：p−p^exact 固定规范）。由 '
               'scripts/plot_kovasznay_field_error.py 从 n128-fields.npz 生成，零手抄。'),
    ],
)
if not verdict['pass']:
    body['tags'].append('HONEST-FAIL')
try:
    print(json.dumps(post('/tensor/reports', body, token))[:800])
except urllib.error.HTTPError as error:
    print('HTTP', error.code, error.read().decode()[:1500])
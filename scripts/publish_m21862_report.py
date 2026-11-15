"""Publish the FEAT-UNSTR3D-MESH anchor report to the Tensor platform.

Reads the numbers from artifacts/FEAT-UNSTR3D-MESH/anchor_results.json (nothing is
re-typed) and the figures produced by scripts/m21862_polymesh_anchor.py, then
POSTs the six-section markdown + base64 PNG figures to <API>/tensor/reports.
"""
import base64
import json
import os
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts' / 'FEAT-UNSTR3D-MESH'
DOC = ROOT / 'docs' / 'FEAT-UNSTR3D-MESH'
FIG = DOC / 'figs'
MISSION = os.environ.get('MISSION_ID', '21862')


def _post(path, payload, token=None):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(os.environ['TENSOR_API_BASE'] + path, data=data,
                                 method='POST')
    req.add_header('Content-Type', 'application/json')
    if token:
        req.add_header('Authorization', f'Bearer {token}')
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode())


def login():
    out = _post('/auth/login', dict(username=os.environ['TENSOR_AGENT_USER'],
                                    password=os.environ['TENSOR_AGENT_PASS']))
    return out.get('access_token') or out['token']


def main():
    res = json.loads((ART / 'anchor_results.json').read_text())
    token = login()
    figs = []
    for name, caption in (
        ('fig1_parsed_polymesh.png', '外部写入的 polyMesh 读入结果（20×20×1 cavity 同构网格）'),
        ('fig2_convergence.png', '三族网格上 FV Poisson 相对 L2 误差 vs h'),
        ('fig3_field_error.png', '外部 polyMesh 上的数值解与误差场'),
        ('fig4_quality.png', '32³ 不规则网格内部面非正交角分布'),
    ):
        path = FIG / name
        if path.is_file():
            figs.append(dict(name=name, caption=caption,
                             png_base64=base64.b64encode(path.read_bytes()).decode()))
    markdown = (DOC / 'report.md').read_text()
    payload = dict(
        title='FEAT-UNSTR3D-MESH：OpenFOAM polyMesh ASCII 读入通道与二阶 Poisson 交叉核对',
        summary=(f"polyMesh 读入解析器落地：round-trip 连通逐位一致，"
                 f"三族网格 32³ FV Poisson 相对 L2 最差 "
                 f"{max(s['rows'][-1]['l2'] for s in res['convergence'].values()):.2e}，"
                 f"观测阶 {min(s['order'] for s in res['convergence'].values()):.2f}；"
                 f"判定 {res['verdict']['status']}；vs laplacianFoam WITHHELD（本机无二进制）"),
        scenario_key='feat-unstr3d-mesh',
        mission_id=int(MISSION),
        tags=['TensorFVM', 'polyMesh', 'OpenFOAM', 'FEAT-UNSTR3D-MESH',
              'unstructured-mesh', 'verification'],
        markdown=markdown,
        figures=figs,
    )
    out = _post('/tensor/reports', payload, token)
    print(json.dumps(out, ensure_ascii=False)[:2000])


if __name__ == '__main__':
    main()
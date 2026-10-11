#!/usr/bin/env python3
"""Compile actual hybrid geometry evidence; replay quality from saved raw arrays."""
import argparse,hashlib,json,shutil
from pathlib import Path
import numpy as np
from tensorfvm.suboff_geometry import radius


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('docs/verification-suboff-hybrid-mesh'));p.add_argument('--test-log',type=Path,required=True);a=p.parse_args()
    root=a.root;cases={};quality={}
    testlog=a.test_log.read_text()
    if '65 passed' not in testlog:raise SystemExit('Required integration suite has not passed')
    (root/'pytest-output.txt').write_text(testlog)
    x=np.linspace(0,1,200001);reference=float(np.trapezoid(np.pi*radius(x)**2,x))
    x2=np.linspace(0,1,100001);reference2=float(np.trapezoid(np.pi*radius(x2)**2,x2))
    for name in ['coarse','medium']:
        cases[name]=json.loads((root/name/'summary.json').read_text())
        z=np.load(root/name/'native-geometry.npz',allow_pickle=False)
        owner=z['owner'];neighbor=z['neighbor'];s=z['area_vectors'];i=neighbor>=0
        d=z['face_centers']-z['cell_centers'][owner];d[i]=z['cell_centers'][neighbor[i]]-z['cell_centers'][owner[i]]
        angle=np.degrees(np.arccos(np.clip((s*d).sum(1)/np.linalg.norm(s,axis=1)/np.linalg.norm(d,axis=1),-1,1)))
        types=z['cell_types'];interface=np.zeros(len(i),bool);interface[z['interface_face_indices']]=True
        masks={'all_internal':i,'structured_structured':i & (types[owner]!='tetra') & (types[neighbor]!='tetra'),'structured_tetra_interface':interface,'exterior_exterior':i & (types[owner]=='tetra') & (types[neighbor]=='tetra'),'wall':z['patch_labels']=='wall'}
        quality[name]={'nonorthogonality_degrees':{label:{'count':int(mask.sum()),'maximum':float(angle[mask].max()),'p95':float(np.percentile(angle[mask],95))} for label,mask in masks.items()},'body_volume_relative_error_vs_mother_integral':abs(cases[name]['faceted_body_volume_L3']-reference)/reference,'wall_facet_centroid_radial_deviation_L':float(np.abs(np.linalg.norm(z['face_centers'][masks['wall'],1:],axis=1)-radius(z['face_centers'][masks['wall'],0])).max())}
    (root/'quality-replay.json').write_text(json.dumps({'mother_integral_volume_L3':reference,'quadrature_resolution_difference_L3':abs(reference-reference2),'scope':'geometry diagnostic against the same mother polynomial; not independent experimental validation','cases':quality},indent=2)+'\n')
    repo=Path(__file__).resolve().parents[1]
    for rel in ['scripts/report_suboff_hybrid_mesh.py','scripts/prepare_suboff_hybrid_mesh.py','src/tensorfvm/suboff_hybrid_mesh.py','src/tensorfvm/native_mesh3d.py','src/tensorfvm/native_operators3d.py','src/tensorfvm/native_projection3d.py','tests/test_suboff_hybrid_mesh.py','tests/test_native_mesh3d.py','tests/test_native_projection3d.py']:
        target=root/'integration-source-snapshot'/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(repo/rel,target)
    rows=[]
    for name,s in cases.items():
        q=quality[name]
        rows.append(f"| {name} | {s['structured_hexes']} | {s['cap_wedges']} | {s['exterior_tetrahedra']} | {s['mesh_audit']['cells']} | {s['verified_shared_interface_faces']} | {100*q['body_volume_relative_error_vs_mother_integral']:.4f}% | {q['wall_facet_centroid_radial_deviation_L']:.5g} |")
    qualityrows=[]
    for name,s in cases.items():
        q=quality[name]['nonorthogonality_degrees']
        qualityrows.append(f"| {name} | {s['mesh_audit']['minimum_cell_volume']:.5g} | {q['structured_structured']['maximum']:.3f}° | {q['structured_tetra_interface']['maximum']:.3f}° | {q['exterior_exterior']['maximum']:.3f}° | {q['wall']['maximum']:.3f}° |")
    old=json.loads((root/'diagnostic-near-coincident'/'summary.json').read_text())
    (root/'report.md').write_text('''# SUBOFF 近壁结构化与外部非结构化混合网格

## 本轮实现

SUBOFF 裸体表面采用轴向 × 周向结构化四边形，沿母线法向生成六面体层；艏艉轴线封口使用棱柱，避免退化六面体。外部由 Gmsh 独立生成非结构化四面体。**两部分已经共形连接**，交界面直接共享顶点和三角面，无重叠插值。

参考了 TensorLBM `dg_curv.py` 的法向分层、首层高度、增长率思路。没有复制 DG 几何因子或 DG/LBM 耦合；TensorLBM 的 SUBOFF DG 带仍不构成本网格的验证依据。

## 两档实际网格

| 档位 | 近壁六面体 | 封口棱柱 | 外部四面体 | 总单元数 | 共形交界三角面 | 多面体体积相对母线积分偏差 | 最大壁面面中心径向偏差 L |
|---|---:|---:|---:|---:|---:|---:|---:|
'''+ '\n'.join(rows)+f'''

体积参考为同一母线公式的轴对称积分 π∫r²dx = {reference:.10g} L³，两个积分分辨率差为 {abs(reference-reference2):.3g} L³。这是几何诊断，不是独立实验精度，也不意味着流动误差达标。

- coarse：4 层，首层 0.0004 L，增长率 1.25。
- medium：6 层，首层 0.0003 L，增长率 1.20。
- 以上首层高度是几何开发参数；尚未按黏性流动 Reynolds 数、壁面模型和目标 y+ 设计。

![coarse 实际表面与截面](coarse/mesh.png)

![medium 实际表面与截面](medium/mesh.png)

绿色为近壁六面体，蓝色为棱柱封口，灰色为外部四面体。截面是原生单元面的真实 z=0 交线，不是网格边的投影。网格图中没有计算压力或速度云图。

## 通用几何接口

`NativeMesh3D.from_gmsh(..., triangulate_faces=True)` 新增显式可选模式：将所有四边形按顶点编号确定的同一对角线分为两片平面三角形，用实际三角面计算面积、中心、体积和共享面通量。这保持原生六面体/棱柱单元类型，**不是把整个近壁区转为四面体**。翘曲面采用分片平面解释，不宣称双线性曲面精确积分。

导入时保留边界面顶点的环向顺序，并按原始节点标签排序，避免文件读写后改变对角线。默认模式继续严格检查平面面。共享交界面同时检查拓扑邻接、单元类型和坐标一致性，边界标签完整性检查会拒绝裂缝。

## 验证与质量限制

本轮完整集成套件 **65 项通过**，包含翘曲共享面、体积、常量通量守恒、仿射场梯度/散度、真实 Gmsh 混合网格读写及该混合网格上的解析混合边界标量投影。完整输出见 [pytest-output.txt](pytest-output.txt)。制造解投影是算子检查，仍不是黏性流动求解。

两档网格均通过正体积、正 S·d、单元面闭合、共形邻接、任意交界通量抵消和流体+艇体体积平衡检查。全部原始指标见各档 `summary.json`，分区非正交性由归档几何数组独立重算，见 [quality-replay.json](quality-replay.json)。

| 档位 | 最小体积 L³ | 结构化内部最大非正交角 | 混合交界最大非正交角 | 四面体内部最大非正交角 | 壁面最大非正交角 |
|---|---:|---:|---:|---:|---:|
'''+ '\n'.join(qualityrows)+f'''

壁面拆成三角面后，三角面中心与薄层单元中心之间的切向位移会使该角度接近 90°；**没有将这一项隐藏或判定为黏性计算合格**。下一步需要结合实际法向扩散/压力边界离散测试，而不是仅用一个几何最大角度下结论。曲面弦差、单元长宽比、边界层到外部尺寸跳变、尾流与计算域均需进一步验证。多面体节点在母线上不代表面内弦差为零；面中心偏差已单独记录。当前艏艉面内几何误差仍大于首层高度，后续要增加按曲率/弦差自适应的母线采样；这两档仅作为混合拓扑开发网格，不能直接认定为最终黏性 benchmark 网格。

## 修复与保留的诊断

首版把母线分段点直接加入余弦采样，尾部分段点与已有采样点过近，造成极小外部四面体。现改为用分段点替换最近的内部采样点。coarse 最小体积从 {old['mesh_audit']['minimum_cell_volume']:.5g} 提升到 {cases['coarse']['mesh_audit']['minimum_cell_volume']:.5g} L³。首版原始网格、几何数组、图和源码仍保留在 [diagnostic-near-coincident](diagnostic-near-coincident/report.md)，不作为当前推荐网格。

## 使用与复现

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src /data/TensorFEM/.venv/bin/python scripts/prepare_suboff_hybrid_mesh.py --output NEW_EMPTY_COARSE --axial 32 --azimuthal 24 --layers 4
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src /data/TensorFEM/.venv/bin/python scripts/prepare_suboff_hybrid_mesh.py --output NEW_EMPTY_MEDIUM --axial 48 --azimuthal 32 --layers 6 --first-height .0003 --growth 1.2 --far-size .3
```

[coarse 分档报告](coarse/report.md)、[medium 分档报告](medium/report.md)包含 Gmsh 文件、原始几何数据、源码快照和 SHA256 清单；各档 `mesh.pdf` 是可导出的网格图。输出目录必须为空，防止覆盖旧证据。

## 当前资格与下一步

**仅完成几何与算子验证，SUBOFF 黏性流动 benchmark 仍未达标。** 尚无附体、真实黏性压力/速度场、Cp/阻力、y+、湍流壁面函数验证或 GPU 加速结论。当前初始化 CLI 仍是原四面体专用工具，不能直接套用此混合网格运行命令。它的速度势也不能当作物理压力。

接下来先补充曲率自适应艏艉采样与边界层/外部平滑过渡，并将通用保守黏性动量方程与匹配的压力/单元面速度耦合接入混合网格，先做制造解与低 Reynolds 数 SUBOFF，再按 y+ 调整边界层、优化外部过渡及尾流并开展网格与计算域独立性研究。原有未通过的圆柱/SUBOFF 流动精度状态不变。
''')
    (root/'manifest.json').write_text(json.dumps({str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in root.rglob('*') if f.is_file() and f!=root/'manifest.json'},indent=2)+'\n')
    print(json.dumps({'tests_passed':65,'quality':quality,'geometry_qualified':all(s['geometry_passed'] for s in cases.values()),'flow_qualified':False},indent=2))

if __name__=='__main__':main()

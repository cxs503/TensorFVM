#!/usr/bin/env python3
"""Compare unchanged SUBOFF wall facets and actual nearest-wall distances."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
from tensorfvm.native_mesh3d import NativeMesh3D
from tensorfvm.native_rans3d import true_triangle_wall_distance
from run_native_suboff_benchmark import digest, write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--coarse',type=Path,required=True)
    p.add_argument('--fine',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output
    if out.exists() and any(out.iterdir()):p.error('output must be empty')
    out.mkdir(parents=True,exist_ok=True)
    surfaces=[];records={}
    for name,path in [('coarse',a.coarse),('fine',a.fine)]:
        m=NativeMesh3D.from_gmsh(path,triangulate_faces=True)
        wall=np.flatnonzero(m.patch_labels=='wall');owners=np.unique(m.owner[wall])
        facets=[tuple(sorted(tuple(point) for point in np.round(m.vertices[list(m.face_vertices[f])],13))) for f in wall]
        surfaces.append(sorted(facets))
        distance=true_triangle_wall_distance(m,points=m.cell_centers[owners])
        records[name]=dict(mesh=str(path),mesh_sha256=digest(path),cells=len(m.cells),wall_faces=len(wall),
            nearest_wall_distance_L=dict(minimum=float(distance.min()),median=float(np.median(distance)),maximum=float(distance.max())),
            cell_types={str(k):int(v) for k,v in zip(*np.unique(m.cell_types,return_counts=True))},
            estimated_maximum_yplus_at_assumed_Cf_0003={str(re):float(re*np.sqrt(.003/2)*distance.max()) for re in [12e6,13218056.809580836]},
            actual_wall_shear_available=False,actual_yplus_qualified=False)
    summary=dict(same_wall_facets_to_13_decimal_places=surfaces[0]==surfaces[1],
        cases=records,physical_validation_passed=False,
        note='Layer thickness and exterior cells also change; this is a wall-layer sensitivity pair, not three-grid spatial convergence.')
    write(out/'comparison.json',summary)
    rows='\n'.join(f"| {name} | {r['cells']:,} | {r['nearest_wall_distance_L']['median']:.8g} | {r['nearest_wall_distance_L']['maximum']:.8g} | {r['estimated_maximum_yplus_at_assumed_Cf_0003']['12000000.0']:.6g} | {r['estimated_maximum_yplus_at_assumed_Cf_0003']['13218056.809580836']:.6g} |" for name,r in records.items())
    (out/'report.md').write_text(f'''# 相同壁面离散下的近壁层对照

壁面实际三角形顶点逐面比较（保留 13 位小数）：**{summary['same_wall_facets_to_13_decimal_places']}**。
距离为壁邻单元中心到实际壁面三角形的最近欧氏距离，使用与 SST 相同的共性模块。

| 网格 | 单元数 | 中位壁距 / L | 最大壁距 / L | Cp 工况估计最大 y+ | 阻力工况估计最大 y+ |
|---|---:|---:|---:|---:|---:|
{rows}

估计式 y+=Re sqrt(Cf/2) d/L，**假定 Cf=0.003**，并非计算壁面应力得到的实际 y+。
此对照保持表面几何离散，改变层数、首层高度和总层厚度；外部四面体也重新生成。因此它用于近壁层敏感性诊断，不构成三档网格收敛。
几何通过不等于 SST 收敛或 Cp/Cd 达标。原始网格哈希与完整数值保存在 comparison.json。
''')
    shutil.copy2(__file__,out/'comparison-source.py')
    write(out/'manifest.json',{path.name:digest(path) for path in out.iterdir() if path.is_file()})
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()

#!/usr/bin/env python3
"""Plot actual polygonal sections of saved mixed-cell pressure diagnostics."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import tensorfvm.native_sections3d as section_module


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fields',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--z',type=float,default=.47)
    args=parser.parse_args();out=args.output
    if out.exists() and any(out.iterdir()):parser.error('plot output must be empty')
    out.mkdir(parents=True,exist_ok=True)
    f=np.load(args.fields,allow_pickle=False)
    offset=f['cell_offsets'];cells=[f['cell_vertices'][a:b] for a,b in zip(offset[:-1],offset[1:])]
    mesh=SimpleNamespace(cells=cells,vertices=f['vertices'],face_vertices=f['face_vertices'],owner=f['owner'],neighbor=f['neighbor'])
    polygons,ids=section_module.cell_section_polygons(mesh,value=args.z)
    if not len(ids):raise ValueError('empty section')
    p=f['pressure']-np.average(f['pressure'],weights=f['cell_volumes'])
    exact=f['exact_pressure']-np.average(f['exact_pressure'],weights=f['cell_volumes'])
    speed=np.linalg.norm(f['velocity'],axis=1)
    bounds=np.concatenate(polygons);low=min(p[ids].min(),exact[ids].min());high=max(p[ids].max(),exact[ids].max())
    fig,axes=plt.subplots(1,4,figsize=(15,3.5),constrained_layout=True)
    for ax in axes:
        ax.set(xlim=(bounds[:,0].min(),bounds[:,0].max()),ylim=(bounds[:,1].min(),bounds[:,1].max()),aspect='equal',xlabel='x',ylabel='y')
    axes[0].add_collection(PolyCollection(polygons,facecolors='none',edgecolors='.3',linewidths=.5))
    axes[0].set_title(f'Actual section z={args.z:g}')
    for ax,data,title in [(axes[1],exact,'Analytic cell pressure'),(axes[2],p,'Computed cell pressure'),(axes[3],speed,'Computed speed')]:
        collection=PolyCollection(polygons,array=data[ids],edgecolors='none',cmap='viridis')
        if ax is not axes[3]:collection.set_clim(low,high)
        ax.add_collection(collection);ax.set_title(title);fig.colorbar(collection,ax=ax)
    fig.savefig(out/'mesh-pressure-speed.png',dpi=180);fig.savefig(out/'mesh-pressure-speed.pdf');plt.close(fig)
    sources=[Path(__file__).resolve(),Path(section_module.__file__).resolve()]
    metadata=dict(plane_z=args.z,cell_ids=ids.tolist(),cells_intersected=len(ids),
        fields_sha256=hashlib.sha256(args.fields.read_bytes()).hexdigest(),
        pressure_gauge='both fields independently volume-mean centered',
        sampling='actual facet intersections, piecewise constant saved cell fields; no spatial interpolation',
        plot_sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    (out/'manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
    for p in sources:(out/p.name).write_bytes(p.read_bytes())


if __name__=='__main__':main()

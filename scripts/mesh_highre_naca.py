"""Systematic NACA grid family, including trailing-edge facet refinement."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--level',type=int,choices=[-1,0,1,2],default=0);p.add_argument('--alpha',type=float,default=0.)
    a=p.parse_args();out=a.output
    if out.exists() and any(out.iterdir()):p.error('Empty output required')
    out.mkdir(parents=True,exist_ok=True)
    root=Path(__file__).resolve().parents[1];source=root/'work/naca0012/make_naca_mesh.py'
    spec=importlib.util.spec_from_file_location('naca_mesh_writer',source)
    writer=importlib.util.module_from_spec(spec);spec.loader.exec_module(writer)
    scale=1.5**(-a.level)
    options=dict(first_layer=2e-6*scale,layer_thickness=.04,ratio=1+.12*scale,
        near=.004*scale,wake_size=.03*scale,wake_distance=.5,
        far_size=25*scale,radius=500.,alpha_split=a.alpha,grading_distance=400.)
    points=writer.resample(writer.dense_profile(),1.5*scale,options['near'],te_size=.001*scale)
    writer.write_geo(out/'mesh.geo',points,**options)
    import gmsh
    gmsh.initialize();gmsh.option.setNumber('General.Terminal',0)
    try:
        gmsh.open(str(out/'mesh.geo'));gmsh.model.mesh.generate(2)
        gmsh.option.setNumber('Mesh.MshFileVersion',2.2);gmsh.write(str(out/'mesh.msh'))
        types,tags,_=gmsh.model.mesh.getElements(2);counts={int(t):len(v) for t,v in zip(types,tags)}
    finally:gmsh.finalize()
    sources={}
    for path in [Path(__file__).resolve(),source]:
        relative=str(path.relative_to(root));target=out/'source-snapshot'/relative
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
        sources[relative]=hashlib.sha256(path.read_bytes()).hexdigest()
    metadata=dict(level=a.level,refinement_scale=scale,options=options,
        trailing_edge_facet_cap=.001*scale,max_turn_deg=1.5*scale,
        wall_points=len(points),cells=sum(counts.values()),element_counts=counts,gmsh_version=gmsh.__version__,
        source_sha256=sources,mesh_sha256=hashlib.sha256((out/'mesh.msh').read_bytes()).hexdigest())
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n');print(json.dumps(metadata),flush=True)


if __name__=='__main__':main()

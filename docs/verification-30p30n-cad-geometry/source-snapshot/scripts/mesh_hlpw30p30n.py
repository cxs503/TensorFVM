"""Generate reproducible original-CAD or legacy polygon 30P30N hybrid grids."""
import argparse,json,shutil
from pathlib import Path
from types import SimpleNamespace
import torch
from tensorfvm.hlpw30p30n import (DATA,metadata,write_mesh_geo,write_cad_mesh_geo,
    write_cad_triangle_mesh_geo,mesh_quality,CAD_FILE,CAD_SCALE_MM_PER_CHORD,CAD_CURVE_TAGS)
from tensorfvm.multi_element import ThreeElementMesh
from tensorfvm.verification.core import sha,write_json,artifact_manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--alpha',type=float,default=8.1)
    p.add_argument('--level',type=int,choices=[0,1,2],default=0)
    p.add_argument('--layer-thickness',type=float,default=.0005)
    p.add_argument('--geometry',choices=['cad','polygon'],default='polygon')
    p.add_argument('--cad-mesh-mode',choices=['hybrid','triangles'],default='hybrid')
    p.add_argument('--curvature-points',type=int,default=48)
    a=p.parse_args();out=a.output
    if not 0<a.layer_thickness<.006:p.error('Layer thickness must be positive and below half the narrowest element gap')
    if out.exists() and any(out.iterdir()):p.error('Mesh output must be empty')
    out.mkdir(parents=True,exist_ok=True);ratio=2**(.5*a.level)
    options=dict(near_size=.003/ratio,far_size=25/ratio,first_layer=2e-6/ratio,
        layer_thickness=a.layer_thickness,layer_ratio=1.18,grading_distance=100,
        alpha=a.alpha,fan_elements=12)
    import gmsh
    failure=None;mesh=None;fan_points=[];body_curve_types={};meshing_curve_tags={}
    try:
        if a.geometry=='cad':
            options['curvature_points']=a.curvature_points
            options['terminate_cove']=True
            options['intersect_metrics']=True
            if a.cad_mesh_mode=='triangles':
                for key in ['first_layer','layer_thickness','layer_ratio','fan_elements','terminate_cove','intersect_metrics']:
                    options.pop(key,None)
            writer=write_cad_triangle_mesh_geo if a.cad_mesh_mode=='triangles' else write_cad_mesh_geo
            geo=writer(out/'mesh.geo',**options)
        else:geo=write_mesh_geo(out/'mesh.geo',**options)
        gmsh.initialize()
        try:
            gmsh.option.setNumber('General.Verbosity',2)
            gmsh.option.setNumber('General.NumThreads',1)
            gmsh.option.setNumber('Mesh.RandomSeed',1)
            gmsh.open(str(geo.resolve()))
            if a.geometry!='cad' or a.cad_mesh_mode=='hybrid':
                fan_points=list(map(int,gmsh.model.mesh.field.getNumbers(3,'FanPointsList')))
            if a.geometry=='cad':
                for dim,physical in gmsh.model.getPhysicalGroups(1):
                    name=gmsh.model.getPhysicalName(dim,physical)
                    if name not in CAD_CURVE_TAGS:continue
                    tags=list(map(int,gmsh.model.getEntitiesForPhysicalGroup(dim,physical)))
                    meshing_curve_tags[name]=tags
                    body_curve_types[name]=[gmsh.model.getType(1,t) for t in tags]
                    if any(t not in ('BSpline','Line','TrimmedCurve') for t in body_curve_types[name]):
                        raise ValueError('Original STEP curve topology changed: '+name)
            gmsh.logger.start()
            try:gmsh.model.mesh.generate(2)
            except Exception:
                import numpy as np
                tags,coordinates,_=gmsh.model.mesh.getNodes()
                np.savez_compressed(out/'failed-mesh-nodes.npz',tags=tags,coordinates=coordinates)
                try:gmsh.write(str((out/'mesh-partial.msh').resolve()))
                except Exception:pass
                raise
            finally:
                (out/'gmsh-log.txt').write_text('\n'.join(gmsh.logger.get())+'\n')
                gmsh.logger.stop()
            gmsh.option.setNumber('Mesh.MshFileVersion',2.2)
            gmsh.option.setNumber('Mesh.Binary',0)
            gmsh.write(str((out/'mesh.msh').resolve()))
        finally:gmsh.finalize()
        torch.set_num_threads(1)
        mesh=ThreeElementMesh(SimpleNamespace(mesh_file=out/'mesh.msh',device='cpu'))
        quality=mesh_quality(mesh,a.alpha)
        quality.update(minimum_cell_area_chords2=float(mesh.volumes.min()),
            triangle_cells=sum(len(v)==3 for v in mesh.cell_nodes),
            quadrilateral_cells=sum(len(v)==4 for v in mesh.cell_nodes))
        if not quality['accepted']:failure='Independent mesh quality rejected'
        root=Path(__file__).resolve().parents[1]
        cad_info={} if a.geometry!='cad' else dict(cad_sha256=sha(CAD_FILE),
            cad_source_relative_path=str(CAD_FILE.relative_to(root)),
            cad_scale_mm_per_chord=CAD_SCALE_MM_PER_CHORD,
            cad_normalization_scale=1/CAD_SCALE_MM_PER_CHORD,
            cad_curve_tags=CAD_CURVE_TAGS,cad_curve_types=body_curve_types,
            cad_meshing_curve_tags=meshing_curve_tags,
            cad_mesh_construction=json.loads((out/'cad-mesh-construction.json').read_text()))
        write_json(out/'mesh-metadata.json',dict(level=a.level,options=options,
            nodes=len(mesh.vertices),cells=len(mesh.cell_nodes),
            wall_faces={n:int(mesh.masks[n].sum()) for n in ['slat','main','flap']},
            quality=quality,geometry=metadata(),gmsh_version=gmsh.__version__,
            geometry_representation='official-step-bspline' if a.geometry=='cad' else 'official-profile-polyline',
            mesh_mode=('cad_triangle' if a.cad_mesh_mode=='triangles' else 'cad_hybrid_research') if a.geometry=='cad' else 'polygon_hybrid',
            boundary_layers=not(a.geometry=='cad' and a.cad_mesh_mode=='triangles'),accuracy_qualified=False,
            actual_fan_point_tags=fan_points,y_plus_measured=False,**cad_info))
    except Exception as error:
        failure=str(error)
    if failure is not None:
        write_json(out/'mesh-failure.json',dict(level=a.level,options=options,
            failure=failure,accepted=False,geometry=metadata(),
            geometry_representation='official-step-bspline' if a.geometry=='cad' else 'official-profile-polyline',
            gmsh_version=gmsh.__version__))
    root=Path(__file__).resolve().parents[1]
    sources=[Path(__file__),root/'src/tensorfvm/hlpw30p30n.py',
        root/'src/tensorfvm/benchmark_30p30n.py',root/'src/tensorfvm/multi_element.py',
        *[p for p in DATA.rglob('*') if p.is_file()]]
    hashes={}
    for source in sources:
        name=str(source.relative_to(root));hashes[name]=sha(source)
        target=out/'source-snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target)
    write_json(out/'manifest.json',dict(source_snapshot_root='source-snapshot',
        source_sha256=hashes,artifacts_sha256=artifact_manifest(out)))
    if failure is not None:raise SystemExit('Mesh failed; native geometry and failure record retained: '+failure)
    print(json.dumps(dict(cells=len(mesh.cell_nodes),nodes=len(mesh.vertices),level=a.level,
        geometry_representation='official-step-bspline' if a.geometry=='cad' else 'official-profile-polyline')))

if __name__=='__main__':main()

"""Common Gmsh cylinder/NACA refinement runner with strict physical gates."""
import argparse
from dataclasses import replace
from pathlib import Path
import importlib.util
import torch
from .external_flow import configuration, run, audit
from .core import provenance, sha, save_run, write_json, artifact_manifest
from ..gmsh_external import GmshExternalMesh, GmshCoupledSolver, GmshExternalSolver
from ..external_high_order import LinearUpwindCoupledSolver, LinearUpwindExternalSolver, ConservativeLinearUpwindCoupledSolver, ConservativeLinearUpwindSolver
from ..mesh_quality import mesh_quality


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',choices=['cylinder','naca'],required=True)
    p.add_argument('--scales',type=float,nargs='+',default=[2,1,.5])
    p.add_argument('--solver',choices=['upwind','linear-upwind','conservative-linear-upwind'],default='conservative-linear-upwind')
    p.add_argument('--max-iterations',type=int,default=1500)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(argv)
    root=Path(__file__).resolve().parents[3]
    output=a.output.resolve()
    if output.exists() and any(output.iterdir()):
        p.error('Output must be empty; existing evidence cannot be overwritten')
    if len(set(a.scales))!=len(a.scales) or any(v<=0 for v in a.scales) or a.max_iterations<1:
        p.error('Unique positive scales and a positive iteration limit required')
    output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    generator=root/'scripts/generate_external_gmsh.py'
    spec=importlib.util.spec_from_file_location('external_mesh_generator',generator)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    source=provenance()
    for name in ['sparse_simple.py','external_linear.py','external_cached.py','external_anderson.py','external_coupled.py','coupled_coarse.py','benchmark_airfoil.py','multi_element.py','gmsh_external.py','external_high_order.py','conservative_pressure.py','mesh_quality.py']:
        source['source_sha256']['src/tensorfvm/'+name]=sha(root/'src/tensorfvm'/name)
    source['source_sha256']['scripts/generate_external_gmsh.py']=sha(generator)
    summary=dict(provenance=source,case=a.case,runs=[],passed=False,grid_convergence_qualified=False)
    classes={'cylinder':dict(upwind=GmshCoupledSolver,**{'linear-upwind':LinearUpwindCoupledSolver,'conservative-linear-upwind':ConservativeLinearUpwindCoupledSolver}),'naca':dict(upwind=GmshExternalSolver,**{'linear-upwind':LinearUpwindExternalSolver,'conservative-linear-upwind':ConservativeLinearUpwindSolver})}
    for scale in sorted(a.scales,reverse=True):
        label=f'scale-{scale:g}'
        mesh=output/'meshes'/f'{label}.msh'
        metadata=module.generate(a.case,mesh,scale)
        c=replace(configuration(a.case,64,24,'cpu',a.max_iterations),nx=1,ny=1,mesh_type='gmsh-cylinder' if a.case=='cylinder' else 'gmsh-airfoil',mesh_file=str(mesh))
        quality=mesh_quality(GmshExternalMesh(c))
        print(label,quality,flush=True)
        if not quality['quality_gate_passed']:
            raise ValueError('Repair mesh generation before solving: quality gate failed')
        raw=run(a.case,1,1,'cpu',a.max_iterations,a.solver,config=c,solver_factory=classes[a.case][a.solver])
        raw.update(resolution=label,directory=label,mesh_scale=scale,mesh_quality=quality,mesh_generation=metadata)
        raw['config']['mesh_file']=str(mesh.relative_to(root)) if mesh.is_relative_to(root) else str(mesh)
        row=save_run(output/label,raw)
        write_json(output/label/'audit.json',audit(output/label,row))
        summary['runs'].append(row)
        summary['passed']=len(summary['runs'])>=3 and all(r['passed'] for r in summary['runs'])
        summary['finest_physical_passed']=row['passed']
        write_json(output/'summary.json',summary)
        write_json(output/'manifest.json',dict(source_sha256=source['source_sha256'],artifacts_sha256=artifact_manifest(output)))
        print(label,row['computed'],'converged',row['converged'],'physical_passed',row['passed'],flush=True)
    return 0 if summary['passed'] else 2


if __name__=='__main__':
    raise SystemExit(main())

"""Hybrid quad wall layers and triangular outer mesh on the matched NACA domain."""
import argparse
from dataclasses import replace
from pathlib import Path
import importlib.util
import torch
from tensorfvm.verification.core import provenance,sha,write_json,save_run,artifact_manifest
from tensorfvm.verification.external_flow import configuration,run,audit
from tensorfvm.external_open_boundary import OpenBoundaryNacaSolver
from tensorfvm.gmsh_external import GmshExternalMesh
from tensorfvm.hybrid_mesh_quality import hybrid_mesh_quality as mesh_quality


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scales',type=float,nargs='+',default=[2,1])
    p.add_argument('--max-iterations',type=int,default=3500)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(argv);root=Path(__file__).resolve().parents[1];output=args.output.resolve()
    if output.exists() and any(output.iterdir()):p.error('Output must be empty')
    if len(set(args.scales))!=len(args.scales) or any(s<=0 for s in args.scales) or args.max_iterations<1:p.error('Unique positive scales and positive iterations required')
    output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
    generator=root/'scripts/generate_external_gmsh.py';spec=importlib.util.spec_from_file_location('generator',generator)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    prov=provenance()
    for name in ['sparse_simple.py','external_linear.py','external_cached.py','external_anderson.py','external_coupled.py','coupled_coarse.py','wall_traction.py','benchmark_airfoil.py','multi_element.py','gmsh_external.py','external_high_order.py','conservative_pressure.py','external_open_boundary.py','mesh_quality.py','hybrid_mesh_quality.py']:
        prov['source_sha256']['src/tensorfvm/'+name]=sha(root/'src/tensorfvm'/name)
    prov['source_sha256']['scripts/generate_external_gmsh.py']=sha(generator)
    prov['source_sha256']['scripts/run_hybrid_naca.py']=sha(Path(__file__))
    prov['source_sha256']['docs/naca-reference-domain-review.json']=sha(root/'docs/naca-reference-domain-review.json')
    summary=dict(provenance=prov,case='naca',runs=[],passed=False,grid_convergence_qualified=False)
    write_json(output/'manifest.json',dict(source_sha256=prov['source_sha256'],artifacts_sha256=artifact_manifest(output)))
    for scale in sorted(args.scales,reverse=True):
        label=f'scale-{scale:g}';mesh=output/'meshes'/f'{label}.msh'
        metadata=module.generate('naca',mesh,scale,length=36,height=16,leading_x=11.75,leading_y=8,geometric_incidence_deg=4,open_boundaries=True,hybrid=True)
        c=replace(configuration('naca',64,26,'cpu',args.max_iterations),nx=1,ny=1,mesh_type='gmsh-airfoil-open',mesh_file=str(mesh),length=36,height=16,airfoil_x=11.75,airfoil_y=8,angle_of_attack=0)
        quality=mesh_quality(GmshExternalMesh(c));print(label,quality,flush=True)
        if not quality['hybrid_gate_passed']:raise ValueError('Repair mesh before calculation')
        raw=run('naca',1,1,'cpu',args.max_iterations,solver='conservative-linear-upwind-open',config=c,solver_factory=OpenBoundaryNacaSolver)
        raw.update(resolution=label,directory=label,mesh_scale=scale,mesh_generation=metadata,mesh_quality=quality,physical_angle_of_attack_deg=4,reference_domain_matched=True,reference_open_boundary_formula_unspecified=True)
        raw['discretization']['boundary']='pressure-open-freestream-backflow'
        raw['config']['mesh_file']=str(mesh.relative_to(root)) if mesh.is_relative_to(root) else str(mesh)
        row=save_run(output/label,raw);write_json(output/label/'audit.json',audit(output/label,row))
        summary['runs'].append(row);summary['passed']=len(summary['runs'])>=3 and all(r['passed'] for r in summary['runs']);summary['finest_physical_passed']=row['passed']
        write_json(output/'summary.json',summary);write_json(output/'manifest.json',dict(source_sha256=prov['source_sha256'],artifacts_sha256=artifact_manifest(output)))
        print(label,row['computed'],'steady',row['converged'],'physical_passed',row['passed'],flush=True)
    return 0 if summary['passed'] else 2


if __name__=='__main__':raise SystemExit(main())

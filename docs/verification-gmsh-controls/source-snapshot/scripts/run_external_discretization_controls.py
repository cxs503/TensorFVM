"""Isolate transport and pressure changes on one identical Gmsh cylinder mesh."""
from dataclasses import replace
from pathlib import Path
import json
import torch
from tensorfvm.verification.external_flow import configuration,run,audit
from tensorfvm.verification.core import provenance,sha,save_run,write_json,artifact_manifest
from tensorfvm.gmsh_external import GmshExternalMesh,GmshCoupledSolver
from tensorfvm.external_high_order import LinearUpwindCoupledSolver,ConservativeLinearUpwindCoupledSolver
from tensorfvm.mesh_quality import mesh_quality

ROOT=Path(__file__).resolve().parents[1]
output=ROOT/'docs/verification-gmsh-controls'
if output.exists() and any(output.iterdir()):raise ValueError('Existing controls cannot be overwritten')
output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
mesh=ROOT/'docs/verification-gmsh-cylinder/meshes/scale-1.msh'
c=replace(configuration('cylinder',64,24,'cpu',300),nx=1,ny=1,mesh_type='gmsh-cylinder',mesh_file=str(mesh))
quality=mesh_quality(GmshExternalMesh(c));prov=provenance()
for name in ['sparse_simple.py','external_linear.py','external_cached.py','external_anderson.py','external_coupled.py','coupled_coarse.py','benchmark_airfoil.py','multi_element.py','gmsh_external.py','external_high_order.py','conservative_pressure.py','mesh_quality.py']:
 prov['source_sha256']['src/tensorfvm/'+name]=sha(ROOT/'src/tensorfvm'/name)
prov['source_sha256']['scripts/run_external_discretization_controls.py']=sha(Path(__file__))
prov['source_sha256'][str(mesh.relative_to(ROOT))]=sha(mesh)
summary=dict(provenance=prov,case='cylinder',study_type='discretization-control',runs=[],passed=False,grid_convergence_qualified=False)
for label,factory in [('upwind',GmshCoupledSolver),('linear-upwind',LinearUpwindCoupledSolver),('conservative-linear-upwind',ConservativeLinearUpwindCoupledSolver)]:
 raw=run('cylinder',1,1,'cpu',300,label,config=c,solver_factory=factory)
 raw.update(resolution=label,directory=label,mesh_scale=1.,mesh_quality=quality,mesh_generation=json.loads(mesh.with_suffix('.json').read_text()))
 raw['config']['mesh_file']=str(mesh.relative_to(ROOT))
 row=save_run(output/label,raw);write_json(output/label/'audit.json',audit(output/label,row));summary['runs'].append(row)
 write_json(output/'summary.json',summary);write_json(output/'manifest.json',dict(source_sha256=prov['source_sha256'],artifacts_sha256=artifact_manifest(output)))
 print(label,row['computed'],row['converged'],row['passed'],flush=True)

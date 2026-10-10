"""Continue saved near-steady NACA fields; preserve the entire initial study."""
import copy
import argparse
from dataclasses import replace
import json
from pathlib import Path
import shutil
import torch
import numpy as np
from tensorfvm.verification.external_flow import run,audit
from tensorfvm.verification.core import provenance,sha,write_json,save_run,artifact_manifest
from tensorfvm.external_open_boundary import OpenBoundaryNacaSolver
from tensorfvm.solver import SolverConfig

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,default=ROOT/'docs/verification-gmsh-naca-matched')
parser.add_argument('--initial',type=Path)
parser.add_argument('--extra-iterations',type=int,default=1500)
args=parser.parse_args()
if args.extra_iterations<1:parser.error('Positive extra iteration count required')
output=args.output.resolve();initial=args.initial.resolve() if args.initial else output.with_name(output.name+'-initial')
def path_name(path):return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
# Preserve current exact sources automatically for a fresh upstream run.
manifest=json.loads((output/'manifest.json').read_text())
if not manifest.get('source_snapshot_root'):
 for key,value in manifest['source_sha256'].items():
  source=ROOT/key
  if sha(source)!=value:raise ValueError('Upstream source changed; recover its exact producer snapshot first')
  target=output/'source-snapshot'/key;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
 manifest['source_snapshot_root']='source-snapshot';manifest['artifacts_sha256']=artifact_manifest(output);write_json(output/'manifest.json',manifest)
if initial.exists():raise ValueError('Initial continuation evidence already exists')
output.rename(initial);shutil.copytree(initial,output)
summary=json.loads((output/'summary.json').read_text());oldprov=copy.deepcopy(summary['provenance'])
(output/manifest['source_snapshot_root']).rename(output/'source-snapshot-initial')
for row in summary['runs']:
 row['producer_provenance']=copy.deepcopy(oldprov);row['source_snapshot_root']='source-snapshot-initial'
old=summary['runs'][-1];directory=initial/old['directory'];c=SolverConfig(**dict(old['config'],mesh_file=str(ROOT/old['config']['mesh_file']),max_iterations=old['iterations']+args.extra_iterations))
with np.load(directory/'fields.npz') as f:restart={name:f[name].copy() for name in ['cell_velocity_m_s','cell_pressure_pa','face_mass_flux']}
history=json.loads((directory/'history.json').read_text());torch.set_num_threads(1)
prov=provenance()
for name in ['sparse_simple.py','external_linear.py','external_cached.py','external_anderson.py','external_coupled.py','coupled_coarse.py','benchmark_airfoil.py','multi_element.py','gmsh_external.py','external_high_order.py','conservative_pressure.py','external_open_boundary.py','wall_traction.py','mesh_quality.py']:
 prov['source_sha256']['src/tensorfvm/'+name]=sha(ROOT/'src/tensorfvm'/name)
prov['source_sha256']['scripts/continue_matched_naca.py']=sha(Path(__file__))
prov['source_sha256']['docs/naca-reference-domain-review.json']=sha(ROOT/'docs/naca-reference-domain-review.json')

def restore(config):
 s=OpenBoundaryNacaSolver(config)
 s.velocity.copy_(torch.from_numpy(restart['cell_velocity_m_s']));s.p.copy_(torch.from_numpy(restart['cell_pressure_pa']));s.mass_flux.copy_(torch.from_numpy(restart['face_mass_flux']))
 s.history=copy.deepcopy(history)
 return s

raw=run('naca',1,1,'cpu',args.extra_iterations,solver='conservative-linear-upwind-open',config=c,solver_factory=restore)
raw.update(resolution=old['resolution'],directory=old['directory'],mesh_scale=old['mesh_scale'],mesh_generation=old['mesh_generation'],mesh_quality=old['mesh_quality'],physical_angle_of_attack_deg=4,reference_domain_matched=True,reference_open_boundary_formula_unspecified=True,producer_provenance=prov,source_snapshot_root='source-snapshot-final',continuation=dict(initial_iterations=old['iterations'],additional_iterations=raw['iterations']-old['iterations'],initial_fields_sha256=sha(directory/'fields.npz'),initial_fields_path=path_name(directory/'fields.npz'),initial_elapsed_s=old['elapsed_s'],additional_elapsed_s=raw['elapsed_s'],initial_numerical_producer_provenance=oldprov,initial_source_snapshot_root='source-snapshot-initial'))
raw['elapsed_s']+=old['elapsed_s'];raw['config']['mesh_file']=old['config']['mesh_file'];raw['discretization']['boundary']='pressure-open-freestream-backflow'
row=save_run(output/old['directory'],raw);write_json(output/old['directory']/'audit.json',audit(output/old['directory'],row))
summary['runs'][-1]=row;summary['provenance']=prov;summary['passed']=all(r['passed'] for r in summary['runs']);summary['finest_physical_passed']=row['passed']
write_json(output/'summary.json',summary)
for key,value in prov['source_sha256'].items():
 source=ROOT/key
 if sha(source)!=value:raise ValueError('Continuation producer changed')
 target=output/'source-snapshot-final'/key;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
write_json(output/'manifest.json',dict(source_sha256=prov['source_sha256'],source_snapshot_root='source-snapshot-final',input_sha256={path_name(directory/'fields.npz'):sha(directory/'fields.npz')},artifacts_sha256=artifact_manifest(output)))
print('Continued actual NACA',row['computed'],row['iterations'],'converged',row['converged'],'accepted',row['passed'],flush=True)

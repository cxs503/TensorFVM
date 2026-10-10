"""Compare an incompressible no-slip wall gradient on immutable solved fields.

This is postprocessing, not a new flow solve. Original cell fields, face fluxes,
histories and exact numerical producer versions remain separately archived.
"""
import copy
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
import torch
from tensorfvm.wall_traction import incompressible_wall_gradient
from tensorfvm.verification.core import provenance,sha,save_run,write_json,artifact_manifest,Metric,evaluate
from tensorfvm.verification.external_flow import audit

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--input',type=Path,default=ROOT/'docs/verification-gmsh-cylinder')
parser.add_argument('--output',type=Path,default=ROOT/'docs/verification-gmsh-cylinder-wall')
args=parser.parse_args();source=args.input.resolve();output=args.output.resolve()
def path_name(path):return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
if output.exists() and any(output.iterdir()):raise ValueError('Existing wall comparison cannot be overwritten')
output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
base=json.loads((source/'summary.json').read_text())
if any(row['case']!='cylinder' for row in base['runs']):raise ValueError('This published wall comparison supports the verified DFG cylinder study')
manifest=json.loads((source/'manifest.json').read_text())
producer=base['provenance'];snapshot=output/'numerical-source-snapshot'
for key,value in producer['source_sha256'].items():
 src=source/manifest['source_snapshot_root']/key if manifest.get('source_snapshot_root') else ROOT/key
 if sha(src)!=value:raise ValueError('Original numerical producer mismatch: '+key)
 dst=snapshot/key;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
prov=provenance();prov['source_sha256']['src/tensorfvm/wall_traction.py']=sha(ROOT/'src/tensorfvm/wall_traction.py');prov['source_sha256']['scripts/postprocess_incompressible_wall.py']=sha(Path(__file__))
summary=dict(provenance=prov,case='cylinder',study_type='wall-traction-postprocessing',runs=[],passed=False,grid_convergence_qualified=False,original_numerical_producer_provenance=producer,numerical_source_snapshot_root='numerical-source-snapshot',input_sha256={})
shutil.copytree(source/'meshes',output/'meshes')
for old in base['runs']:
 row=copy.deepcopy(old);directory=source/old['directory'];input_path=directory/'fields.npz'
 with np.load(input_path) as archive:fields={key:archive[key].copy() for key in archive.files}
 mask=fields['surface_mask'];S=fields['face_area_vectors_m'][mask];own=fields['face_owner'][mask];d=fields['surface_displacement_m'];u=fields['cell_velocity_m_s'][own]
 g=incompressible_wall_gradient(torch.from_numpy(u),torch.from_numpy(d),torch.from_numpy(S)).numpy()
 c=row['config'];length=2*c['cylinder_radius'];mu=c['density']*c['inlet_velocity']*length/c['reynolds']
 fields['surface_gradient']=g
 fields['surface_viscous_force']=-np.einsum('fij,fj->fi',mu*(g+g.transpose(0,2,1)),S)
 fields['surface_wall_velocity_m_s']=np.zeros_like(u)
 fields['surface_reconstruction_remainder_m_s']=u+np.einsum('fi,fij->fj',d,g)
 coeff=(fields['surface_pressure_force']+fields['surface_viscous_force']).sum(0)/(.5*c['density']*c['inlet_velocity']**2*length)
 row['computed'].update(drag=float(coeff[0]),lift=float(coeff[1]))
 row['discretization']['wall_gradient']='incompressible-normal'
 row.update(wall_postprocessed_without_resolving=True,upstream_fields_path=path_name(input_path),upstream_fields_sha256=sha(input_path))
 summary['input_sha256'][row['upstream_fields_path']]=row['upstream_fields_sha256']
 metrics=[Metric(name+'_relative_error',abs(row['computed'][name]-ref)/abs(ref),.03,'Strict physical relative error; same immutable solved field, incompressible wall traction') for name,ref in row['reference'].items()]
 last=json.loads((directory/'history.json').read_text())[-1]
 metrics += [Metric('solver_residual',max(last[k] for k in ('momentum','continuity','mass_imbalance')),c['tolerance'],'Unchanged actual solved equations'),Metric('convergence_failure',0 if row['converged'] else 1,.5,'Explicit steady convergence')]
 row['metrics'],row['passed']=evaluate(metrics)
 row['fields']=fields;row['history']=json.loads((directory/'history.json').read_text());save_run(output/row['directory'],row)
 write_json(output/row['directory']/'audit.json',audit(output/row['directory'],row));summary['runs'].append(row)
 print(row['resolution'],row['computed'],'accepted',row['passed'],flush=True)
summary['passed']=len(summary['runs'])>=3 and all(row['passed'] for row in summary['runs']);summary['finest_physical_passed']=summary['runs'][-1]['passed']
write_json(output/'summary.json',summary)
write_json(output/'manifest.json',dict(source_sha256=prov['source_sha256'],input_sha256=summary['input_sha256'],artifacts_sha256=artifact_manifest(output)))

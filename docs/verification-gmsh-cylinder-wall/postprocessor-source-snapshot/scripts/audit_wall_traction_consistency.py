"""Analytic wall-shear consistency, separate from solved benchmark fields."""
import json
from pathlib import Path
import shutil
import numpy as np
from tensorfvm.verification.core import sha,write_json,artifact_manifest

ROOT=Path(__file__).resolve().parents[1];output=ROOT/'docs/verification-gmsh-cylinder-wall'
summary=json.loads((output/'summary.json').read_text());records=[]
for row in summary['runs']:
 with np.load(output/row['directory']/'fields.npz') as f:
  mask=f['surface_mask'];S=f['face_area_vectors_m'][mask];normal=S/np.linalg.norm(S,axis=1)[:,None];own=f['face_owner'][mask];center=f['cell_centers_m'].reshape(-1,2)[own];d=f['surface_displacement_m'];h=(d*normal).sum(1)
  radial=center-[.2,.2];r=np.linalg.norm(radial,axis=1);er=radial/r[:,None]
  manufactured_velocity=(r-.05)[:,None]*np.c_[-er[:,1],er[:,0]]
  tangent=manufactured_velocity-(manufactured_velocity*normal).sum(1)[:,None]*normal
  shear=np.linalg.norm(tangent,axis=1)/h
  error=float(np.linalg.norm(shear-1)/np.sqrt(len(shear)))
  g=f['surface_gradient'];actual_u=f['cell_velocity_m_s'][own];ut=actual_u-(actual_u*normal).sum(1)[:,None]*normal
  expected=-normal[:,:,None]*ut[:,None,:]/h[:,None,None]
  np.testing.assert_allclose(g,expected,rtol=0,atol=1e-12)
  trace=float(np.max(abs(np.trace(g,axis1=1,axis2=2))))
  tangent_direction=np.c_[-normal[:,1],normal[:,0]]
  tangent_derivative=float(np.max(abs(np.einsum('fi,fij->fj',tangent_direction,g))))
  records.append(dict(resolution=row['resolution'],exact_manufactured_wall_shear=1.,manufactured_wall_shear_relative_l2_error=error,solved_wall_gradient_trace_max=trace,solved_wall_tangential_derivative_max=tangent_derivative))
errors=[r['manufactured_wall_shear_relative_l2_error'] for r in records]
orders=[float(np.log(a/b)/np.log(2)) for a,b in zip(errors,errors[1:])]
passed=all(a>b for a,b in zip(errors,errors[1:])) and errors[-1]<.03 and min(orders)>.8 and all(r['solved_wall_gradient_trace_max']<1e-10 and r['solved_wall_tangential_derivative_max']<1e-10 for r in records)
if not passed:raise ValueError('Wall consistency failed')
write_json(output/'wall-reconstruction-audit.json',dict(passed=passed,runs=records,observed_orders=orders,scope='Analytic divergence-free circular swirl u_theta=r-R, exact curved-wall shear 1, evaluated on saved geometry; analytic values never substitute for solved external-flow fields. Actual wall gradients separately satisfy incompressibility and zero tangential derivative identities.'))
m=json.loads((output/'manifest.json').read_text());key='scripts/audit_wall_traction_consistency.py';m['source_sha256'][key]=sha(Path(__file__))
target=output/m['source_snapshot_root']/key;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(Path(__file__),target)
m['artifacts_sha256']=artifact_manifest(output);write_json(output/'manifest.json',m)
print('Wall consistency audited:',errors,'orders',orders)

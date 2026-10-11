"""Experimental linear-only scalability probe; production solver unchanged."""
import json,time,sys,hashlib
from pathlib import Path
from dataclasses import replace
import numpy as np
out=Path(__file__).resolve().parent
run=out.parent/'quadratic-assembled-small'
sys.path.insert(0,str((run/'source-snapshot/src').resolve()))
from tensorfvm.native_mesh3d import NativeMesh3D
from tensorfvm.native_collocated3d import NativeSteadySettings,NativeSteadySolver3D
from tensorfvm.native_quadratic_flow3d import NativeQuadraticSteadySolver3D
from tensorfvm.native_quadratic_system3d import current_matrix
from tensorfvm.native_manufactured3d import abc_fields,exact_boundary_flux,exact_boundary_velocity_means
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
start=time.perf_counter();config=json.loads((run/'config.json').read_text())
bindings={}
for name,module in list(sys.modules.items()):
 if name=='tensorfvm' or name.startswith('tensorfvm.'):
  path=Path(module.__file__).resolve();relative=str(path.relative_to((run/'source-snapshot').resolve()))
  if config['source_sha256'].get(relative)!=sha(path):raise ValueError('unbound numerical source')
  bindings[relative]=sha(path)
(out/'config.json').write_text(json.dumps(dict(producer=str(run),source_sha256=bindings,input_sha256={name:sha(run/name) for name in ['config.json','input.msh','fields.npz']},method='experimental existing base Schur preconditioner on full quadratic matrix',timeout_seconds=240,production_quadratic_method_enabled=False,scope='one frozen linear system only; not a nonlinear flow solve'),indent=2)+'\n')
m=NativeMesh3D.from_gmsh(run/'input.msh',triangulate_faces=True);settings=NativeSteadySettings(**config['settings']);field=lambda x:abc_fields(x,nu=settings.viscosity);exact=field(m.cell_centers)
s=NativeQuadraticSteadySolver3D(m,m.neighbor<0,field(m.face_centers)['velocity'],settings=settings,body_force=exact['acceleration'],boundary_flux=exact_boundary_flux(m,field),gauge_value=exact['pressure'][0],boundary_velocity_means=exact_boundary_velocity_means(m,field),boundary_velocity_gradients=field(m.face_centers)['velocity_gradient'],quadratic_linear_solver='assembled')
with np.load(run/'fields.npz') as f:s.u=f['velocity'].copy();s.p=f['pressure'].copy();flux=f['flux'].copy()
old=np.r_[s.u.ravel(),s.p];_,_,a,_,_=s._system(flux);s._linear_flux=flux;s._frozen_covariance_divergence=s._covariance_divergence();matrix=current_matrix(s,flux,a)
s.u=np.zeros_like(s.u);s.p=np.zeros_like(s.p);_,base_rhs,_,_,_=NativeSteadySolver3D._system(s,flux);zero,_=s._corrections(flux,a);target=base_rhs+zero;del s._frozen_covariance_divergence
print(json.dumps(dict(stage='matrix_ready',nnz=matrix.nnz,elapsed_seconds=time.perf_counter()-start)),flush=True)
# Deliberately restricted to a read-only linear feasibility experiment. The
# production quadratic constructor and runner still reject this method.
s.settings=replace(settings,linear_method='schur');linear_start=time.perf_counter()
try:
 answer,true,iterations=NativeSteadySolver3D._linear(s,matrix,target,a)
 result=dict(linear_solve_passed=True,true_relative_residual=true,iterations=iterations,change_from_converged_state_max=float(abs(answer-old).max()))
 np.savez_compressed(out/'linear-answer.npz',answer=answer)
except Exception as error:
 result=dict(linear_solve_passed=False,failure=dict(type=type(error).__name__,message=str(error)))
result.update(linear_seconds=time.perf_counter()-linear_start,total_seconds=time.perf_counter()-start,flow_solved=False,production_enabled=False)
(out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

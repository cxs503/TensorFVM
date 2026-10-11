"""Compare SuperLU orderings on a frozen full quadratic system, without a flow solve."""
import json,time,sys,hashlib
from pathlib import Path
import numpy as np
import argparse,resource,signal
from scipy import sparse
from scipy.sparse.linalg import splu
p=argparse.ArgumentParser();p.add_argument('--ordering',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--run',type=Path,required=True);args=p.parse_args()
signal.alarm(600)
if args.ordering not in ('COLAMD','MMD_ATA','MMD_AT_PLUS_A','NATURAL'):p.error('Unsupported SuperLU ordering')
out=args.output;out.mkdir(parents=True,exist_ok=False);run=args.run.resolve()
(out/'executed-probe.py').write_bytes(Path(__file__).read_bytes())
sys.path.insert(0,str((run/'source-snapshot/src').resolve()))
from tensorfvm.native_mesh3d import NativeMesh3D
from tensorfvm.native_collocated3d import NativeSteadySettings,NativeSteadySolver3D
from tensorfvm.native_quadratic_flow3d import NativeQuadraticSteadySolver3D
from tensorfvm.native_quadratic_system3d import current_matrix
from tensorfvm.native_manufactured3d import abc_fields,exact_boundary_flux,exact_boundary_velocity_means
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
start=time.perf_counter();config=json.loads((run/'config.json').read_text())
if config.get('mode')!='manufactured' or config.get('reconstruction')!='quadratic' or config.get('merge_coplanar_faces',False):raise ValueError('Requires triangular-face quadratic manufactured producer')
if sha(run/'input.msh')!=config['mesh_sha256']:raise ValueError('Mesh binding mismatch')
bindings={}
for name,module in list(sys.modules.items()):
 if name=='tensorfvm' or name.startswith('tensorfvm.'):
  path=Path(module.__file__).resolve();relative=str(path.relative_to((run/'source-snapshot').resolve()))
  if config['source_sha256'].get(relative)!=sha(path):raise ValueError('unbound numerical source')
  bindings[relative]=sha(path)
(out/'config.json').write_text(json.dumps(dict(producer=str(run),source_sha256=bindings,input_sha256={name:sha(run/name) for name in ['config.json','input.msh','fields.npz']},method='SuperLU ordering comparison on identical full quadratic system',ordering=args.ordering,timeout_seconds=600,scope='one frozen linear system only; not a nonlinear flow solve'),indent=2)+'\n')
m=NativeMesh3D.from_gmsh(run/'input.msh',triangulate_faces=True);settings=NativeSteadySettings(**config['settings']);field=lambda x:abc_fields(x,nu=settings.viscosity);exact=field(m.cell_centers)
s=NativeQuadraticSteadySolver3D(m,m.neighbor<0,field(m.face_centers)['velocity'],settings=settings,body_force=exact['acceleration'],boundary_flux=exact_boundary_flux(m,field),gauge_value=exact['pressure'][0],boundary_velocity_means=exact_boundary_velocity_means(m,field),boundary_velocity_gradients=field(m.face_centers)['velocity_gradient'],quadratic_linear_solver='assembled')
with np.load(run/'fields.npz') as f:s.u=f['velocity'].copy();s.p=f['pressure'].copy();flux=f['flux'].copy()
old=np.r_[s.u.ravel(),s.p];_,_,a,_,_=s._system(flux);s._linear_flux=flux;s._frozen_covariance_divergence=s._covariance_divergence();matrix=current_matrix(s,flux,a)
s.u=np.zeros_like(s.u);s.p=np.zeros_like(s.p);_,base_rhs,_,_,_=NativeSteadySolver3D._system(s,flux);zero,_=s._corrections(flux,a);target=base_rhs+zero;del s._frozen_covariance_divergence
print(json.dumps(dict(stage='matrix_ready',nnz=matrix.nnz,elapsed_seconds=time.perf_counter()-start)),flush=True)
linear_start=time.perf_counter()
scale=np.maximum(np.asarray(abs(matrix).sum(1)).ravel(),1e-30)
scaled=(sparse.diags(1/scale)@matrix).tocsc();target_scaled=target/scale
matrix_digest=hashlib.sha256()
for data in (scaled.data,scaled.indices,scaled.indptr,target_scaled):matrix_digest.update(data.tobytes())
matrix_hash=matrix_digest.hexdigest()
(out/'operator.json').write_text(json.dumps(dict(scaled_matrix_and_rhs_sha256=matrix_hash,nnz=matrix.nnz))+'\n')
try:
 factor_start=time.perf_counter();lu=splu(scaled,permc_spec=args.ordering);factor_seconds=time.perf_counter()-factor_start
 answer=lu.solve(target_scaled)
 true=float(np.linalg.norm(scaled@answer-target_scaled)/np.linalg.norm(target_scaled))
 result=dict(linear_solve_passed=bool(np.isfinite(answer).all() and true<1e-9),true_relative_residual=true,
  factor_seconds=factor_seconds,factor_nnz=int(lu.L.nnz+lu.U.nnz),matrix_nnz=matrix.nnz,
  change_from_input_state_max=float(abs(answer-old).max()),ordering=args.ordering,
  scaled_matrix_and_rhs_sha256=matrix_hash)
 np.savez_compressed(out/'linear-answer.npz',answer=answer)
except Exception as error:
 result=dict(linear_solve_passed=False,failure=dict(type=type(error).__name__,message=str(error)))
result.update(linear_seconds=time.perf_counter()-linear_start,total_seconds=time.perf_counter()-start,
 peak_process_rss_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,flow_solved=False)
(out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
(out/'manifest.json').write_text(json.dumps({f.name:sha(f) for f in out.iterdir() if f.is_file() and f.name!='manifest.json'},indent=2)+'\n')

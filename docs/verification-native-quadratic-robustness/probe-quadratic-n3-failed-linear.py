import sys,json,time
from pathlib import Path
import numpy as np
from scipy.sparse.linalg import splu,gmres,LinearOperator
sys.path.insert(0,'/home/jsyc/tensor-suite-development/TensorFVM/docs/verification-native-abc3d-quadratic/source-snapshot/src')
from tensorfvm.native_mesh3d import NativeMesh3D
from tensorfvm.native_collocated3d import NativeSteadySettings
from tensorfvm.native_manufactured3d import abc_fields,exact_boundary_flux,exact_boundary_velocity_means
from tensorfvm.native_quadratic_flow3d import NativeQuadraticSteadySolver3D
n=3; shear=np.array([[1,.18,.12],[0,1,.2],[0,0,1]])
points=np.array([[x,y,z] for z in np.linspace(0,1,n+1) for y in np.linspace(0,1,n+1) for x in np.linspace(0,1,n+1)])@shear.T
ix=lambda x,y,z:z*(n+1)**2+y*(n+1)+x
cells=[]
for z in range(n):
 for y in range(n):
  for x in range(n):cells.append([ix(x,y,z),ix(x+1,y,z),ix(x+1,y+1,z),ix(x,y+1,z),ix(x,y,z+1),ix(x+1,y,z+1),ix(x+1,y+1,z+1),ix(x,y+1,z+1)])
m=NativeMesh3D.from_cells(points,cells,['hexahedron']*len(cells),triangulate_faces=True);exact=abc_fields(m.cell_centers)
s=NativeQuadraticSteadySolver3D(m,m.neighbor<0,abc_fields(m.face_centers)['velocity'],boundary_velocity_means=exact_boundary_velocity_means(m,abc_fields),boundary_velocity_gradients=abc_fields(m.face_centers)['velocity_gradient'],body_force=exact['acceleration'],boundary_flux=exact_boundary_flux(m,abc_fields),gauge_value=exact['pressure'][0],settings=NativeSteadySettings(viscosity=.1,density=1.,nonlinear_tolerance=1e-9,linear_tolerance=1e-11,max_iterations=100,relaxation=.8))
saved={};original=s._linear
def capture(block,rhs,a):
 try:return original(block,rhs,a)
 except Exception as exc:
  saved.update(block=block,rhs=rhs.copy(),a=a,flux=s._pending_flux.copy(),error=str(exc),history=list(s.history));raise
s._linear=capture
try:s.solve()
except RuntimeError:pass
if not saved:raise RuntimeError('expected failure was not reproduced')
block,rhs,a=saved['block'],saved['rhs'],saved['a'];flux=saved['flux'];s._linear_flux=flux;s._frozen_covariance_divergence=s._covariance_divergence();old_u,old_p=s.u.copy(),s.p.copy()
extra,_=s._corrections(flux,a);base_rhs=rhs-extra;s.u[:]=0;s.p[:]=0;zero,_=s._corrections(flux,a)
scale=np.maximum(np.asarray(abs(block).sum(1)).ravel(),1e-30);target=(base_rhs+zero)/scale
scaled=block.multiply((1/scale)[:,None]).tocsc();lu=splu(scaled)
def action(x):
 s.u=x[:3*s.N].reshape(s.N,3);s.p=x[3*s.N:];effect,_=s._corrections(flux,a);return (block@x-effect+zero)/scale
size=block.shape[0];eye=np.eye(size);A=np.column_stack([action(eye[:,j]) for j in range(size)])
rng=np.random.default_rng(789);x=rng.normal(size=size);y=rng.normal(size=size);sv=np.linalg.svd(A,compute_uv=False)
answer=np.linalg.solve(A,target);true=np.linalg.norm(action(answer)-target)/np.linalg.norm(target);preA=lu.solve(A);pre_sv=np.linalg.svd(preA,compute_uv=False);ev=np.linalg.eigvals(preA)
output={'failure':saved['error'],'completed_nonlinear_iterations':len(saved['history']),'last_nonlinear_history':saved['history'][-1] if saved['history'] else None,'cells':s.N,'matrix_size':size,'rank':int(np.linalg.matrix_rank(A)),'condition':float(sv[0]/sv[-1]),'singular_min':float(sv[-1]),'singular_max':float(sv[0]),'linearity_error':float(np.max(abs(action(x+y)-action(x)-action(y)))),'action_vs_explicit_error':float(np.max(abs(action(x)-A@x))),'zero_action_error':float(np.max(abs(action(np.zeros(size))))),'direct_original_equation_relative_residual':float(true),'preconditioned_condition':float(pre_sv[0]/pre_sv[-1]),'preconditioned_eigenvalue_abs_min':float(abs(ev).min()),'preconditioned_eigenvalue_abs_max':float(abs(ev).max()),'preconditioned_non_normality':float(np.linalg.norm(preA.T@preA-preA@preA.T)/np.linalg.norm(preA)**2),'gmres':[]}
for restart in [60,108]:
 for matrix_free in [False,True]:
  pass
for preconditioner in ['cached','fresh']:
 for restart in [60,108]:
  matrix_free=True
  apply_pre=(lambda x:s._quadratic_preconditioner.solve(x*scale/s._quadratic_preconditioner_scale)) if preconditioner=='cached' else lu.solve
  history=[];operator=LinearOperator(A.shape,matvec=action) if matrix_free else A
  ans,info=gmres(operator,target,M=LinearOperator(A.shape,matvec=apply_pre),x0=np.r_[old_u.ravel(),old_p],rtol=1e-11,atol=1e-14,restart=restart,maxiter=30,callback=lambda r:history.append(float(r)),callback_type='pr_norm')
  output['gmres'].append({'preconditioner':preconditioner,'restart':restart,'matrix_free':matrix_free,'info':int(info),'iterations':len(history),'true_relative_residual':float(np.linalg.norm(action(ans)-target)/np.linalg.norm(target)),'last_preconditioned_residual':history[-1]})
print(json.dumps(output,indent=2));Path('/tmp/probe-quadratic-n3-failed-linear.json').write_text(json.dumps(output,indent=2))
np.savez('/tmp/probe-quadratic-n3-failed-linear.npz',matrix=A,target=target,direct_solution=answer,baseline=scaled.toarray())

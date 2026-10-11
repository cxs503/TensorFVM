import sys,json,time
from pathlib import Path
import numpy as np
from scipy.sparse.linalg import splu,gmres,LinearOperator
sys.path.insert(0,'/home/jsyc/tensor-suite-development/TensorFVM/docs/verification-native-abc3d-quadratic/source-snapshot/src')
from tensorfvm.native_mesh3d import NativeMesh3D
from tensorfvm.native_collocated3d import NativeSteadySettings,NativeSteadySolver3D
from tensorfvm.native_manufactured3d import abc_fields,exact_boundary_flux,exact_boundary_velocity_means,volume_weighted_error
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


s.flux=s.velocity_flux_matrix@s.u.ravel()+s.velocity_flux_source
history=[];failure=None;converged=False;started=time.perf_counter()
def assess(u,p,q):
 s.u,s.p,s.flux=u,p,q
 B,rhs,a,C,src=s._system(q)
 consistent=s.velocity_flux_matrix@u.ravel()+C@p+src
 residual=B@np.r_[u.ravel(),p]-rhs
 momentum=float(np.linalg.norm(residual[:3*s.N])/max(np.linalg.norm(s.force_integrated)+np.linalg.norm(s.diffusion_source)+np.linalg.norm(s.p_force_source),1))
 div=s.viscous.op.integrate(q)/m.cell_volumes
 current_div=s.viscous.op.integrate(consistent)/m.cell_volumes
 coupling=(q-consistent)/np.linalg.norm(m.area_vectors,axis=1)
 merit=float(np.sqrt(momentum**2+np.sum(div**2*m.cell_volumes)/m.cell_volumes.sum()+np.mean(coupling**2)))
 return B,rhs,a,C,src,merit,dict(momentum_residual=momentum,native_divergence_max=float(abs(div).max()),current_reconstructed_divergence_max=float(abs(current_div).max()),face_flux_consistency_max=float(abs(coupling).max()))
try:
 for iteration in range(100):
  old_u,old_p,old_q=s.u.copy(),s.p.copy(),s.flux.copy()
  B,rhs,a,C,src,merit,old_metrics=assess(old_u,old_p,old_q)
  answer,linear,li=s._linear(B,rhs,a)
  nu=answer[:3*s.N].reshape(s.N,3);npres=answer[3*s.N:]
  nq=s.velocity_flux_matrix@nu.ravel()+C@npres+src
  trials=[];accepted=False
  for backtrack in range(16):
   omega=.8*2**(-backtrack)
   u=old_u+omega*(nu-old_u);p=old_p+omega*(npres-old_p);q=old_q+omega*(nq-old_q)
   *_,trial_merit,metrics=assess(u,p,q)
   trials.append(dict(omega=omega,merit=trial_merit))
   if trial_merit <= merit*(1-1e-4*omega):accepted=True;break
  if not accepted:
   s.u,s.p,s.flux=old_u,old_p,old_q;failure='No residual-decreasing step in 16 backtracking trials';history.append(dict(iteration=iteration+1,accepted=False,old_merit=merit,trials=trials,old_metrics=old_metrics));break
  changes=dict(velocity_change=float(np.linalg.norm(u-old_u)/max(np.linalg.norm(u),1e-30)),pressure_change=float(np.linalg.norm(p-old_p)/max(np.linalg.norm(p),1)),flux_change=float(np.linalg.norm(q-old_q)/max(np.linalg.norm(q),1e-30)))
  row=dict(iteration=iteration+1,accepted=True,omega=omega,merit=trial_merit,old_merit=merit,trials=trials,true_linear_relative_residual=float(linear),linear_iterations=li,**changes,**metrics);history.append(row)
  print(json.dumps(row),flush=True)
  if max(*changes.values(),*metrics.values()) < 1e-9:converged=True;break
except Exception as exc:failure=str(exc)
errors={field:volume_weighted_error(value,exact[field],m.cell_volumes,remove_mean=field=='pressure') for field,value in [('velocity',s.u),('pressure',s.p)]}
errors.update({f'velocity_{axis}':volume_weighted_error(s.u[:,j],exact['velocity'][:,j],m.cell_volumes) for j,axis in enumerate('xyz')})
result=dict(kind='Independent zero initial velocity, constant pressure gauge and residual-decreasing backtracking; no production solver changes',converged=converged,failure=failure,history=history,elapsed=time.perf_counter()-started,errors=errors,mms_qualified=converged and all(e<.03 for e in errors.values()),acceptance_limit=.03,nonlinear_tolerance=1e-9,linear_tolerance=1e-11)
Path('/tmp/probe-quadratic-n3-zero-linesearch.json').write_text(json.dumps(result,indent=2));np.savez('/tmp/probe-quadratic-n3-zero-linesearch.npz',velocity=s.u,pressure=s.p,flux=s.flux);print(json.dumps({k:v for k,v in result.items() if k!='history'},indent=2))

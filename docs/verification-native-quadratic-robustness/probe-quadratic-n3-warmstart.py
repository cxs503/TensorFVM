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

kwargs=dict(body_force=exact['acceleration'],boundary_flux=exact_boundary_flux(m,abc_fields),gauge_value=exact['pressure'][0],settings=NativeSteadySettings(viscosity=.1,density=1.,nonlinear_tolerance=1e-9,linear_tolerance=1e-11,max_iterations=100,relaxation=.8))
baseline=NativeSteadySolver3D(m,m.neighbor<0,abc_fields(m.face_centers)['velocity'],**kwargs)
baseline_state=baseline.solve()
s.u=baseline_state['velocity'].copy();s.p=baseline_state['pressure'].copy()
records={'initial_kind':'converged numerical first-order LS native solution, not analytic interior fields','baseline_converged':baseline_state['converged'],'baseline_history':baseline_state['history'],'baseline_error':{key:volume_weighted_error(baseline_state[field],exact[field],m.cell_volumes,remove_mean=field=='pressure') for key,field in [('velocity','velocity'),('pressure','pressure')]}}
started=time.perf_counter()
try:
 state=s.solve(callback=lambda row:print(json.dumps(row),flush=True))
 records.update(converged=state['converged'],errors={field:volume_weighted_error(state[field],exact[field],m.cell_volumes,remove_mean=field=='pressure') for field in ['velocity','pressure']})
except Exception as exc:
 records.update(converged=False,error=str(exc))
records.update(quadratic_history=s.history,elapsed=time.perf_counter()-started)
Path('/tmp/probe-quadratic-n3-warmstart.json').write_text(json.dumps(records,indent=2));np.savez('/tmp/probe-quadratic-n3-warmstart.npz',baseline_velocity=baseline.u,baseline_pressure=baseline.p,quadratic_velocity=s.u,quadratic_pressure=s.p)
print(json.dumps({k:v for k,v in records.items() if k not in ['quadratic_history','baseline_history']},indent=2))

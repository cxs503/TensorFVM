import numpy as np,time,json
from tensorfvm.native_mesh3d import NativeMesh3D
from tensorfvm.native_collocated3d import NativeSteadySolver3D,NativeSteadySettings
from tensorfvm.native_manufactured3d import abc_fields,exact_boundary_flux,volume_weighted_error
path='docs/verification-suboff-hybrid-mesh/coarse/suboff-hybrid.msh'
start=time.perf_counter();m=NativeMesh3D.from_gmsh(path,triangulate_faces=True);print('MESH_READ',len(m.cells),time.perf_counter()-start,flush=True)
f=abc_fields(m.cell_centers);bu=abc_fields(m.face_centers)['velocity'];q=exact_boundary_flux(m,abc_fields);print('BC_QUADRATURE',q.sum(),flush=True)
s=NativeSteadySolver3D(m,m.neighbor<0,bu,body_force=f['acceleration'],boundary_flux=q,gauge_value=f['pressure'][0],settings=NativeSteadySettings(viscosity=.1,linear_method='direct',nonlinear_tolerance=1e-7,max_iterations=60,relaxation=.7))
print('OPERATOR_READY',time.perf_counter()-start,flush=True)
def callback(row):
 print(json.dumps(row),flush=True)
 np.savez_compressed('/tmp/native-suboff-mms-current-state.npz',velocity=s.u,pressure=s.p,flux=s.flux)
r=s.solve(callback=callback)
print('ERROR',volume_weighted_error(r['velocity'],f['velocity'],m.cell_volumes),volume_weighted_error(r['pressure'],f['pressure'],m.cell_volumes,remove_mean=True),r['converged'],time.perf_counter()-start,flush=True)
np.savez_compressed('/tmp/native-suboff-mms-probe.npz',velocity=r['velocity'],pressure=r['pressure'],flux=r['flux'])

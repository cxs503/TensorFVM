import json
from pathlib import Path
import numpy as np
import tensorfvm.native_scalar_mmp_newton3d as newton
import tensorfvm.native_hybrid_scalar3d as scalar
from tensorfvm.native_scalar_mmp3d import ScalarMMPConvergenceError
from tensorfvm.native_mesh3d import NativeMesh3D
from tensorfvm.suboff_hybrid_mesh import structured_layers
source=Path(newton.__file__).read_text()
source=source.replace("state=np.r_[initial['cell_values'],initial['face_values']].astype(np.longdouble)","saved=np.load('results/native-mmp-extended-r2/suboff_layer_affine-r2-nonlinear_mmp-fields.npz');state=np.r_[saved['cell_values'],saved['face_values']].astype(np.longdouble)")
source=source.replace('(1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9,1e-10,0.)','(1e-11,1e-12,1e-13,1e-14,1e-15,1e-16,0.)')
source=source.replace('            history.append(row)','            history.append(row)\n            print(row,flush=True)')
exec(compile(source,newton.__file__,'exec'),newton.__dict__)
scalar.solve_mmp_newton=newton.solve_mmp_newton
r=2;growth=1.2**(1/r)
p,c,t,*_=structured_layers(axial=12*r,azimuthal=8*r,layers=2*r,first_height=.0022/sum(growth**np.arange(2*r)),growth=growth)
m=NativeMesh3D.from_cells(p,c,t,triangulate_faces=True)
s=scalar.NativeHybridScalar3D(m,correction='nonlinear_mmp')
try:
 result=s.solve(boundary_values=1+m.face_centers@np.array([.1,-.2,.3]));print('CONVERGED',flush=True)
except ScalarMMPConvergenceError as e:result=e.scalar_result;print('REJECTED',result['failure'],flush=True)
Path('/tmp/fvm-finer-continuation-history.json').write_text(json.dumps(result['nonlinear_history'],indent=2))
np.savez_compressed('/tmp/fvm-finer-continuation-state.npz',**{k:v for k,v in result.items() if k!='nonlinear_history'})
print({k:result[k] for k in ['current_nonlinear_residual','final_smoothing','work_balance','cell_balance_max','interface_flux_defect_max']},flush=True)

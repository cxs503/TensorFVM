import numpy as np
from tensorfvm.native_hybrid_scalar3d import NativeHybridScalar3D
from tensorfvm.verification.native_hybrid_pressure import pressure_mesh
for n in (2,4,8):
 m=pressure_mesh(n);g=np.array([.1,-.2,.3]);bd=1+m.face_centers@g
 r=NativeHybridScalar3D(m,2.,correction='nonlinear_mmp').solve(boundary_values=bd)
 print(n,np.max(np.abs(r['cell_values']-1-m.cell_centers@g)),np.max(np.abs(r['diffusive_flux']-2*m.area_vectors@g)),len(r['nonlinear_history']),flush=True)

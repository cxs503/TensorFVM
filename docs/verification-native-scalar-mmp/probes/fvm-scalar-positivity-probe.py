import numpy as np
from tensorfvm.verification.native_hybrid_rans import coupled_shear_solver
from tensorfvm.native_hybrid_sst3d import HybridSSTAdmissibilityError
from tensorfvm.native_hybrid_scalar3d import NativeHybridScalar3D
for n in (2,4,8):
 m,s=coupled_shear_solver(n,'low_re',triangulate_faces=True)
 for dt in (1.,.01,.0001):
  try:
   r=s.transport.step(s.k,s.omega,s.k_faces,s.omega_faces,s.flow.volume_flux(),s.wall_distance,s.nu,s.flow.velocity_gradient(),dt,**s.boundary)
   print('STEP',n,dt,'pass',r['k_faces'].min(),r['omega_faces'].min(),flush=True)
  except HybridSSTAdmissibilityError as e:print('STEP',n,dt,e.scalar_name,e.scalar_result['cell_min'],e.scalar_result['face_min'],flush=True)
 # The exact frozen steady reaction problem does not depend on initialization/dt.
 c=s._closure();op=NativeHybridScalar3D(m,s.nu+c['sigma_k']*c['nu_t'])
 r=op.solve(s.flow.volume_flux(),source=np.maximum(c['production_limited'],0),sink=.09*s.omega,
            boundary_values=s.boundary['k_boundary'],inflow_values=s.boundary['k_inflow'])
 print('STEADY',n,r['cell_min'],r['face_min'],flush=True)
 local=op.diffusion.local[0][3];off=local.copy();np.fill_diagonal(off,0)
 print('MATRIX',n,'positive_offdiagonal',off.max(),flush=True)

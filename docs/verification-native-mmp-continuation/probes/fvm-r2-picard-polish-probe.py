import copy
import numpy as np
from tensorfvm.native_hybrid_scalar3d import NativeHybridScalar3D
from tensorfvm.native_mesh3d import NativeMesh3D
from tensorfvm.suboff_hybrid_mesh import structured_layers
r=2;growth=1.2**(1/r)
p,c,t,*_=structured_layers(axial=12*r,azimuthal=8*r,layers=2*r,first_height=.0022/sum(growth**np.arange(2*r)),growth=growth)
m=NativeMesh3D.from_cells(p,c,t,triangulate_faces=True);s=NativeHybridScalar3D(m,correction='nonlinear_mmp')
f=np.load('results/native-mmp-extended-r2/suboff_layer_affine-r2-nonlinear_mmp-fields.npz')
u=np.r_[f['cell_values'],f['face_values']];bd=1+m.face_centers@np.array([.1,-.2,.3])
for iteration in range(20):
 linear=copy.copy(s);linear.mmp=None;linear.diffusion,_=s.mmp.operator(u)
 r=linear.solve(boundary_values=bd);v=np.r_[r['cell_values'],r['face_values']]
 audit=s.steady_residual(v[:s.N],v[s.N:],boundary_values=bd)
 print(iteration,'unrelaxed_change',float(np.linalg.norm(v-u)/np.linalg.norm(v)),'current_residual',audit['relative_residual'],flush=True)
 u=v
np.savez_compressed('/tmp/fvm-r2-picard-polish-state.npz',state=u)

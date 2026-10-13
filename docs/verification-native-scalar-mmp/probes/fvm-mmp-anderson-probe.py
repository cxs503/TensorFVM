import copy,numpy as np
from scipy.optimize import anderson,NoConvergence
from tensorfvm.suboff_hybrid_mesh import structured_layers
from tensorfvm.native_mesh3d import NativeMesh3D
from tensorfvm.native_hybrid_scalar3d import NativeHybridScalar3D
p,c,t,_,_,_=structured_layers(axial=12,azimuthal=8,layers=2,first_height=.001,growth=1.2)
m=NativeMesh3D.from_cells(p,c,t,triangulate_faces=True);g=np.array([.1,-.2,.3]);bd=1+m.face_centers@g
op=NativeHybridScalar3D(m,correction='nonlinear_mmp');linear=copy.copy(op);linear.mmp=None
r=linear.solve(boundary_values=bd);state=np.r_[r['cell_values'],r['face_values']];free=linear.diffusion.free;count=0

def residual(x):
 global count
 u=state.copy();u[free]=x
 linear.diffusion,_=op.mmp.operator(u)
 r=linear.solve(boundary_values=bd);delta=np.r_[r['cell_values'],r['face_values']][free]-x;count+=1
 if count%20==0:print('EVAL',count,np.max(np.abs(delta)),flush=True)
 return delta
try:x=anderson(residual,state[free],alpha=.5,M=8,w0=.01,maxiter=400,f_tol=1e-12,line_search='armijo')
except NoConvergence as e:x=e.args[0];print('NO_CONVERGENCE',flush=True)
state[free]=x;r=op.steady_residual(state[:op.N],state[op.N:],boundary_values=bd)
print('FINAL',count,r['relative_residual'],np.max(np.abs(residual(x))),np.max(np.abs(x[:op.N]-1-m.cell_centers@g)),flush=True)

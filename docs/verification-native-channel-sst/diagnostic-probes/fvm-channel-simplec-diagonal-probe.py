from tensorfvm.verification.native_developed_channel import developed_channel_rans
from tensorfvm.native_hybrid_stokes3d import NativeHybridStokes3D
for relaxation in (.5,.9,1.):
 m,s,b=developed_channel_rans(16,axial=2,length=.1,span=.1)
 s._refresh_viscosity();f=s.flow;f.A=f.viscous.matrix;f.rhs=(f.load-f.viscous.full_matrix@f.prescribed)[f.viscous.free]
 f.momentum_relaxation_basis="diagonal"
 NativeHybridStokes3D.solve(f,algorithm='simplec',relaxation=relaxation,tolerance=1e-9)
 print(relaxation,f.converged,f.history[-1],flush=True)

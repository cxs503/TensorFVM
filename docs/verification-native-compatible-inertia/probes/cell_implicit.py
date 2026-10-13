import numpy as np
from scipy import sparse
from tensorfvm.verification.native_hybrid_pressure import pressure_mesh
from tensorfvm.native_hybrid_ns3d import NativeHybridNavierStokes3D
from tensorfvm.native_manufactured3d import exact_boundary_velocity_means,volume_weighted_error
class Implicit(NativeHybridNavierStokes3D):
 def _current_system(self):
  c,b,_,_=self.convection.assemble(self.volume_flux(),self.velocity_gradient())
  q=sparse.hstack((c,sparse.csr_matrix((3*self.N,3*self.F))),format='csr')
  return self.viscous.full_matrix+self.face_mean.matrix.T@q,self.face_mean.matrix.T@b

def field(points):
 x,y,z=points.T;u=np.column_stack((1+y*z,1+x*z,1+x*y))
 return dict(velocity=u,pressure=-.5*np.sum(u*u,axis=1))
for nu in (.1,.01,.001):
 m=pressure_mesh(4);e=field(m.cell_centers);bd=exact_boundary_velocity_means(m,field)
 s=Implicit(m,m.neighbor<0,bd,viscosity=nu,gauge_value=e['pressure'][0],convection_order=2,coupled_linear_method='reused_lu')
 r=s.solve(algorithm='coupled',outer_relaxation=.7,max_outer_iterations=200,tolerance=1e-8)
 print(nu,r['numerically_converged'],volume_weighted_error(r['velocity'],e['velocity'],m.cell_volumes),volume_weighted_error(r['pressure'],e['pressure'],m.cell_volumes,remove_mean=True),r['outer_history'][-1],r['energy_balance_error'],r['local_momentum_balance_max'],flush=True)

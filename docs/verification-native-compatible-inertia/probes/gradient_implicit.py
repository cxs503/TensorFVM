import numpy as np
from scipy import sparse
from tensorfvm.verification.native_hybrid_pressure import pressure_mesh
from tensorfvm.native_hybrid_ns3d import NativeHybridNavierStokes3D
from tensorfvm.native_manufactured3d import exact_boundary_velocity_means,volume_weighted_error
class Implicit(NativeHybridNavierStokes3D):
 def _current_system(self):
  q=self.volume_flux();c,b,t,src=self.convection.assemble(q,np.zeros((self.N,3,3)))
  active=(self.mesh.neighbor>=0)|(q>=0);ids=np.flatnonzero(active)
  up=np.where(q>=0,self.mesh.owner,self.mesh.neighbor)[active]
  r=self.mesh.face_centers[active]-self.mesh.cell_centers[up]
  h=sparse.csr_matrix((self.F,self.F))
  for j in range(3):
   a=sparse.coo_matrix((q[ids]*r[:,j],(ids,up)),shape=(self.F,self.N)).tocsr()
   g=sparse.diags(1/self.mesh.cell_volumes)@self.convection.D@sparse.diags(self.mesh.area_vectors[:,j])
   h+=a@g
  matrix=sparse.kron(self.convection.D@sparse.hstack((t,h)),sparse.eye(3),format='csr')
  return self.viscous.full_matrix+self.face_mean.matrix.T@matrix,self.face_mean.matrix.T@b

def field(points):
 x,y,z=points.T;u=np.column_stack((1+y*z,1+x*z,1+x*y))
 return dict(velocity=u,pressure=-.5*np.sum(u*u,axis=1))
for nu in (.1,.01,.001):
 m=pressure_mesh(4);e=field(m.cell_centers);bd=exact_boundary_velocity_means(m,field)
 s=Implicit(m,m.neighbor<0,bd,viscosity=nu,gauge_value=e['pressure'][0],convection_order=2,coupled_linear_method='reused_lu')
 r=s.solve(algorithm='coupled',outer_relaxation=.7,max_outer_iterations=200,tolerance=1e-8)
 print(nu,r['numerically_converged'],volume_weighted_error(r['velocity'],e['velocity'],m.cell_volumes),volume_weighted_error(r['pressure'],e['pressure'],m.cell_volumes,remove_mean=True),r['outer_history'][-1],r['energy_balance_error'],r['local_momentum_balance_max'],flush=True)

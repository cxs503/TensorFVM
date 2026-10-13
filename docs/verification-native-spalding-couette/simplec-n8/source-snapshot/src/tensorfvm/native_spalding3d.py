"""Cell-sampled Spalding traction on native hybrid boundary equations.

The tangential relative velocity is sampled at the wall-adjacent cell. Its
nonlinear traction enters the boundary FACE equation. The shared face trace
is an auxiliary bulk-flow trace; it is not imposed as physical no-slip data.
Normal velocity must be constrained separately. This nonlocal boundary law
is not the same discrete operator as OpenFOAM's boundary-nut construction.
The difference between cell-sampled dissipation and face work is reported.
"""
import numpy as np
from scipy import sparse
from .wall_functions import wall_traction_linearization


class NativeSpaldingWall3D:
    def __init__(self,mesh,faces,distance,*,density=1.,molecular_viscosity=1.,wall_velocity=0.):
        self.mesh=mesh;self.N=len(mesh.cells);self.F=len(mesh.owner)
        mask=np.asarray(faces,bool)
        if mask.shape!=(self.F,) or not mask.any() or np.any(mask & (mesh.neighbor>=0)):
            raise ValueError('nonempty boundary face mask required')
        self.faces=np.flatnonzero(mask);self.mask=mask.copy();self.owner=mesh.owner[self.faces]
        self.area=np.linalg.norm(mesh.area_vectors[self.faces],axis=1)
        self.normal=mesh.area_vectors[self.faces]/self.area[:,None]
        self.projection=np.eye(3)-self.normal[:,:,None]*self.normal[:,None,:]
        self.distance=np.broadcast_to(np.asarray(distance,float),(len(self.faces),)).copy()
        self.wall_velocity=np.broadcast_to(np.asarray(wall_velocity,float),(len(self.faces),3)).copy()
        self.density=float(density);self.molecular_viscosity=float(molecular_viscosity)
        if (not np.isfinite(self.distance).all() or np.any(self.distance<=0)
            or not np.isfinite(self.wall_velocity).all()
            or not np.isfinite([self.density,self.molecular_viscosity]).all()
            or min(self.density,self.molecular_viscosity)<=0):
            raise ValueError('finite wall velocities and positive distance/density/molecular viscosity required')

    def validate_flow(self,flow,boundary_traction=0.):
        if flow.mesh is not self.mesh:raise ValueError('wall and flow must use identical mesh')
        if flow.convection.density!=self.density:raise ValueError('wall/flow density mismatch')
        if flow.open_normal_boundary[self.mask].any():raise ValueError('Spalding wall requires exact normal-velocity constraints')
        masks=flow.viscous.dirichlet_components[self.mask]
        if np.max(abs(self.projection*masks[:,None,:]))>1e-12:
            raise ValueError('Spalding tangential components must remain free')
        if np.any(flow.viscous.robin_coefficient[self.mask]!=0):raise ValueError('Spalding wall cannot also carry Robin friction')
        imposed=flow.prescribed[3*self.N:].reshape(self.F,3)[self.mask]
        if np.max(abs(np.sum((imposed-self.wall_velocity)*self.normal,axis=1)))>1e-12:
            raise ValueError('wall velocity normal component differs from prescribed normal velocity')
        traction=np.broadcast_to(np.asarray(boundary_traction,float),(self.F,3))[self.mask]
        if np.max(abs(np.einsum('fij,fj->fi',self.projection,traction)))>1e-12:
            raise ValueError('Spalding wall cannot also prescribe tangential traction')

    def evaluate(self,state):
        state=np.asarray(state,float)
        if state.shape!=(3*(self.N+self.F),) or not np.isfinite(state).all():raise ValueError('finite complete hybrid velocity state required')
        cells=state[:3*self.N].reshape(self.N,3)[self.owner]
        relative=np.einsum('fij,fj->fi',self.projection,cells-self.wall_velocity)
        value=wall_traction_linearization(relative,self.distance,self.density,self.molecular_viscosity)
        value.update(sampled_velocity=cells,relative_tangential_velocity=relative)
        return value

    def _matrix(self,blocks):
        f,i,j=np.nonzero(blocks)
        return sparse.coo_matrix((self.area[f]*blocks[f,i,j],
            (3*(self.N+self.faces[f])+i,3*self.owner[f]+j)),
            shape=(3*(self.N+self.F),)*2).tocsr()

    def system(self,state):
        """Secant system for Picard/SIMPLE, exact at the supplied state."""
        value=self.evaluate(state)
        blocks=value['tangential_secant_coefficient'][:,None,None]*self.projection
        matrix=self._matrix(blocks);source=np.zeros(3*(self.N+self.F))
        source[3*self.N:].reshape(self.F,3)[self.mask]=-self.area[:,None]*np.einsum('fij,fj->fi',blocks,self.wall_velocity)
        return matrix,source

    def jacobian(self,state):
        value=self.evaluate(state)
        blocks=-np.einsum('fij,fjk->fik',value['traction_jacobian'],self.projection)
        return self._matrix(blocks)

    def diagnostics(self,state,total_traction):
        value=self.evaluate(state);target=self.area[:,None]*value['traction_pa']
        face=np.asarray(state)[3*self.N:].reshape(self.F,3)[self.mask]
        actual=np.einsum('fij,fj->fi',self.projection,np.asarray(total_traction)[self.mask])
        sampled=value['sampled_velocity'];wall=self.wall_velocity
        boundary_power=float(np.sum(face*target));wall_power=float(np.sum(wall*target))
        dissipation=-float(np.sum((sampled-wall)*target))
        transfer=float(np.sum((face-sampled)*target))
        return dict(faces=self.faces.copy(),traction_integrated=target,
            actual_tangential_traction_integrated=actual,traction_defect_max=float(np.max(abs(actual-target))),
            sampled_velocity=sampled,relative_tangential_velocity=value['relative_tangential_velocity'],
            friction_velocity=value['friction_velocity_m_s'],y_plus=value['y_plus'],
            boundary_power=boundary_power,wall_motion_power=wall_power,
            sample_dissipation=dissipation,sampling_transfer_power=transfer,
            work_decomposition_error=abs(boundary_power-wall_power+dissipation-transfer),
            sampling='wall-adjacent cell; face trace used for discrete boundary work',
            physical_benchmark_passed=False)

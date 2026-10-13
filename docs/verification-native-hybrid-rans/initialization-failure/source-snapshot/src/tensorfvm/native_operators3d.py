"""Conservative face integration and affine-exact reconstruction on native 3D meshes."""
import numpy as np

class NativeOperators3D:
    def __init__(self,mesh):
        self.mesh=mesh; self.internal=mesh.neighbor>=0; self.boundary=~self.internal
        self.d=mesh.face_centers-mesh.cell_centers[mesh.owner]
        self.d[self.internal]=mesh.cell_centers[mesh.neighbor[self.internal]]-mesh.cell_centers[mesh.owner[self.internal]]
        self.weights=1/np.einsum('ij,ij->i',self.d,self.d)
        normal=self.integrate(self.weights[:,None,None]*self.d[:,:,None]*self.d[:,None,:],signed=False)
        if np.any(np.linalg.eigvalsh(normal)[:,0]<=1e-13*np.linalg.eigvalsh(normal)[:,-1]):raise ValueError('rank-deficient gradient stencil')
        self.inverse_normal=np.linalg.inv(normal)

    def integrate(self,face_values,signed=True):
        """Sum outward integrated shared face quantities; internal contributions cancel."""
        m=self.mesh; f=np.asarray(face_values)
        if f.shape[0]!=len(m.owner):raise ValueError('one quantity per native face required')
        result=np.zeros((len(m.cells),)+f.shape[1:],dtype=np.result_type(f,float))
        np.add.at(result,m.owner,f)
        np.add.at(result,m.neighbor[self.internal],(-1 if signed else 1)*f[self.internal])
        return result

    def least_squares_gradient(self,values,boundary_face_values):
        """Weighted least squares using explicit Dirichlet data on every boundary face.

        Boundary data are mandatory: an absent wall gradient must not silently
        become a zero-normal condition for manufactured affine fields.
        """
        m=self.mesh;v=np.asarray(values);b=np.asarray(boundary_face_values)
        if v.shape!=(len(m.cells),) or b.shape!=(self.boundary.sum(),):raise ValueError('scalar cell values and compact boundary data required')
        delta=np.empty(len(m.owner));delta[self.internal]=v[m.neighbor[self.internal]]-v[m.owner[self.internal]];delta[self.boundary]=b-v[m.owner[self.boundary]]
        rhs=self.integrate((self.weights*delta)[:,None]*self.d,signed=False)
        return np.einsum('cij,cj->ci',self.inverse_normal,rhs)

    def reconstruct_faces(self,values,gradients,boundary_values):
        """Geometric interpolation plus gradient skew correction, exact for affine p."""
        m=self.mesh;v=np.asarray(values);g=np.asarray(gradients);result=np.empty(len(m.owner))
        i=self.internal;d=self.d[i];s=m.area_vectors[i]
        weight=np.einsum('ij,ij->i',m.face_centers[i]-m.cell_centers[m.owner[i]],s)/np.einsum('ij,ij->i',d,s)
        linear_point=m.cell_centers[m.owner[i]]+weight[:,None]*d
        gf=(1-weight[:,None])*g[m.owner[i]]+weight[:,None]*g[m.neighbor[i]]
        result[i]=(1-weight)*v[m.owner[i]]+weight*v[m.neighbor[i]]+np.einsum('ij,ij->i',gf,m.face_centers[i]-linear_point)
        result[self.boundary]=boundary_values
        return result

    def gauss_gradient(self,face_values):
        return self.integrate(np.asarray(face_values)[:,None]*self.mesh.area_vectors)/self.mesh.cell_volumes[:,None]

    def audit_affine(self):
        m=self.mesh;gradient=np.array([.7,-1.3,2.1]);v=1.7+m.cell_centers@gradient;exact=1.7+m.face_centers@gradient
        ls=self.least_squares_gradient(v,exact[self.boundary]);faces=self.reconstruct_faces(v,ls,exact[self.boundary]);gauss=self.gauss_gradient(faces)
        flux=m.area_vectors@np.array([.3,.8,-.4]); balance=self.integrate(flux)
        # Vector field u=(x,2y,3z) has divergence 6, independent of grid.
        affine_flux=np.einsum('ij,ij->i',m.face_centers*np.array([1.,2.,3.]),m.area_vectors)
        return dict(least_squares_gradient_max_error=float(np.max(np.abs(ls-gradient))),face_affine_max_error=float(np.max(np.abs(faces-exact))),gauss_gradient_max_error=float(np.max(np.abs(gauss-gradient))),constant_vector_divergence_max_error=float(np.max(np.abs(balance/m.cell_volumes))),affine_vector_divergence_max_error=float(np.max(np.abs(self.integrate(affine_flux)/m.cell_volumes-6))),global_flux_balance_error=float(abs(balance.sum()-flux[self.boundary].sum())))

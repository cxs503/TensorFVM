"""Homogeneous sparse action of the actual quadratic native FVM equations.

Known boundary data and frozen nonlinear face covariance enter the source,
not a difference between two large affine matrix-vector evaluations.
"""
import numpy as np
from scipy import sparse
from .native_quadratic_maps3d import face_mean_matrix,face_gradient_matrices


def prepare_geometry(solver):
    m=solver.mesh;N=solver.N;F=solver.F;D=solver.viscous.divergence_matrix
    op=solver.quadratic_velocity;p=solver.quadratic_pressure;s=m.area_vectors
    mean=face_mean_matrix(op,prescribed_boundary_means=solver.boundary_velocity_means is not None)
    pressure_mean=face_mean_matrix(p,prescribed_boundary_means=True)
    gradient=face_gradient_matrices(op)
    normal=sum(sparse.diags(s[:,k])@gradient[k] for k in range(3))
    mu=sparse.diags(solver.viscous.face_viscosity)
    stress=[]
    for a in range(3):
        row=[]
        for b in range(3):
            block=sparse.diags(s[:,b])@gradient[a]-(2/3 if solver.viscous.deviatoric else 0)*sparse.diags(s[:,a])@gradient[b]
            if a==b:block=block+normal
            row.append(-D@mu@block)
        stress.append(row)
    order=np.arange(3*N).reshape(3,N).T.ravel()
    diffusion=sparse.bmat(stress,format='csr')[order,:][:,order].tocsr()
    pressure_force=sparse.vstack([D@sparse.diags(s[:,k])@pressure_mean for k in range(3)],format='csr')[order,:]
    velocity_flux=sparse.hstack([sparse.diags(s[:,k]*(~solver.fixed))@mean for k in range(3)],format='csr')[:,order]
    # Compact/noncompact Hessian terms simplify analytically. Avoid subtracting
    # large nearly equal face-gradient matrices on very thin wall layers.
    f=np.arange(F);owner=sparse.csr_matrix((np.ones(F),(f,m.owner)),shape=(F,N));neighbor=sparse.csr_matrix((np.ones(F),(f,p.neighbor_safe)),shape=(F,N))
    J=sparse.diags(1-p.weight)@owner+sparse.diags(p.weight)@neighbor
    K=sparse.diags(1-p.weight)@owner-sparse.diags(p.weight)@neighbor
    delta=sparse.diags(p.internal.astype(float))@(neighbor-owner)-sparse.diags(p.dirichlet.astype(float))@owner
    gauss=[sparse.diags(1/m.cell_volumes)@D@sparse.diags(s[:,k])@pressure_mean for k in range(3)]
    defect=sum(sparse.diags(p.d[:,k])@J@gauss[k] for k in range(3))-delta
    for j in range(3):
        for k in range(3):defect+=.5*sparse.diags(p.d[:,j]*p.d[:,k])@K@p.hessian_matrices[j][k]
    coefficient=np.sum(s*s,axis=1)/p.projection*(p.internal|p.dirichlet)
    defect=(sparse.diags(coefficient)@defect).tocsr()
    return dict(diffusion=diffusion,pressure_force=pressure_force,velocity_flux=velocity_flux,
        pressure_defect=defect,velocity_face_mean=mean)


def current_matrix(solver,flux,a):
    if not hasattr(solver,'_assembled_geometry'):solver._assembled_geometry=prepare_geometry(solver)
    geometry=solver._assembled_geometry;m=solver.mesh;N=solver.N;F=solver.F
    D=solver.viscous.divergence_matrix;op=solver.quadratic_velocity
    momentum=geometry['diffusion']
    if solver.settings.convection:
        internal=m.neighbor>=0;indices=np.flatnonzero(internal)
        up=np.where(flux[internal]>=0,m.owner[internal],m.neighbor[internal])
        selector=sparse.csr_matrix((np.ones(len(up)),(indices,up)),shape=(F,N))
        offset=np.zeros((F,3));offset[internal]=m.face_centers[internal]-m.cell_centers[up]
        moment=offset[:,:,None]*offset[:,None,:]+op.face_covariance
        advected=selector.copy()
        for j in range(3):advected+=sparse.diags(offset[:,j])@selector@op.gradient_matrices[j]
        for j in range(3):
            for k in range(3):advected+=.5*sparse.diags(moment[:,j,k])@selector@op.hessian_matrices[j][k]
        advected+=sparse.diags((~internal).astype(float))@geometry['velocity_face_mean']
        scalar=D@sparse.diags(flux*solver.settings.density)@advected
        momentum=momentum+sparse.kron(scalar,sparse.eye(3),format='csr')
    diagonal=solver._mobility_reference+(a-solver._diffusion).diagonal().reshape(N,3).mean(1)
    alpha=np.asarray(solver.viscous.interpolation@(m.cell_volumes/diagonal));alpha[solver.fixed]=0
    correction=sparse.diags(alpha)@geometry['pressure_defect']
    matrix=sparse.bmat([[momentum,geometry['pressure_force']],[D@geometry['velocity_flux'],D@correction]],format='csr')
    if solver.gauge:
        matrix=matrix.tolil();matrix[3*N,:]=0;matrix[3*N,3*N]=1;matrix=matrix.tocsr()
    matrix.eliminate_zeros()
    return matrix

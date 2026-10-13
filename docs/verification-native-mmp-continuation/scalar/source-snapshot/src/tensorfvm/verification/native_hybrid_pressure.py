"""Independent skew mixed-cell pressure/stress controls for the hybrid backend."""
import numpy as np
from tensorfvm.verification.native_hybrid_flow import channel_mesh
from tensorfvm.native_mesh3d import NativeMesh3D
from tensorfvm.native_hybrid_stokes3d import NativeHybridStokes3D
from tensorfvm.native_manufactured3d import abc_fields,exact_boundary_velocity_means,volume_weighted_error

def pressure_mesh(n,mixed=True,skew=True):
 base=channel_mesh(n);cells=[];kinds=[]
 pieces=((0,1,2,6),(0,2,3,6),(0,3,7,6),(0,7,4,6),(0,4,5,6),(0,5,1,6))
 for i,c in enumerate(base.cells):
  if not mixed or (i%n+i//n%n+i//n**2)%2:cells.append(c);kinds.append('hexahedron')
  else:cells.extend([[c[j] for j in t] for t in pieces]);kinds.extend(['tetra']*6)
 points=base.vertices@(np.array([[1,0,0],[.2,1,0],[.1,.1,1]]) if skew else np.eye(3))
 return NativeMesh3D.from_cells(points,cells,kinds,triangulate_faces=True)

def pressure_fields(points,kind):
 if kind not in ('hydro','shear','abc'):raise ValueError('unknown pressure control')
 x,y,z=np.asarray(points).T;u=np.zeros((len(x),3));p=np.zeros(len(x));f=u.copy()
 if kind=='hydro':p=(x*x+y*y+z*z)/2;f=points.copy()
 if kind=='shear':u=np.column_stack((y*y,z*z,x*x));f=np.full_like(u,-.2)
 if kind=='abc':
  e=abc_fields(points);u=e['velocity'];p=e['pressure'];f=e['acceleration']-np.einsum('cij,cj->ci',e['velocity_gradient'],u)
 return dict(velocity=u,pressure=p,force=f)


def pressure_solver(n,kind='abc',*,stabilization=1.,navier_stokes=False):
    from tensorfvm.native_hybrid_ns3d import NativeHybridNavierStokes3D
    mesh=pressure_mesh(n)
    field=lambda points:pressure_fields(points,kind)
    exact=field(mesh.cell_centers)
    boundary=exact_boundary_velocity_means(mesh,field)
    force=exact['force']
    options={}
    cls=NativeHybridStokes3D
    if navier_stokes:
        if kind!='abc':raise ValueError('NS diagnostic currently supports ABC only')
        cls=NativeHybridNavierStokes3D
        force=abc_fields(mesh.cell_centers)['acceleration']
        options['convection_order']=2
    solver=cls(mesh,mesh.neighbor<0,boundary,viscosity=.1,body_force=force,
        gauge_value=exact['pressure'][0],gradient_jump_stabilization=stabilization,
        coupled_linear_method='reused_lu',**options)
    return mesh,solver,exact


def gradient_norm_matrix(viscous):
    """Independent full-gradient + original residual norm, same free dofs.

    Used only as a denominator in the discrete strain coercivity diagnostic.
    It is not substituted for physical symmetric viscous stress.
    """
    from scipy import sparse
    rows=[];cols=[];values=[]
    for cell,(_,_,ids,_,g,residual,penalty) in enumerate(viscous.local):
        scalar=viscous.mu[cell]*viscous.mesh.cell_volumes[cell]*(g.T@g)
        scalar+=residual.T@(penalty[:,None]*residual)
        block=np.kron(scalar,np.eye(3))
        rows.append(np.repeat(ids,len(ids)));cols.append(np.tile(ids,len(ids)));values.append(block.ravel())
    norm=sparse.coo_matrix((np.concatenate(values),(np.concatenate(rows),np.concatenate(cols))),
                          shape=viscous.full_matrix.shape).tocsr()
    return norm[viscous.free][:,viscous.free].tocsc()


def strain_spectrum(viscous):
    """Report unconstrained, full-strain and divergence-free energy ratios.

    The deviatoric operator alone does not control dilation. The kernel test
    explicitly enforces the same cellwise incompressibility as the flow solve.
    The positive samples do not prove a mesh-independent bound on all meshes.
    """
    from scipy import sparse
    from scipy.sparse.linalg import eigsh,eigs,splu,LinearOperator
    mesh=viscous.mesh;N=viscous.N;F=viscous.F;ids=np.arange(F);internal=mesh.neighbor>=0
    cells=np.r_[mesh.owner,mesh.neighbor[internal]];faces=np.r_[ids,ids[internal]]
    vectors=np.vstack((mesh.area_vectors,-mesh.area_vectors[internal]))
    full_B=sparse.coo_matrix((vectors.ravel(),(np.repeat(cells,3),
        (3*(N+faces[:,None])+np.arange(3)).ravel())),shape=(N,3*(N+F))).tocsr()
    B=full_B[:,viscous.free];norm=gradient_norm_matrix(viscous);A=viscous.matrix
    start=np.random.default_rng(42).normal(size=A.shape[0])
    raw=eigsh(A,M=norm,k=1,which='SA',tol=1e-8,maxiter=2000,v0=start,return_eigenvectors=False)[0]
    bulk=(2/3)*B.T@sparse.diags(viscous.mu/mesh.cell_volumes)@B
    complete=eigsh(A+bulk,M=norm,k=1,which='SA',tol=1e-8,maxiter=2000,v0=start,return_eigenvectors=False)[0]
    independent=B[1:];nv=A.shape[0];np_=independent.shape[0]
    block=sparse.bmat([[A,independent.T],[independent,None]],format='csc');factor=splu(block)
    response=LinearOperator(block.shape,matvec=lambda vector:factor.solve(np.r_[norm@vector[:nv],np.zeros(np_)]))
    theta,mode=eigs(response,k=1,which='LM',tol=1e-8,maxiter=2000,
                    v0=np.random.default_rng(42).normal(size=block.shape[0]))
    if abs(theta[0].imag)>1e-8:raise RuntimeError('complex constrained energy eigenvalue')
    ratio=float(1/theta[0].real);velocity=mode[:nv,0].real;multiplier=mode[nv:,0].real
    residual=float(np.linalg.norm(A@velocity+independent.T@multiplier-ratio*(norm@velocity))/np.linalg.norm(A@velocity))
    divergence=float(np.max(np.abs(B@velocity/mesh.cell_volumes)))
    if residual>1e-6 or divergence>1e-8:raise RuntimeError('constrained energy eigenproblem did not converge')
    return dict(unconstrained_deviatoric_ratio=float(raw),full_strain_ratio=float(complete),
        minimum_strain_to_gradient_energy_ratio=ratio,eigen_relative_residual=residual,
        eigen_divergence_max=divergence,ratio_definition='divergence-free strain energy / full gradient plus original residual energy'),velocity

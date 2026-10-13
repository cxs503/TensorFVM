"""Independent polynomial Navier--Stokes manufacture with nonzero convection."""
import numpy as np
from .native_hybrid_flow import channel_mesh
from tensorfvm.native_manufactured3d import exact_boundary_velocity_means
from tensorfvm.native_hybrid_ns3d import NativeHybridNavierStokes3D


def polynomial_fields(points, reynolds=10.):
    x, y, z = np.asarray(points).T
    a = .2
    velocity = np.column_stack((1+a*x*x, -2*a*x*y, np.zeros_like(x)))
    pressure = x-.5*y+.25*z
    gradient = np.zeros((len(x),3,3))
    gradient[:,0,0] = 2*a*x
    gradient[:,1,0] = -2*a*y
    gradient[:,1,1] = -2*a*x
    convection = np.column_stack((2*a*x*(1+a*x*x), -2*a*y+2*a*a*x*x*y, np.zeros_like(x)))
    force = convection+np.array([1.,-.5,.25])-np.array([2*a,0,0])/reynolds
    return dict(velocity=velocity, pressure=pressure, gradient=gradient,
                convection=convection, force=force)


def polynomial_solver(n, reynolds=10., *, convection_order=1, body_force_discretization='face'):
    if not np.isfinite(reynolds) or reynolds <= 0:
        raise ValueError('positive finite Reynolds number required')
    mesh = channel_mesh(n)
    exact = polynomial_fields(mesh.cell_centers, reynolds)
    # Prescribe true face area means, giving globally balanced boundary flux.
    boundary = exact_boundary_velocity_means(mesh, polynomial_fields)
    solver = NativeHybridNavierStokes3D(mesh, mesh.neighbor < 0, boundary,
        viscosity=1/reynolds, body_force=exact['force'], gauge_value=exact['pressure'][0],
        convection_order=convection_order,body_force_discretization=body_force_discretization)
    return mesh, solver, exact


def kovasznay_fields(points, reynolds=40.):
    """Unforced steady Navier--Stokes solution; rho=1 and nu=1/Re.

    exp(lambda*x) decays downstream. The identity lambda**2-Re*lambda
    = 4*pi**2 makes convection + grad(p) - nu*laplacian(u) exactly zero.
    No source calculated from the discrete operator is used.
    """
    if not np.isfinite(reynolds) or reynolds <= 0:
        raise ValueError('positive finite Reynolds number required')
    x, y, _ = np.asarray(points).T
    k = 2*np.pi
    lam = -2*k*k/(reynolds+np.sqrt(reynolds**2+4*k*k))
    e = np.exp(lam*x); c = np.cos(k*y); s = np.sin(k*y)
    velocity = np.column_stack((1-e*c, lam/k*e*s, np.zeros_like(x)))
    pressure = .5*(1-e*e)
    gradient = np.zeros((len(x),3,3))
    gradient[:,0,0] = -lam*e*c
    gradient[:,0,1] = k*e*s
    gradient[:,1,0] = lam*lam/k*e*s
    gradient[:,1,1] = lam*e*c
    return dict(velocity=velocity, pressure=pressure, gradient=gradient,
                force=np.zeros_like(velocity))


def kovasznay_solver(n, reynolds=40., *, convection_order=2, triangulate_faces=True,
                     coupled_linear_method='direct', symmetry_extrusion=False):

    """Unit cube segment of an analytic 2D flow, fixed exact boundary means."""
    mesh = channel_mesh((n,n,1) if symmetry_extrusion else n,triangulate_faces=triangulate_faces)
    fields = lambda points: kovasznay_fields(points,reynolds)
    exact = fields(mesh.cell_centers)
    boundary = exact_boundary_velocity_means(mesh,fields)
    fixed = np.repeat((mesh.neighbor<0)[:,None],3,axis=1)
    if symmetry_extrusion:
        z = mesh.face_centers[:,2]
        planes = (mesh.neighbor<0) & ((z<1e-10)|(z>1-1e-10))
        fixed[planes,:2] = False
    solver = NativeHybridNavierStokes3D(mesh,fixed,boundary,
        viscosity=1/reynolds,body_force=0.,gauge_value=exact['pressure'][0],
        convection_order=convection_order,coupled_linear_method=coupled_linear_method)
    return mesh,solver,exact


def potential_fields(points):
    """Unforced genuinely 3D irrotational flow at any constant viscosity.

    U = grad(x+y+z+xyz), div U = lap U = curl U = 0,
    p = -|U|^2/2, so (U dot grad) U + grad p - nu lap U = 0.
    All three velocity components and all three coordinates are active.
    """
    x,y,z = np.asarray(points).T
    velocity = np.column_stack((1+y*z,1+x*z,1+x*y))
    gradient = np.zeros((len(x),3,3))
    gradient[:,0,1]=gradient[:,1,0]=z
    gradient[:,0,2]=gradient[:,2,0]=y
    gradient[:,1,2]=gradient[:,2,1]=x
    return dict(velocity=velocity,pressure=-.5*np.sum(velocity**2,axis=1),
                gradient=gradient,force=np.zeros_like(velocity))


def potential_solver(n, viscosity=.1):
    from .native_hybrid_pressure import pressure_mesh
    mesh=pressure_mesh(n);exact=potential_fields(mesh.cell_centers)
    boundary=exact_boundary_velocity_means(mesh,potential_fields)
    solver=NativeHybridNavierStokes3D(mesh,mesh.neighbor<0,boundary,
        viscosity=viscosity,body_force=0.,gauge_value=exact['pressure'][0],
        convection_order=2,coupled_linear_method='reused_lu')
    return mesh,solver,exact

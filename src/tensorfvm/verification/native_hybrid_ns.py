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

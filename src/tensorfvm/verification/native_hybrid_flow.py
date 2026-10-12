"""Analytic pressure-driven Stokes channel for the native hybrid backend."""
import numpy as np
from tensorfvm.native_mesh3d import NativeMesh3D
from tensorfvm.native_hybrid_stokes3d import NativeHybridStokes3D


def channel_mesh(n):
    if not isinstance(n, int) or n < 1:
        raise ValueError('positive integer mesh subdivision required')
    grid = np.linspace(0., 1., n+1)
    points = np.array([[x, y, z] for z in grid for y in grid for x in grid])
    index = lambda x, y, z: z*(n+1)**2+y*(n+1)+x
    cells = []
    for z in range(n):
        for y in range(n):
            for x in range(n):
                cells.append([index(x,y,z), index(x+1,y,z), index(x+1,y+1,z), index(x,y+1,z),
                              index(x,y,z+1), index(x+1,y,z+1), index(x+1,y+1,z+1), index(x,y+1,z+1)])
    return NativeMesh3D.from_cells(points, cells, ['hexahedron']*len(cells), triangulate_faces=True)


def channel_exact(points):
    x, y, _ = np.asarray(points).T
    velocity = np.zeros((len(x), 3)); velocity[:, 0] = y*(1-y)
    pressure = 2*(1-x)
    stress = np.zeros((len(x), 3, 3))
    stress[:, 0, 1] = stress[:, 1, 0] = 1-2*y
    return velocity, pressure, stress


def channel_solver(n):
    """No body force. Inlet/outlet prescribe total traction, y/z planes velocity.

    z planes impose the analytic profile of an infinitely wide channel; this
    is not a rectangular duct with no-slip sidewalls. Boundary face values are
    midpoint samples; their quadrature error is included in refinement error.
    """
    mesh = channel_mesh(n)
    x = mesh.face_centers[:, 0]
    boundary = mesh.neighbor < 0
    ends = boundary & ((x < 1e-10) | (x > 1-1e-10))
    velocity, pressure, stress = channel_exact(mesh.face_centers)
    normal = mesh.area_vectors/np.linalg.norm(mesh.area_vectors, axis=1)[:, None]
    traction = np.einsum('fij,fj->fi', stress, normal)-pressure[:, None]*normal
    solver = NativeHybridStokes3D(mesh, boundary & ~ends, velocity, boundary_traction=traction)
    return mesh, solver


def channel_metrics(mesh, result):
    velocity, pressure, _ = channel_exact(mesh.cell_centers)
    volume = mesh.cell_volumes
    relative = lambda error, exact: float(np.sqrt(np.sum(volume*error**2)/np.sum(volume*exact**2)))
    speed_error = relative(np.linalg.norm(result['velocity']-velocity, axis=1), np.linalg.norm(velocity, axis=1))
    pressure_error = relative(result['pressure']-pressure, pressure)
    wall = (mesh.neighbor < 0) & ((mesh.face_centers[:, 1] < 1e-10) | (mesh.face_centers[:, 1] > 1-1e-10))
    # Both channel walls exert integrated streamwise traction -1 each.
    wall_force = float(result['total_traction_integrated'][wall, 0].sum())
    return dict(cells=len(mesh.cells), faces=len(mesh.owner),
                velocity_relative_l2=speed_error, pressure_relative_l2=pressure_error,
                wall_streamwise_force=wall_force, wall_force_relative_error=abs(wall_force+2)/2,
                numerically_converged=result['numerically_converged'],
                iterations=len(result['history']),
                energy_balance_error=result['energy_balance_error'],
                interface_traction_defect_max=result['interface_traction_defect_max'],
                final_residuals=result['history'][-1])

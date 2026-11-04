"""Order-of-accuracy study for the native 3-D viscous (stress) operator.

The target is :class:`tensorfvm.native_viscous3d.NativeViscous3D`, the operator
that the collocated / RANS / SST solvers actually assemble.  Accuracy is probed
with a manufactured field on a family of refined hexahedral boxes, so every
error below is measured against an analytic divergence of the viscous stress
tensor and not against a reference solver.

Three independent measurements are provided:

``truncation``
    cell divergence of the assembled face traction, split into interior cells
    and Dirichlet-wall-adjacent cells;
``face_traction``
    per-unit-area traction error on internal and boundary faces, plus the
    wall-normal derivative error that sets wall shear;
``solution``
    the discrete Dirichlet diffusion problem ``div(tau(u)) = 0`` solved for the
    cell values, compared with the manufactured field.

Nothing here asserts a desired order: the reported numbers are whatever the
code produces, and the accompanying tests only pin what is measured.
"""
import numpy as np

from tensorfvm.native_mesh3d import NativeMesh3D
from tensorfvm.native_viscous3d import NativeViscous3D

PI = np.pi


def hex_box(n, warp=0.0, triangulate_faces=True):
    """Unit cube cut into ``n**3`` hexahedra; ``warp`` shifts vertices sinusoidally."""
    xs = np.linspace(0.0, 1.0, n + 1)
    x, y, z = np.meshgrid(xs, xs, xs, indexing='ij')
    if warp:
        x = x + warp*0.06*np.sin(3*y)*np.sin(2*z)
    vertices = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)
    span = (n + 1)*(n + 1)

    def vid(i, j, k):
        return i*span + j*(n + 1) + k

    cells = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                cells.append([vid(i, j, k), vid(i, j, k + 1), vid(i, j + 1, k + 1),
                              vid(i, j + 1, k), vid(i + 1, j, k), vid(i + 1, j, k + 1),
                              vid(i + 1, j + 1, k + 1), vid(i + 1, j + 1, k)])
    return NativeMesh3D.from_cells(vertices, cells, ['hexahedron']*len(cells),
                                   triangulate_faces=triangulate_faces)


def manufactured(points):
    """Smooth, non-polynomial velocity used as the manufactured field."""
    p = np.atleast_2d(np.asarray(points, dtype=float))
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    return np.stack([np.sin(PI*x)*np.cos(PI*y/2)*np.exp(z),
                     np.cos(PI*x/2)*np.sin(PI*y)*np.exp(-z),
                     np.sin(PI*x)*np.sin(PI*y)*z], axis=1)


def manufactured_gradient(points):
    """Analytic grad u of :func:`manufactured` as (n, 3, 3), ``[i, a, b] = du_a/dx_b``."""
    p = np.atleast_2d(np.asarray(points, dtype=float))
    x, y, z = p[:, 0], p[:, 1], np.atleast_1d(p[:, 2])
    s, c, e1, e2 = np.sin, np.cos, np.exp(z), np.exp(-z)
    g = np.zeros((len(x), 3, 3))
    g[:, 0, 0] = PI*c(PI*x)*c(PI*y/2)*e1
    g[:, 0, 1] = -PI/2*s(PI*x)*s(PI*y/2)*e1
    g[:, 0, 2] = s(PI*x)*c(PI*y/2)*e1
    g[:, 1, 0] = -PI/2*s(PI*x/2)*s(PI*y)*e2
    g[:, 1, 1] = PI*c(PI*x/2)*c(PI*y)*e2
    g[:, 1, 2] = -c(PI*x/2)*s(PI*y)*e2
    g[:, 2, 0] = PI*c(PI*x)*s(PI*y)*z
    g[:, 2, 1] = PI*s(PI*x)*c(PI*y)*z
    g[:, 2, 2] = s(PI*x)*s(PI*y)
    return g


def stress(points, viscosity=1.0):
    """Analytic symmetric deviatoric stress mu*(grad u + grad u^T - 2/3 tr I)."""
    g = manufactured_gradient(points)
    return viscosity*(g + g.transpose(0, 2, 1)
                      - 2/3*np.trace(g, axis1=1, axis2=2)[:, None, None]*np.eye(3))


def divergence_of_stress(points, viscosity=1.0):
    """Analytic div tau = mu*(lap u + grad(div u)/3), components [0..2]."""
    p = np.atleast_2d(np.asarray(points, dtype=float))
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    s, c, e1, e2 = np.sin, np.cos, np.exp(z), np.exp(-z)
    u = manufactured(p)
    lap = np.stack([(1 - 1.25*PI**2)*u[:, 0], (1 - 1.25*PI**2)*u[:, 1],
                    -2*PI**2*u[:, 2]], axis=1)
    grad_div = np.stack([
        -PI**2*s(PI*x)*c(PI*y/2)*e1 - PI**2/2*s(PI*x/2)*c(PI*y)*e2 + PI*c(PI*x)*s(PI*y),
        -PI**2/2*c(PI*x)*s(PI*y/2)*e1 - PI**2*c(PI*x/2)*s(PI*y)*e2 + PI*s(PI*x)*c(PI*y),
        PI*c(PI*x)*c(PI*y/2)*e1 - PI*c(PI*x/2)*c(PI*y)*e2], axis=1)
    return viscosity*(lap + grad_div/3.0)


def harmonic_velocity(points, k=PI):
    """Exact solution of the homogeneous Dirichlet problem ``div(tau(u)) = 0``.

    Built as ``u = curl A`` with a component-wise harmonic potential
    ``A = (e^{k(z-1/2)}cos(ky), e^{k(z-1/2)}sin(kx), e^{k(z-1/2)}cos(kx))``.
    Because every potential component is harmonic, ``lap u = 0`` and
    ``div u = 0`` hold identically, so ``div tau = mu*(lap u + grad(div u)/3)``
    vanishes pointwise: this field *is* the solution the Dirichlet solve is
    supposed to reproduce.  ``field_scale`` here is O(1)-O(20) on the unit box.
    """
    p = np.atleast_2d(np.asarray(points, dtype=float))
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    e = np.exp(k*(z - 0.5))
    return np.stack([-k*e*np.sin(k*x),
                     k*e*(np.cos(k*y) + np.sin(k*x)),
                     k*e*(np.cos(k*x) + np.sin(k*y))], axis=1)


def harmonic_gradient(points, k=PI):
    """Analytic ``grad u`` of :func:`harmonic_velocity`, ``[i, a, b] = du_a/dx_b``."""
    p = np.atleast_2d(np.asarray(points, dtype=float))
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    e = k*np.exp(k*(z - 0.5))
    g = np.zeros((len(x), 3, 3))
    g[:, 0, 0] = -k*e*np.cos(k*x)
    g[:, 0, 2] = -k*e*np.sin(k*x)
    g[:, 1, 0] = k*e*np.cos(k*x)
    g[:, 1, 1] = -k*e*np.sin(k*y)
    g[:, 1, 2] = k*e*(np.cos(k*y) + np.sin(k*x))
    g[:, 2, 0] = -k*e*np.sin(k*x)
    g[:, 2, 1] = k*e*np.cos(k*y)
    g[:, 2, 2] = k*e*(np.cos(k*x) + np.sin(k*y))
    return g


def _wall_mask(mesh, n, layers=1.5):
    """Cells whose centre lies within ``layers`` cells of any box face."""
    half = layers/n
    c = mesh.cell_centers
    return np.any((c < half) | (c > 1.0 - half), axis=1)


def _operator(mesh, viscosity):
    return NativeViscous3D(mesh, mesh.neighbor < 0, viscosity)


def _boundary_data(op, viscosity=1.0):
    data = np.zeros((op.F, 3))
    data[op.dirichlet] = manufactured(op.mesh.face_centers[op.dirichlet])
    return data


def truncation_study(n, warp=0.0, viscosity=1.3):
    """Cell divergence error of the assembled traction, interior vs wall cells."""
    mesh = hex_box(n, warp)
    op = _operator(mesh, viscosity)
    data = _boundary_data(op)
    flux = op.momentum_flux(manufactured(mesh.cell_centers), dirichlet_values=data)
    discrete = op.op.integrate(flux)/mesh.cell_volumes[:, None]
    exact = divergence_of_stress(mesh.cell_centers, viscosity)
    err = np.linalg.norm(discrete - exact, axis=1)
    wall = _wall_mask(mesh, n)
    # Which faces carry the cell-divergence deficit?  The wall-cell error is
    # attributed to the Dirichlet faces alone if ``l2_wall_wall_faces_only``
    # matches ``l2_wall``: that is the signature of a wall-flux closure error,
    # not of an interior operator error.
    deficit = flux - np.einsum('fij,fj->fi', stress(mesh.face_centers, viscosity),
                               mesh.area_vectors)
    owner_share = np.zeros((op.N, 3))
    np.add.at(owner_share, mesh.owner, deficit)
    inner = np.where(op.internal)[0]
    np.add.at(owner_share, mesh.neighbor[inner], -deficit[inner])
    wall_only = np.zeros((op.N, 3))
    np.add.at(wall_only, mesh.owner[op.dirichlet], deficit[op.dirichlet])
    per_volume = mesh.cell_volumes[:, None]
    wall_flux_err = np.linalg.norm(wall_only/per_volume, axis=1)
    return dict(n=n, h=1.0/n, warp=warp, cells=int(len(mesh.cells)),
                faces=int(len(mesh.owner)), wall_cells=int(wall.sum()),
                field_scale=float(np.linalg.norm(exact, axis=1).max()),
                l2_global=float(np.sqrt(np.mean(err**2))),
                linf_global=float(err.max()),
                l2_interior=float(np.sqrt(np.mean(err[~wall]**2))),
                linf_interior=float(err[~wall].max()),
                l2_wall=float(np.sqrt(np.mean(err[wall]**2))),
                linf_wall=float(err[wall].max()),
                l2_wall_wall_faces_only=float(np.sqrt(np.mean(wall_flux_err[wall]**2))),
                l2_interior_wall_faces_only=float(
                    np.sqrt(np.mean(wall_flux_err[~wall]**2))))


def face_traction_study(n, warp=0.0, viscosity=1.3):
    """Per-unit-area traction error on internal and boundary faces, plus wall du/dn.

    The full face gradient is the assembled part (``face_gradient_matrices``) plus
    the boundary-data part (``face_gradient_source``); using only the source part
    would compare a perturbation term with the whole gradient.
    """
    mesh = hex_box(n, warp)
    op = _operator(mesh, viscosity)
    data = _boundary_data(op)
    values = manufactured(mesh.cell_centers)
    flux = op.momentum_flux(values, dirichlet_values=data)
    exact_flux = np.einsum('fij,fj->fi', stress(mesh.face_centers, viscosity), mesh.area_vectors)
    per_area = np.linalg.norm(flux - exact_flux, axis=1)/op.area
    exact_per_area = np.linalg.norm(exact_flux, axis=1)/op.area
    gradient = np.empty((op.F, 3, 3))
    for a in range(3):
        source = op.face_gradient_source(0.0, data[:, a])
        for k in range(3):
            gradient[:, a, k] = op.face_gradient_matrices[k] @ values[:, a] + source[:, k]
    exact_normal = np.einsum('fab,fb->fa', manufactured_gradient(mesh.face_centers), op.normal)
    normal_err = np.linalg.norm(np.einsum('fab,fb->fa', gradient, op.normal) - exact_normal, axis=1)
    inner, wall = op.internal, op.boundary
    return dict(n=n, h=1.0/n, warp=warp,
                linf_internal_per_area=float(per_area[inner].max()),
                rel_internal_per_area=float(per_area[inner].max()/exact_per_area[inner].max()),
                linf_boundary_per_area=float(per_area[wall].max()),
                rel_boundary_per_area=float(per_area[wall].max()/exact_per_area[wall].max()),
                wall_normal_derivative_error=float(normal_err[wall].max()),
                rel_wall_normal_derivative=float(normal_err[wall].max()
                                                / np.linalg.norm(exact_normal[wall], axis=1).max()),
                interior_normal_derivative_error=float(normal_err[inner].max()))


def solution_study(n, warp=0.0, viscosity=1.3):
    """Solve div(tau(u))=0 with exact Dirichlet data and compare with the field."""
    from scipy.sparse.linalg import spsolve

    mesh = hex_box(n, warp)
    op = _operator(mesh, viscosity)
    data = _boundary_data(op)
    rhs = -(op.vector_divergence_matrix @
            op.momentum_flux_source(dirichlet_values=data).ravel())
    solved = spsolve(op.momentum_diffusion_matrix.tocsc(), rhs)
    values = solved.reshape(len(mesh.cells), 3)
    exact = manufactured(mesh.cell_centers)
    err = np.linalg.norm(values - exact, axis=1)
    wall = _wall_mask(mesh, n)
    return dict(n=n, h=1.0/n, warp=warp,
                l2_global=float(np.sqrt(np.mean(err**2))),
                linf_global=float(err.max()),
                l2_interior=float(np.sqrt(np.mean(err[~wall]**2))),
                field_scale=float(np.linalg.norm(exact, axis=1).max()),
                residual=float(np.linalg.norm(
                    op.momentum_diffusion_matrix @ solved - rhs)
                    / max(np.linalg.norm(rhs), 1e-30)))


def manufactured_solution_study(n, warp=0.0, viscosity=1.3):
    """Dirichlet diffusion solve of the *consistent* manufactured problem.

    ``solution_study`` solves ``div(tau(u)) = 0`` and compares the result with
    :func:`manufactured`, which does not satisfy that homogeneous equation
    (``divergence_of_stress`` is O(1) everywhere).  That comparison therefore
    measures the distance between two different fields, not a discretisation
    error.  Here the analytic divergence of the stress of the manufactured
    field is imposed as the cell source, so the exact discrete solution really
    is ``manufactured`` and the error below is a discretisation error.
    """
    from scipy.sparse.linalg import spsolve

    mesh = hex_box(n, warp)
    op = _operator(mesh, viscosity)
    data = _boundary_data(op)
    rhs = -(op.vector_divergence_matrix @
            op.momentum_flux_source(dirichlet_values=data).ravel())
    source = divergence_of_stress(mesh.cell_centers, viscosity)*mesh.cell_volumes[:, None]
    rhs = rhs + source.ravel()
    solved = spsolve(op.momentum_diffusion_matrix.tocsc(), rhs)
    values = solved.reshape(len(mesh.cells), 3)
    exact = manufactured(mesh.cell_centers)
    err = np.linalg.norm(values - exact, axis=1)
    wall = _wall_mask(mesh, n)
    return dict(n=n, h=1.0/n, warp=warp,
                l2_global=float(np.sqrt(np.mean(err**2))),
                linf_global=float(err.max()),
                l2_interior=float(np.sqrt(np.mean(err[~wall]**2))),
                l2_wall=float(np.sqrt(np.mean(err[wall]**2))),
                field_scale=float(np.linalg.norm(exact, axis=1).max()),
                residual=float(np.linalg.norm(op.momentum_diffusion_matrix @ solved - rhs)
                               / max(np.linalg.norm(rhs), 1e-30)))


def harmonic_solution_study(n, warp=0.0, viscosity=1.3, k=PI):
    """Dirichlet solve against a field that *is* an exact solution.

    :func:`solution_study` feeds the manufactured field, which satisfies
    ``div(tau(u)) = mu*lap u != 0``; no refinement can make the discrete
    solution approach it.  Here the Dirichlet data come from
    :func:`harmonic_velocity`, for which ``div(tau) = 0`` identically, so the
    measured error is a genuine discretisation error and its slope is the
    solution order of the operator.
    """
    from scipy.sparse.linalg import spsolve

    mesh = hex_box(n, warp)
    op = _operator(mesh, viscosity)
    data = np.zeros((op.F, 3))
    data[op.dirichlet] = harmonic_velocity(mesh.face_centers[op.dirichlet], k)
    rhs = -(op.vector_divergence_matrix @
            op.momentum_flux_source(dirichlet_values=data).ravel())
    solved = spsolve(op.momentum_diffusion_matrix.tocsc(), rhs)
    values = solved.reshape(len(mesh.cells), 3)
    exact = harmonic_velocity(mesh.cell_centers, k)
    err = np.linalg.norm(values - exact, axis=1)
    wall = _wall_mask(mesh, n)
    return dict(n=n, h=1.0/n, warp=warp,
                l2_global=float(np.sqrt(np.mean(err**2))),
                linf_global=float(err.max()),
                l2_interior=float(np.sqrt(np.mean(err[~wall]**2))),
                linf_interior=float(err[~wall].max()),
                l2_wall=float(np.sqrt(np.mean(err[wall]**2))),
                field_scale=float(np.linalg.norm(exact, axis=1).max()),
                residual=float(np.linalg.norm(
                    op.momentum_diffusion_matrix @ solved - rhs)
                    / max(np.linalg.norm(rhs), 1e-30)))


def wall_second_derivative(face_centers, normals, step=1e-4):
    """Analytic d2u_a/dn2 at the given points along the supplied unit normals."""
    normals = np.asarray(normals, dtype=float)
    points = np.asarray(face_centers, dtype=float)
    second = np.zeros((len(points), 3))
    for a in range(3):
        plus = np.einsum('fab,fb->fa', manufactured_gradient(points + step*normals), normals)
        centre = np.einsum('fab,fb->fa', manufactured_gradient(points), normals)
        minus = np.einsum('fab,fb->fa', manufactured_gradient(points - step*normals), normals)
        second[:, a] = (plus[:, a] - 2*centre[:, a] + minus[:, a])/step**2
    return second


def wall_error_model(n, warp=0.0, viscosity=1.3):
    """Leading-order model of the wall-cell divergence error.

    The Dirichlet face reconstruction supplies ``2*(u_D - u_P)/|d|`` as the
    normal derivative, i.e. the central difference over the half-cell interval
    *evaluated at the wall*; its leading error is ``(|d|/2) d2u/dn2``.  Per face
    that error contributes ``mu*err*area`` to the outflux, and the cell-volume
    normalisation of a uniform box cancels it exactly, so the wall-cell
    divergence error should approach ``mu*|d|/2*|d2u/dn2|`` with no ``h`` decay.
    """
    mesh = hex_box(n, warp)
    op = _operator(mesh, viscosity)
    wall = op.dirichlet
    distance = np.linalg.norm(mesh.face_centers[wall] - mesh.cell_centers[mesh.owner[wall]], axis=1)
    second = wall_second_derivative(mesh.face_centers[wall], op.normal[wall])
    predicted = viscosity*distance[:, None]/2*second
    return dict(n=n, h=1.0/n, warp=warp, mean_distance=float(distance.mean()),
                predicted_l2=float(np.sqrt(np.mean(np.sum(predicted**2, axis=1)))),
                predicted_linf=float(np.linalg.norm(predicted, axis=1).max()))


def orders(rows, key):
    """Successive convergence orders of ``key`` over a refined family."""
    values = [row[key] for row in rows]
    return [float(np.log2(values[i]/values[i + 1])) for i in range(len(values) - 1)]
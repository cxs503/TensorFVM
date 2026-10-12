"""Two-cell viscous passivity diagnostic, independent of any flow benchmark."""
import numpy as np

from tensorfvm.native_mesh3d import NativeMesh3D
from tensorfvm.native_viscous3d import NativeViscous3D


def two_tetrahedra(skew=1.):
    """Deform two valid tetrahedra sharing z=0; skew=1 is the known counterexample."""
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0],
                       [2*skew, 2*skew, 1-.75*skew],
                       [-skew, skew, -1-skew]], dtype=float)
    return NativeMesh3D.from_cells(points, [[0, 1, 2, 3], [0, 2, 1, 4]], ['tetra']*2)


def energy_diagnostic(mesh):
    """Return matrix and separately assembled face-stress energy balances.

    All boundaries are at rest, mu=rho=1; no pressure or turbulence is used.
    The prescribed cell velocity is (1,0,0) in cell 0 and zero elsewhere.
    This velocity is not constrained to be incompressible.
    """
    viscous = NativeViscous3D(mesh, mesh.neighbor < 0, 1.)
    velocity = np.zeros((len(mesh.cells), 3))
    velocity[0, 0] = 1.
    matrix = -viscous.momentum_diffusion_matrix.toarray()
    # Assemble gradients and stress without using momentum_flux_matrix.
    gradient = np.stack([g @ velocity for g in viscous.face_gradient_matrices], axis=2)
    stress = gradient + gradient.transpose(0, 2, 1)
    stress -= (2/3)*np.trace(gradient, axis1=1, axis2=2)[:, None, None]*np.eye(3)
    traction = np.einsum('fij,fj->fi', stress, mesh.area_vectors)
    integrated = viscous.op.integrate(traction)
    rate = float(np.sum(velocity*integrated))
    form = float(velocity.ravel() @ matrix @ velocity.ravel())
    np.testing.assert_allclose(rate, -form, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(traction, viscous.momentum_flux(velocity), atol=1e-12)
    cosine = np.sum(mesh.area_vectors*viscous.op.d, axis=1)/(
        np.linalg.norm(mesh.area_vectors, axis=1)*np.linalg.norm(viscous.op.d, axis=1))
    scalar = -viscous.scalar_diffusion_matrix.toarray()
    return dict(cells=len(mesh.cells), faces=len(mesh.owner),
                vertices=mesh.vertices.tolist(), connectivity=[list(c) for c in mesh.cells],
                cell_volumes=mesh.cell_volumes.tolist(), velocity=velocity.tolist(),
                maximum_nonorthogonality_degrees=float(np.degrees(np.arccos(np.clip(cosine, -1, 1))).max()),
                diffusion_energy_form=form, kinetic_energy_rate=rate,
                energy_balance_error=abs(rate+form),
                symmetric_minimum_eigenvalue=float(np.linalg.eigvalsh((matrix+matrix.T)/2)[0]),
                scalar_diffusion_diagonal_min=float(scalar.diagonal().min()),
                momentum_matrix=matrix.tolist(), scalar_matrix=scalar.tolist(),
                face_traction=traction.tolist(), passed=bool(rate <= 1e-12))

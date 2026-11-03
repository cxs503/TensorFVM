"""Two-cell viscous passivity diagnostic, independent of any flow benchmark.

The implicit viscous operator used by :mod:`tensorfvm.native_collocated3d` and
:mod:`tensorfvm.native_rans3d` is ``L = -momentum_diffusion_matrix``.  A
cell-centred finite-volume discretisation of ``-div(mu (grad u + grad u^T))``
with homogeneous Dirichlet data must satisfy the discrete passivity (kinetic
energy dissipation) inequality ``x^T L x <= 0`` for *every* velocity vector
``x``; otherwise the implicit step can inject kinetic energy.

``energy_diagnostic`` measures that inequality directly (assembled face
traction, energy identity, symmetric spectrum).  ``passivity_audit`` reports the
signed spectra, ``correction_family_scan`` scans the deferred nonorthogonal
correction factor, ``two_point_control`` builds the strictly passive two-point
reference, and ``random_shear_survey`` samples sheared tetrahedral pairs so the
passivity boundary can be expressed as a function of mesh nonorthogonality.
"""
import numpy as np
from scipy.sparse import diags

from tensorfvm.native_mesh3d import NativeMesh3D
from tensorfvm.native_viscous3d import NativeViscous3D

#: Correction directions accepted by :class:`NativeViscous3D` as strings.  Any
#: other numeric value is used as a raw multiplier ``gamma`` of the area vector,
#: i.e. ``correction_direction = gamma * s`` (over-relaxed is ``1/(s.d)``).
CORRECTIONS = ('over-relaxed', 'minimum', 'none')


def two_tetrahedra(skew=1.):
    """Deform two valid tetrahedra sharing z=0; skew=1 is the known counterexample."""
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0],
                       [2*skew, 2*skew, 1-.75*skew],
                       [-skew, skew, -1-skew]], dtype=float)
    return NativeMesh3D.from_cells(points, [[0, 1, 2, 3], [0, 2, 1, 4]], ['tetra']*2)


def energy_diagnostic(mesh, correction='over-relaxed'):
    """Return matrix and separately assembled face-stress energy balances.

    All boundaries are at rest, mu=rho=1; no pressure or turbulence is used.
    The prescribed cell velocity is (1,0,0) in cell 0 and zero elsewhere.
    This velocity is not constrained to be incompressible.
    """
    viscous = NativeViscous3D(mesh, mesh.neighbor < 0, 1., correction=correction)
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


def _signed_spectrum(dissipation):
    """Classify the symmetric spectrum of ``dissipation = -L``.

    The discrete energy rate of a velocity vector ``x`` is ``x^T L x`` with
    ``L`` the implicit diffusion matrix, so passivity (no kinetic-energy
    creation) is exactly ``dissipation = -L`` positive semidefinite, i.e.
    ``lambda_min >= 0``.  ``creation_rate`` is the worst-case energy injection
    per unit ``|x|^2`` (``max(0, -lambda_min)``).
    """
    symmetric = (dissipation+dissipation.T)/2
    eigenvalues = np.linalg.eigvalsh(symmetric)
    scale = max(1., float(np.abs(eigenvalues).max()))
    creation = float(max(0., -eigenvalues[0]))
    dissipation_rate = float(max(0., eigenvalues[-1]))
    total = creation+dissipation_rate
    return dict(modes=int(eigenvalues.size), lambda_max=float(eigenvalues[-1]),
                lambda_min=float(eigenvalues[0]),
                creation_rate=creation, dissipation_rate=dissipation_rate,
                creation_fraction=float(creation/total) if total > 0 else 0.,
                wrong_sign_modes=int(np.count_nonzero(eigenvalues < -1e-12*scale)),
                passive=bool(eigenvalues[0] >= -1e-12*scale))


def passivity_audit(mesh, correction='over-relaxed', deviatoric=True, viscosity=1.):
    """Signed spectra of both implicit viscous operators on ``mesh``.

    ``scalar`` is the shared-face scalar Laplacian (the Rhie--Chow mobility
    reference), ``vector`` the full symmetric stress momentum operator.
    """
    viscous = NativeViscous3D(mesh, mesh.neighbor < 0, viscosity,
                             deviatoric=deviatoric, correction=correction)
    scalar = -viscous.scalar_diffusion_matrix.toarray()
    vector = -viscous.momentum_diffusion_matrix.toarray()
    cosine = np.sum(mesh.area_vectors*viscous.op.d, axis=1)/(
        np.linalg.norm(mesh.area_vectors, axis=1)*np.linalg.norm(viscous.op.d, axis=1))
    return dict(correction=str(correction), deviatoric=bool(deviatoric),
                cells=len(mesh.cells), faces=len(mesh.owner),
                maximum_nonorthogonality_degrees=float(
                    np.degrees(np.arccos(np.clip(cosine, -1, 1))).max()),
                scalar=_signed_spectrum(scalar), vector=_signed_spectrum(vector),
                passive=bool(_signed_spectrum(scalar)['passive']
                             and _signed_spectrum(vector)['passive']),
                scalar_matrix=scalar.tolist(), momentum_matrix=vector.tolist())


def correction_family_scan(mesh, scales=None, deviatoric=True):
    """Scan ``correction_direction = gamma * s`` for any passively signable gamma.

    The deferred-correction family is one-parameter: ``gamma -> 0`` drops the
    skewness correction, ``gamma = 1/(s.d)`` is the over-relaxed minimum form and
    ``gamma = (s.d)/(|d|^2 |s|^2)`` the orthogonal (minimum) correction.  If no
    sample is passive the energy defect is structural for this scheme family and
    cannot be removed by re-weighting the correction.
    """
    if scales is None:
        viscous = NativeViscous3D(mesh, mesh.neighbor < 0, 1.)
        projection = np.einsum('ij,ij->i', mesh.area_vectors, viscous.op.d)
        scales = tuple(sorted(set(
            [0.] + [float(x) for x in np.geomspace(1e-3, 4., 9)]
            + [float(1/projection.max()), float(1/projection.min())])))
    rows = []
    for gamma in scales:
        audit = passivity_audit(mesh, correction=float(gamma), deviatoric=deviatoric)
        rows.append(dict(gamma=float(gamma),
                         scalar_lambda_max=audit['scalar']['lambda_max'],
                         scalar_creation_fraction=audit['scalar']['creation_fraction'],
                         vector_lambda_max=audit['vector']['lambda_max'],
                         vector_creation_fraction=audit['vector']['creation_fraction'],
                         passive=audit['passive']))
    return dict(scales=[row['gamma'] for row in rows], rows=rows,
                any_passive=bool(any(row['passive'] for row in rows)))


def affine_consistency(mesh, correction='over-relaxed', viscosity=1., affine_gradient=(.7, -.3, .2),
                       deviatoric=True):
    """Relative error of the assembled face traction against the exact affine one.

    For an affine velocity the traction is exactly ``mu * (s . (G + G^T))`` with
    ``G`` the constant gradient, and the deferred nonorthogonal correction
    vanishes.  A scheme family that keeps this error at round-off is
    second-order consistent; the two-point reference is not.
    """
    viscous = NativeViscous3D(mesh, mesh.neighbor < 0, viscosity,
                             deviatoric=deviatoric, correction=correction)
    gradient = np.outer(np.asarray(affine_gradient, dtype=float), [1., 1., 1.])
    cells = mesh.cell_centers @ gradient.T
    faces = mesh.face_centers @ gradient.T
    stress = gradient+gradient.T
    if deviatoric:
        stress = stress-(2/3)*np.trace(gradient)*np.eye(3)
    exact = viscosity*np.einsum('aj,fj->fa', stress, mesh.area_vectors)
    computed = viscous.momentum_flux(cells, 0., faces)
    return dict(correction=str(correction),
                maximum_absolute_error=float(np.abs(computed-exact).max()),
                maximum_relative_error=float(np.abs(computed-exact).max()
                                             / max(np.abs(exact).max(), 1e-300)))


def two_point_control(mesh, viscosity=1., affine_gradient=(.7, -.3, .2)):
    """Strictly passive two-point reference and its affine-flux inconsistency.

    ``coef_f = mu (s.d)/|d|^2`` makes the operator an M-matrix graph Laplacian,
    so the discrete dissipation inequality holds by construction.  The price is
    that the face flux no longer reproduces the exact affine traction
    ``mu s.g``: the normal derivative is projected on ``d`` instead of ``s``.
    """
    viscous = NativeViscous3D(mesh, mesh.neighbor < 0, viscosity)
    area, d = mesh.area_vectors, viscous.op.d
    projection = np.einsum('ij,ij->i', area, d)
    coefficient = viscosity*projection/np.einsum('ij,ij->i', d, d)
    matrix = -(viscous.divergence_matrix @ diags(coefficient) @ viscous.delta_matrix).toarray()
    gradient = np.asarray(affine_gradient, dtype=float)
    exact = viscosity*(area @ gradient)
    two_point = coefficient*(d @ gradient)
    scale = float(np.abs(exact).max())
    return dict(spectrum=_signed_spectrum(matrix),
                affine_flux_max_relative_error=float(np.abs(two_point-exact).max()/scale),
                coefficient_min_max=(float(coefficient.min()), float(coefficient.max())),
                passive=bool(_signed_spectrum(matrix)['passive']))


def skew_sweep(skews=None, corrections=('over-relaxed', 'minimum')):
    """Deterministic passivity boundary of the two-tetrahedra family."""
    if skews is None:
        skews = tuple(float(x) for x in np.linspace(0., 1.25, 11))
    rows = []
    for skew in skews:
        try:
            mesh = two_tetrahedra(skew)
        except ValueError:
            rows.append(dict(skew=float(skew), valid=False, nonorthogonality_degrees=None,
                             corrections={name: None for name in corrections}))
            continue
        audit = {name: passivity_audit(mesh, name) for name in corrections}
        rows.append(dict(skew=float(skew), valid=True,
                         nonorthogonality_degrees=audit[corrections[0]]['maximum_nonorthogonality_degrees'],
                         corrections={name: dict(audit[name]['vector'], scalar=audit[name]['scalar'])
                                      for name in corrections}))
    return dict(skews=[row['skew'] for row in rows], corrections=list(corrections), rows=rows,
                valid=[row['skew'] for row in rows if row['valid']])


def passivity_boundary(correction='over-relaxed', lo=0., hi=1.25, iterations=20):
    """Bisect the skew at which the first energy-creation mode appears.

    Returns the largest passive skew and its nonorthogonality angle together
    with the first skew that creates energy, so the validity range can be quoted
    as an angle instead of a family parameter.
    """
    def creation(skew):
        return passivity_audit(two_tetrahedra(skew), correction)['vector']['creation_rate']
    if creation(lo) > 0:
        raise ValueError('reference skew already creates energy')
    if creation(hi) <= 0:
        return dict(correction=str(correction), passive_skew=float(hi), creation_skew=None,
                    passive_degrees=float(passivity_audit(two_tetrahedra(hi), correction)[
                        'maximum_nonorthogonality_degrees']), creation_degrees=None)
    low, high = lo, hi
    for _ in range(iterations):
        middle = (low+high)/2
        if creation(middle) > 0:
            high = middle
        else:
            low = middle
    return dict(correction=str(correction), passive_skew=float(low), creation_skew=float(high),
                passive_degrees=float(passivity_audit(two_tetrahedra(low), correction)[
                    'maximum_nonorthogonality_degrees']),
                creation_degrees=float(passivity_audit(two_tetrahedra(high), correction)[
                    'maximum_nonorthogonality_degrees']))


def random_shear_survey(levels=(0.1, 0.2, 0.3, 0.4, 0.5, 0.7), samples=24, seed=20261103,
                        base_skew=0.6):
    """Sheared tetrahedral pairs: passivity versus nonorthogonality, by level.

    Shears ``I + M`` with ``M`` uniform on ``[-level, level]`` off the diagonal
    are applied to the ``base_skew`` reference pair.  Invalid meshes
    (non-positive projection or volume) are discarded, so every retained sample
    is a legal native mesh; ``levels`` therefore spans the transition.
    """
    generator = np.random.default_rng(seed)
    points = two_tetrahedra(base_skew).vertices
    rows = []
    for level in levels:
        for _ in range(samples):
            matrix = np.eye(3)
            matrix[~np.eye(3, dtype=bool)] = generator.uniform(-level, level, 6)
            try:
                mesh = NativeMesh3D.from_cells(points @ matrix.T, [[0, 1, 2, 3], [0, 2, 1, 4]],
                                               ['tetra']*2)
                over = passivity_audit(mesh, 'over-relaxed')
                minimum = passivity_audit(mesh, 'minimum')
            except ValueError:
                continue
            rows.append(dict(shear_level=float(level),
                             nonorthogonality_degrees=over['maximum_nonorthogonality_degrees'],
                             over_relaxed=over['vector'], minimum=minimum['vector'],
                             scalar_over_relaxed=over['scalar'], scalar_minimum=minimum['scalar']))
    angle = np.array([row['nonorthogonality_degrees'] for row in rows])
    over = np.array([row['over_relaxed']['creation_rate'] for row in rows])
    minimum = np.array([row['minimum']['creation_rate'] for row in rows])
    return dict(samples=len(rows), levels=[float(x) for x in levels], rows=rows,
                max_nonorthogonality_degrees=float(angle.max()) if len(rows) else 0.,
                over_relaxed_failures=int(np.count_nonzero(over > 0)),
                minimum_failures=int(np.count_nonzero(minimum > 0)),
                over_relaxed_max_creation_rate=float(over.max()) if len(rows) else 0.,
                minimum_max_creation_rate=float(minimum.max()) if len(rows) else 0.,
                threshold_degrees_estimate=_passivity_threshold(angle, over),
                minimum_threshold_degrees_estimate=_passivity_threshold(angle, minimum),
                angle_creation_correlation=(float(np.corrcoef(angle, over)[0, 1])
                                            if len(rows) > 1 and over.std() > 0 else None))


def _passivity_threshold(angle, creation):
    """Largest sampled angle below which every sample is passive (None if none)."""
    if not len(angle):
        return None
    passive = creation <= 0
    if passive.all():
        return float(angle.max())
    first_failure = float(angle[~passive].min())
    below = angle[passive & (angle < first_failure)]
    return float(below.max()) if below.size else None
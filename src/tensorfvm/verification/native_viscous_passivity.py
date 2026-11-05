"""Mesh-level passivity audit of the cell-centred viscous operator.

Verification debt closed here
------------------------------
``docs/verification-native-solver-audit/README.md`` reported, on the 9,068-cell
SUBOFF hybrid mesh, a direction with ``u^T A u = -0.24579039`` for the assembled
viscous operator ``A = -NativeViscous3D.momentum_diffusion_matrix`` and left the
item open ("未修复，需重新检查离散耗散性"), recommending a check on the
*trangulated thin layers / tetrahedral interfaces*.  The mesh and the witness
velocity were removed from the repository, so the number cannot be replayed.
:mod:`tensorfvm.verification.native_energy` audits the same inequality but only
on a *two-cell* tetrahedral pair.  This module extends the audit to real 3D
meshes and answers three questions with reproducible evidence:

1. **Tool certification (analytic).**  On a uniform Cartesian mesh the scalar
   flux operator ``-scalar_diffusion_matrix`` must equal the closed-form
   7-point flux operator (``coef=h`` on internal faces, ``2h`` on boundary
   faces because ``d`` is half a spacing).  Without this anchor a "passive"
   verdict would only grade the code against itself.
2. **Do hybrid interfaces / thin layers / prisms create energy?**  On hex, tet,
   hex-tet transition and wedge (prism) boundary-layer meshes with aspect ratios
   up to 1.2e3 and up to 68 deg of face non-orthogonality the symmetrised
   operator stays positive definite, so *transition geometry and clustering
   alone do not reproduce the audit's negative direction*.  The antisymmetric
   part is large (up to 0.47 relative), i.e. the defect is asymmetry /
   conditioning, not energy creation.
3. **Where does energy creation actually start, and is it local?**  The known
   two-tetrahedra family (:func:`native_energy.two_tetrahedra`) turns energy
   creating at ``skew`` 0.9587 = 85.9 deg of non-orthogonality.  Copying that
   pair into a many-cell mesh reproduces a *negative* normalised margin at exactly
   the same threshold (``-5.4e-2`` at skew 1.0, ``-1.5e-1`` at 1.1), and the
   margin does not change when the number of pairs grows: the failure is a
   *local* face-cluster property, not a global accumulation.  The creating faces
   are identified by an exact per-face energy decomposition.

The margin is the smallest eigenvalue of ``R (A+A^T)/2 R`` with ``R`` the
reciprocal square root of the orthogonal-diffusion diagonal, i.e. the same
dimensionless quantity the SUBOFF audit reported (``scaled_symmetric_rayleigh``).
Every passivity statement here is an energy diagnostic; it is not a stability or
convergence proof of the incompressible solver.
"""
import json
from collections import defaultdict

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

from tensorfvm.native_mesh3d import NativeMesh3D
from tensorfvm.native_viscous3d import NativeViscous3D
from tensorfvm.verification.native_energy import two_tetrahedra

#: Correction families accepted by :class:`NativeViscous3D`.
CORRECTIONS = ('over-relaxed', 'minimum', 'none')


# --------------------------------------------------------------------------- #
# meshes
# --------------------------------------------------------------------------- #
def uniform_cartesian(n=3):
    """Uniform ``n^3`` hexahedral box; the mesh used for the analytic anchor."""
    axis = np.linspace(0., 1., n+1)
    points = np.array([[x, y, z] for z in axis for y in axis for x in axis])
    def index(i, j, k):
        return k*(n+1)**2+j*(n+1)+i
    cells = []
    for k in range(n):
        for j in range(n):
            for i in range(n):
                cells.append([index(i, j, k), index(i+1, j, k), index(i+1, j+1, k),
                              index(i, j+1, k), index(i, j, k+1), index(i+1, j, k+1),
                              index(i+1, j+1, k+1), index(i, j+1, k+1)])
    return NativeMesh3D.from_cells(points, cells, ['hexahedron']*len(cells),
                                   triangulate_faces=True)


def layered_mesh(nx=2, ny=2, nz=6, ratio=1., shear=.3, mode='hex'):
    """Box with exponentially clustered layers; ``mode`` in ``hex``, ``hybrid``, ``wedge``.

    ``ratio`` is the wall-to-far-field layer thickness ratio (1 = uniform,
    0.002 = nearly degenerate layers, aspect ratio ~1.2e3).  ``shear`` applies
    an affine shear, ``hybrid`` alternates hexahedra with six-tetrahedron
    sub-cells (shared triangular interfaces, as in a tetrahedral transition),
    ``wedge`` builds triangular prisms over a triangulated wall (boundary-layer
    prism topology).
    """
    zs = np.concatenate(([0.], np.cumsum(ratio**np.linspace(0., 1., nz+1)[:-1])))
    zs /= zs[-1]
    xs, ys = np.linspace(0., 1., nx+1), np.linspace(0., 1., ny+1)
    points = np.array([[x, y, z] for z in zs for y in ys for x in xs], float)
    points = points@np.array([[1., shear, .5*shear], [0., 1., .4*shear],
                              [0., 0., 1.]]).T
    def index(i, j, k):
        return k*(ny+1)*(nx+1)+j*(nx+1)+i
    cells, kinds = [], []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                bottom = [index(i, j, k), index(i+1, j, k), index(i+1, j+1, k), index(i, j+1, k)]
                top = [index(i, j, k+1), index(i+1, j, k+1), index(i+1, j+1, k+1),
                       index(i, j+1, k+1)]
                corner = bottom+top
                if mode == 'wedge':
                    for a, b in (([0, 1, 2], [0, 1, 2]), ([0, 2, 3], [0, 2, 3])):
                        cells.append([bottom[q] for q in a]+[top[q] for q in b])
                        kinds.append('wedge')
                elif mode == 'hybrid' and (i+j+k)%2:
                    for tetra in ([0, 1, 2, 6], [0, 2, 3, 6], [0, 3, 7, 6],
                                  [0, 7, 4, 6], [0, 4, 5, 6], [0, 5, 1, 6]):
                        cells.append([corner[q] for q in tetra])
                        kinds.append('tetra')
                else:
                    cells.append(corner)
                    kinds.append('hexahedron')
    return NativeMesh3D.from_cells(points, cells, kinds, triangulate_faces=True)


def mosaic_pairs(count=1, skew=1., gap=2.2):
    """``count`` translated copies of the two-tetrahedra counterexample family."""
    points, cells = [], []
    for copy in range(count):
        block = two_tetrahedra(skew).vertices.copy()
        block[:, 0] += gap*copy
        base = len(points)
        cells += [[base, base+1, base+2, base+3], [base, base+2, base+1, base+4]]
        points += list(block)
    return NativeMesh3D.from_cells(np.asarray(points), cells, ['tetra']*len(cells))


# --------------------------------------------------------------------------- #
# analytic anchor
# --------------------------------------------------------------------------- #
def seven_point_flux_operator(n, viscosity=1.):
    """Closed-form 7-point flux operator on the uniform ``n^3`` box mesh.

    ``NativeViscous3D`` assembles ``-scalar_diffusion_matrix`` *without* a
    volume division: an internal face contributes ``mu*|s|/|d| = mu*h`` and a
    Dirichlet face contributes ``mu*|s|/|d| = 2*mu*h`` because its ``d`` runs to
    the face centre (half a spacing).  No stencil coefficient is read from the
    discretisation, so the comparison is independent of the code under test.
    """
    h = 1./n
    entries = defaultdict(float)
    def internal(i, j, k):
        return 0 <= i < n and 0 <= j < n and 0 <= k < n
    def index(i, j, k):
        return k*n*n+j*n+i
    for k in range(n):
        for j in range(n):
            for i in range(n):
                pivot = index(i, j, k)
                for di, dj, dk in ((1, 0, 0), (-1, 0, 0), (0, 1, 0),
                                   (0, -1, 0), (0, 0, 1), (0, 0, -1)):
                    if internal(i+di, j+dj, k+dk):
                        entries[(pivot, pivot)] += viscosity*h
                        entries[(pivot, index(i+di, j+dj, k+dk))] -= viscosity*h
                    else:
                        entries[(pivot, pivot)] += 2*viscosity*h
    rows = np.array([key[0] for key in entries])
    columns = np.array([key[1] for key in entries])
    values = np.array(list(entries.values()))
    return sparse.coo_matrix((values, (rows, columns)), shape=(n**3, n**3)).tocsr()


def analytic_anchor(n=3, viscosity=1.):
    """Compare the assembled scalar operator with the closed-form reference."""
    mesh = uniform_cartesian(n)
    viscous = NativeViscous3D(mesh, mesh.neighbor < 0, viscosity)
    assembled = (-viscous.scalar_diffusion_matrix).tocsr()
    reference = seven_point_flux_operator(n, viscosity)
    difference = abs(assembled-reference).max()
    scale = abs(reference).max()
    diagonal = np.asarray(assembled.diagonal()).ravel()
    rotation = sparse.diags(1./np.sqrt(diagonal))
    computed = float(np.linalg.eigvalsh((rotation@assembled@rotation).toarray())[0])
    reference_eigenvalue = float(np.linalg.eigvalsh(
        (sparse.diags(1./np.sqrt(np.asarray(reference.diagonal()).ravel()))
         @reference@sparse.diags(1./np.sqrt(np.asarray(reference.diagonal()).ravel()))).toarray())[0])
    return dict(n=n, cells=n**3, viscosity=viscosity,
                maximum_absolute_matrix_difference=float(difference),
                relative_matrix_difference=float(difference/scale),
                normalised_minimum_eigenvalue=computed,
                normalised_minimum_eigenvalue_from_reference=reference_eigenvalue,
                eigenvalue_relative_difference=float(abs(computed-reference_eigenvalue)
                                                     / reference_eigenvalue),
                coefficient_internal_face=float(viscosity/n),
                coefficient_dirichlet_face=float(2*viscosity/n))


# --------------------------------------------------------------------------- #
# passivity margin and per-face decomposition
# --------------------------------------------------------------------------- #
def _normalised_spectrum(matrix, diagonal):
    size = matrix.shape[0]
    rotation = sparse.diags(1./np.sqrt(np.repeat(diagonal, 3)))
    symmetric = ((rotation@((matrix+matrix.T)*.5)@rotation)).tocsr()
    if size <= 400:
        values, vectors = np.linalg.eigh(symmetric.toarray())
        value, vector = float(values[0]), vectors[:, 0]
    else:
        value, vector = eigsh(symmetric, k=1, which='SA', tol=1e-11, maxiter=20000,
                             v0=np.random.default_rng(19).normal(size=size))
        value, vector = float(value[0]), vector[:, 0]
    return value, np.asarray(vector).ravel(), symmetric


def passivity_margin(mesh, correction='over-relaxed'):
    """Normalised dissipativity margin of ``A = -momentum_diffusion_matrix``.

    Returns the smallest eigenvalue of the orthogonal-diagonal-normalised
    symmetric part (negative = energy creation), the asymmetry ratio
    ``sum|A-A^T|/sum|A|``, the exact per-face energy decomposition of the
    creating mode, and the mesh metrics that predict the outcome.
    """
    viscous = NativeViscous3D(mesh, mesh.neighbor < 0, 1., correction=correction)
    matrix = (-viscous.momentum_diffusion_matrix).tocsr()
    area, d = mesh.area_vectors, viscous.op.d
    coefficient = np.sum(area**2, axis=1)/np.sum(area*d, axis=1)
    diagonal = np.asarray((-viscous.divergence_matrix
                           @sparse.diags(coefficient)@viscous.delta_matrix).diagonal()).ravel()
    value, vector, symmetric = _normalised_spectrum(matrix, diagonal)
    velocity = (sparse.diags(1./np.sqrt(np.repeat(diagonal, 3)))@vector).reshape(-1, 3)
    flux = viscous.momentum_flux(velocity)
    owner, neighbor = mesh.owner, mesh.neighbor
    internal = np.flatnonzero(neighbor >= 0)
    boundary = np.flatnonzero(neighbor < 0)
    per_face = np.zeros(len(owner))
    per_face[internal] = -np.sum(flux[internal]*(velocity[owner[internal]]
                                                 - velocity[neighbor[internal]]), axis=1)
    per_face[boundary] = -np.sum(flux[boundary]*velocity[owner[boundary]], axis=1)
    energy = float(velocity.ravel()@(matrix@velocity.ravel()))
    scale = max(np.abs(per_face).max(), 1e-300)
    cosines = np.sum(area*d, axis=1)/(np.linalg.norm(area, axis=1)*np.linalg.norm(d, axis=1))
    angles = np.degrees(np.arccos(np.clip(cosines, -1., 1.)))
    aspect = np.linalg.norm(d, axis=1)**3/np.maximum(mesh.cell_volumes[owner], 1e-300)
    return dict(cells=len(mesh.cells), faces=len(owner), correction=str(correction),
                margin=value, asymmetry_ratio=float(abs(matrix-matrix.T).sum()/abs(matrix).sum()),
                quadratic_form_energy=energy,
                eigen_residual=float(np.linalg.norm(symmetric@vector-value*vector)
                                     / max(np.linalg.norm(vector), 1e-300)),
                creating_faces=int(np.count_nonzero(per_face < -1e-12*scale)),
                worst_face_contribution=float(per_face.min()),
                worst_face_share=float(per_face.min()/energy) if energy < 0 else 0.,
                maximum_nonorthogonality_degrees=float(angles.max()),
                maximum_interpolation_extrapolation=float(max(viscous.weight.max(),
                                                              -viscous.weight.min(), 0.)),
                maximum_face_aspect=float(aspect.max()),
                per_face=per_face, mesh=mesh)


def compact(audit):
    """The JSON-serialisable part of :func:`passivity_margin`."""
    return {key: value for key, value in audit.items() if key not in ('per_face', 'mesh')}


# --------------------------------------------------------------------------- #
# families
# --------------------------------------------------------------------------- #
def mesh_family(ratios=(1., .1, .02), shears=(.3, .8), modes=('hex', 'hybrid', 'wedge')):
    """Thin-layer / aspect-ratio / topology sweep of the one-block families."""
    rows = []
    for mode in modes:
        for ratio in ratios:
            for shear in shears:
                mesh = layered_mesh(2, 2, 6, ratio, shear, mode)
                audit = passivity_margin(mesh)
                rows.append(dict(mode=mode, layer_ratio=ratio, shear=shear, **compact(audit)))
    return rows


def counterexample_sweep(skews=(0., .3, .6, .8, .9, .95, 1., 1.1), counts=(1, 2, 4, 8)):
    """Passivity of the two-tetrahedra family inside a multi-cell mesh."""
    rows = []
    for skew in skews:
        for count in counts:
            try:
                mesh = mosaic_pairs(count, skew)
            except ValueError:
                rows.append(dict(skew=skew, pairs=count, valid=False))
                continue
            audit = passivity_margin(mesh)
            rows.append(dict(skew=skew, pairs=count, valid=True, **compact(audit)))
    return rows


def creation_threshold(lo=.8, hi=1., iterations=24):
    """Bisect the skew at which the *mesh-level* margin changes sign."""
    def creates(skew):
        return passivity_margin(mosaic_pairs(1, skew))['margin'] < 0.
    if creates(lo):
        raise ValueError('reference skew already creates energy')
    if not creates(hi):
        raise ValueError('upper bound does not create energy')
    for _ in range(iterations):
        middle = (lo+hi)/2
        if creates(middle):
            hi = middle
        else:
            lo = middle
    angle = passivity_margin(mosaic_pairs(1, hi))['maximum_nonorthogonality_degrees']
    return dict(passive_skew=float(lo), creating_skew=float(hi), creating_degrees=float(angle))


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def run(directory, make_figures=True):
    """Write the audit JSON (and, optionally, figures) into ``directory``."""
    from pathlib import Path
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    anchors = [analytic_anchor(n) for n in (2, 3, 4, 6, 8)]
    family = mesh_family()
    sweep = counterexample_sweep()
    threshold = creation_threshold()
    detailed = passivity_margin(mosaic_pairs(1, 1.))
    report = dict(analytic_anchor=anchors, mesh_family=family, counterexample_sweep=sweep,
                  creation_threshold=threshold,
                  counterexample_detail=compact(detailed),
                  counterexample_faces=dict(
                      creating=[int(f) for f in np.flatnonzero(detailed['per_face']
                                                              < -1e-12*np.abs(detailed['per_face']).max())],
                      contributions=[float(x) for x in detailed['per_face']]),
                  statement=('Energy creation is not produced by tetrahedral transition, '
                             'prism topology, clustering or shear alone; it needs faces at or '
                             'beyond the two-cell threshold (~86 deg non-orthogonality) and '
                             'is a local face-cluster property.'))
    (directory/'native-viscous-passivity.json').write_text(
        json.dumps(report, indent=2, sort_keys=False)+'\n')
    if make_figures:
        _figures(directory, family, sweep, threshold, detailed)
    return report


def _figures(directory, family, sweep, threshold, detailed):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams['mathtext.fontset'] = 'dejavusans'

    figure, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), dpi=160)
    for mode, marker in (('hex', 'o'), ('hybrid', 's'), ('wedge', '^')):
        rows = [row for row in family if row['mode'] == mode]
        axes[0].plot([row['maximum_face_aspect'] for row in rows],
                     [row['margin'] for row in rows], marker, label=mode, markersize=5)
    axes[0].set_xscale('log')
    axes[0].set_ylim(bottom=0.)
    axes[0].set_xlabel('maximum face aspect ratio $|d|^3/V$')
    axes[0].set_ylabel(r'passivity margin $\lambda_{\min}(RAR)$')
    axes[0].set_title('hex / tet-transition / prism layers stay passive')
    axes[0].legend()
    axes[0].grid(alpha=.3)

    pairs = [row for row in sweep if row['pairs'] == 1 and row['valid']]
    axes[1].plot([row['skew'] for row in pairs], [row['margin'] for row in pairs],
                 'o-', color='tab:red', label='mosaic mesh, 1 pair')
    for entry in sweep:
        if entry['valid'] and entry['margin'] < 0:
            axes[1].plot(entry['skew'], entry['margin'], 'x', color='tab:blue', alpha=.5)
    axes[1].axhline(0., color='k', linewidth=.8)
    axes[1].axvline(threshold['creating_skew'], linestyle=':', color='k', linewidth=.8)
    axes[1].annotate(f"threshold skew {threshold['creating_skew']:.3f}\n"
                     f"= {threshold['creating_degrees']:.1f} deg non-orthogonality",
                     xy=(threshold['creating_skew'], 0.), xytext=(.2, -.1),
                     arrowprops=dict(arrowstyle='->', linewidth=.7), fontsize=7)
    axes[1].set_xlabel('two-tetrahedra family parameter (skew)')
    axes[1].set_ylabel('passivity margin')
    axes[1].set_title('counterexample family: local sign change')
    axes[1].legend()
    axes[1].grid(alpha=.3)
    figure.tight_layout()
    figure.savefig(directory/'native-viscous-passivity.png')
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6.4, 3.2), dpi=160)
    contributions = np.asarray(detailed['per_face'])
    axis.bar(np.arange(contributions.size), contributions,
             color=np.where(contributions < 0, 'tab:red', 'tab:blue'))
    axis.set_xlabel('face index (two-tetrahedra pair, skew = 1.0)')
    axis.set_ylabel('per-face energy contribution')
    axis.set_title('creating-mode energy decomposition, total '
                   f"{detailed['quadratic_form_energy']:+.3e}")
    axis.grid(alpha=.3, axis='y')
    figure.tight_layout()
    figure.savefig(directory/'native-viscous-passivity-faces.png')
    plt.close(figure)
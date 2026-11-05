"""Thin-layer passivity envelope of the implicit viscous operator.

Motivation
----------
`docs/verification-native-solver-audit/README.md` leaves one defect unfixed: the
assembled viscous operator `L = -momentum_diffusion_matrix` is not dissipative
on the 9,068-cell hybrid SUBOFF mesh (`u^T L u = -0.2458`), and its "next steps"
section asks for the *thin-layer / tetrahedral interface* regime specifically.
`tensorfvm.verification.native_energy` only covers two-cell tetrahedral pairs.

What this script adds
---------------------
A deterministic, mesh-quality-indexed envelope.  Families: Kuhn-split tetrahedra,
wedges (prism boundary layers) and hexahedra, with planar quad faces or
triangulated (split) quads, swept over aspect ratio `AR` in ``1..50`` and lateral
shear in ``0..0.8``.  For every mesh it reports the signed spectrum of the
vector (momentum) dissipation operator for the three documented corrections
(``over-relaxed``, ``minimum``, ``none``), then bisects

* the face-wise correction *damping* at which a failing configuration regains
  passivity, and
* the uniform correction scale ``gamma`` at which it regains passivity,

and finally verifies that damping costs nothing in affine-flux consistency (the
deferred correction is proportional to the affine residual, so it vanishes for
affine fields for every damping value).  No flow solve, no pressure system; the
whole audit is small dense linear algebra (<= 486 unknowns).

Usage
-----
    PYTHONPATH=src python scripts/audit_thin_layer_viscous_passivity.py \
        --output results/thin-layer-viscous-passivity
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from tensorfvm.native_mesh3d import NativeMesh3D
from tensorfvm.native_viscous3d import NativeViscous3D
from tensorfvm.verification.native_energy import affine_consistency, passivity_audit

CORRECTIONS = ('over-relaxed', 'minimum', 'none')
#: Kuhn (6-tetrahedra) decomposition of a hexahedron, vertex order of
#: :class:`NativeMesh3D` hexahedra.
KUHN = ((0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6), (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6))


def _box(n, ar, shear, query):
    """Node lattice of ``[0,1]x[0,1]x[0,1/ar]`` sheared by ``x += shear*z``."""
    xs, ys = np.linspace(0., 1., n+1), np.linspace(0., 1., n+1)
    zs = np.linspace(0., 1./ar, n+1)
    points = np.array([[x+shear*z, y, z] for z in zs for y in ys for x in xs], dtype=float)
    return points, query(n)


def _lattice(n):
    def index(i, j, k):
        return k*(n+1)**2+j*(n+1)+i
    return index


def tetra_box(n=2, ar=1., shear=0.):
    """Kuhn-split tetrahedra: every face is a triangle, 6 tets per lattice cube."""
    points, index = _box(n, ar, shear, _lattice)
    cells, kinds = [], []
    for k in range(n):
        for j in range(n):
            for i in range(n):
                cube = [index(i, j, k), index(i+1, j, k), index(i+1, j+1, k), index(i, j+1, k),
                        index(i, j, k+1), index(i+1, j, k+1), index(i+1, j+1, k+1),
                        index(i, j+1, k+1)]
                for local in KUHN:
                    cells.append(tuple(cube[q] for q in local))
                    kinds.append('tetra')
    return NativeMesh3D.from_cells(points, cells, kinds)


def wedge_box(n=2, ar=1., shear=0., triangulate=False):
    """Prism (wedge) boundary-layer stack; two wedges per lattice cube."""
    points, index = _box(n, ar, shear, _lattice)
    cells, kinds = [], []
    for k in range(n):
        for j in range(n):
            for i in range(n):
                cells.append((index(i, j, k), index(i+1, j, k), index(i, j+1, k),
                              index(i, j, k+1), index(i+1, j, k+1), index(i, j+1, k+1)))
                cells.append((index(i+1, j, k), index(i+1, j+1, k), index(i, j+1, k),
                              index(i+1, j, k+1), index(i+1, j+1, k+1), index(i, j+1, k+1)))
                kinds.extend(('wedge', 'wedge'))
    return NativeMesh3D.from_cells(points, cells, kinds, triangulate_faces=triangulate)


def hexa_box(n=2, ar=1., shear=0., triangulate=True):
    """Hexahedral stack, by default with triangulated (split) quad faces."""
    points, index = _box(n, ar, shear, _lattice)
    cells = []
    for k in range(n):
        for j in range(n):
            for i in range(n):
                cells.append((index(i, j, k), index(i+1, j, k), index(i+1, j+1, k),
                              index(i, j+1, k), index(i, j, k+1), index(i+1, j, k+1),
                              index(i+1, j+1, k+1), index(i, j+1, k+1)))
    return NativeMesh3D.from_cells(points, cells, ['hexahedron']*len(cells),
                                   triangulate_faces=triangulate)


FAMILIES = {
    'tetra': lambda n, ar, sh: tetra_box(n, ar, sh),
    'wedge-quad': lambda n, ar, sh: wedge_box(n, ar, sh, triangulate=False),
    'wedge-tri': lambda n, ar, sh: wedge_box(n, ar, sh, triangulate=True),
    'hexa-tri': lambda n, ar, sh: hexa_box(n, ar, sh, triangulate=True),
}


def quality(mesh):
    """Nonorthogonality, projection and aspect-ratio descriptors of ``mesh``."""
    viscous = NativeViscous3D(mesh, mesh.neighbor < 0, 1.)
    area, d = mesh.area_vectors, viscous.op.d
    norm_area, norm_d = np.linalg.norm(area, axis=1), np.linalg.norm(d, axis=1)
    cosine = np.clip(np.sum(area*d, axis=1)/(norm_area*norm_d), -1., 1.)
    projection = np.einsum('ij,ij->i', area, d)
    volumes = mesh.cell_volumes
    # Characteristic cell size per cell: largest face area / cell volume, i.e.
    # the inverse of the thinnest direction, normalised by the cell's own scale.
    faces = [i for i, (own, nei) in enumerate(zip(mesh.owner, mesh.neighbor))
             if own >= 0]
    area_max = np.zeros(len(mesh.cells))
    for f in faces:
        area_max[mesh.owner[f]] = max(area_max[mesh.owner[f]], norm_area[f])
    # Face-area / volume descriptor: large for thin layers, O(1) for cubes.
    aspect = area_max**1.5/np.maximum(volumes, 1e-300)
    return dict(max_nonorthogonality_degrees=float(np.degrees(np.arccos(cosine)).max()),
                min_projection_ratio=float(projection.min()/projection.max()),
                min_cell_volume=float(volumes.min()),
                max_aspect_descriptor=float(aspect.max()))


def scan_family(name, builder, aspects, shears, corrections=CORRECTIONS, cells=2):
    rows = []
    for ar in aspects:
        for shear in shears:
            try:
                mesh = builder(cells, ar, shear)
            except ValueError as error:            # invalid / folded lattice
                rows.append(dict(family=name, aspect_ratio=ar, shear=shear,
                                 valid=False, reason=str(error)))
                continue
            audit = {c: passivity_audit(mesh, c) for c in corrections}
            row = dict(family=name, aspect_ratio=ar, shear=shear, valid=True,
                       cells=len(mesh.cells), faces=len(mesh.owner),
                       quality=quality(mesh),
                       corrections={c: dict(
                           vector={k: audit[c]['vector'][k] for k in
                                   ('lambda_min', 'creation_rate', 'creation_fraction',
                                    'passive', 'wrong_sign_modes')},
                           scalar_creation_fraction=audit[c]['scalar']['creation_fraction'])
                           for c in corrections})
            rows.append(row)
    return rows


def _damped_passivity(mesh, correction, damping):
    """Passivity of ``mesh`` with ``correction`` scaled by ``damping``."""
    viscous = NativeViscous3D(mesh, mesh.neighbor < 0, 1., correction=correction,
                              correction_damping=damping)
    dissipation = -viscous.momentum_diffusion_matrix.toarray()
    eigenvalues = np.linalg.eigvalsh((dissipation+dissipation.T)/2)
    scale = max(1., float(np.abs(eigenvalues).max()))
    return dict(lambda_min=float(eigenvalues[0]),
                creation_rate=float(max(0., -eigenvalues[0])),
                passive=bool(eigenvalues[0] >= -1e-12*scale))


def damping_line(mesh, correction='over-relaxed', samples=21, iterations=24):
    """Smallest damping that restores passivity, and the spectrum along the line."""
    grid = [float(x) for x in np.linspace(0., 1., samples)]
    rows = [dict(damping=d, **_damped_passivity(mesh, correction, d)) for d in grid]
    if rows[-1]['passive'] and not rows[0]['passive']:
        low, high = 0., 1.
        for _ in range(iterations):
            middle = (low+high)/2
            if _damped_passivity(mesh, correction, middle)['passive']:
                high = middle
            else:
                low = middle
        threshold = high
    elif rows[0]['passive']:
        threshold = 0.
    else:
        threshold = None
    return dict(correction=correction, rows=rows,
                full_damping_passive=rows[-1]['passive'],
                minimum_safe_damping=threshold,
                no_damping_creation_rate=rows[0]['creation_rate'],
                full_damping_creation_rate=rows[-1]['creation_rate'])


def gamma_line(mesh, lo=-8., hi=6., samples=29, iterations=30):
    """Uniform ``gamma`` line: where does the correction family become passive?

    ``gamma`` multiplies the face area vector (``correction_direction = gamma*s``),
    so ``gamma = 0`` is ``none`` and ``gamma = 1/(s.d)`` is ``over-relaxed``.
    """
    viscous = NativeViscous3D(mesh, mesh.neighbor < 0, 1.)
    projection = np.einsum('ij,ij->i', mesh.area_vectors, viscous.op.d)
    scales = [float(x) for x in np.geomspace(10.**lo, 10.**hi, samples)]
    rows = []
    for gamma in scales:
        audit = passivity_audit(mesh, correction=float(gamma))
        rows.append(dict(gamma=gamma, creation_rate=audit['vector']['creation_rate'],
                         passive=audit['vector']['passive']))
    # Bisect the smallest passive gamma above the largest failing one.
    passive = [r['gamma'] for r in rows if r['passive']]
    failing = [r['gamma'] for r in rows if not r['passive']]
    threshold = None
    if passive and failing:
        high = min(g for g in passive if g > max(failing))
        low = max(failing)
        for _ in range(iterations):
            middle = math.sqrt(low*high)
            if passivity_audit(mesh, correction=float(middle))['vector']['passive']:
                high = middle
            else:
                low = middle
        threshold = high
    return dict(rows=rows, any_passive=bool(passive), passive_fraction=len(passive)/len(rows),
                minimum_passive_gamma=threshold,
                over_relaxed_gamma=(float((1./projection).min()), float((1./projection).max())),
                minimum_gamma=float((projection/(np.einsum('ij,ij->i', viscous.op.d,
                                                           viscous.op.d)
                                      * np.einsum('ij,ij->i', mesh.area_vectors,
                                                  mesh.area_vectors))).min()))


def damping_affine_cost(mesh):
    """Affine-flux error of the over-relaxed correction versus damping factor.

    The deferred correction is proportional to the affine residual
    ``u_N-u_P-d.grad``, which is identically zero for affine fields, so the
    affine traction must be reproduced to round-off for every damping value.
    """
    rows = []
    for damping in (0., 0.25, 0.5, 0.75, 1.):
        viscous = NativeViscous3D(mesh, mesh.neighbor < 0, 1., correction='over-relaxed',
                                  correction_damping=damping)
        gradient = np.outer(np.array([.7, -.3, .2]), [1., 1., 1.])
        stress = gradient+gradient.T-(2/3)*np.trace(gradient)*np.eye(3)
        exact = np.einsum('aj,fj->fa', stress, mesh.area_vectors)
        computed = viscous.momentum_flux(mesh.cell_centers @ gradient.T, 0.,
                                         mesh.face_centers @ gradient.T)
        rows.append(dict(damping=damping,
                         maximum_relative_error=float(np.abs(computed-exact).max()
                                                      / np.abs(exact).max()),
                         zero_damping_reference=affine_consistency(
                             mesh, correction='over-relaxed')['maximum_relative_error']))
    return rows


def _rounded(value, digits=6):
    """Recursively round floats for a compact, diff-stable JSON artefact."""
    if isinstance(value, float):
        return float(f'{value:.{digits}g}')
    if isinstance(value, dict):
        return {key: _rounded(item, digits) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_rounded(item, digits) for item in value]
    return value


def summarise(rows):
    """Envelope per (family, correction): first failing aspect ratio and worst rate."""
    envelope = {}
    for row in rows:
        if not row['valid']:
            continue
        for correction, spectral in row['corrections'].items():
            key = f"{row['family']}/{correction}"
            entry = envelope.setdefault(key, dict(first_failing_aspect_ratio=None,
                                                 first_failing_angle=None,
                                                 worst_creation_fraction=0.,
                                                 samples=0, passive_samples=0))
            entry['samples'] += 1
            vector = spectral['vector']
            if vector['passive']:
                entry['passive_samples'] += 1
            else:
                if entry['first_failing_aspect_ratio'] is None:
                    entry['first_failing_aspect_ratio'] = row['aspect_ratio']
                    entry['first_failing_angle'] = row['quality'][
                        'max_nonorthogonality_degrees']
                entry['worst_creation_fraction'] = max(
                    entry['worst_creation_fraction'], vector['creation_fraction'])
    return envelope


def figure(rows, damping, gamma, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    colors = {'tetra': 'C0', 'wedge-quad': 'C1', 'wedge-tri': 'C2', 'hexa-tri': 'C3'}
    styles = {'over-relaxed': '-', 'minimum': '--', 'none': ':'}
    ax = axes[0]
    for family in FAMILIES:
        for correction in CORRECTIONS:
            xs, ys = [], []
            for row in rows:
                if not row['valid'] or row['family'] != family:
                    continue
                if row['shear'] != 0.4:
                    continue
                xs.append(row['aspect_ratio'])
                ys.append(row['corrections'][correction]['vector']['creation_fraction'])
            if xs:
                ax.plot(xs, ys, styles[correction], color=colors[family], marker='o', ms=3,
                        label=f'{family} / {correction}')
    ax.set_xscale('log')
    ax.set_yscale('symlog', linthresh=1e-9)
    ax.set_xlabel('cell aspect ratio (in-plane / thickness)')
    ax.set_ylabel('creation fraction  max(0,-$\\lambda_{{min}}$)/$\\lambda_{{max}}$')
    ax.set_title('shear=0.4 passivity envelope')
    ax.grid(alpha=.3)
    ax.legend(fontsize=6, ncol=2)

    ax = axes[1]
    for name, line, color in (('over-relaxed', damping['over-relaxed'], 'C4'),
                              ('minimum', damping['minimum'], 'C1')):
        ax.plot([r['damping'] for r in line['rows']],
                [r['creation_rate'] for r in line['rows']], 'o-', color=color,
                ms=3, label=f'{name}')
        if line['minimum_safe_damping'] is not None:
            ax.axvline(line['minimum_safe_damping'], color=color, ls='--',
                       label=f"{name} safe $\\eta^*$={line['minimum_safe_damping']:.3f}")
    ax.set_yscale('symlog', linthresh=1e-9)
    ax.set_xlabel('correction damping  $\\eta$')
    ax.set_ylabel('creation rate')
    ax.set_title('damping line (worst failing configuration)')
    ax.grid(alpha=.3)
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.loglog([r['gamma'] for r in gamma['rows']],
              [max(r['creation_rate'], 1e-12) for r in gamma['rows']], 'o-', color='C5')
    if gamma['minimum_passive_gamma']:
        ax.axvline(gamma['minimum_passive_gamma'], color='k', ls='--',
                   label=f"$\\gamma^*$={gamma['minimum_passive_gamma']:.3g}")
    low, high = gamma['over_relaxed_gamma']
    ax.axvspan(low, high, color='C2', alpha=.15, label='over-relaxed $1/(s\\cdot d)$')
    ax.set_xlabel('uniform correction scale  $\\gamma$')
    ax.set_ylabel('creation rate')
    ax.set_title('$\\gamma$ threshold')
    ax.grid(alpha=.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = Path(output)/'thin-layer-viscous-passivity.png'
    fig.savefig(path, dpi=118)
    plt.close(fig)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--output', type=Path, default=Path('results/thin-layer-viscous-passivity'))
    parser.add_argument('--aspects', type=float, nargs='+',
                        default=[1., 2., 3., 5., 10., 20., 50.])
    parser.add_argument('--shears', type=float, nargs='+', default=[0., .2, .4, .6, .8])
    parser.add_argument('--worst-family', default='tetra')
    parser.add_argument('--worst-aspect', type=float, default=20.)
    parser.add_argument('--worst-shear', type=float, default=0.)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, builder in FAMILIES.items():
        rows.extend(scan_family(name, builder, args.aspects, args.shears))
    envelope = summarise(rows)

    worst_mesh = FAMILIES[args.worst_family](2, args.worst_aspect, args.worst_shear)
    damping = {name: damping_line(worst_mesh, name) for name in ('over-relaxed', 'minimum')}
    gamma = gamma_line(worst_mesh)
    affine = damping_affine_cost(worst_mesh)
    reference = passivity_audit(worst_mesh, 'over-relaxed')

    payload = dict(
        description='Thin-layer passivity envelope of the implicit native viscous operator',
        families=list(FAMILIES), corrections=list(CORRECTIONS),
        aspects=[float(x) for x in args.aspects], shears=[float(x) for x in args.shears],
        cells_per_family=2, mesh_rows=rows,
        envelope=envelope,
        worst_configuration=dict(family=args.worst_family, aspect_ratio=args.worst_aspect,
                                 shear=args.worst_shear,
                                 max_nonorthogonality_degrees=reference[
                                     'maximum_nonorthogonality_degrees'],
                                 over_relaxed=reference['vector'],
                                 minimum=passivity_audit(worst_mesh, 'minimum')['vector'],
                                 none=passivity_audit(worst_mesh, 'none')['vector']),
        damping_lines=damping, gamma_line=gamma, damping_affine_cost=affine,
    )
    (args.output/'passivity-envelope.json').write_text(
        json.dumps(_rounded(payload), indent=1, sort_keys=True))
    path = figure(rows, damping, gamma, args.output)
    over = envelope['tetra/over-relaxed']
    print(f"families={len(FAMILIES)} samples={sum(1 for r in rows if r['valid'])}")
    print(f"tetra/over-relaxed: passive {over['passive_samples']}/{over['samples']} samples, "
          f"first failing AR={over['first_failing_aspect_ratio']}")
    print(f"worst config: AR={args.worst_aspect} angle="
          f"{reference['maximum_nonorthogonality_degrees']:.2f} deg, "
          f"over-relaxed creation={reference['vector']['creation_rate']:.3e}")
    for name, line in damping.items():
        print(f"damping line {name}: safe damping={line['minimum_safe_damping']}, "
              f"creation at eta=0 {line['no_damping_creation_rate']:.3e} -> "
              f"eta=1 {line['full_damping_creation_rate']:.3e}")
    print(f"gamma*={gamma['minimum_passive_gamma']}")
    print(f"wrote {args.output/'passivity-envelope.json'} and {path}")


if __name__ == '__main__':
    main()
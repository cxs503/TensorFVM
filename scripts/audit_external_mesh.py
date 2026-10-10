"""Require physical body conformity in saved external-flow benchmark meshes.

Independent NumPy geometry checks; no production mesh or solver imports.
"""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def audit(case, config, fields):
    expected = 'body-fitted' if case == 'cylinder' else 'c-grid'
    if config['mesh_type'] != expected:
        raise ValueError('External-flow benchmarks require body-fitted geometry')
    v = fields['vertices_m']
    wall = v[0]
    if case == 'cylinder':
        error = np.max(np.abs(np.linalg.norm(wall - [config['cylinder_x'], config['cylinder_y']], axis=1) - config['cylinder_radius']))
    else:
        if config['airfoil_code'] != '0012':
            raise ValueError('Independent profile audit currently supports NACA0012')
        x = (1 - np.cos(np.linspace(0, np.pi, max(128, config['nx'] * 4) + 1))) / 2
        y = .6 * (.2969*np.sqrt(x) - .126*x - .3516*x**2 + .2843*x**3 - .1036*x**4)
        profile = np.concatenate([np.stack([x[::-1], y[::-1]], axis=1), np.stack([x[1:], -y[1:]], axis=1)])
        profile = profile * config['airfoil_chord'] + [config['airfoil_x'], config['airfoil_y']]
        a, b = profile[:-1], profile[1:]
        d = b - a
        delta = wall[:, None, :] - a
        t = np.clip(np.sum(delta*d, axis=-1)/np.sum(d*d, axis=-1), 0, 1)
        error = np.max(np.min(np.linalg.norm(delta-t[..., None]*d, axis=-1), axis=1))
    polygons = np.stack([v[:-1, :-1], v[1:, :-1], v[1:, 1:], v[:-1, 1:]], axis=2)
    following = np.roll(polygons, -1, axis=2)
    signed = .5*np.sum(polygons[..., 0]*following[..., 1]-polygons[..., 1]*following[..., 0], axis=2)
    volumes = fields['cell_volumes_m2'].reshape(signed.shape)
    area_error = np.max(np.abs(np.abs(signed)-volumes))
    seam_error = np.max(np.abs(v[:, 0]-v[:, -1]))
    surface_error = np.max(np.abs(fields['surface_centers_m']-(wall[:-1]+wall[1:])/2))
    vectors = fields['face_area_vectors_m']
    oblique = int(np.count_nonzero(np.all(np.abs(vectors)>1e-12, axis=1)))
    passed = bool(error < 1e-10 and area_error < 1e-10 and seam_error < 1e-10 and surface_error < 1e-10 and np.all(signed > 0) and np.all(volumes > 0) and oblique > 0 and np.count_nonzero(fields['surface_mask']) == config['nx'])
    result = dict(body_fitted=passed, wall_vertex_distance_m=float(error), area_error_m2=float(area_error), periodic_seam_error_m=float(seam_error), wall_face_midpoint_error_m=float(surface_error), min_cell_area_m2=float(volumes.min()), oblique_faces=oblique, cells=int(volumes.size), topology='periodic radial O topology', physical_accuracy_qualified=False)
    if not passed:
        raise ValueError(f'Mesh generation must be repaired before benchmark acceptance: {result}')
    return result


def main():
    runs = []
    for case in ('cylinder', 'naca'):
        directory = ROOT/'docs'/f'verification-external-{case}'
        for row in json.loads((directory/'summary.json').read_text())['runs']:
            path = directory/row['resolution']/'fields.npz'
            with np.load(path) as fields:
                result = audit(case, row['config'], fields)
            runs.append(dict(case=case, resolution=row['resolution'], fields_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), **result))
    output = ROOT/'docs/verification-external-mesh'
    output.mkdir(exist_ok=True)
    (output/'audit.json').write_text(json.dumps(dict(requirement='Body-fitted physical geometry; Cartesian meshes prohibited', runs=runs), indent=2)+'\n')
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    (output/'manifest.json').write_text(json.dumps(dict(source_sha256={'scripts/audit_external_mesh.py': sha(Path(__file__))}, artifacts_sha256={'audit.json': sha(output/'audit.json')}), indent=2)+'\n')
    print(json.dumps(runs, indent=2))


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Compare bound native mesh artifacts, including tail/interface size jumps."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--meshes', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output must be new')
    rows, bindings = [], {}
    for directory in args.meshes:
        manifest = json.loads((directory / 'manifest.json').read_text())
        for name in ('summary.json', 'config.json', 'native-geometry.npz', 'suboff-hybrid.msh'):
            path = directory / name
            actual = digest(path)
            if manifest.get(name) != actual:
                raise ValueError('Mesh artifact hash mismatch: ' + str(path))
            bindings[str(path.resolve())] = actual
        summary = json.loads((directory / 'summary.json').read_text())
        config = json.loads((directory / 'config.json').read_text())
        with np.load(directory / 'native-geometry.npz') as data:
            volumes = data['cell_volumes']
            centers = data['cell_centers']
            owner, neighbor = data['owner'], data['neighbor']
            kinds = data['cell_types']
            areas, face_centers = data['area_vectors'], data['face_centers']
        internal = neighbor >= 0
        ni = np.maximum(neighbor, 0)
        interface = internal & ((kinds[owner] == 'tetra') != (kinds[ni] == 'tetra'))
        tail = (centers[:, 0] >= .9) & (centers[:, 0] <= 1.025) & (np.linalg.norm(centers[:, 1:], axis=1) <= .025)
        tail_faces = internal & (tail[owner] | tail[ni])
        displacement = face_centers - centers[owner]
        displacement[internal] = centers[neighbor[internal]] - centers[owner[internal]]
        cosine = np.einsum('ij,ij->i', areas, displacement) / np.linalg.norm(areas, axis=1) / np.linalg.norm(displacement, axis=1)
        angles = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
        ratios = np.maximum(volumes[owner] / volumes[ni], volumes[ni] / volumes[owner])
        stats = {}
        for name, mask in (('all_internal_faces', internal), ('layer_tetra_interface', interface), ('fixed_tail_internal_faces', tail_faces)):
            values = ratios[mask]
            stats[name] = dict(faces=int(mask.sum()), neighbor_volume_ratio_max=float(values.max()),
                               neighbor_volume_ratio_p95=float(np.percentile(values, 95)),
                               nonorthogonality_max_degrees=float(angles[mask].max()),
                               nonorthogonality_p95_degrees=float(np.percentile(angles[mask], 95)))
        if len(volumes) != summary['mesh_audit']['cells'] or interface.sum() != summary['verified_shared_interface_faces']:
            raise ValueError('Summary topology disagrees with stored native geometry')
        rows.append(dict(directory=str(directory), cells=len(volumes), faces=len(owner),
            cell_types={str(k): int(n) for k, n in zip(*np.unique(kinds, return_counts=True))},
            axial_intervals=summary['axial_stations']-1, azimuthal=summary['azimuthal'],
            layers=summary['layers'], first_height_L=summary['first_height_L'], growth=summary['growth'],
            total_thickness_L=summary['total_thickness_L'], far_size_L=config['far_size'],
            geometry_passed=summary['geometry_passed'], generation_and_import_seconds=summary['generation_and_import_seconds'],
            faceted_body_volume_L3=summary['faceted_body_volume_L3'],
            maximum_relative_closure=summary['mesh_audit']['maximum_relative_closure'],
            domain_volume_balance_error_L3=summary['domain_volume_balance_error_L3'],
            quality_regions=stats))
    args.output.mkdir(parents=True)
    rows.sort(key=lambda row: row['cells'])
    result = dict(meshes=rows, tail_window_L=dict(x=[.9, 1.025], radius_max=.025),
                  definition='Size jumps compare adjacent actual cell volumes. Fixed-tail faces touch at least one cell centroid in the cylinder. Faceted FV face angles can differ from recovered parent-quad angles.',
                  flow_qualification_inferred=False, physical_qualification_inferred=False)
    (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    shutil.copy2(__file__, args.output / 'executed-audit.py')
    artifacts = {path.name: digest(path) for path in args.output.iterdir() if path.is_file()}
    (args.output / 'manifest.json').write_text(json.dumps(dict(input_sha256=bindings, artifact_sha256=artifacts), indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Preflight quadratic reconstruction on a real mesh; no flow solve."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np
from tensorfvm.native_mesh3d import NativeMesh3D
from tensorfvm.native_quadratic3d import NativeQuadratic3D


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mesh', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output must be new')
    args.output.mkdir(parents=True)
    root = Path(__file__).resolve().parents[1]
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    sources = {Path(__file__).resolve()} | {Path(m.__file__).resolve()
        for name, m in sys.modules.copy().items()
        if (name == 'tensorfvm' or name.startswith('tensorfvm.')) and getattr(m, '__file__', None)}
    for source in sources:
        target = args.output / 'source-snapshot' / source.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    config = dict(mesh=str(args.mesh), mesh_sha256=sha(args.mesh),
                  maximum_absolute_polynomial_derivative_tolerance=1e-7,
                  scope='P2 gradient/Hessian reproduction and stencil construction only; no NS flow or accuracy qualification')
    (args.output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    start = time.perf_counter()
    mesh = NativeMesh3D.from_gmsh(args.mesh, triangulate_faces=True)
    # This fixed polynomial is independent of the flow benchmark solution.
    hessian = np.array([[.2, .07, -.05], [.07, -.4, .09], [-.05, .09, .3]])
    linear = np.array([.3, -.2, .4])
    polynomial = lambda x: .1 + x @ linear + .5 * np.einsum('ni,ij,nj->n', x, hessian, x)
    values, boundary_values = polynomial(mesh.cell_centers), polynomial(mesh.face_centers)
    expected_gradient = linear + mesh.cell_centers @ hessian
    results, arrays = {}, {}
    for name, mask in (('free_pressure', np.zeros(len(mesh.owner), dtype=bool)),
                       ('prescribed_velocity', mesh.neighbor < 0)):
        op = NativeQuadratic3D(mesh, mask)
        gradient_error = op.gradient(values, boundary_values) - expected_gradient
        hessian_error = op.hessian(values, boundary_values) - hessian
        results[name] = dict(stencil_diagnostics=op.stencil_diagnostics,
                            maximum_absolute_gradient_error=float(abs(gradient_error).max()),
                            maximum_absolute_hessian_error=float(abs(hessian_error).max()))
        arrays[name + '_gradient_error_max_per_cell'] = np.max(abs(gradient_error), axis=1)
        arrays[name + '_hessian_error_max_per_cell'] = np.max(abs(hessian_error), axis=(1, 2))
        arrays[name + '_stencil_conditions'] = op.stencil_conditions
        del op
    np.savez_compressed(args.output / 'polynomial-errors.npz', **arrays)
    passed = all(max(r['maximum_absolute_gradient_error'], r['maximum_absolute_hessian_error']) < 1e-7 for r in results.values())
    result = dict(cells=len(mesh.cells), faces=len(mesh.owner), polynomial_reproduction_passed=passed,
                  results=results, elapsed_seconds=time.perf_counter()-start,
                  flow_solved=False, mesh_convergence_qualified=False, experimental_accuracy_qualified=False)
    (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    manifest = {str(p.relative_to(args.output)): sha(p) for p in args.output.rglob('*') if p.is_file()}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)
    return 0 if passed else 2


if __name__ == '__main__':
    raise SystemExit(main())

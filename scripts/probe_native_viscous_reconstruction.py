#!/usr/bin/env python3
"""Compare quadratic/cubic viscous truncation on identical native geometry.

This read-only exact-state diagnostic is not a converged flow calculation.
Analytic derivatives are used for independent reference integration only.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from tensorfvm.native_mesh3d import NativeMesh3D
from tensorfvm.native_quadratic3d import NativeQuadratic3D
from tensorfvm.native_cubic_gradient3d import NativeCubicFaceGradient3D
from tensorfvm.native_manufactured3d import abc_fields
from audit_native_quadratic_terms import face_integrals


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mesh', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); out = args.output
    if out.exists(): p.error('Output must be new')
    out.mkdir(parents=True)
    sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    repo = Path(__file__).resolve().parents[1]
    sources = {Path(__file__).resolve(), Path(__file__).with_name('audit_native_quadratic_terms.py').resolve()} | {
        Path(module.__file__).resolve() for name, module in list(sys.modules.items())
        if name == 'tensorfvm' or name.startswith('tensorfvm.')}
    bindings = {}
    for source in sorted(sources):
        relative = source.relative_to(repo); target = out/'source-snapshot'/relative
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, target)
        bindings[str(relative)] = sha(target)
    shutil.copy2(args.mesh, out/'input.msh')
    m = NativeMesh3D.from_gmsh(out/'input.msh', triangulate_faces=True)
    def integrate(flux):
        value = np.zeros((len(m.cells), 3)); np.add.at(value, m.owner, flux)
        internal = m.neighbor >= 0; np.add.at(value, m.neighbor[internal], -flux[internal])
        return value
    field = lambda x: abc_fields(x, nu=.1)
    exact = field(m.cell_centers); boundary = field(m.face_centers)
    stress8 = face_integrals(m, field, .1, 1., 8)[3]
    stress6 = face_integrals(m, field, .1, 1., 6)[3]
    wall = np.zeros(len(m.cells), bool); wall[np.unique(m.owner[m.patch_labels == 'wall'])] = True
    tail = (m.cell_centers[:, 0] >= .9)&(m.cell_centers[:, 0] <= 1.025)&(np.linalg.norm(m.cell_centers[:, 1:], axis=1) <= .025)
    masks = dict(all=np.ones(len(m.cells), bool), wall=wall, tail=tail,
                 wall_tail_wedge=wall&tail&(np.asarray(m.cell_types) == 'wedge'))
    results = {}; arrays = {}
    for name, cls in [('quadratic', NativeQuadratic3D), ('cubic', NativeCubicFaceGradient3D)]:
        start = time.perf_counter(); op = cls(m, m.neighbor < 0)
        g = op.face_gradient(exact['velocity'], boundary['velocity'])
        stress = .1*np.einsum('fij,fj->fi', g+g.swapaxes(-1,-2)-(2/3)*np.trace(g,axis1=-2,axis2=-1)[:,None,None]*np.eye(3), m.area_vectors)
        residual = -integrate(stress-stress8)/m.cell_volumes[:, None]
        results[name] = dict(seconds=time.perf_counter()-start, stencil=op.stencil_diagnostics,
            regions={key: dict(acceleration_RMS=float(np.sqrt(np.sum(residual[mask]**2*m.cell_volumes[mask, None])/m.cell_volumes[mask].sum())),
                               acceleration_max=float(np.linalg.norm(residual[mask], axis=1).max())) for key, mask in masks.items()})
        arrays[name+'_acceleration_residual'] = residual
        print(json.dumps({name: results[name]}), flush=True)
    result = dict(scope='read-only exact-state viscous truncation comparison, not a flow solve', cells=len(m.cells),
        mesh_sha256=sha(out/'input.msh'), source_sha256=bindings, comparison=results,
        reference_quadrature_6_vs_8_acceleration_max=float(np.linalg.norm(integrate(stress8-stress6)/m.cell_volumes[:,None],axis=1).max()))
    for source in sources:
        if sha(source) != bindings[str(source.relative_to(repo))]: raise ValueError('Source changed during probe')
    np.savez_compressed(out/'residuals.npz', **arrays)
    (out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    (out/'manifest.json').write_text(json.dumps({str(path.relative_to(out)): sha(path) for path in out.rglob('*') if path.is_file()}, indent=2)+'\n')


if __name__ == '__main__': main()

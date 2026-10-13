#!/usr/bin/env python3
"""Directional Jacobian checks on an archived native SUBOFF numerical state."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
from tensorfvm.native_mesh3d import NativeMesh3D
from tensorfvm.native_hybrid_ns3d import NativeHybridNavierStokes3D
from tensorfvm.verification.native_run import NativeRunEvidence,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();evidence=NativeRunEvidence(a.output,__file__)
    cfg=json.loads((a.run/'settings.json').read_text());summary=json.loads((a.run/'summary.json').read_text())
    if cfg['backend']!='native_hybrid_cell_and_shared_face_velocity':raise ValueError('native hybrid state required')
    for name,sha in [('input.msh',cfg['mesh_sha256']),('fields.npz',summary['fields_sha256'])]:
        if hashlib.sha256((a.run/name).read_bytes()).hexdigest()!=sha:raise ValueError('modified input')
        shutil.copy2(a.run/name,a.output/name)
    for name in ('settings.json','summary.json'):shutil.copy2(a.run/name,a.output/('input-'+name))
    m=NativeMesh3D.from_gmsh(a.output/'input.msh',triangulate_faces=True)
    wall=m.patch_labels=='wall';fixed=(m.neighbor<0)&(m.patch_labels!='outlet')
    boundary=np.tile([1.,0.,0.],(len(m.owner),1));boundary[wall]=0
    s=NativeHybridNavierStokes3D(m,fixed,boundary,viscosity=cfg['nu'],
        convection_order=cfg['convection_order'],backflow_velocity=cfg['outlet_backflow_velocity'])
    with np.load(a.output/'fields.npz',allow_pickle=False) as f:
        s.state=np.r_[f['velocity'].ravel(),f['face_velocity'].ravel()];s.p=f['pressure'].copy()
        fixed_dofs=np.setdiff1d(np.arange(len(s.state)),s.viscous.free)
        np.testing.assert_allclose(s.state[fixed_dofs],s.prescribed[fixed_dofs],atol=1e-14)
    free=s.viscous.free;old=s.state.copy();old_p=s.p.copy()
    jac=s.viscous.full_matrix+s.convection.hybrid_jacobian(s.state,s.face_mean.matrix)
    rng=np.random.default_rng(72491);rows=[]
    def residual(state,pressure):
        matrix,source=s._current_system(state)
        return (matrix@state+source-s.divergence.T@pressure-s.load)[free]
    for direction in range(3):
        du=np.zeros_like(s.state);du[free]=rng.normal(size=len(free));du/=np.linalg.norm(du)
        dp=rng.normal(size=s.N);dp/=np.linalg.norm(dp)
        exact=(jac@du-s.divergence.T@dp)[free]
        for h in (1e-3,1e-4,1e-5,1e-6,1e-7):
            plus=s.state+h*du;minus=s.state-h*du
            diff=(residual(plus,s.p+h*dp)-residual(minus,s.p-h*dp))/(2*h)
            changes=int(np.count_nonzero((s.volume_flux(plus)>=0)!=(s.volume_flux(minus)>=0)))
            row=dict(direction=direction,step=h,relative_derivative_error=float(np.linalg.norm(diff-exact)/np.linalg.norm(exact)),
                max_absolute_derivative_error=float(np.max(abs(diff-exact))),upwind_sign_changes=changes)
            rows.append(row);print(row,flush=True)
    np.testing.assert_array_equal(s.state,old);np.testing.assert_array_equal(s.p,old_p)
    evidence.verify_sources()
    passed=all(any(r['direction']==i and r['upwind_sign_changes']==0 and r['relative_derivative_error']<1e-6 for r in rows) for i in range(3))
    write_json(a.output/'summary.json',dict(directional_checks_passed=passed,input_numerically_converged=summary['numerically_converged'],
        Re_L=cfg['Re_L'],cells=s.N,rows=rows,source_integrity_verified=True,
        physical_suboff_benchmark_passed=False,scope='Three deterministic directions at one archived state; does not prove global nonlinear convergence'))
    raise SystemExit(0 if passed else 2)


if __name__=='__main__':main()

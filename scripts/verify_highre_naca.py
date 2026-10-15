"""Reproducible NACA diagnostic using the production unstructured SIMPLE/SA solver.

Numerical convergence and physical qualification are separate. This runner never
declares physical qualification; the independent reference/grid report does that.
"""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from tensorfvm.solver import SolverConfig
from tensorfvm.multi_element_skew_pressure import SkewPressureMultiElementSolver
from tensorfvm.hlpw30p30n import mesh_quality
from tensorfvm.verification.native_run import NativeRunEvidence, write_json


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mesh', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--alpha', type=float, default=10.)
    p.add_argument('--reynolds', type=float, default=6e6)
    p.add_argument('--iterations', type=int, default=2500)
    p.add_argument('--save-every', type=int, default=50)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--u-relaxation', type=float, default=.85)
    p.add_argument('--sa-relaxation', type=float, default=.9)
    p.add_argument('--linear-budget-scale', type=float, default=1.)
    a = p.parse_args()
    if a.save_every < 1 or a.linear_budget_scale <= 0:
        p.error('positive save interval and linear budget required')
    torch.set_num_threads(1)
    out = a.output.resolve()
    c = SolverConfig(nx=1, ny=1, length=1000., height=1000., cylinder_radius=None,
        mesh_type='three-element', mesh_file=str(a.mesh.resolve()), single_element=True,
        airfoil_chord=1., inlet_velocity=1., density=1., reynolds=a.reynolds,
        angle_of_attack=a.alpha, turbulence_model='spalart-allmaras', sa_freestream_ratio=3.,
        velocity_relaxation=a.u_relaxation, pressure_relaxation=.3,
        turbulence_relaxation=a.sa_relaxation, pseudo_time_step=.02,
        max_iterations=a.iterations, tolerance=1e-5, device='cpu')
    settings = dict(config=asdict(c), mesh_sha256=sha(a.mesh),
        linear_backend='krylov', linear_budget_scale=a.linear_budget_scale,
        lu_refresh_interval=10, sa_inner_iterations=3, sa_linear_upwind=False,
        pressure_gradient='gauss-skew-corrected', convection='bounded linear upwind',
        ramp='blend=(iteration-40)/60 clipped to [0,1]; pseudo dt .02 until iteration 100',
        physical_acceptance='not evaluated by runner; requires reference and grid report')
    s = SkewPressureMultiElementSolver(c)
    bounds = s.mesh.vertices.max(0).values-s.mesh.vertices.min(0).values
    s.config = replace(c, length=float(bounds[0]), height=float(bounds[1]))
    s.linear_backend='krylov'; s.lu_refresh_interval=10
    s.linear_budget_scale=a.linear_budget_scale
    quality=mesh_quality(s.mesh,a.alpha)
    if not quality['accepted']:
        raise ValueError('Mesh quality rejected: '+json.dumps(quality))
    root=Path(__file__).resolve().parents[1]
    if a.resume:
        old=json.loads((out/'settings.json').read_text())
        old['config']['max_iterations']=settings['config']['max_iterations']
        if old != settings:
            raise ValueError('Restart mesh or numerical settings changed')
        hashes=json.loads((out/'sources.json').read_text())
        for name,digest in hashes.items():
            if sha(root/name)!=digest or sha(out/'source-snapshot'/name)!=digest:
                raise ValueError('Restart source mismatch: '+name)
        progress=json.loads((out/'progress.json').read_text())
        if sha(out/'checkpoint.npz')!=progress['checkpoint_sha256']:
            raise ValueError('Restart checkpoint hash mismatch')
        if sha(out/'mesh.msh')!=settings['mesh_sha256']:
            raise ValueError('Saved mesh hash mismatch')
        with np.load(out/'checkpoint.npz') as f:
            for name,target in [('velocity',s.velocity),('pressure',s.p),('mass_flux',s.mass_flux),('nu_tilde',s.nu_tilde)]:
                if f[name].shape!=tuple(target.shape) or not np.isfinite(f[name]).all():
                    raise ValueError('Invalid restart field '+name)
                target.copy_(torch.from_numpy(f[name]))
            s.history=json.loads(str(f['history'].item()))
            records=json.loads(str(f['records'].item()))
        s.turbulent_kinematic_viscosity=s._sa_eddy_viscosity(s.nu_tilde)
        write_json(out/'restart-request.json',settings)
    else:
        evidence=NativeRunEvidence(out,__file__)
        # Include lazy-loaded numerical dependencies as well as current imports.
        for source in list((root/'src/tensorfvm').rglob('*.py'))+[root/'work/naca0012/make_naca_mesh.py']:
            name=str(source.relative_to(root)); target=out/'source-snapshot'/name
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
            evidence.hashes[name]=sha(source)
        hashes=evidence.hashes;write_json(out/'sources.json',hashes)
        write_json(out/'settings.json',settings);shutil.copy2(a.mesh,out/'mesh.msh')
        for name in ['metadata.json','mesh.geo']:
            if (a.mesh.parent/name).exists():shutil.copy2(a.mesh.parent/name,out/name)
        s.history=[];records=[]
    write_json(out/'mesh-quality.json',quality)
    def verify_sources():
        for name,digest in hashes.items():
            if sha(root/name)!=digest:raise RuntimeError('Source changed during run: '+name)
    def save():
        np.savez_compressed(out/'checkpoint-next.npz',velocity=s.velocity.numpy(),pressure=s.p.numpy(),
            nu_tilde=s.nu_tilde.numpy(),mass_flux=s.mass_flux.numpy(),
            history=json.dumps(s.history),records=json.dumps(records))
        (out/'checkpoint-next.npz').replace(out/'checkpoint.npz')
        write_json(out/'history.json',records)
        wall=s.mesh.masks['wall'];owners=s.o[wall]
        normal=-s.mesh.face_normals[wall]
        tangent=torch.stack([normal[:,1],-normal[:,0]],1)
        distance=abs(((s.mesh.centers.reshape(-1,2)[owners]-s.mesh.face_centers[wall])*normal).sum(1))
        tau=s.kinematic_viscosity*(s.velocity[owners]*tangent).sum(1)/distance
        wall_values=np.column_stack([s.mesh.face_centers[wall].numpy(),normal.numpy(),
            (2*s.reconstructed_pressure_faces(s.p)[wall]).numpy(),(2*tau).numpy(),
            (distance*torch.sqrt(abs(tau))/s.kinematic_viscosity).numpy()])
        np.savetxt(out/'wall.csv',wall_values,delimiter=',',header='x,y,nx,ny,Cp,Cf_tangent,y_plus',comments='')
        write_json(out/'progress.json',dict(iterations=len(s.history),cells=s.count,
            elapsed_seconds=time.perf_counter()-start,last=records[-1] if records else None,
            solver_converged=s.converged,physical_qualified=False,checkpoint_sha256=sha(out/'checkpoint.npz')))
    start=time.perf_counter();failure=None
    print(json.dumps(dict(cells=s.count,mesh_quality=quality,settings=settings)),flush=True)
    try:
        for i in range(len(s.history),a.iterations):
            s.convection_blend=min(1.,max(0.,(i-40)/60))
            if i>=100:s.config=replace(s.config,pseudo_time_step=None)
            metrics=s.step()
            if not all(np.isfinite(v) for v in metrics.values()):raise RuntimeError('Nonfinite metrics')
            if i%10==0 or s.converged or i+1==a.iterations:
                row=dict(iteration=i+1,elapsed_seconds=time.perf_counter()-start,
                    coefficients=s._aerodynamic_coefficients(),metrics=metrics)
                records.append(row);print(json.dumps(row),flush=True)
            if (i+1)%a.save_every==0 or s.converged:save()
            if s.converged:break
    except (RuntimeError,ValueError,KeyboardInterrupt) as e:
        failure=dict(type=type(e).__name__,message=str(e))
        # Preserve the last complete checkpoint; do not overwrite it with a failed step.
    verify_sources()
    if failure is None:save()
    write_json(out/'status.json',dict(failure=failure,iterations=len(s.history),solver_converged=s.converged and failure is None,
        physical_qualified=False,source_integrity_verified=True,elapsed_seconds=time.perf_counter()-start,
        checkpoint_sha256=sha(out/'checkpoint.npz') if (out/'checkpoint.npz').exists() else None))
    return 0 if s.converged and failure is None else 2


if __name__=='__main__':raise SystemExit(main())

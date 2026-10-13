#!/usr/bin/env python3
"""Native hybrid SUBOFF laminar startup; no experimental accuracy qualification."""
import argparse
import hashlib
from pathlib import Path
import shutil
import time
import numpy as np
from tensorfvm.native_mesh3d import NativeMesh3D
from tensorfvm.native_hybrid_ns3d import NativeHybridNavierStokes3D
from tensorfvm.native_boundary_constraints3d import require_rigid_motion_constraints
from tensorfvm.verification.native_run import NativeRunEvidence,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mesh',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--reynolds',type=float,required=True)
    p.add_argument('--algorithm',choices=['newton','simplec'],default='newton')
    p.add_argument('--max-iterations',type=int,default=40)
    a=p.parse_args()
    if not np.isfinite(a.reynolds) or a.reynolds<=0:p.error('positive finite Reynolds number required')
    if a.max_iterations<1:p.error('positive iteration limit required')
    evidence=NativeRunEvidence(a.output,__file__);out=a.output
    shutil.copy2(a.mesh,out/'input.msh');mesh_hash=hashlib.sha256((out/'input.msh').read_bytes()).hexdigest()
    write_json(out/'settings.json',dict(backend='native_hybrid_cell_and_shared_face_velocity',
        flow='steady laminar; no SST or wall function',Re_L=a.reynolds,L=1.,U=1.,rho=1.,nu=1/a.reynolds,
        algorithm=a.algorithm,max_iterations=a.max_iterations,convection_order=2,body_force=[0,0,0],
        inlet_and_farfield_velocity=[1,0,0],wall_velocity=[0,0,0],outlet='zero total outward traction',
        outlet_backflow_velocity=[1,0,0],initialization='native Stokes numerical solve, no reference interior state',
        numerical_tolerance=1e-8,absolute_integrated_ledger_tolerance=1e-8,
        reference_area='L^2=1; no experimental Cd normalization assumed',mesh_sha256=mesh_hash,
        wall_pressure_status='owner cell pressure and normal traction proxy saved separately; wall Cp not certified',
        physical_suboff_benchmark_passed=False))
    tick=time.perf_counter();print('IMPORT_MESH',flush=True)
    m=NativeMesh3D.from_gmsh(out/'input.msh',triangulate_faces=True)
    b=m.neighbor<0;wall=m.patch_labels=='wall';outlet=m.patch_labels=='outlet'
    if not wall.any() or not outlet.any() or set(m.patch_labels[b])!={'wall','inlet','outlet','farfield'}:
        raise ValueError('expected complete SUBOFF physical boundary patches')
    fixed=b & ~outlet;velocity=np.tile([1.,0.,0.],(len(b),1));velocity[wall]=0.
    constraint=require_rigid_motion_constraints(m,np.repeat(fixed[:,None],3,axis=1))
    write_json(out/'mesh-audit.json',dict(**m.audit(),
        cell_type_counts={k:m.cell_types.count(k) for k in set(m.cell_types)},rigid_constraint_rank=constraint['rank']))
    print(dict(stage='BUILD_NATIVE_HYBRID',cells=len(m.cells),faces=len(b)),flush=True)
    s=NativeHybridNavierStokes3D(m,fixed,velocity,viscosity=1/a.reynolds,
        boundary_traction=0.,backflow_velocity=[1,0,0],convection_order=2)
    failure=None
    try:
        print('INITIALIZE_STOKES',flush=True);initial=s.initialize_stokes()
        write_json(out/'initialization.json',initial)
        print(dict(stage='SOLVE_NS',initial=initial),flush=True)
        def progress(row):
            write_json(out/'progress.json',row);print(row,flush=True)
        if a.algorithm=='newton':r=s.solve_newton(max_iterations=a.max_iterations,tolerance=1e-8,callback=progress)
        else:r=s.solve(algorithm='simplec',max_outer_iterations=a.max_iterations,tolerance=1e-8,callback=progress)
    except (RuntimeError,ValueError) as exc:
        failure=f'{type(exc).__name__}: {exc}';s.nonlinear_converged=False;r=s.result()
        print(dict(failure=failure),flush=True)
    area=np.linalg.norm(m.area_vectors,axis=1);normal=m.area_vectors/area[:,None]
    t=r['total_traction_integrated'];wall_force=-t[wall].sum(0)
    pressure_force=(r['pressure'][m.owner[wall],None]*m.area_vectors[wall]).sum(0)
    wall_cp_owner=2*r['pressure'][m.owner[wall]]
    wall_cp_normal_traction=-2*np.sum(t[wall]*normal[wall],axis=1)/area[wall]
    internal=m.neighbor>=0
    def div(q):
        v=np.zeros((len(m.cells),)+q.shape[1:]);np.add.at(v,m.owner,q);np.add.at(v,m.neighbor[internal],-q[internal]);return v
    mass=float(np.max(np.abs(div(r['flux'])/m.cell_volumes)))
    momentum=float(np.max(np.abs(div(r['momentum_flux_integrated']))))
    other_momentum=r['momentum_flux_integrated'][b & ~wall].sum(0)
    force_balance=float(np.max(np.abs(wall_force+other_momentum)))
    accepted=bool(r['numerically_converged'] and failure is None and max(mass,momentum,force_balance,
        r['interface_traction_defect_max'],r['energy_balance_error'])<1e-8)
    fields={k:r[k] for k in ('velocity','face_velocity','pressure','flux','total_traction_integrated','momentum_flux_integrated')}
    np.savez_compressed(out/'fields.npz',vertices=m.vertices,face_vertices=np.asarray(m.face_vertices),
        cell_vertices=np.concatenate(m.cells),cell_offsets=np.r_[0,np.cumsum([len(c) for c in m.cells])],
        cell_centers=m.cell_centers,cell_volumes=m.cell_volumes,face_centers=m.face_centers,
        area_vectors=m.area_vectors,owner=m.owner,neighbor=m.neighbor,patch_labels=m.patch_labels,
        wall_cp_owner=wall_cp_owner,wall_cp_normal_traction_proxy=wall_cp_normal_traction,**fields)
    history=r.get('newton_history',getattr(s,'newton_history',r['outer_history']))
    write_json(out/'history.json',history)
    evidence.verify_sources()
    summary=dict(numerically_converged=accepted,solver_converged=bool(r['numerically_converged']),
        failure=failure or r.get('newton_failure'),cells=len(m.cells),faces=len(b),Re_L=a.reynolds,
        mass_divergence_max=mass,local_integrated_momentum_max=momentum,force_balance_max=force_balance,
        interface_traction_defect_max=r['interface_traction_defect_max'],energy_balance_error=r['energy_balance_error'],
        wall_force=wall_force.tolist(),wall_owner_pressure_force=pressure_force.tolist(),
        wall_remaining_numerical_force=(wall_force-pressure_force).tolist(),Cd_L2=float(2*wall_force[0]),
        wall_Cp_certified=False,grid_convergence_certified=False,physical_suboff_benchmark_passed=False,
        max_cell_speed=float(np.linalg.norm(r['velocity'],axis=1).max()),
        source_integrity_verified=True,fields_sha256=hashlib.sha256((out/'fields.npz').read_bytes()).hexdigest(),
        runtime_seconds=time.perf_counter()-tick)
    write_json(out/'summary.json',summary);print(summary,flush=True)
    raise SystemExit(0 if accepted else 2)


if __name__=='__main__':main()

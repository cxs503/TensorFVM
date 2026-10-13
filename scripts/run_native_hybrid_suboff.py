#!/usr/bin/env python3
"""Native hybrid SUBOFF laminar startup; no experimental accuracy qualification."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
import numpy as np
from tensorfvm.native_mesh3d import NativeMesh3D
from tensorfvm.native_block_solver3d import NativeBlockSolveError
from tensorfvm.native_hybrid_ns3d import NativeHybridNavierStokes3D
from tensorfvm.native_boundary_constraints3d import require_rigid_motion_constraints
from tensorfvm.verification.native_run import NativeRunEvidence,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mesh',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--reynolds',type=float,required=True)
    p.add_argument('--algorithm',choices=['newton','simplec'],default='newton')
    p.add_argument('--inexact-newton',action='store_true',help='Permit iteration-limited AMG directions only after original matrix checks; nonlinear gates unchanged')
    p.add_argument('--max-iterations',type=int,default=40)
    p.add_argument('--tolerance',type=float,default=1e-8,help='May tighten but cannot relax the 1e-8 benchmark gate')
    p.add_argument('--initial-run',type=Path,help='Previous native laminar numerical state on the identical mesh; Re may differ for continuation')
    p.add_argument('--linear-method',choices=['direct','reused_lu','block_amg'],default='direct')
    p.add_argument('--momentum-iterations',type=int,default=3000)
    p.add_argument('--momentum-relaxation',type=float,default=.5)
    p.add_argument('--momentum-relaxation-basis',choices=['absolute_row_sum','diagonal'],default='absolute_row_sum')
    a=p.parse_args()
    if not np.isfinite(a.reynolds) or a.reynolds<=0:p.error('positive finite Reynolds number required')
    if not np.isfinite(a.tolerance) or not 0<a.tolerance<=1e-8:p.error('tolerance must be positive and <=1e-8')
    if a.inexact_newton and (a.algorithm!='newton' or a.linear_method!='block_amg'):p.error('--inexact-newton requires newton and block_amg')
    if a.max_iterations<1:p.error('positive iteration limit required')
    evidence=NativeRunEvidence(a.output,__file__);out=a.output
    shutil.copy2(a.mesh,out/'input.msh');mesh_hash=hashlib.sha256((out/'input.msh').read_bytes()).hexdigest()
    initial_binding=None
    if a.initial_run is not None:
        old_config=json.loads((a.initial_run/'settings.json').read_text());old_summary=json.loads((a.initial_run/'summary.json').read_text())
        if old_config.get('backend')!='native_hybrid_cell_and_shared_face_velocity' or old_config.get('flow')!='steady laminar; no SST or wall function':
            raise ValueError('restart requires a native hybrid laminar numerical run')
        if old_config['mesh_sha256']!=mesh_hash:raise ValueError('restart mesh mismatch')
        if not old_summary['source_integrity_verified']:raise ValueError('unverified restart source')
        if hashlib.sha256((a.initial_run/'fields.npz').read_bytes()).hexdigest()!=old_summary['fields_sha256']:raise ValueError('restart fields hash mismatch')
        initial_binding=dict(previous_Re_L=old_config['Re_L'],previous_numerically_converged=old_summary['numerically_converged'],origin=str(a.initial_run),files={})
        for name in ('fields.npz','settings.json','summary.json'):
            shutil.copy2(a.initial_run/name,out/('initial-'+name))
            initial_binding['files'][name]=hashlib.sha256((out/('initial-'+name)).read_bytes()).hexdigest()
    write_json(out/'settings.json',dict(backend='native_hybrid_cell_and_shared_face_velocity',
        flow='steady laminar; no SST or wall function',Re_L=a.reynolds,L=1.,U=1.,rho=1.,nu=1/a.reynolds,
        algorithm=a.algorithm,max_iterations=a.max_iterations,convection_order=2,body_force=[0,0,0],
        momentum_iterations=a.momentum_iterations,momentum_relaxation=a.momentum_relaxation,
        momentum_relaxation_basis=a.momentum_relaxation_basis,linear_method=a.linear_method,
        initial_run=initial_binding,inexact_newton=a.inexact_newton,
        inlet_and_farfield_velocity=[1,0,0],wall_velocity=[0,0,0],outlet='zero total outward traction',
        outlet_backflow_velocity=[1,0,0],initialization='previous numerical iterate; current equations rebuilt' if initial_binding else 'native Stokes numerical solve, no reference interior state',
        numerical_tolerance=a.tolerance,initialization_tolerance=1e-8,absolute_integrated_ledger_tolerance=1e-8,
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
        boundary_traction=0.,backflow_velocity=[1,0,0],convection_order=2,
        momentum_relaxation_basis=a.momentum_relaxation_basis,coupled_linear_method=a.linear_method)
    failure=None
    try:
        if initial_binding is None:
            print('INITIALIZE_STOKES',flush=True);initial=s.initialize_stokes(tolerance=1e-8)
        else:
            with np.load(out/'initial-fields.npz',allow_pickle=False) as previous:
                state=np.r_[previous['velocity'].ravel(),previous['face_velocity'].ravel()];pressure=previous['pressure'].copy()
                if state.shape!=s.state.shape or pressure.shape!=s.p.shape or not np.isfinite(state).all() or not np.isfinite(pressure).all():
                    raise ValueError('invalid restart state shape/values')
                np.testing.assert_array_equal(previous['owner'],m.owner);np.testing.assert_array_equal(previous['neighbor'],m.neighbor)
                np.testing.assert_allclose(previous['face_velocity'][fixed],velocity[fixed],rtol=0,atol=1e-14)
                np.testing.assert_allclose(previous['flux'],np.sum(previous['face_velocity']*m.area_vectors,axis=1),rtol=0,atol=1e-14)
                s.state=state;s.p=pressure;s.converged=False;s.nonlinear_converged=False
            initial=dict(method='previous_native_numerical_state',**initial_binding)

        write_json(out/'initialization.json',initial)
        print(dict(stage='SOLVE_NS',initial=initial),flush=True)
        def progress(row):
            force=-s.result()['total_traction_integrated'][wall].sum(0)
            row.update(wall_force=force.tolist(),Cd_L2=float(2*force[0]))
            write_json(out/'progress.json',row);print(row,flush=True)
        if a.algorithm=='newton':r=s.solve_newton(max_iterations=a.max_iterations,tolerance=a.tolerance,callback=progress,allow_inexact_linear_steps=a.inexact_newton)
        else:r=s.solve(algorithm='simplec',max_outer_iterations=a.max_iterations,
            max_iterations=a.momentum_iterations,relaxation=a.momentum_relaxation,tolerance=a.tolerance,callback=progress)
    except (RuntimeError,ValueError) as exc:
        failure=f'{type(exc).__name__}: {exc}';s.nonlinear_converged=False;r=s.result()
        if isinstance(exc,NativeBlockSolveError):
            np.savez_compressed(out/'rejected-linear-candidate.npz',candidate=exc.candidate,
                equation_defect=exc.defect,free_velocity_dofs=s.viscous.free,
                cell_volumes=m.cell_volumes,gauge=np.asarray(s.gauge))
            write_json(out/'rejected-linear-candidate.json',dict(**exc.record,
                fields_sha256=hashlib.sha256((out/'rejected-linear-candidate.npz').read_bytes()).hexdigest(),
                changes_accepted_state=False))
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
    write_json(out/'linear-history.json',list(s.history))
    write_json(out/'coupled-linear-history.json',list(s.coupled_linear_history))
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

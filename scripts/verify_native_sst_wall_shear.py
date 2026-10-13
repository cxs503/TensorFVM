#!/usr/bin/env python3
"""Actual coupled native SST/Spalding shear diagnostic; no physical reference claim."""
import argparse,hashlib,time
from pathlib import Path
import numpy as np
from tensorfvm.verification.native_run import NativeRunEvidence,write_json
from tensorfvm.verification.native_sst_wall import sst_wall_shear_solver


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--layers',type=int,default=4);p.add_argument('--nu',type=float,default=.001)
    p.add_argument('--steps',type=int,default=120);p.add_argument('--dt',type=float,default=1.)
    p.add_argument('--momentum-method',choices=['newton','simplec'],default='newton')
    a=p.parse_args();evidence=NativeRunEvidence(a.output,__file__);start=time.perf_counter()
    s=sst_wall_shear_solver(a.layers,a.nu);m=s.mesh;flow=s.flow
    write_json(a.output/'settings.json',dict(kind=__doc__,layers=a.layers,nu=a.nu,rho=1.,Re_H_deltaU=1/a.nu,
        cells=flow.N,faces=flow.F,body_force=0.,steps=a.steps,dt=a.dt,relaxation=.7,tolerance=1e-10,
        k_inflow=.01,omega_inflow=2.,inlet_provenance='explicit numerical diagnostic assumptions, not experiment',
        velocity_inflow='fixed Spalding Couette profile with reference bulk viscosity .02; inlet data only',
        initial_bulk_viscosity=.02,bulk_viscosity_during_RANS='nu + current SST nu_t; no prescribed eddy viscosity',
        wall_treatment=s.wall_treatment,upper_wall='moving no-slip U=1; k=0; omega=60nu/(beta1*d1^2)',
        scalar_correction='nonlinear_mmp',scalar_precision='longdouble',scalar_work_balance_policy='relative',
        scalar_warm_start=True,require_steady_ledgers=True,momentum_method=a.momentum_method,
        physical_suboff_benchmark_passed=False))
    failure=None
    try:
        def progress(row):write_json(a.output/'progress.json',row);print(row,flush=True)
        s.solve(max_iterations=a.steps,dt=a.dt,tolerance=1e-10,relaxation=.7,
            momentum_method=a.momentum_method,require_steady_ledgers=True,callback=progress)
    except (ValueError,RuntimeError) as exc:
        failure=dict(type=type(exc).__name__,message=str(exc));print(failure,flush=True)
    r=s.result();current=s.current_residual(with_ledgers=True)
    acceptance=s._steady_ledger_acceptance(current,flow.result())
    fields=dict(vertices=m.vertices,face_vertices=np.asarray(m.face_vertices),owner=m.owner,neighbor=m.neighbor,
        area_vectors=m.area_vectors,face_centers=m.face_centers,cell_centers=m.cell_centers,cell_volumes=m.cell_volumes,
        wall_distance=s.wall_distance,k_boundary=s.boundary['k_boundary'],omega_boundary=s.boundary['omega_boundary'],
        k_dirichlet=s.transport.k_mask,omega_dirichlet=s.transport.omega_mask,
        wall_faces=flow.wall_model.faces,wall_sampling_distance=flow.wall_model.distance,
        wall_velocity=flow.wall_model.wall_velocity)
    for key in ('velocity','face_velocity','modified_pressure','physical_pressure','k','omega','k_faces','omega_faces','nu_t','flux','total_traction_integrated','momentum_flux_integrated'):
        fields[key]=r[key]
    for key in ('cells','omega','production','face_omega','face_production','y_plus_k','physical_wall_gradient_magnitude'):
        fields['wall_'+key]=current['transport']['closure']['wall_model'][key]
    for name in ('k','omega'):
        for key in ('total_flux','cell_balance','cell_constraint_reaction','work_balance_cell_terms','work_balance_interface_terms'):
            fields[name+'_'+key]=current['transport'][name]['steady_ledger'][key]
    np.savez_compressed(a.output/'fields.npz',**fields)
    write_json(a.output/'history.json',s.history)
    evidence.verify_sources()
    summary=dict(source_integrity_verified=True,completed_steps=len(s.history),failure=failure,
        solver_converged=r['numerically_converged'],current_steady_ledgers=acceptance,
        current_equations={key:value for key,value in current.items() if key!='transport'},
        wall_y_plus_min=float(r['wall_model']['y_plus'].min()),wall_y_plus_max=float(r['wall_model']['y_plus'].max()),
        viscosity_min=float(flow.viscous.mu.min()),viscosity_max=float(flow.viscous.mu.max()),
        fields_sha256=hashlib.sha256((a.output/'fields.npz').read_bytes()).hexdigest(),runtime_seconds=time.perf_counter()-start,
        physical_suboff_benchmark_passed=False,physical_channel_benchmark_passed=False)
    write_json(a.output/'summary.json',summary);print(summary,flush=True)
    raise SystemExit(0 if failure is None and r['numerically_converged'] and acceptance['passed'] else 2)


if __name__=='__main__':main()

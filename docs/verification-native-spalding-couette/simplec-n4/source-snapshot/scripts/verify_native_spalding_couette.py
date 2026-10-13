#!/usr/bin/env python3
"""Unforced native NS with cell-sampled Spalding stress; not SST qualification."""
import argparse,hashlib,time
from pathlib import Path
import numpy as np
from tensorfvm.verification.native_run import NativeRunEvidence,write_json
from tensorfvm.verification.native_spalding import spalding_couette_solver


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--algorithm',choices=['newton','simple','simplec'],default='simplec')
    p.add_argument('--layers',type=int,default=4)
    p.add_argument('--bulk-viscosity',type=float,default=.02)
    p.add_argument('--molecular-viscosity',type=float,default=.001)
    p.add_argument('--wall-speed',type=float,default=0.)
    a=p.parse_args();evidence=NativeRunEvidence(a.output,__file__);start=time.perf_counter()
    m,s,expected,tau,low,top=spalding_couette_solver(a.layers,a.bulk_viscosity,a.molecular_viscosity,a.wall_speed)
    write_json(a.output/'settings.json',dict(kind=__doc__,algorithm=a.algorithm,cells=s.N,faces=s.F,layers=a.layers,
        bulk_viscosity=a.bulk_viscosity,molecular_viscosity=a.molecular_viscosity,rho=1.,wall_speed=a.wall_speed,
        upper_wall_speed=a.wall_speed+1.,cell_sampling_distance=.5/a.layers,body_force=0.,
        incoming_velocity='analytical one-dimensional Couette profile used as physical incoming boundary data only',
        initialization='native Stokes, without nonlinear wall stress; no analytical interior field assigned',
        momentum_relaxation_basis='absolute_row_sum',relaxation=.5,tolerance=1e-9,
        sst_coupled=False,physical_suboff_benchmark_passed=False))
    s.initialize_stokes()
    if a.algorithm=='newton':r=s.solve_newton(max_iterations=30,tolerance=1e-9)
    else:r=s.solve(algorithm=a.algorithm,max_outer_iterations=80,max_iterations=3000,relaxation=.5,tolerance=1e-9)
    wall=r['wall_model'];inside=m.neighbor>=0
    balance=np.zeros((s.N,3));np.add.at(balance,m.owner,r['momentum_flux_integrated'])
    np.add.at(balance,m.neighbor[inside],-r['momentum_flux_integrated'][inside])
    divergence=np.zeros(s.N);np.add.at(divergence,m.owner,r['flux'])
    np.add.at(divergence,m.neighbor[inside],-r['flux'][inside])
    area=np.linalg.norm(m.area_vectors,axis=1)
    independent=dict(velocity_max_error=float(np.max(abs(r['velocity']-expected))),
        wall_shear_max_error=float(np.max(abs(r['total_traction_integrated'][low,0]/area[low]+tau))),
        mass_divergence_max=float(np.max(abs(divergence/m.cell_volumes))),
        local_momentum_balance_max=float(np.max(abs(balance))),
        global_momentum_balance_max=float(np.max(abs(r['momentum_flux_integrated'][~inside].sum(0)))),
        wall_normal_flux_max=float(np.max(abs(r['flux'][low]))))
    small={key:wall[key] for key in ('traction_defect_max','boundary_power','wall_motion_power','sample_dissipation','sampling_transfer_power','work_decomposition_error','sampling')}
    passed=bool(r['numerically_converged'] and independent['velocity_max_error']<1e-7
        and all(v<1e-8 for k,v in independent.items() if k!='velocity_max_error')
        and r['energy_balance_error']<1e-8 and r['interface_traction_defect_max']<1e-8
        and small['traction_defect_max']<1e-8 and small['work_decomposition_error']<1e-10)
    np.savez_compressed(a.output/'fields.npz',vertices=m.vertices,face_vertices=np.asarray(m.face_vertices),
        cell_centers=m.cell_centers,face_centers=m.face_centers,owner=m.owner,neighbor=m.neighbor,area_vectors=m.area_vectors,
        velocity=r['velocity'],face_velocity=r['face_velocity'],pressure=r['pressure'],reference_velocity=expected,
        flux=r['flux'],momentum_flux=r['momentum_flux_integrated'],total_traction=r['total_traction_integrated'],
        wall_faces=wall['faces'],wall_target_traction=wall['traction_integrated'],wall_y_plus=wall['y_plus'])
    write_json(a.output/'history.json',s.newton_history if a.algorithm=='newton' else s.outer_history)
    write_json(a.output/'momentum-history.json',s.history)
    evidence.verify_sources()
    summary=dict(source_integrity_verified=True,solver_converged=r['numerically_converged'],boundary_verification_passed=passed,
        independent=independent,wall=small,reference_shear=tau,energy_balance_error=r['energy_balance_error'],
        interface_traction_defect_max=r['interface_traction_defect_max'],wall_y_plus_min=float(wall['y_plus'].min()),
        wall_y_plus_max=float(wall['y_plus'].max()),bulk_viscous_energy=r['bulk_viscous_energy'],
        fields_sha256=hashlib.sha256((a.output/'fields.npz').read_bytes()).hexdigest(),
        runtime_seconds=time.perf_counter()-start,physical_suboff_benchmark_passed=False,sst_coupled=False)
    write_json(a.output/'summary.json',summary);print(summary,flush=True)
    raise SystemExit(0 if passed else 2)


if __name__=='__main__':main()

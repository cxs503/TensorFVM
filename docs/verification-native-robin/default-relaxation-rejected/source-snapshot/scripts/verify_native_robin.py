#!/usr/bin/env python3
"""Validate native Robin wall traction, velocity, force, and mechanical power."""
import argparse
import hashlib
import time
from pathlib import Path
import numpy as np
from tensorfvm.verification.native_robin import slip_couette_solver
from tensorfvm.verification.native_run import NativeRunEvidence,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output;e=NativeRunEvidence(out,__file__)
    write_json(out/'settings.json',dict(case='unforced finite-slip Couette',mu=.7,density=1.,height=1.,wall_speeds=[0,1],
        friction=[2,5],robin_sign='outward total traction + C u = g',normal_velocity='zero separately imposed at planar walls',
        body_force=[0,0,0],reference_pressure=0.,analytic_interior_seed=False,wall_function=False,
        full_curved_impermeable_wall_supported=False,numerical_and_absolute_ledger_gate=1e-8,accuracy_gate=.03,
        physical_suboff_benchmark_passed=False))
    rows=[]
    for n in (4,8,16):
        for algorithm in ('coupled','simplec','newton'):
            tick=time.perf_counter();m,s,exact,tau,low,high,g=slip_couette_solver(n,ns=algorithm=='newton')
            if algorithm=='newton':s.initialize_stokes();r=s.solve_newton(tolerance=1e-10)
            else:r=s.solve(algorithm=algorithm,tolerance=1e-10,max_iterations=6000)
            area=np.linalg.norm(m.area_vectors,axis=1);wall=low|high
            robin_defect=r['total_traction_integrated']+r['robin_reaction_integrated']-area[:,None]*g
            slip=r['face_velocity'].copy();slip[high,0]-=1.
            slip_power=float(np.einsum('fi,fij,fj,f->',slip,s.viscous.robin_coefficient,slip,area))
            wall_power=float(r['total_traction_integrated'][high,0].sum())
            ledger=dict(robin_boundary=float(np.max(abs(robin_defect[s.viscous.traction_components]))),
                divergence=float(np.max(abs(s.divergence@s.state/m.cell_volumes))),
                global_force=float(np.max(abs(r['total_traction_integrated'][m.neighbor<0].sum(0)))),
                interface_traction=r['interface_traction_defect_max'],algebraic_energy=r['energy_balance_error'],
                mechanical_energy=abs(wall_power-r['bulk_viscous_energy']-slip_power))
            velocity=float(np.sqrt(np.sum(m.cell_volumes[:,None]*(r['velocity']-exact)**2)/np.sum(m.cell_volumes[:,None]*exact**2)))
            forces=[float(r['total_traction_integrated'][low,0].sum()),float(r['total_traction_integrated'][high,0].sum())]
            force_error=max(abs(forces[0]+tau),abs(forces[1]-tau))/tau
            pressure=float(np.max(abs(r['pressure'])))
            row=dict(case=f'n{n}-{algorithm}',n=n,algorithm=algorithm,cells=len(m.cells),faces=len(m.owner),
                solver_converged=bool(r['numerically_converged']),numerically_converged=bool(r['numerically_converged'] and max(ledger.values())<1e-8),
                velocity_relative_l2=velocity,wall_force_relative_error=force_error,pressure_absolute_error=pressure,
                exact_shear=tau,wall_forces=forces,bulk_dissipation=r['bulk_viscous_energy'],relative_slip_dissipation=slip_power,
                moving_wall_power=wall_power,ledger=ledger,runtime_seconds=time.perf_counter()-tick,
                physical_suboff_benchmark_passed=False)
            row['control_passed']=bool(row['numerically_converged'] and max(velocity,force_error)<.03 and pressure<1e-8)
            path=out/(row['case']+'-fields.npz')
            np.savez_compressed(path,vertices=m.vertices,cell_vertices=np.concatenate(m.cells),cell_offsets=np.r_[0,np.cumsum([len(c) for c in m.cells])],
                face_vertices=np.asarray(m.face_vertices),cell_centers=m.cell_centers,cell_volumes=m.cell_volumes,
                face_centers=m.face_centers,area_vectors=m.area_vectors,owner=m.owner,neighbor=m.neighbor,
                dirichlet_components=s.viscous.dirichlet_components,robin_coefficient=s.viscous.robin_coefficient,boundary_load=g,
                lower_wall=low,upper_wall=high,reference_velocity=exact,
                **{k:r[k] for k in ('velocity','face_velocity','pressure','flux','total_traction_integrated','robin_reaction_integrated')})
            row['fields_sha256']=hashlib.sha256(path.read_bytes()).hexdigest();rows.append(row)
            write_json(out/(row['case']+'-history.json'),r.get('newton_history',r['history']))
            write_json(out/'cases.json',rows);print(row,flush=True)
    e.verify_sources();summary=dict(source_integrity_verified=True,all_controls_passed=all(r['control_passed'] for r in rows),
        physical_suboff_benchmark_passed=False,turbulence_wall_function_verified=False,
        scope='linear slip boundary and work audit only; no general curved normal constraint')
    write_json(out/'summary.json',summary);print(summary)
    raise SystemExit(0 if summary['all_controls_passed'] else 2)


if __name__=='__main__':main()

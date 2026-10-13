#!/usr/bin/env python3
"""Native NS nonlinear and accuracy verification with frozen source/field evidence."""
import argparse
import gzip
import json
from pathlib import Path
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorfvm.verification.native_hybrid_pressure import pressure_solver
from tensorfvm.verification.native_hybrid_ns import potential_solver,kovasznay_solver
from tensorfvm.verification.native_run import NativeRunEvidence,write_json
from tensorfvm.native_manufactured3d import volume_weighted_error


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',choices=['potential','abc','kovasznay'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--levels',type=int,nargs='+',default=[2,4,6])
    parser.add_argument('--viscosities',type=float,nargs='+',default=[.001])
    parser.add_argument('--initialization',choices=['zero','stokes'],default='stokes')
    parser.add_argument('--method',choices=['newton','picard'],default='newton')
    parser.add_argument('--max-iterations',type=int,default=50)
    args=parser.parse_args();out=args.output
    if args.case=='abc' and args.viscosities!=[.1]:parser.error('ABC uses fixed viscosity 0.1')
    if args.case=='kovasznay' and args.viscosities!=[.025]:parser.error('Kovasznay uses fixed Re=40, viscosity 0.025')
    evidence=NativeRunEvidence(out,__file__);start=time.perf_counter();rows=[]
    settings=vars(args).copy();settings['output']=str(out)
    settings.update(tolerance=1e-8,accuracy_threshold=.03,gpu_tested=False,
        pressure_metric='volume weighted mean-free relative L2; also fixed gauge for Kovasznay',
        kovasznay_mesh='n x n x 1 with spanwise symmetry; not genuine 3D refinement',
        newton_linear_solver='native coupled sparse LU/reused LU; not SIMPLE/SIMPLEC')
    write_json(out/'settings.json',settings)
    for mu in args.viscosities:
        for n in args.levels:
            name=f'{args.case}-n{n}-mu{mu:g}';tick=time.perf_counter()
            if args.case=='abc':m,s,e=pressure_solver(n,navier_stokes=True)
            elif args.case=='potential':m,s,e=potential_solver(n,mu)
            else:m,s,e=kovasznay_solver(n,symmetry_extrusion=True,coupled_linear_method='reused_lu')
            init=s.initialize_stokes() if args.initialization=='stokes' else None
            initial_state=s.state.copy();initial_pressure=s.p.copy()
            def progress(row):
                print('ITER',name,{k:v for k,v in row.items() if k!='trials'},flush=True)
            if args.method=='newton':
                r=s.solve_newton(max_iterations=args.max_iterations,tolerance=1e-8,callback=progress)
                history=r['newton_history']
            else:
                r=s.solve(algorithm='coupled',max_outer_iterations=args.max_iterations,
                          outer_relaxation=.7,tolerance=1e-8,callback=progress)
                history=r['outer_history']
            uerr=volume_weighted_error(r['velocity'],e['velocity'],m.cell_volumes)
            perr=volume_weighted_error(r['pressure'],e['pressure'],m.cell_volumes,remove_mean=True)
            fixed_perr=volume_weighted_error(r['pressure'],e['pressure'],m.cell_volumes)
            ledger={k:float(r[k]) for k in ('energy_balance_error','convection_energy_identity_error',
                'interface_traction_defect_max','local_momentum_balance_max','global_momentum_balance_max')}
            metric=fixed_perr if args.case=='kovasznay' else perr
            passed=bool(r['numerically_converged'] and max(uerr,metric)<.03 and max(ledger.values())<1e-7)
            row=dict(case=name,cells=s.N,faces=s.F,velocity_unknowns=len(s.viscous.free),n=n,viscosity=mu,
                numerically_converged=r['numerically_converged'],failure=r.get('newton_failure'),
                velocity_relative_l2=uerr,pressure_mean_free_relative_l2=perr,pressure_fixed_gauge_relative_l2=fixed_perr,
                analytic_accuracy_and_ledger_passed=passed,iterations=len(history),
                final_residuals={k:v for k,v in history[-1].items() if k!='trials'},
                initialization=init,runtime_seconds=time.perf_counter()-tick,**ledger)
            rows.append(row);write_json(out/'cases.json',rows);print('RESULT',row,flush=True)
            write_json(out/f'{name}-history.json',history)
            write_json(out/f'{name}-linear.json',s.coupled_linear_history)
            (out/f'{name}-inner-history.json.gz').write_bytes(gzip.compress(json.dumps(s.history,allow_nan=False).encode(),mtime=0))
            np.savez_compressed(out/f'{name}-fields.npz',vertices=m.vertices,
                cell_vertices=np.concatenate(m.cells),cell_offsets=np.r_[0,np.cumsum([len(c) for c in m.cells])],
                cell_types=np.asarray(m.cell_types),face_vertices=np.asarray(m.face_vertices),
                owner=m.owner,neighbor=m.neighbor,area_vectors=m.area_vectors,cell_centers=m.cell_centers,
                cell_volumes=m.cell_volumes,body_force=s.body_force,exact_velocity=e['velocity'],exact_pressure=e['pressure'],
                initialization_velocity=initial_state[:3*s.N].reshape(s.N,3),
                initialization_face_velocity=initial_state[3*s.N:].reshape(s.F,3),initialization_pressure=initial_pressure,
                **{k:r[k] for k in ('velocity','pressure','face_velocity','flux','total_traction_integrated',
                    'neighbor_total_traction_integrated','momentum_flux_integrated','reconstructed_velocity')})
    fig,axes=plt.subplots(1,2,figsize=(9,3.5),constrained_layout=True)
    pressure_key='pressure_fixed_gauge_relative_l2' if args.case=='kovasznay' else 'pressure_mean_free_relative_l2'
    for mu in args.viscosities:
        selected=[r for r in rows if r['viscosity']==mu]
        for ax,key in zip(axes,('velocity_relative_l2',pressure_key)):
            for converged,style in ((True,'o-'),(False,'x')):
                subset=[r for r in selected if r['numerically_converged']==converged]
                if subset:ax.loglog([r['cells'] for r in subset],[100*r[key] for r in subset],style,
                    label=f'mu={mu:g}'+('' if converged else ' UNCONVERGED'))
    for ax,title in zip(axes,('Velocity','Fixed-gauge pressure' if args.case=='kovasznay' else 'Centered pressure')):
        ax.axhline(3,color='k',ls='--',lw=.7);ax.set(xlabel='Cells',ylabel='Relative L2 error (%)',title=title);ax.legend()
    fig.savefig(out/'errors.png',dpi=180);fig.savefig(out/'errors.pdf');plt.close(fig)
    evidence.verify_sources()
    write_json(out/'summary.json',dict(source_integrity_verified=True,all_cases_converged=all(r['numerically_converged'] for r in rows),
        all_cases_analytic_accuracy_and_ledger_passed=all(r['analytic_accuracy_and_ledger_passed'] for r in rows),
        sst_integrated=False,wall_functions_integrated=False,physical_suboff_benchmark_passed=False,
        runtime_seconds=time.perf_counter()-start))
    return 0 if all(r['analytic_accuracy_and_ledger_passed'] for r in rows) else 2


if __name__=='__main__':raise SystemExit(main())

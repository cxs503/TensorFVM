#!/usr/bin/env python3
"""Reproducible native 3D pressure/inertia accuracy and conservation audit."""
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
from tensorfvm.verification.native_hybrid_ns import potential_solver
from tensorfvm.verification.native_run import NativeRunEvidence,write_json
from tensorfvm.native_manufactured3d import volume_weighted_error


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--case',choices=['abc','potential'],required=True)
    p.add_argument('--levels',type=int,nargs='+',default=[2,4,6])
    p.add_argument('--viscosities',type=float,nargs='+',default=[.1])
    p.add_argument('--algorithm',choices=['coupled','simple','simplec'],default='coupled')
    p.add_argument('--outer-relaxation',type=float,default=.7)
    p.add_argument('--max-outer',type=int,default=300)
    args=p.parse_args()
    if args.case=='abc' and args.viscosities!=[.1]:p.error('ABC fixed at viscosity 0.1')
    out=args.output;evidence=NativeRunEvidence(out,__file__);rows=[];tick=time.perf_counter()
    write_json(out/'settings.json',dict(case=args.case,levels=args.levels,viscosities=args.viscosities,
        algorithm=args.algorithm,outer_relaxation=args.outer_relaxation,max_outer=args.max_outer,
        nonlinear_tolerance=1e-8,accuracy_threshold=.03,initialization='zero free velocity, fixed boundary means',
        pressure_metric='volume weighted relative L2; independent volume mean removal'))
    for mu in args.viscosities:
        for n in args.levels:
            name=f'{args.case}-n{n}-mu{mu:g}';start=time.perf_counter()
            m,s,e=(pressure_solver(n,navier_stokes=True) if args.case=='abc' else potential_solver(n,mu))
            def progress(row):
                if row['iteration']%25==0:print('ITER',name,row,flush=True)
            r=s.solve(algorithm=args.algorithm,max_outer_iterations=args.max_outer,tolerance=1e-8,
                      outer_relaxation=args.outer_relaxation,callback=progress)
            uerr=volume_weighted_error(r['velocity'],e['velocity'],m.cell_volumes)
            perr=volume_weighted_error(r['pressure'],e['pressure'],m.cell_volumes,remove_mean=True)
            ledger_keys=('energy_balance_error','convection_energy_identity_error','convection_global_balance_error',
                         'interface_traction_defect_max','local_momentum_balance_max','global_momentum_balance_max')
            ledger={key:float(r[key]) for key in ledger_keys}
            passed=bool(r['numerically_converged'] and max(uerr,perr)<.03 and max(ledger.values())<1e-7)
            row=dict(case=name,n=n,cells=s.N,faces=s.F,velocity_unknowns=len(s.viscous.free),viscosity=mu,
                numerically_converged=r['numerically_converged'],velocity_relative_l2=uerr,
                pressure_mean_free_relative_l2=perr,analytic_accuracy_and_ledger_passed=passed,
                final_residuals=r['outer_history'][-1],total_inner_iterations=len(s.history),
                max_inner_true_linear_residual=max(h['true_linear_relative_residual'] for h in s.history),runtime_seconds=time.perf_counter()-start,**ledger)
            rows.append(row);write_json(out/'cases.json',rows);print('RESULT',row,flush=True)
            write_json(out/f'{name}-history.json',r['outer_history'])
            write_json(out/f'{name}-linear.json',s.coupled_linear_history)
            (out/f'{name}-inner-history.json.gz').write_bytes(gzip.compress(
                json.dumps(s.history,allow_nan=False).encode(),mtime=0))
            np.savez_compressed(out/f'{name}-fields.npz',vertices=m.vertices,
                cell_vertices=np.concatenate(m.cells),cell_offsets=np.r_[0,np.cumsum([len(c) for c in m.cells])],
                cell_types=np.asarray(m.cell_types),face_vertices=np.asarray(m.face_vertices),
                owner=m.owner,neighbor=m.neighbor,area_vectors=m.area_vectors,cell_centers=m.cell_centers,
                cell_volumes=m.cell_volumes,body_force=s.body_force,exact_velocity=e['velocity'],exact_pressure=e['pressure'],
                **{k:r[k] for k in ('velocity','pressure','face_velocity','flux','total_traction_integrated',
                    'neighbor_total_traction_integrated','momentum_flux_integrated','reconstructed_velocity')})
    fig,axes=plt.subplots(1,2,figsize=(9,3.5),constrained_layout=True)
    for mu in args.viscosities:
        selected=[row for row in rows if row['viscosity']==mu]
        for ax,key in zip(axes,('velocity_relative_l2','pressure_mean_free_relative_l2')):
            valid=[r for r in selected if r['numerically_converged']]
            invalid=[r for r in selected if not r['numerically_converged']]
            if valid:ax.loglog([r['cells'] for r in valid],[100*r[key] for r in valid],'o-',label=f'mu={mu:g}')
            if invalid:ax.loglog([r['cells'] for r in invalid],[100*r[key] for r in invalid],'x',label=f'mu={mu:g} UNCONVERGED')
    for ax,title in zip(axes,('Velocity','Centered pressure')):
        ax.axhline(3,color='k',ls='--',lw=.7);ax.set(xlabel='Cells',ylabel='Relative L2 error (%)',title=title);ax.legend()
    fig.savefig(out/'errors.png',dpi=180);fig.savefig(out/'errors.pdf');plt.close(fig)
    evidence.verify_sources()
    write_json(out/'summary.json',dict(source_integrity_verified=True,
        all_cases_converged=all(r['numerically_converged'] for r in rows),
        all_cases_analytic_accuracy_and_ledger_passed=all(r['analytic_accuracy_and_ledger_passed'] for r in rows),
        physical_suboff_benchmark_passed=False,sst_integrated=False,wall_functions_integrated=False,
        runtime_seconds=time.perf_counter()-tick))
    return 0 if all(r['numerically_converged'] for r in rows) else 2


if __name__=='__main__':raise SystemExit(main())

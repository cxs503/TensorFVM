#!/usr/bin/env python3
"""Native unforced Re=40 verification, with explicit 2D extrusion limits."""
import argparse
from pathlib import Path
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorfvm.verification.native_hybrid_ns import kovasznay_solver, kovasznay_fields
from tensorfvm.verification.native_run import NativeRunEvidence, write_json
from tensorfvm.native_manufactured3d import volume_weighted_error


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--fine-n',type=int,default=64)
    args=p.parse_args()
    if args.fine_n<=32:p.error('fine-n must exceed 32')
    evidence=NativeRunEvidence(args.output,__file__);out=args.output;start=time.perf_counter()
    rows=[];comparisons=[];reference=None;fine=None
    cases=[(3,True,a) for a in ('coupled','simple','simplec')]
    cases += [(n,False,'coupled') for n in (4,8)]
    cases += [(n,True,'coupled') for n in (8,16,32,args.fine_n)]
    for n,symmetry,algorithm in cases:
        label=f'{"symmetry" if symmetry else "cube"}-n{n}-{algorithm}'
        tick=time.perf_counter()
        mesh,s,exact=kovasznay_solver(n,symmetry_extrusion=symmetry,coupled_linear_method='reused_lu')
        r=s.solve(algorithm=algorithm,outer_relaxation=.7,tolerance=1e-8,
            max_outer_iterations=120,max_iterations=6000,
            callback=lambda row:print(label,row,flush=True))
        err=lambda key:volume_weighted_error(r[key],exact[key],mesh.cell_volumes)
        row=dict(case=label,n=n,symmetry_extrusion=symmetry,cells=len(mesh.cells),faces=len(mesh.owner),
            velocity_unknowns=len(s.viscous.free),pressure_unknowns=s.N,algorithm=algorithm,
            coupled_linear_method='reused_lu' if algorithm=='coupled' else None,
            reynolds=40.,body_force_max=float(np.max(np.abs(s.body_force))),
            numerically_converged=r['numerically_converged'],velocity_relative_l2=err('velocity'),
            pressure_relative_l2=err('pressure'),pressure_mean_free_relative_l2=volume_weighted_error(
                r['pressure'],exact['pressure'],mesh.cell_volumes,remove_mean=True),
            maximum_z_velocity=float(np.max(np.abs(r['velocity'][:,2]))),
            outer_iterations=len(r['outer_history']),inner_iterations=len(r['history']),
            final_current_residuals=r['outer_history'][-1],energy_balance_error=r['energy_balance_error'],
            convection_energy_identity_error=r['convection_energy_identity_error'],
            interface_traction_defect_max=r['interface_traction_defect_max'],
            global_momentum_balance_error=float(np.max(np.abs(r['momentum_flux_integrated'][mesh.neighbor<0].sum(0)))),
            elapsed_seconds=time.perf_counter()-tick)
        row['analytic_3_percent_passed']=bool(r['numerically_converged'] and
            max(row['velocity_relative_l2'],row['pressure_relative_l2'])<.03)
        rows.append(row);write_json(out/'cases.json',rows)
        for kind,data in [('outer',r['outer_history']),('inner',r['history']),('linear',s.coupled_linear_history)]:
            write_json(out/f'{label}-{kind}.json',data)
        np.savez_compressed(out/f'{label}-fields.npz',vertices=mesh.vertices,cells=np.asarray(mesh.cells),
            face_vertices=np.asarray(mesh.face_vertices),owner=mesh.owner,neighbor=mesh.neighbor,
            area_vectors=mesh.area_vectors,cell_centers=mesh.cell_centers,cell_volumes=mesh.cell_volumes,
            exact_velocity=exact['velocity'],exact_pressure=exact['pressure'],
            velocity_dirichlet_components=s.viscous.dirichlet_components,
            body_force=s.body_force,**{key:r[key] for key in ('velocity','pressure','face_velocity',
            'flux','total_traction_integrated','momentum_flux_integrated')})
        if n==3:
            if reference is None:reference=r
            else:comparisons.append(dict(algorithm=algorithm,maximum_absolute_difference={
                key:float(np.max(np.abs(r[key]-reference[key]))) for key in ('velocity','pressure','face_velocity','flux')}))
        if n==args.fine_n:fine=r
        print('RESULT',row,flush=True)
    write_json(out/'algorithm-comparison.json',comparisons)
    n=args.fine_n;u=fine['velocity'].reshape(n,n,3);pressure=fine['pressure'].reshape(n,n)
    fig,axes=plt.subplots(2,3,figsize=(12,7.5),constrained_layout=True)
    x=(np.arange(n)+.5)/n
    for ax in axes[0]:ax.set(xlim=(0,1),ylim=(0,1),aspect='equal',xlabel='x',ylabel='y')
    for c in np.linspace(0,1,n+1):
        axes[0,0].axvline(c,color='.4',lw=.35);axes[0,0].axhline(c,color='.4',lw=.35)
    axes[0,0].set_title(f'Actual {n} x {n} x 1 mesh section')
    for ax,values,title in [(axes[0,1],pressure,'Computed pressure'),(axes[0,2],np.linalg.norm(u,axis=2),'Computed speed')]:
        im=ax.imshow(values,origin='lower',extent=(0,1,0,1),interpolation='nearest');ax.set_title(title);fig.colorbar(im,ax=ax)
    dense=np.linspace(0,1,400);y=x[n//2]
    analytic=kovasznay_fields(np.column_stack((dense,np.full(len(dense),y),np.full(len(dense),.5))))
    for ax,key,values in [(axes[1,0],'velocity',u[n//2,:,0]),(axes[1,1],'pressure',pressure[n//2,:])]:
        ref=analytic[key][:,0] if key=='velocity' else analytic[key]
        ax.plot(dense,ref,label='analytic');ax.plot(x,values,'o',markersize=3,label='native NS')
        ax.set(xlabel='x',ylabel='u_x' if key=='velocity' else 'p',title=f'y={y:.5f}, z=0.5');ax.legend()
    selected=[r for r in rows if r['symmetry_extrusion'] and r['n']>=8]
    for key,label in [('velocity_relative_l2','Velocity'),('pressure_relative_l2','Pressure (fixed gauge)')]:
        axes[1,2].loglog([r['cells'] for r in selected],[100*r[key] for r in selected],'o-',label=label)
    axes[1,2].axhline(3,color='k',linestyle='--',label='3%');axes[1,2].legend()
    axes[1,2].set(xlabel='Cells (n x n x 1)',ylabel='Relative L2 error (%)',title='In-plane refinement')
    fig.savefig(out/'mesh-fields-refinement.png',dpi=180);fig.savefig(out/'mesh-fields-refinement.pdf');plt.close(fig)
    evidence.verify_sources()
    passed=bool(rows[-1]['analytic_3_percent_passed'] and all(r['numerically_converged'] for r in rows)
        and all(max(r['maximum_absolute_difference'].values())<1e-6 for r in comparisons))
    write_json(out/'summary.json',dict(unforced_2d_analytic_gate_passed=passed,
        fully_3d_grid_refinement_passed=False,sst_integrated=False,wall_functions_integrated=False,
        physical_suboff_benchmark_passed=False,gpu_tested=False,source_integrity_verified=True,
        runtime_seconds=time.perf_counter()-start))
    return 0 if passed else 2


if __name__=='__main__':raise SystemExit(main())

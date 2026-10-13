#!/usr/bin/env python3
"""Freeze old/new Korn spectra and independent 3D Stokes/NS diagnostics."""
import argparse
import json
import hashlib
from pathlib import Path
import time
import numpy as np
from scipy.sparse.linalg import eigsh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorfvm.native_hybrid_viscous3d import NativeHybridViscous3D
from tensorfvm.verification.native_hybrid_pressure import pressure_mesh,pressure_solver,strain_spectrum
from tensorfvm.verification.native_run import NativeRunEvidence,write_json
from tensorfvm.native_manufactured3d import volume_weighted_error


def plot_trends(out,rows,spectra):
    fig,axes=plt.subplots(1,3,figsize=(13,3.6),constrained_layout=True)
    for strength,label in [(0.,'Original'),(1.,'With jump energy')]:
        subset=[r for r in spectra if r['stabilization']==strength]
        axes[0].loglog([r['cells'] for r in subset],[r['minimum_strain_to_gradient_energy_ratio'] for r in subset],'o-',label=label)
        for ns,ax in [(False,axes[1]),(True,axes[2])]:
            subset=[r for r in rows if r['kind']=='abc' and r['stabilization']==strength and r['navier_stokes']==ns]
            ax.loglog([r['cells'] for r in subset],[100*r['pressure_mean_free_relative_l2'] for r in subset],'o-',label=label)
            ax.set(xlabel='Cells',ylabel='Pressure relative L2 error (%)',title='ABC NS' if ns else 'ABC Stokes')
            ax.axhline(3,color='k',ls='--',lw=.7)
    axes[0].set(xlabel='Cells',ylabel='Minimum energy ratio',title='Divergence-free strain control')
    for ax in axes:
        ax.legend()
        ax.set_xticks([28,224,756],labels=['28','224','756'])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    fig.savefig(out/'spectra-pressure.png',dpi=180);fig.savefig(out/'spectra-pressure.pdf');plt.close(fig)
    write_json(out/'plot-sources.json',dict(
        cases_sha256=hashlib.sha256((out/'cases.json').read_bytes()).hexdigest(),
        spectra_sha256=hashlib.sha256((out/'spectra.json').read_bytes()).hexdigest(),
        plot_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    (out/'plot-script.py').write_bytes(Path(__file__).read_bytes())


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--plots-only',action='store_true',help='regenerate trend figures from saved JSON without recomputing flow')
    args=parser.parse_args();out=args.output
    if args.plots_only:
        plot_trends(out,json.loads((out/'cases.json').read_text()),json.loads((out/'spectra.json').read_text()))
        return 0
    evidence=NativeRunEvidence(out,__file__);tick=time.perf_counter()
    spectra=[];rows=[];answers={}
    for n in (2,4,6):
        mesh=pressure_mesh(n)
        for strength in (0.,1.):
            op=NativeHybridViscous3D(mesh,.1,gradient_jump_stabilization=strength)
            spectral,v=strain_spectrum(op)
            spectra.append(dict(n=n,cells=len(mesh.cells),stabilization=strength,**spectral))
            write_json(out/'spectra.json',spectra)
            np.savez_compressed(out/f'n{n}-jump{strength:g}-mode.npz',free_dofs=op.free,mode=v,
                                vertices=mesh.vertices,cell_centers=mesh.cell_centers,face_centers=mesh.face_centers)
            for kind,ns in [('hydro',False),('shear',False),('abc',False),('abc',True)]:
                name=f'n{n}-jump{strength:g}-{kind}-{"ns" if ns else "stokes"}'
                m,s,exact=pressure_solver(n,kind,stabilization=strength,navier_stokes=ns)
                options=dict(algorithm='coupled',tolerance=1e-8 if ns else 1e-10)
                if ns:options.update(max_outer_iterations=100,outer_relaxation=.7)
                result=s.solve(**options)
                du=result['velocity']-exact['velocity'];dp=result['pressure']-exact['pressure']
                dp-=np.average(dp,weights=m.cell_volumes)
                velocity_abs=float(np.sqrt(np.sum(m.cell_volumes[:,None]*du**2)/m.cell_volumes.sum()))
                pressure_abs=float(np.sqrt(np.average(dp**2,weights=m.cell_volumes)))
                velocity_relative=volume_weighted_error(result['velocity'],exact['velocity'],m.cell_volumes) if kind!='hydro' else None
                pressure_relative=volume_weighted_error(result['pressure'],exact['pressure'],m.cell_volumes,remove_mean=True) if kind!='shear' else None
                boundary_flux=result['momentum_flux_integrated'] if ns else -result['total_traction_integrated']
                force=(m.cell_volumes[:,None]*s.body_force).sum(0)
                row=dict(case=name,n=n,cells=len(m.cells),kind=kind,navier_stokes=ns,stabilization=strength,
                    numerically_converged=result['numerically_converged'],velocity_absolute_l2=velocity_abs,
                    pressure_mean_free_absolute_l2=pressure_abs,velocity_relative_l2=velocity_relative,
                    pressure_mean_free_relative_l2=pressure_relative,
                    energy_balance_error=result['energy_balance_error'],gradient_jump_energy=result['gradient_jump_energy'],
                    interface_traction_defect_max=result['interface_traction_defect_max'],
                    global_momentum_balance_error=float(np.max(np.abs(boundary_flux[m.neighbor<0].sum(0)-force))),
                    final_residuals=result['outer_history'][-1] if ns else result['history'][-1])
                row['analytic_3_percent_passed']=bool(kind=='abc' and result['numerically_converged']
                    and max(velocity_relative,pressure_relative)<.03)
                rows.append(row);write_json(out/'cases.json',rows);print('RESULT',row,flush=True)
                write_json(out/f'{name}-history.json',result['outer_history'] if ns else result['history'])
                np.savez_compressed(out/f'{name}-fields.npz',vertices=m.vertices,
                    cell_vertices=np.concatenate(m.cells),cell_offsets=np.r_[0,np.cumsum([len(c) for c in m.cells])],
                    cell_types=np.asarray(m.cell_types),face_vertices=np.asarray(m.face_vertices),
                    owner=m.owner,neighbor=m.neighbor,area_vectors=m.area_vectors,cell_centers=m.cell_centers,
                    cell_volumes=m.cell_volumes,body_force=s.body_force,exact_velocity=exact['velocity'],exact_pressure=exact['pressure'],
                    **{k:result[k] for k in ('velocity','pressure','face_velocity','flux','total_traction_integrated')})
                if n==6 and kind=='abc':answers[(strength,ns)]=(m,result,exact)
    plot_trends(out,rows,spectra)
    # Display actual cell-center values near the middle of the skew domain.
    m,answer,exact=answers[(1.,True)];z=m.cell_centers[:,2]
    selected=np.abs(z-.5)<.08
    fig,axes=plt.subplots(1,3,figsize=(12,3.6),constrained_layout=True)
    computed=answer['pressure']-np.average(answer['pressure'],weights=m.cell_volumes)
    reference=exact['pressure']-np.average(exact['pressure'],weights=m.cell_volumes)
    low=min(computed.min(),reference.min());high=max(computed.max(),reference.max())
    for ax,values,title in [(axes[0],reference,'Analytic centered pressure'),(axes[1],computed,'Computed centered pressure'),(axes[2],np.linalg.norm(answer['velocity'],axis=1),'Computed speed')]:
        options=dict(vmin=low,vmax=high) if ax is not axes[2] else {}
        im=ax.scatter(*m.cell_centers[selected,:2].T,c=values[selected],s=15,**options)
        ax.set(xlabel='x',ylabel='y',title=title,aspect='equal');fig.colorbar(im,ax=ax)
    fig.savefig(out/'abc-cell-values.png',dpi=180);fig.savefig(out/'abc-cell-values.pdf');plt.close(fig)
    evidence.verify_sources()
    fixed=[r for r in rows if r['stabilization']==1]
    passed=bool(all(r['numerically_converged'] and r['global_momentum_balance_error']<1e-7 and r['energy_balance_error']<1e-7 for r in fixed)
        and all(r['minimum_strain_to_gradient_energy_ratio']>.35 for r in spectra if r['stabilization']==1))
    write_json(out/'summary.json',dict(sampled_strain_stability_and_ledger_passed=passed,
        abc_3d_pressure_accuracy_passed=all(r['analytic_3_percent_passed'] for r in fixed if r['kind']=='abc' and r['n']==6),
        sst_integrated=False,wall_functions_integrated=False,physical_suboff_benchmark_passed=False,
        source_integrity_verified=True,runtime_seconds=time.perf_counter()-tick))
    return 0 if passed else 2


if __name__=='__main__':raise SystemExit(main())

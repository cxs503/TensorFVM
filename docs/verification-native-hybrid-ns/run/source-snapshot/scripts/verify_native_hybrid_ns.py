#!/usr/bin/env python3
"""Freeze native finite-Re manufactured flow, load-balance audit and plots."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy

from tensorfvm.native_hybrid_stokes3d import NativeHybridStokes3D
from tensorfvm.verification.native_hybrid_flow import channel_mesh
from tensorfvm.verification.native_hybrid_ns import polynomial_solver


def write(path, data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--fine-n',type=int,default=10)
    args=parser.parse_args();out=args.output
    if args.fine_n<=8: parser.error('fine-n must exceed 8')
    if out.exists() and any(out.iterdir()): parser.error('output must be empty')
    out.mkdir(parents=True,exist_ok=True);repo=Path(__file__).resolve().parents[1]
    sources={Path(__file__).resolve()} | {Path(m.__file__).resolve() for name,m in list(sys.modules.items())
        if name.startswith('tensorfvm') and getattr(m,'__file__',None) and Path(m.__file__).suffix=='.py'}
    hashes={}
    for source in sorted(sources):
        relative=source.relative_to(repo);target=out/'source-snapshot'/relative
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        hashes[str(relative)]=hashlib.sha256(target.read_bytes()).hexdigest()
    write(out/'sources.json',hashes)
    write(out/'environment.json',dict(python=sys.version,executable=sys.executable,
        numpy=np.__version__,scipy=scipy.__version__,matplotlib=matplotlib.__version__,
        platform=platform.platform(),device='CPU; GPU untested'))
    rows=[];comparisons=[];start=time.perf_counter()
    # A hydrostatic control isolates pressure/body-force compatibility from
    # advection, turbulence, wall functions and nonlinear iteration.
    controls=[];mesh=channel_mesh(2);g=np.array([.7,-.3,.2]);p=.4+mesh.cell_centers@g
    for mode in ('cell','face'):
        solver=NativeHybridStokes3D(mesh,mesh.neighbor<0,body_force=g,
            gauge_value=p[0],body_force_discretization=mode)
        r=solver.solve(algorithm='coupled',tolerance=1e-10)
        controls.append(dict(load=mode,maximum_speed=float(np.linalg.norm(r['velocity'],axis=1).max()),
            pressure_maximum_absolute_error=float(np.max(np.abs(r['pressure']-p))),
            numerically_converged=r['numerically_converged']))
        np.savez_compressed(out/f'hydrostatic-{mode}.npz',velocity=r['velocity'],pressure=r['pressure'],
                            cell_centers=mesh.cell_centers,exact_pressure=p)
    write(out/'hydrostatic.json',controls)
    reference=None;fine=None
    cases=[(3,2,'face',a) for a in ('coupled','simple','simplec')]
    cases += [(8,1,'cell','coupled'),(8,1,'face','coupled')]
    cases += [(n,2,'face','coupled') for n in (4,8,args.fine_n)]
    for n,order,load,algorithm in cases:
        label=f'n{n}-o{order}-{load}-{algorithm}'
        mesh,solver,exact=polynomial_solver(n,convection_order=order,body_force_discretization=load)
        def progress(row):
            print(json.dumps(dict(case=label,**row)),flush=True)
        r=solver.solve(algorithm=algorithm,tolerance=1e-8,max_outer_iterations=50,
                       max_iterations=4000,callback=progress)
        volume=mesh.cell_volumes
        error=lambda field:float(np.sqrt(np.sum(volume[:,None]*(r[field]-exact[field])**2)
            /np.sum(volume[:,None]*exact[field]**2))) if field=='velocity' else float(np.sqrt(
            np.sum(volume*(r[field]-exact[field])**2)/np.sum(volume*exact[field]**2)))
        boundary_momentum=r['momentum_flux_integrated'][mesh.neighbor<0].sum(0)
        applied_force=(volume[:,None]*exact['force']).sum(0)
        row=dict(case=label,n=n,cells=len(mesh.cells),order=order,body_force_discretization=load,
            algorithm=algorithm,reynolds=10.,numerically_converged=r['numerically_converged'],
            velocity_relative_l2=error('velocity'),pressure_relative_l2=error('pressure'),
            maximum_speed=float(np.linalg.norm(r['velocity'],axis=1).max()),
            outer_iterations=len(r['outer_history']),inner_iterations=len(r['history']),
            final_current_residuals=r['outer_history'][-1],
            energy_balance_error=r['energy_balance_error'],convection_energy_identity_error=r['convection_energy_identity_error'],
            convection_power=r['convection_power'],upwind_dissipation=r['upwind_dissipation'],
            deferred_correction_power=r['deferred_correction_power'],
            interface_traction_defect_max=r['interface_traction_defect_max'],
            global_momentum_balance_error=float(np.max(np.abs(boundary_momentum-applied_force))),
            boundary_momentum_flux=boundary_momentum.tolist(),applied_volume_force=applied_force.tolist(),
            elapsed_seconds=time.perf_counter()-start)
        row['analytic_3_percent_passed']=bool(r['numerically_converged'] and max(row['velocity_relative_l2'],row['pressure_relative_l2'])<.03)
        rows.append(row);write(out/'cases.json',rows)
        write(out/f'{label}-outer.json',r['outer_history']);write(out/f'{label}-inner.json',r['history'])
        np.savez_compressed(out/f'{label}-fields.npz',vertices=mesh.vertices,cells=np.asarray(mesh.cells),
            owner=mesh.owner,neighbor=mesh.neighbor,area_vectors=mesh.area_vectors,cell_centers=mesh.cell_centers,
            cell_volumes=volume,exact_velocity=exact['velocity'],exact_pressure=exact['pressure'],
            body_force=exact['force'],**{k:r[k] for k in ('velocity','pressure','face_velocity','flux',
            'momentum_flux_integrated','total_traction_integrated')})
        print('RESULT '+json.dumps(row),flush=True)
        if n==3:
            if reference is None: reference=r
            else: comparisons.append(dict(algorithm=algorithm,maximum_absolute_difference={key:float(np.max(np.abs(r[key]-reference[key]))) for key in ('velocity','pressure','face_velocity','flux')}))
        if n==args.fine_n: fine=r
    write(out/'algorithm-comparison.json',comparisons)
    passed=bool(rows[-1]['analytic_3_percent_passed'] and all(r['numerically_converged'] for r in rows)
        and controls[-1]['maximum_speed']<1e-10 and controls[-1]['pressure_maximum_absolute_error']<1e-10
        and all(max(r['maximum_absolute_difference'].values())<1e-6 for r in comparisons))
    write(out/'summary.json',dict(native_finite_re_verification_passed=passed,
        legacy_collocated_viscosity_fixed=False,sst_integrated=False,wall_functions_integrated=False,
        physical_suboff_benchmark_passed=False,gpu_tested=False,runtime_seconds=time.perf_counter()-start))
    n=args.fine_n;layer=n//2-1
    p=fine['pressure'].reshape(n,n,n)[layer];u=fine['velocity'].reshape(n,n,n,3)[layer]
    fig,axes=plt.subplots(1,3,figsize=(12,3.7),constrained_layout=True)
    for axis in axes:axis.set(xlim=(0,1),ylim=(0,1),xlabel='x',ylabel='y',aspect='equal')
    for coordinate in np.linspace(0,1,n+1):
        axes[0].axvline(coordinate,color='.4',lw=.6);axes[0].axhline(coordinate,color='.4',lw=.6)
    axes[0].set_title(f'Actual cell section ({n**3} cells)')
    for axis,data,title in zip(axes[1:],[p,np.linalg.norm(u,axis=2)],['Computed pressure','Computed speed']):
        im=axis.imshow(data,origin='lower',extent=(0,1,0,1),interpolation='nearest');axis.set_title(title);fig.colorbar(im,ax=axis)
    fig.savefig(out/'mesh-fields.png',dpi=180);fig.savefig(out/'mesh-fields.pdf');plt.close(fig)
    center=(np.arange(n)+.5)/n;dense=np.linspace(0,1,300);y=center[n//2];z=center[layer]
    fig,axes=plt.subplots(1,3,figsize=(12,3.6),constrained_layout=True)
    axes[0].plot(dense,1+.2*dense**2,label='analytic');axes[0].plot(center,u[n//2,:,0],'o',label='native NS')
    axes[0].set(xlabel='x',ylabel='u_x',title='Velocity');axes[0].legend()
    axes[1].plot(dense,dense-.5*y+.25*z,label='analytic');axes[1].plot(center,p[n//2,:],'o',label='native NS')
    axes[1].set(xlabel='x',ylabel='p',title='Pressure');axes[1].legend()
    selected=[r for r in rows if r['order']==2 and r['n']>=4]
    for key,label in [('velocity_relative_l2','Velocity'),('pressure_relative_l2','Pressure')]:
        axes[2].loglog([r['cells'] for r in selected],[100*r[key] for r in selected],'o-',label=label)
    axes[2].axhline(3,color='k',linestyle='--',label='3%');axes[2].legend();axes[2].set(xlabel='Cells',ylabel='Relative L2 error (%)',title='Refinement')
    fig.savefig(out/'profiles-refinement.png',dpi=180);fig.savefig(out/'profiles-refinement.pdf');plt.close(fig)
    for relative,expected in hashes.items():
        if hashlib.sha256((repo/relative).read_bytes()).hexdigest()!=expected:
            raise RuntimeError(f'Source changed during run: {relative}')
    return 0 if passed else 2


if __name__=='__main__':raise SystemExit(main())

"""Audit a saved high-Re NACA state and compare full primary-source Cp tables.

A single-grid report deliberately cannot certify grid convergence. Reference
agreement, equation convergence and grid qualification are separate gates.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def zones(path):
    result={};key=None
    for line in Path(path).read_text().splitlines():
        if line.lower().startswith('zone'):
            key=float(re.search(r'alpha=([\d.+-]+)',line).group(1));result[key]=[]
        elif key is not None:
            result[key].append([float(x) for x in line.split()])
    return {k:np.asarray(v) for k,v in result.items()}


def reference_surfaces(path,alpha):
    q=zones(path)[alpha];leading=int(q[:,0].argmin())
    # TMR file goes lower TE -> LE -> upper TE, as confirmed by its alpha=10 Cp.
    return {'lower':q[:leading+1][::-1], 'upper':q[leading:]}


def curve_error(x,values,reference):
    order=np.argsort(x);x=np.asarray(x)[order];v=np.asarray(values)[order]
    x,unique=np.unique(x,return_index=True);v=v[unique]
    lo=max(float(x[0]),float(reference[0,0]));hi=min(float(x[-1]),float(reference[-1,0]))
    if hi<=lo:raise ValueError('No common curve support')
    knots=np.unique(np.r_[lo,hi,x[(x>lo)&(x<hi)],reference[(reference[:,0]>lo)&(reference[:,0]<hi),0]])
    actual=np.interp(knots,x,v);ref=np.interp(knots,reference[:,0],reference[:,1])
    # Exact integral of squared piecewise-linear interpolants on the union grid.
    def square_integral(a):return np.sum(np.diff(knots)*(a[:-1]**2+a[:-1]*a[1:]+a[1:]**2)/3)
    denominator=square_integral(ref)
    if denominator<=0:raise ValueError('Zero curve reference requires a different metric')
    return dict(relative_l2=float(np.sqrt(square_integral(actual-ref)/denominator)),
        coverage_chord=hi-lo,x_min=lo,x_max=hi,maximum_absolute=float(np.max(abs(actual-ref)))),np.c_[knots,actual,ref]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path);p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists() and any(a.output.iterdir()):p.error('Empty report output required')
    a.output.mkdir(parents=True,exist_ok=True)
    root=Path(__file__).resolve().parents[1];run=a.run
    settings=json.loads((run/'settings.json').read_text());sources=json.loads((run/'sources.json').read_text())
    for name,digest in sources.items():
        if sha(root/name)!=digest or sha(run/'source-snapshot'/name)!=digest:raise ValueError('Source mismatch: '+name)
    if sha(run/'mesh.msh')!=settings['mesh_sha256']:raise ValueError('Mesh mismatch')
    data=(run/'checkpoint.npz').read_bytes();checkpoint_hash=hashlib.sha256(data).hexdigest()
    progress=json.loads((run/'progress.json').read_text())
    if progress['checkpoint_sha256']!=checkpoint_hash:raise ValueError('Checkpoint changed while reading; retry after save')
    manifest=json.loads((a.reference/'manifest.json').read_text())
    for name,item in manifest.items():
        if sha(a.reference/name)!=item['sha256']:raise ValueError('Reference changed: '+name)
    shutil.copy2(__file__,a.output/Path(__file__).name)
    import torch
    from dataclasses import replace
    from tensorfvm.solver import SolverConfig
    from tensorfvm.multi_element_skew_pressure import SkewPressureMultiElementSolver
    torch.set_num_threads(1)
    config=dict(settings['config']);config['mesh_file']=str(run/'mesh.msh')
    s=SkewPressureMultiElementSolver(SolverConfig(**config));m=s.mesh
    bounds=m.vertices.max(0).values-m.vertices.min(0).values
    with np.load(io.BytesIO(data)) as f:
        for name,target in [('velocity',s.velocity),('pressure',s.p),('nu_tilde',s.nu_tilde),('mass_flux',s.mass_flux)]:target.copy_(torch.from_numpy(f[name]))
        history=json.loads(str(f['history'].item()));records=json.loads(str(f['records'].item()))
    iteration=len(history);s.convection_blend=min(1.,max(0.,(iteration-1-40)/60))
    s.config=replace(s.config,length=float(bounds[0]),height=float(bounds[1]),pseudo_time_step=None if iteration>100 else .02)
    s.turbulent_kinematic_viscosity=s._sa_eddy_viscosity(s.nu_tilde)
    s.sa_linear_upwind=settings['sa_linear_upwind'];s.sa_inner_iterations=settings['sa_inner_iterations']
    current=s._true_metrics();coeff=s._aerodynamic_coefficients()
    for name,value in current.items():
        np.testing.assert_allclose(value,history[-1][name],rtol=1e-7,atol=1e-10,err_msg='Saved residual mismatch: '+name)
    wall=m.masks['wall'];own=s.o[wall];fc=m.face_centers[wall].numpy();normal=-m.face_normals[wall]
    tangent=torch.stack([normal[:,1],-normal[:,0]],1)
    distance=abs(((m.centers.reshape(-1,2)[own]-m.face_centers[wall])*normal).sum(1))
    tau=s.kinematic_viscosity*(s.velocity[own]*tangent).sum(1)/distance
    cp=(2*s.reconstructed_pressure_faces(s.p)[wall]).numpy();cf=(2*tau).numpy()
    yp=(distance*torch.sqrt(abs(tau))/s.kinematic_viscosity).numpy()
    alpha=config['angle_of_attack'];reference=reference_surfaces(a.reference/'n0012cp_cfl3d_sa.dat',alpha)
    cf_ref=zones(a.reference/'n0012cf_cfl3d_sa.dat')[alpha]
    cf_ref=cf_ref[np.argsort(cf_ref[:,0])]
    coefficient_table=np.loadtxt(a.reference/'n0012clcd_cfl3d_sa.dat',skiprows=4)
    ref_row=coefficient_table[np.isclose(coefficient_table[:,0],alpha)][0]
    cl_error=abs(coeff['lift']-ref_row[1])/(abs(ref_row[1]) if alpha else 1.)
    cd_error=abs(coeff['drag']-ref_row[2])/abs(ref_row[2])
    errors={};curves={}
    for side,mask in [('upper',fc[:,1]>0),('lower',fc[:,1]<0)]:
        errors[side],curves[side]=curve_error(fc[mask,0],cp[mask],reference[side])
        np.savetxt(a.output/(side+'-cp.csv'),curves[side],delimiter=',',header='x,Cp_computed,Cp_reference',comments='')
    upper=fc[:,1]>0;cf_error,cf_curve=curve_error(fc[upper,0],cf[upper],cf_ref)
    recent=[r for r in records if r['iteration']>=iteration-100]
    force_ranges={name:float(np.ptp([r['coefficients'][name] for r in recent])) if recent else None for name in ['lift','drag']}
    force_stable=bool(len(recent)>=10 and force_ranges['drag']<.001*abs(ref_row[2]) and force_ranges['lift']<(.001*abs(ref_row[1]) if alpha else .0001))
    equation_passed=bool(s.convection_blend==1 and max(current[k] for k in ['momentum','turbulence','continuity','mass_imbalance','rhie_chow_max_normal_velocity_defect'])<config['tolerance'] and current['rhie_chow_flux_defect']<1e-7)
    reference_passed=bool(cl_error<(.03 if alpha else .001) and cd_error<.03 and all(e['relative_l2']<.03 and e['coverage_chord']>.999 for e in errors.values()))
    terminal=json.loads((run/'status.json').read_text()) if (run/'status.json').exists() else None
    terminal_success=bool(terminal and terminal['failure'] is None and terminal['solver_converged'] and terminal['checkpoint_sha256']==checkpoint_hash)
    summary=dict(iterations=iteration,cells=s.count,alpha=alpha,reynolds=config['reynolds'],
        checkpoint_sha256=checkpoint_hash,mesh_sha256=settings['mesh_sha256'],source_integrity_verified=True,
        sources_sha256=sha(run/'sources.json'),settings_sha256=sha(run/'settings.json'),
        reference_manifest=manifest,reference_scope='TMR essentially incompressible case; CFL3D SA M=0.15 with vortex farfield; native incompressible SA at radius 500c, no vortex correction. No coefficient correction applied.',
        coefficients=coeff,reference_coefficients=dict(lift=float(ref_row[1]),drag=float(ref_row[2])),
        lift_error=float(cl_error),lift_error_kind='relative' if alpha else 'absolute',drag_relative_error=float(cd_error),
        cp_errors=errors,cf_upper_diagnostic=cf_error,wall_y_plus=dict(max=float(yp.max()),p99=float(np.quantile(yp,.99))),
        current_equations=current,velocity_change=history[-1]['velocity_change'],force_ranges_last_100=force_ranges,
        equation_passed=equation_passed,force_stable=force_stable,
        producer_terminal_status=terminal,
        numerical_passed=terminal_success and equation_passed and force_stable and history[-1]['velocity_change']<1e-6,
        reference_passed=reference_passed,wall_resolution_passed=bool(yp.max()<1),
        grid_convergence_passed=False,qualified=False,
        qualification_reason='Single-grid diagnostic; requires convergence and a three-grid study. Reference agreement alone is insufficient.')
    (a.output/'audit.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    np.savez_compressed(a.output/'reconstructed-fields.npz',velocity=s.velocity.numpy(),pressure=s.p.numpy(),
        nu_tilde=s.nu_tilde.numpy(),nu_t=s.turbulent_kinematic_viscosity.numpy(),centers=m.centers.numpy(),
        surface_centers=fc,cp=cp,cf=cf,y_plus=yp)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    vertices=m.vertices.numpy();polygons=[vertices[list(cell)] for cell in m.cell_nodes]
    centers=m.centers.reshape(-1,2).numpy();visible=(centers[:,0]>-.3)&(centers[:,0]<1.5)&(abs(centers[:,1])<.6)
    selected=[poly for poly,keep in zip(polygons,visible) if keep]
    fig,axes=plt.subplots(2,2,figsize=(12,7))
    values=[None,2*s.p.numpy(),np.linalg.norm(s.velocity.numpy(),axis=1),s.turbulent_kinematic_viscosity.numpy()/s.kinematic_viscosity]
    titles=['Actual computational mesh','Pressure coefficient in flow','Speed / U infinity','Eddy viscosity / molecular viscosity']
    for ax,value,title in zip(axes.flat,values,titles):
        pc=PolyCollection(selected,facecolors='none',edgecolors='k',linewidths=.15) if value is None else PolyCollection(selected,array=value[visible],edgecolors='none',cmap='viridis',rasterized=True)
        ax.add_collection(pc);ax.set(xlim=(-.2,1.4),ylim=(-.45,.45),xlabel='x/c',ylabel='y/c',title=title);ax.set_aspect('equal')
        if value is not None:fig.colorbar(pc,ax=ax)
    fig.suptitle(f'NACA0012 Re=6e6, alpha={alpha:g}, {s.count:,} cells, iteration {iteration}; diagnostic')
    fig.tight_layout();fig.savefig(a.output/'mesh-fields.png',dpi=220);fig.savefig(a.output/'mesh-fields.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for side,color in [('upper','tab:blue'),('lower','tab:orange')]:
        q=curves[side];axes[0].plot(q[:,0],q[:,1],color=color,label=side+' TensorFVM');axes[0].plot(q[:,0],q[:,2],'--',color=color,label=side+' CFL3D')
    axes[0].invert_yaxis();axes[0].set(xlabel='x/c',ylabel='Cp');axes[0].legend()
    axes[1].plot(cf_curve[:,0],cf_curve[:,1],label='TensorFVM upper');axes[1].plot(cf_curve[:,0],cf_curve[:,2],'--',label='CFL3D upper');axes[1].set(xlabel='x/c',ylabel='Cf');axes[1].legend()
    fig.tight_layout();fig.savefig(a.output/'surface-comparison.png',dpi=220);fig.savefig(a.output/'surface-comparison.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for name in ['momentum','turbulence','velocity_change','rhie_chow_max_normal_velocity_defect']:
        axes[0].semilogy(range(1,iteration+1),[max(h[name],1e-18) for h in history],label=name)
    axes[0].legend();axes[0].set(xlabel='Iteration',ylabel='Current residual / change')
    for name in ['lift','drag']:axes[1].plot([r['iteration'] for r in records],[r['coefficients'][name] for r in records],label=name)
    axes[1].legend();axes[1].set(xlabel='Iteration',ylabel='Coefficient')
    fig.tight_layout();fig.savefig(a.output/'convergence.png',dpi=220);fig.savefig(a.output/'convergence.pdf');plt.close(fig)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()

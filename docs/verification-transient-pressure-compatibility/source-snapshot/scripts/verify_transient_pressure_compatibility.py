#!/usr/bin/env python3
"""Actual native 3-D pressure/velocity compatibility experiments, no smoothing."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from tensorfvm.body_fitted_transient3d import BodyFittedTransient3D,TransientSettings
from tensorfvm.solver3d import Cylinder3DConfig
from tensorfvm.verification.core import Metric,evaluate,save_run,write_json,sha,provenance,artifact_manifest


def scatter(owner,neighbor,values,n):
    result=np.zeros((n,)+values.shape[1:])
    np.add.at(result,owner,values)
    internal=neighbor>=0
    np.add.at(result,neighbor[internal],-values[internal])
    return result


def independent_replay(s,old_u,old_flux):
    conv,visc,_=s.transport(old_u,old_flux)
    facep=s.pressure_face_values(s.pressure)
    o,n=s.owner.numpy(),s.neighbor.numpy();area=s.area.numpy();vol=s.volumes.numpy()
    flux=s.flux.numpy();u=s.velocity.numpy();uf=s.face_values(s.velocity).numpy()
    native=scatter(o,n,flux,s.N)/vol
    cell=scatter(o,n,(uf*area).sum(1),s.N)/vol
    normal=np.abs(flux-(uf*area).sum(1))/np.linalg.norm(area,axis=1)/s.config.inlet_velocity
    rhs=scatter(o,n,(visc-conv).numpy()-facep.numpy()[:,None]*area,s.N)
    expected=old_u.numpy()+s.config.time_step/s.config.density*rhs/vol[:,None]
    momentum=s.config.density*((u-old_u.numpy())*vol[:,None]).sum(0)
    boundary=(s.config.time_step*((visc-conv).numpy()-facep.numpy()[:,None]*area))[n<0].sum(0)
    return dict(native_divergence_max=float(np.abs(native).max()),cell_interpolated_divergence_max=float(np.abs(cell).max()),
        interior_normal_defect_max=float(normal[n>=0].max()),interior_normal_defect_p95=float(np.quantile(normal[n>=0],.95)),
        momentum_update_max_error=float(np.abs(u-expected).max()),momentum_ledger_max=float(np.abs(momentum-boundary).max()),
        mass_imbalance=float(flux[n<0].sum())),conv,visc,facep


def fields(s):
    return {k:v.numpy() for k,v in dict(cell_velocity=s.velocity,cell_pressure=s.pressure,cell_centers=s.centers,cell_volumes=s.volumes,
        face_flux=s.flux,face_owner=s.owner,face_neighbor=s.neighbor,face_area_vectors=s.area,face_centers=s.fc,
        vertices=s.mesh.vertices,cell_connectivity=s.mesh.connectivity,wall_face_mask=s.wall,outlet_face_mask=s.outlet).items()}


def manufacture(out):
    c=Cylinder3DConfig(nx=48,ny=16,nz=4,length=24,height=20,cylinder_x=8,cylinder_y=10,body_fitted_stretching=7,mesh_type='body-fitted',time_step=1e-5)
    s=BodyFittedTransient3D(c,TransientSettings(initial_perturbation=0,sgs='none',pressure_tolerance=1e-10,nonorthogonal_iterations=150))
    g=torch.tensor([.7,-.3,0.],dtype=torch.float64);exact=s.centers@g+.2;face_exact=s.fc@g+.2
    # Deliberately retain the incompatible homogeneous-Neumann response as a
    # baseline rather than disguising it as the same manufactured BC.
    legacy_face_error=float((s.pressure_face_values(exact)-face_exact).abs().max())
    legacy_gradient_error=float((s.gradient(exact,pressure=True)-g).abs().max())
    s.set_pressure_boundary_conditions(neumann=(s.area/s.area_mag[:,None])@g,outlet_values=face_exact)
    gradient_error=float((s.gradient(exact,pressure=True)-g).abs().max())
    face_error=float((s.pressure_face_values(exact)-face_exact).abs().max())
    flux_error=float((s.pressure_flux_gradient(exact)-s.area@g).abs().max())
    solved,q,iterations,residual=s.project(s.area@g,correction_scale=1.)
    corrected=g-s.integrate(s.pressure_face_values(solved)[:,None]*s.area)/s.volumes[:,None]
    metrics,passed=evaluate([Metric('all_cell_affine_pressure_gradient_error',gradient_error,1e-10,'LS with actual nonzero face-normal derivatives'),
        Metric('all_face_affine_pressure_value_error',face_error,1e-10,'Actual shared pressure traction, including wall and outlet'),
        Metric('all_face_affine_pressure_flux_error',flux_error,1e-10,'Physical face-normal pressure derivative'),
        Metric('solved_affine_pressure_error',float((solved-exact).abs().max()),1e-8,'Independent prescribed affine pressure'),
        Metric('corrected_cell_velocity_max_error',float(corrected.abs().max()),1e-8,'Constant pressure-force impulse removed using conservative shared-face traction'),
        Metric('corrected_native_face_flux_max',float(q.abs().max()),1e-8,'Constant pressure-gradient face impulse removed'),
        Metric('native_divergence_max',residual,1e-10,'Actual nonlinear nonorthogonal pressure solve')])
    raw=fields(s);raw.update(exact_pressure=exact.numpy(),solved_pressure=solved.numpy(),exact_face_pressure=face_exact.numpy(),solved_flux=q.numpy(),corrected_cell_velocity=corrected.numpy(),pressure_neumann=s.pressure_neumann.numpy(),pressure_outlet_values=s.pressure_outlet_values.numpy())
    return save_run(out/'manufactured-affine',dict(case='3d-affine-pressure-with-physical-Neumann',config=asdict(c),metrics=metrics,passed=passed,projection_iterations=iterations,legacy_homogeneous_neumann_face_error=legacy_face_error,legacy_homogeneous_neumann_gradient_error=legacy_gradient_error,fields=raw,history=[]))


def run_case(out,name,grid,dt,final_time,initialize):
    nx,ny,nz,stretch=grid
    steps=int(round(final_time/dt))
    if abs(steps*dt-final_time)>1e-12:raise ValueError('final time must contain an integer number of steps')
    c=Cylinder3DConfig(nx=nx,ny=ny,nz=nz,length=24,height=20,cylinder_x=8,cylinder_y=10,span=np.pi,body_fitted_stretching=stretch,mesh_type='body-fitted',time_step=dt,max_steps=steps)
    settings=TransientSettings(initial_projection=initialize)
    start=time.perf_counter();s=BodyFittedTransient3D(c,settings)
    initial_velocity=s.velocity.numpy().copy();initial_flux=s.flux.numpy().copy()
    initial_potential=getattr(s,'initial_velocity_potential',torch.zeros(s.N,dtype=torch.float64)).numpy().copy()
    failure=None;old_u=s.velocity.clone();old_flux=s.flux.clone()
    for i in range(steps):
        old_u=s.velocity.clone();old_flux=s.flux.clone()
        try:s.step()
        except (ValueError,RuntimeError)as exc:failure=str(exc);break
        if (i+1)%25==0:print(json.dumps(dict(case=name,step=i+1,continuity=s.history[-1]['continuity'],normal_defect=s.history[-1]['interior_cell_face_normal_defect_max'])),flush=True)
    replay,conv,visc,facep=independent_replay(s,old_u,old_flux)
    metrics,passed=evaluate([Metric('native_face_divergence',replay['native_divergence_max'],settings.pressure_tolerance,'Shared-face incompressibility','1/s'),
        Metric('independent_momentum_update_error',replay['momentum_update_max_error'],1e-11,'NumPy face-scatter replay of entire cell momentum update','m/s'),
        Metric('independent_global_momentum_ledger',replay['momentum_ledger_max'],1e-10,'Domain momentum equals physical boundary traction impulse'),
        Metric('interior_cell_face_normal_defect',replay['interior_normal_defect_max'],.03,'Maximum native versus interpolated normal velocity divided by U'),
        Metric('cell_interpolated_divergence_scaled_by_D_over_U',replay['cell_interpolated_divergence_max']*c.diameter/c.inlet_velocity,.03,'Separate strict screening of interpolated cell divergence; not replaced by native continuity')])
    raw=fields(s);raw.update(initial_velocity=initial_velocity,initial_flux=initial_flux,initial_velocity_potential=initial_potential,
        previous_velocity=old_u.numpy(),previous_flux=old_flux.numpy(),last_convective_face_momentum=conv.numpy(),last_viscous_face_momentum=visc.numpy(),actual_face_pressure=facep.numpy())
    result=save_run(out/name,dict(case=name,config=asdict(c),settings=asdict(settings),initialization=s.initialization,elapsed_seconds=time.perf_counter()-start,
        completed_steps=len(s.history),requested_steps=steps,time=s.time,failure=failure,metrics=metrics,passed=bool(passed and failure is None and len(s.history)==steps),
        numerical_qualified=False,physical_accuracy_qualified=False,independent_replay=replay,last_force=s.forces[-1]if s.forces else None,fields=raw,history=s.history))
    print(json.dumps(dict(case=name,done=True,passed=result['passed'],replay=replay)),flush=True)
    return result


def wall_curve(out,name):
    r=json.loads((out/name/'result.json').read_text());raw=np.load(out/name/'fields.npz');c=r['config'];wall=raw['wall_face_mask'];xyz=raw['face_centers'][wall];p=raw['actual_face_pressure'][wall]
    theta=np.degrees(np.arctan2(xyz[:,1]-c['cylinder_y'],xyz[:,0]-c['cylinder_x']))%360
    rounded=np.round(theta,10);angles=np.unique(rounded)
    cp=np.array([2*np.average(p[rounded==a],weights=np.linalg.norm(raw['face_area_vectors'][wall][rounded==a],axis=1))/(c['density']*c['inlet_velocity']**2) for a in angles])
    return angles,cp


def publish(out,results,manufactured,summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.collections import PolyCollection
    figures=[]
    fig,ax=plt.subplots(figsize=(11,5))
    for name in results:
        theta,cp=wall_curve(out,name);ax.plot(theta,cp,label=name)
    ax.set(xlabel='Wall angle (deg; zero downstream)',ylabel='Reconstructed wall-face Cp',title='Actual startup pressure, same physical time; no smoothing or experimental qualification')
    ax.legend(fontsize=8);ax.grid(alpha=.2);fig.tight_layout();figures.append(('wall-pressure',fig))
    fig,axes=plt.subplots(2,3,figsize=(16,9))
    for col,name in enumerate(['medium-legacy','medium-hodge','fine-hodge']):
        raw=np.load(out/name/'fields.npz');c=results[name]['config'];nx,ny,nz=[c[k]for k in ('nx','ny','nz')]
        cells=raw['cell_connectivity'].reshape(nz,ny,nx,8)[0].reshape(-1,8)
        polys=raw['vertices'].reshape(-1,3)[cells[:,:4],:2]-[c['cylinder_x'],c['cylinder_y']]
        pressure=raw['cell_pressure'].reshape(nz,ny,nx)[0];speed=np.linalg.norm(raw['cell_velocity'].reshape(nz,ny,nx,3)[0],axis=-1)
        for row,values,title in ((0,pressure,'p/(rho U²)'),(1,speed,'|u|/U')):
            pc=PolyCollection(polys,array=values.ravel(),edgecolors='none',cmap='coolwarm'if row==0 else'turbo')
            axes[row,col].add_collection(pc);axes[row,col].set(xlim=(-1,3),ylim=(-1.5,1.5),title=name+' '+title,xlabel='(x-xc)/D',ylabel='(y-yc)/D');axes[row,col].set_aspect('equal');fig.colorbar(pc,ax=axes[row,col])
    fig.tight_layout();figures.append(('native-fields',fig))
    fig,ax=plt.subplots(figsize=(10,5))
    for name,r in results.items():
        hist=json.loads((out/name/'history.json').read_text());ax.plot([h['time'] for h in hist],[h['interior_cell_face_normal_defect_max'] for h in hist],label=name)
    ax.axhline(.03,color='black',linestyle=':',label='3% normal-defect screen')
    ax.set(xlabel='t U/D',ylabel='Max interior normal velocity mismatch / U',title='Actual face/cell mismatch retained as a separate gate');ax.legend(fontsize=8);ax.grid(alpha=.2);fig.tight_layout();figures.append(('cell-face-compatibility',fig))
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,name in zip(axes,['medium-hodge','fine-hodge']):
        raw=np.load(out/name/'fields.npz');c=results[name]['config'];nx,ny,nz=[c[k]for k in ('nx','ny','nz')]
        cells=raw['cell_connectivity'].reshape(nz,ny,nx,8)[0].reshape(-1,8);polys=raw['vertices'].reshape(-1,3)[cells[:,:4],:2]-[c['cylinder_x'],c['cylinder_y']]
        ax.add_collection(PolyCollection(polys,facecolors='none',edgecolors='black',linewidths=.3));ax.set(xlim=(-.8,.8),ylim=(-.8,.8),title=name+' actual mesh',xlabel='(x-xc)/D',ylabel='(y-yc)/D');ax.set_aspect('equal')
    fig.tight_layout();figures.append(('near-wall-mesh',fig))
    with PdfPages(out/'report.pdf')as pdf:
        page=plt.figure(figsize=(8.3,11.7));page.text(.07,.94,'3-D transient pressure / velocity compatibility',fontsize=16)
        text=('Development verification: physical Re3900 qualification remains FALSE.\n\n'
              'The original uniform-cell startup sets wall mass flux to zero, while\n'
              'using homogeneous pressure Neumann data for its impulse.\n'
              'The new opt-in startup solves a dt-independent velocity-potential\n'
              'projection with the actual wall-normal impulse as Neumann data.\n'
              'Shared-face pressure traction updates cell momentum conservatively.\n'
              'Phi is not physical pressure; no elapsed time or force sample is added.\n\n'
              f"Affine manufacture: passed={manufactured['passed']}.\n"
              f"Medium normal defect legacy / Hodge: {results['medium-legacy']['independent_replay']['interior_normal_defect_max']:.6g} / {results['medium-hodge']['independent_replay']['interior_normal_defect_max']:.6g} U.\n"
              f"Fine Hodge normal defect: {results['fine-hodge']['independent_replay']['interior_normal_defect_max']:.6g} U.\n"
              f"Medium interpolated divergence: {results['medium-hodge']['independent_replay']['cell_interpolated_divergence_max']:.6g} U/D.\n\n"
              'The normal-defect improvement does not make the cell and face\n'
              'pressure operators identical. Strict cell divergence and startup\n'
              'Cp/grid response are reported separately, including failures.\n'
              'Old and Hodge starts have different initial cell fields; this is an\n'
              'initial-condition correction, not an identical-start time comparison.\n'
              'No velocity filtering, nonconservative reconstruction, or Cp fitting.\n\n'
              'Ordinary time steps still use homogeneous pressure Neumann data.\n'
              'Full momentum-compatible wall pressure and long-time Navier-Stokes\n'
              'validation remain required. CPU only; no LES, CUDA, or SUBOFF\n'
              'hydrodynamic qualification is claimed. See report.md / summary.json.')
        page.text(.07,.88,text,fontsize=11,va='top',linespacing=1.5);pdf.savefig(page);plt.close(page)
        for name,fig in figures:fig.savefig(out/(name+'.png'),dpi=180);fig.savefig(out/(name+'.svg'));pdf.savefig(fig);plt.close(fig)
    lines=['# Three-dimensional transient pressure–velocity compatibility','',
        '**Restricted initialization correction. Numerical and physical qualification remain false.**','',
        '## Problem and pressure boundary conditions','',
        'Uniform cell velocity combined with zero wall face flux is not an incompressible initial state around an impermeable cylinder. The previous startup projects that mismatch while imposing a homogeneous pressure normal derivative, also used in its LS and wall-traction reconstruction. This creates an inconsistent cell impulse even when the native face divergence is small.','',
        'A boundary pressure gradient may be nonzero when it is needed to achieve the prescribed boundary mass flux. The official OpenFOAM `fixedFluxPressure` boundary type expresses this principle; this implementation is an independently derived static shared-face projection, not a port of the OpenFOAM solver.','',
        '[Official OpenFOAM boundary source](https://github.com/OpenFOAM/OpenFOAM-dev/blob/master/src/finiteVolume/fields/fvPatchFields/derived/fixedFluxPressure/fixedFluxPressureFvPatchScalarField.H)','',
        '## Conservative correction','',
        'For the startup velocity potential phi, q_new = q_raw − grad(phi)·S. The Neumann condition on impermeable faces is d(phi)/dn = (q_raw − q_target)/|S|, with q_target=0. Other flux boundaries use their prescribed data; the outlet potential is zero. Its correction scale is one and does not depend on dt or density. Cell velocity changes only by −V⁻¹ sum(phi_face S), with one traction per shared face. The domain impulse therefore equals the physical boundary potential traction. Phi is archived separately from pressure. Time remains zero and no force/history sample is fabricated. Tangential no-slip is subsequently imposed by viscous boundary stress; a Hodge projection alone cannot create a resolved viscous startup layer.','',
        'The same supplied derivative enters three places: the LS normal-derivative row, the nonorthogonal pressure face flux, and the normal displacement contribution to face pressure. Nonzero outlet Dirichlet data enter both the Poisson RHS and the outlet gradient/traction. Default physical-step Neumann data remain homogeneous; the optional startup correction is not a complete momentum-compatible wall-pressure solver.','',
        '## Independent manufactured response','',
        'The manufacture p=.7x−.3y+.2 prescribes its actual face-normal derivative and actual outlet pressure on the real 48×16×4 body-fitted mesh. A complete Poisson solve removes its constant pressure impulse from both cell momentum and native face flux. This independently prescribed affine solution checks the boundary and interior operators; it is not a Re3900 flow reference.','',
        '| Manufactured metric | Actual value | Gate | Passed |','|---|---:|---:|---|']
    for m in manufactured['metrics']:lines.append(f"| {m['name']} | {m['value']:.6g} | {m['limit']:.6g} | {m['passed']} |")
    lines+=['',f"Applying homogeneous Neumann data to this different boundary problem produces face-pressure error {manufactured['legacy_homogeneous_neumann_face_error']:.6g} and LS gradient error {manufactured['legacy_homogeneous_neumann_gradient_error']:.6g}. These are incompatibility diagnostics, not fair errors for the homogeneous-boundary PDE.",'',
        '## Actual Re3900 startup experiments','',
        'All cases solve shared-face transport and pressure projection to tU/D=0.0005, with WALE, upwind fraction .1, and the same original .001 perturbation before optional projection. Legacy versus Hodge starts have different initial cell fields by construction. Time-step response compares complete Hodge runs from bitwise identical projected initial fields. Coarse/medium/fine meshes differ in angular, radial, spanwise resolution and stretching, so no asymptotic convergence order is assigned.','',
        '| Case | Cells | Native divergence | Cell interpolation divergence | Max normal defect / U | Momentum ledger | All gates |','|---|---:|---:|---:|---:|---:|---|']
    for name,r in results.items():
        a=r['independent_replay'];lines.append(f"| {name} | {np.prod([r['config'][k] for k in ('nx','ny','nz')])} | {a['native_divergence_max']:.6g} | {a['cell_interpolated_divergence_max']:.6g} | {a['interior_normal_defect_max']:.6g} | {a['momentum_ledger_max']:.6g} | {r['passed']} |")
    lines+=['','![Actual wall-face pressure](wall-pressure.png)','','![Unsmoothed native-cell pressure and velocity](native-fields.png)','','![Actual cell/face mismatch history](cell-face-compatibility.png)','','![Actual near-wall meshes](near-wall-mesh.png)','',
        '## Grid and complete time-step responses','',
        'Wall pressure means use actual wall-face pressures from the conservative traction, area-weighted over span at each angle. Spatial curves are compared by periodic interpolation of the coarser angular samples. Relative L2 is norm(coarse−fine)/norm(fine), not agreement with experiment. The actual metric values and definitions are saved in summary.json.','',
        '```json',json.dumps(summary['responses'],indent=2),'```','',
        '## Remaining failures and scope','',
        'Normal mismatch may pass a 3% local screen while the separate interpolated-cell divergence screen fails. Both remain visible. Small native continuity and a global momentum ledger do not qualify the cell/face coupling or the physical pressure. The face-history predictor preserves any remaining initial interpolation defect and integrates the difference of cell and face pressure operators. This patch does not eliminate that mechanism, provide a full mimetic pressure–velocity pair, or prove long-time compatibility. No physical Re3900 Cp, drag, Strouhal, LES, CUDA, or SUBOFF accuracy claim is made. Existing failed reports and raw fields remain unchanged.','',
        '## Reproduction','',
        '```bash','export PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1','/data/TensorFEM/.venv/bin/python scripts/verify_transient_pressure_compatibility.py --output results/transient-pressure-compatibility','```','',
        'Each case has common Metric/evaluate/save_run outputs, actual fields.npz, history.json/CSV and result.json. The producer snapshot and SHA256 manifest cover solver, mesh parents, report script and tests. Raw fields include pre- and post-step velocities, face momentum terms, actual pressure traction, mesh connectivity, initial velocity potential and native flux. Independent NumPy face-scatter checks the last local and global momentum balance and both divergence definitions. The report does not smooth cells or alter numerical results.']
    (out/'report.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=Path('docs/verification-transient-pressure-compatibility'));p.add_argument('--final-time',type=float,default=.0005);p.add_argument('--dt',type=float,default=1e-5);a=p.parse_args()
    if a.output.exists()and any(a.output.iterdir()):p.error('output must be empty')
    a.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
    repo=Path(__file__).resolve().parents[1];env=provenance();sources={repo/k:v for k,v in env['source_sha256'].items()}
    for name in ['scripts/verify_transient_pressure_compatibility.py','src/tensorfvm/body_fitted_transient3d.py','src/tensorfvm/body_fitted3d.py','src/tensorfvm/solver3d.py','src/tensorfvm/mesh3d.py','tests/test_body_fitted_transient3d.py']:sources[repo/name]=sha(repo/name)
    source_map={}
    for source,digest in sorted(sources.items()):
        target=a.output/'source-snapshot'/source.relative_to(repo);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        if sha(target)!=digest:raise RuntimeError('producer source changed during snapshot')
        source_map[str(source.relative_to(repo))]=digest
    env['source_sha256']=source_map;write_json(a.output/'environment.json',env)
    manufactured=manufacture(a.output)
    coarse=(48,16,8,3.5);medium=(64,32,12,7.);fine=(96,48,12,7.)
    definitions=[('coarse-legacy',coarse,a.dt,False),('medium-legacy',medium,a.dt,False),('coarse-hodge',coarse,a.dt,True),('medium-hodge',medium,a.dt,True),('medium-hodge-half-dt',medium,a.dt/2,True),('fine-hodge',fine,a.dt,True)]
    results={name:run_case(a.output,name,grid,dt,a.final_time,init)for name,grid,dt,init in definitions}
    curves={name:wall_curve(a.output,name)for name in results}
    def comparison(first,second):
        theta,ref=curves[second];angle,coarse_values=curves[first]
        pred=np.interp(theta,angle,coarse_values,period=360)
        return dict(wall_face_cp_relative_l2=float(np.linalg.norm(pred-ref)/np.linalg.norm(ref)),definition='norm(periodic_interp(first)-second)/norm(second)',reference='internal sensitivity only, not experiment')
    initial_a=np.load(a.output/'medium-hodge'/'fields.npz');initial_b=np.load(a.output/'medium-hodge-half-dt'/'fields.npz')
    identical=all(np.array_equal(initial_a[k],initial_b[k])for k in ('initial_velocity','initial_flux','initial_velocity_potential'))
    responses={'legacy_coarse_to_medium':comparison('coarse-legacy','medium-legacy'),'hodge_coarse_to_medium':comparison('coarse-hodge','medium-hodge'),'hodge_medium_to_fine':comparison('medium-hodge','fine-hodge'),'hodge_complete_dt_to_half_dt':comparison('medium-hodge','medium-hodge-half-dt'),'dt_independent_projected_initial_fields_bitwise_identical':identical,'asymptotic_grid_convergence_qualified':False}
    summary=dict(manufactured_affine_passed=manufactured['passed'],responses=responses,cases=results,numerical_qualified=False,physical_accuracy_qualified=False,initialization_default_enabled=False,scope='optional conservative initialization and pressure boundary interface; not complete NS coupling')
    write_json(a.output/'summary.json',summary);publish(a.output,results,manufactured,summary)
    write_json(a.output/'manifest.json',dict(schema='tensorfvm.transient-pressure-compatibility/1',source_snapshot_root='source-snapshot',source_sha256=source_map,artifact_sha256=artifact_manifest(a.output)))
    print(json.dumps(dict(manufactured_passed=manufactured['passed'],responses=responses),indent=2),flush=True)

if __name__=='__main__':main()

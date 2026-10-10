#!/usr/bin/env python3
"""Replay unsmoothed saved pressure and native mesh; compare startup coupling."""
import argparse,json,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.backends.backend_pdf import PdfPages
from tensorfvm.collocated_diagnostics import flux_coupling,angular_pressure
from tensorfvm.verification.core import sha,write_json

def replay(path):
    z=np.load(path/'checkpoint.npz');c=json.loads((path/'config.json').read_text())['config']
    o,n=z['owner'],z['neighbor'];safe=n.clip(min=0);b=n<0;wall=z['wall'];outlet=z['outlet']
    slip=b&~wall&~outlet&(np.isclose(z['face_centers'][:,1],0,atol=1e-9)|np.isclose(z['face_centers'][:,1],c['height'],atol=1e-9))
    inlet=b&~wall&~outlet&~slip
    uf=.5*(z['velocity'][o]+z['velocity'][safe]);uf[wall]=0;uf[outlet]=z['velocity'][o[outlet]];uf[slip]=z['velocity'][o[slip]];uf[slip,1]=0;uf[inlet]=[c['inlet_velocity'],0,0]
    metrics=dict(**flux_coupling(o,n,z['area'],z['volumes'],z['flux'],uf,c['inlet_velocity']),**angular_pressure(z['pressure'],c['nx'],c['ny'],c['nz']),time=float(z['time']),cells=len(z['volumes']),pressure_min=float(z['pressure'].min()),pressure_max=float(z['pressure'].max()))
    iq=(uf*z['area']).sum(1)
    net=np.bincount(o,weights=iq,minlength=len(z['volumes']))-np.bincount(n[n>=0],weights=iq[n>=0],minlength=len(z['volumes']))
    worst=int(np.argmax(np.abs(net/z['volumes'])))
    metrics.update(worst_interpolated_divergence_cell=worst,worst_cell_volume=float(z['volumes'][worst]),worst_cell_center=z['centers'][worst].tolist())
    return z,c,metrics

def surface(path):
    data=np.genfromtxt(path/'wall-pressure.csv',delimiter=',',names=True)
    angle=np.unique(data['theta_degrees']);area=np.sqrt(sum(data['area_'+k]**2 for k in ['x','y','z']))
    return angle,np.array([np.average(data['Cp'][data['theta_degrees']==t],weights=area[data['theta_degrees']==t])for t in angle])

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True);a=p.parse_args();r=a.input
    cases={};metrics={}
    for grid in ['coarse','medium']:
        for kind,path in [('old',a.baseline/grid),('new',r/grid),('half-dt',r/(grid+'-half-dt'))]:
            z,c,m=replay(path);cases[grid,kind]=(z,c);metrics[grid+'-'+kind]=m
    refinement={}
    for grid in ['coarse','medium']:
        z,c=cases[grid,'new'];zh,_=cases[grid,'half-dt']
        pa=z['pressure'];pb=zh['pressure']
        walla=pa.reshape(c['nz'],c['ny'],c['nx'])[:,0].mean(0);wallb=pb.reshape(c['nz'],c['ny'],c['nx'])[:,0].mean(0)
        refinement[grid]=dict(same_physical_time=float(z['time']),pressure_relative_l2=float(np.linalg.norm(pa-pb)/np.linalg.norm(pb)),wall_owner_pressure_relative_l2=float(np.linalg.norm(walla-wallb)/np.linalg.norm(wallb)),definition='norm(dt - dt/2)/norm(dt/2); all steps rerun from the same startup, not a frozen-state response')
    coarse_theta,coarse_cp=surface(r/'coarse');fine_theta,fine_cp=surface(r/'medium')
    interpolated=np.interp(fine_theta,coarse_theta,coarse_cp,period=360)
    spatial=dict(wall_face_cp_relative_l2=float(np.linalg.norm(interpolated-fine_cp)/np.linalg.norm(fine_cp)),qualified=False,definition='coarse reconstructed wall-face Cp periodically interpolated to medium wall angles; relative L2 normalized by medium; two-grid startup diagnostic, not a convergence order')
    write_json(r/'comparison.json',dict(metrics=metrics,time_step_refinement=refinement,spatial_wall_cp_diagnostic=spatial,numerical_qualified=False,physical_accuracy_qualified=False,baseline_manifest_sha256=sha(a.baseline/'manifest.json')))
    figures=[]
    f,axes=plt.subplots(2,3,figsize=(15,8))
    z,c=cases['medium','old'];bound=max(abs(z['pressure'].min()),abs(z['pressure'].max()))
    for col,kind in enumerate(['old','new','half-dt']):
        z,c=cases['medium',kind];nx,ny,nz=[c[k]for k in ['nx','ny','nz']];k=nz//2
        cells=z['connectivity'].reshape(nz,ny,nx,8)[k].reshape(-1,8)
        polys=z['vertices'].reshape(-1,3)[cells[:,:4],:2]-[c['cylinder_x'],c['cylinder_y']]
        for row,(values,label,limits)in enumerate([(z['pressure'].reshape(nz,ny,nx)[k],'p / (rho U²)',(-bound,bound)),(np.linalg.norm(z['velocity'].reshape(nz,ny,nx,3)[k],axis=-1),'speed / U',(0,2.))]):
            pc=PolyCollection(polys,array=values.ravel(),cmap='coolwarm'if row==0 else'turbo',edgecolors='none');pc.set_clim(*limits);axes[row,col].add_collection(pc)
            axes[row,col].set(xlim=(-1,3),ylim=(-1.5,1.5),xlabel='x/D',ylabel='y/D',title=f'{kind}, t*=0.0005');axes[row,col].set_aspect('equal');f.colorbar(pc,ax=axes[row,col],label=label)
    f.suptitle('Unsmoothed native-cell fields: identical pressure / speed color scales');f.tight_layout();figures.append(('fields-comparison',f))
    f,axes=plt.subplots(1,2,figsize=(12,4.5))
    for ax,grid in zip(axes,['coarse','medium']):
        for kind in ['old','new','half-dt']:
            z,c=cases[grid,kind];nx,ny,nz=[c[k]for k in ['nx','ny','nz']]
            cp=2*z['pressure'].reshape(nz,ny,nx)[:,0].mean(0)/(c['density']*c['inlet_velocity']**2)
            centers=z['centers'].reshape(nz,ny,nx,3)[0,0];angle=np.degrees(np.arctan2(centers[:,1]-c['cylinder_y'],centers[:,0]-c['cylinder_x']))%360;order=np.argsort(angle)
            ax.plot(angle[order],cp[order],label=kind)
        ax.set(xlabel='theta (deg); 0 downstream',ylabel='owner-ring Cp',title=grid);ax.grid(alpha=.2);ax.legend()
    f.suptitle('Identical adjacent-cell definition for old/new comparison; no experiment implied');f.tight_layout();figures.append(('pressure-curves',f))
    f,ax=plt.subplots(figsize=(10,4.5))
    for grid in ['coarse','medium']:
        for suffix in ['','-half-dt']:
            theta,cp=surface(r/(grid+suffix));ax.plot(theta,cp,label=grid+suffix)
    ax.set(xlabel='theta (deg); 0 downstream',ylabel='reconstructed wall-face Cp',title='New actual wall traction pressure; spatial convergence NOT qualified');ax.legend();ax.grid(alpha=.2);f.tight_layout();figures.append(('new-surface-cp',f))
    f,axes=plt.subplots(1,3,figsize=(15,4.5));z,c=cases['medium','new'];nx,ny,nz=[c[k]for k in ['nx','ny','nz']];k=nz//2
    cells=z['connectivity'].reshape(nz,ny,nx,8)[k].reshape(-1,8);polys=z['vertices'].reshape(-1,3)[cells[:,:4],:2]-[c['cylinder_x'],c['cylinder_y']]
    axes[0].add_collection(PolyCollection(polys,facecolors='none',edgecolors='black',linewidths=.3));axes[0].set(xlim=(-.8,.8),ylim=(-.8,.8),title='Actual medium-grid near-wall cells')
    pressure=z['pressure'].reshape(nz,ny,nx)[k];speed=np.linalg.norm(z['velocity'].reshape(nz,ny,nx,3)[k],axis=-1)
    for ax,values,title in zip(axes[1:],[pressure,speed],['New pressure (own color scale)','New speed / U']):
        pc=PolyCollection(polys,array=values.ravel(),edgecolors='none',cmap='turbo');ax.add_collection(pc);ax.set(xlim=(-1,3),ylim=(-1.5,1.5),title=title);f.colorbar(pc,ax=ax)
    for ax in axes:ax.set_aspect('equal');ax.set(xlabel='x/D',ylabel='y/D')
    f.tight_layout();figures.append(('new-fields-and-mesh',f))
    with PdfPages(r/'report.pdf')as pdf:
        notice=plt.figure(figsize=(8.3,11.7));notice.text(.08,.92,'Re3900 pressure coupling: development verification',fontsize=15)
        old=metrics['medium-old'];new=metrics['medium-new'];ref=refinement['medium']
        notice.text(.08,.83,f"Matched startup: 24,576 native body-fitted cells, tU/D=0.0005\n\nOld high-angular-frequency pressure fraction: {100*old['high_angular_frequency_energy_fraction']:.3f}%\nNew fraction: {100*new['high_angular_frequency_energy_fraction']:.5f}%\nPressure L2 dt versus dt/2: {100*ref['pressure_relative_l2']:.4f}%\n\nChanges: retained native face-flux history, boundary-consistent\nmomentum increments, conservative skew-correct face pressure.\n\nRemaining cell/face normal mismatch: {new['interior_max_normal_velocity_difference']:.3f} U\nInterpolated cell-velocity divergence: {new['interpolated_cell_velocity_divergence_max']:.3g} U/D\nNative face divergence: {new['native_face_divergence_max']:.3g} U/D\n\nNUMERICAL / PHYSICAL QUALIFICATION: FALSE\nShort-time pressure improvement is not verified Re3900 LES.\nCell/face compatibility, grid refinement and long-time statistics\nremain required. See report.md and comparison.json for full evidence.",fontsize=12,va='top',linespacing=1.6)
        pdf.savefig(notice);plt.close(notice)
        for name,f in figures:f.savefig(r/(name+'.png'),dpi=220);f.savefig(r/(name+'.svg'));pdf.savefig(f);plt.close(f)
    lines=['# Re3900 pressure/mesh coupling correction','',
      '**Development verification only. Earlier continuity/momentum passes did not verify pressure coupling. Neither old nor new runs qualify Re3900 accuracy or LES convergence.**','',
      '## Failure mechanism and changes','',
      'The previous predictor interpolated the complete old cell velocity every step, discarding the last native pressure-corrected face flux. The interpolation defect entered the next pressure RHS divided by dt, sustaining near-Nyquist angular pressure modes even with tiny native face divergence. The predictor now advances the retained native face flux using the difference of tentative and old cell-velocity interpolations. Taking a difference cancels prescribed inlet values correctly.','',
      'Pressure traction previously used arithmetic owner/neighbor averaging on stretched/skew faces. It now interpolates at the projected face position and reconstructs the skew displacement with the LS gradient, preserving one conservative traction per shared face. Pressure LS Neumann rows now use actual face-normal directions and a matching normal matrix, rather than the skew owner-to-face displacement. This is a least-squares boundary derivative constraint, not a strong pointwise condition on every reconstructed cell gradient. Outlet pressure remains zero. Independent analytic tests verify the interior affine pressure force and p=x at skew horizontal slip faces. No Cp fitting, filtering or plot smoothing is applied.','',
      '## Actual fields and mesh','',
      'Identical scales in this comparison expose the original oscillations. All fields come from saved checkpoints; all cases reach the same tU/D=0.0005 from identical startup conditions. The lower row is speed, not a pressure-derived proxy.','',
      '![Native pressure and speed, old/new/time-step refinement](fields-comparison.png)','',
      'The following shows the new pressure on its own scale and the actual body-fitted cells. Smoothness alone is not accuracy.','',
      '![Actual mesh and new computed fields](new-fields-and-mesh.png)','',
      '![Adjacent owner-cell Cp with one definition for all cases](pressure-curves.png)','',
      'The comparison uses span-mean adjacent owner-cell pressures for both algorithms. New run wall-pressure.csv instead contains reconstructed face pressure actually used for wall traction; its definition is recorded in pressure-coupling.json. These are startup curves, not stationary Cp or experimental comparisons.','',
      '## Independent checkpoint replay','',
      '| Grid / algorithm | Pressure min / max | High angular energy fraction | Native div max | Interpolated cell div max | Max normal flux defect / U |','|---|---:|---:|---:|---:|---:|']
    for name,m in metrics.items():lines.append(f"| {name} | {m['pressure_min']:.5g} / {m['pressure_max']:.5g} | {m['high_angular_frequency_energy_fraction']:.6g} | {m['native_face_divergence_max']:.4g} | {m['interpolated_cell_velocity_divergence_max']:.4g} | {m['interior_max_normal_velocity_difference']:.4g} |")
    lines+=['','High angular energy sums Fourier modes k >= ceil(Ntheta/4) in the span-mean innermost pressure ring. This is a startup diagnostic, not a universal turbulence threshold. A single exact alternating-mode test would miss the observed k=29/31 modes on Ntheta=64.','',
      '## Complete time-step refinement','',
      '| Mesh | Pressure relative L2 dt vs dt/2 | Owner-ring pressure relative L2 |','|---|---:|---:|']
    for grid,m in refinement.items():lines.append(f"| {grid} | {100*m['pressure_relative_l2']:.6g}% | {100*m['wall_owner_pressure_relative_l2']:.6g}% |")
    lines+=['','The dt=1e-5 run takes 50 steps; dt=5e-6 takes 100. This verifies short-time pressure sensitivity only; no stationary statistics or spatial convergence is inferred.','',
      '## Spatial pressure comparison remains unqualified','',
      '![New reconstructed surface Cp on both grids and both time steps](new-surface-cp.png)','',
      f'The actual reconstructed wall-face Cp differs by {100*spatial["wall_face_cp_relative_l2"]:.4f}% in relative L2 after periodic interpolation of coarse wall angles to the medium wall angles. This exceeds 3%. These are two startup grids with different radial and span resolution, not a demonstrated convergence sequence. Both pressure smoothness and small dt sensitivity are insufficient for physical accuracy.','',
      '## Remaining qualification blockers','',
      'Retaining native face history removes pressure reinjection, but the cell/face discrepancy persists, including the initial projection mismatch. Both divergence measures and normal defects are exposed above. Native continuity and a momentum ledger alone cannot qualify the coupled solution. Conservative face reconstruction improves affine response but does not make the two pressure operators identical. Further spatial refinement and long-time compatibility verification remain required before any 3% physical accuracy claim. New summaries and acceptance audits therefore record mass/momentum checks separately and keep numerical/physical qualification false.','',
      'Existing baseline checkpoints and producer manifests are unchanged. The old report is explicitly withdrawn as evidence of verified pressure coupling. First-step incompressible startup pressure can legitimately scale as 1/dt; the defect is repeated high-frequency pressure after that projection, not the mere presence of a startup impulse.','',
      '## Reproduction','',
      '```bash','export PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1',
      'python scripts/run_body_fitted_cylinder3900.py --output results/pressure-medium --nx 64 --ny 32 --nz 12 --stretching 7 --dt 1e-5 --steps 50',
      'python scripts/run_body_fitted_cylinder3900.py --output results/pressure-medium-half-dt --nx 64 --ny 32 --nz 12 --stretching 7 --dt 5e-6 --steps 100',
      'python scripts/report_cylinder3900_pressure_coupling.py --input docs/verification-cylinder3900-pressure-coupling --baseline docs/verification-cylinder3900-wall-refinement','```']
    (r/'report.md').write_text('\n'.join(lines)+'\n')
    repo=Path(__file__).resolve().parents[1];sources=[Path(__file__).resolve(),repo/'src/tensorfvm/collocated_diagnostics.py',repo/'tests/test_body_fitted_transient3d.py'];bindings={}
    for src in sources:
        target=r/'report-source-snapshot'/src.relative_to(repo);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,target);bindings[str(src.relative_to(repo))]=sha(target)
    write_json(r/'manifest.json',dict(report_source_sha256=bindings,artifact_sha256={str(f.relative_to(r)):sha(f)for f in r.rglob('*')if f.is_file()and f.name!='manifest.json'},child_manifests={name:sha(r/name/'manifest.json')for name in ['coarse','medium','coarse-half-dt','medium-half-dt']}))
    print(json.dumps(dict(time_step_refinement=refinement,metrics=metrics),indent=2))

if __name__=='__main__':main()

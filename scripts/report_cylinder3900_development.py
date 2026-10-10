#!/usr/bin/env python3
"""Native wall preflight and matched-time startup/cost development report."""
import argparse,json,shutil,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import PolyCollection
from tensorfvm.verification.core import Metric,evaluate,write_json,sha
from tensorfvm.cylinder3900_statistics import statistics

def coupling(path):
    z=np.load(path/'checkpoint.npz');c=json.loads((path/'config.json').read_text())['config'];u=z['velocity'];o=z['owner'];n=z['neighbor'];safe=n.clip(min=0);area=z['area'];q=z['flux'];b=n<0
    uf=.5*(u[o]+u[safe]);wall=z['wall'];outlet=z['outlet'];slip=b&~wall&~outlet&(np.isclose(z['face_centers'][:,1],0,atol=1e-9)|np.isclose(z['face_centers'][:,1],c['height'],atol=1e-9));inlet=b&~wall&~outlet&~slip
    uf[wall]=0;uf[outlet]=u[o[outlet]];uf[slip]=u[o[slip]];uf[slip,1]=0;uf[inlet]=[c['inlet_velocity'],0,0]
    delta=q-(uf*area).sum(1);normal=abs(delta[n>=0])/np.linalg.norm(area[n>=0],axis=1)/c['inlet_velocity']
    return dict(interior_max_normal_velocity_difference=float(normal.max()),interior_p95_normal_velocity_difference=float(np.quantile(normal,.95)),
                interior_flux_relative_L2=float(np.linalg.norm(delta[n>=0])/np.linalg.norm(q[n>=0])),physical_coupling_qualified=False,
                description='current cell-velocity interpolation versus authoritative projected face flux, actual boundary values; diagnostic only, not continuity acceptance')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);a=p.parse_args();r=a.input
    if (r/'report.md').exists():p.error('report exists; preserve evidence')
    probes=json.loads((r/'preflight/probes.json').read_text());cases={name:json.loads((r/name/'summary.json').read_text())for name in ['coarse','medium']}
    config={name:json.loads((r/name/'config.json').read_text())['config']for name in cases}
    rows=[]
    for x in probes:
        geometry,passed=evaluate([Metric('closed_cell_area',x['closed_cell_area_max'],1e-12,'native closed 3D face-area sum'),Metric('nonpositive_cell_volume_count',float(x['minimum_volume']<=0),.5,'native volume minimum strictly positive')])
        rows.append(dict(cells=x['cells'],stretch=x['config']['body_fitted_stretching'],metrics=geometry,geometry_passed=passed,physical_accuracy_qualified=False))
    diag={name:coupling(r/name)for name in cases};write_json(r/'cell-face-coupling.json',diag);write_json(r/'preflight-common-metrics.json',rows)
    h=[dict(time=float(t),Cd=1.015,Cl=float(np.sin(2*np.pi*.215*t)))for t in np.arange(0,601,.002)]
    spectral=statistics(h,discard_time=0);write_json(r/'spectral-resolution-verification.json',dict(kind='synthetic operator signal, not CFD or standard field',input=dict(St=.215,dt=.002,duration=601),output=spectral,
               old_fixed_4096_bin=.1220703125,physical_accuracy_qualified=False))
    figs=[]
    f,axes=plt.subplots(1,3,figsize=(12,4))
    groups=sorted({(x['config']['nx'],x['config']['ny'],x['config']['nz'])for x in probes})
    for group in groups:
        pp=sorted([x for x in probes if tuple(x['config'][k]for k in ('nx','ny','nz'))==group],key=lambda x:x['config']['body_fitted_stretching']);ss=[x['config']['body_fitted_stretching']for x in pp];label=' x '.join(map(str,group))
        for ax,yy in zip(axes,[[x['wall_distance_D']['max']for x in pp],[x['initial_yplus']['max']for x in pp],[x['explicit_diffusion_dt_screen_limit']for x in pp]]):ax.semilogy(ss,yy,'o-',label=label);ax.grid(alpha=.2);ax.set_xlabel('Radial stretching')
    for ax,title in zip(axes,['Actual maximum wall distance / D','Initial y+ indicator; NOT qualification','Initial explicit diffusion dt ceiling']):ax.set_title(title,fontsize=10)
    axes[1].axhline(1,color='black',ls='--');axes[0].legend(fontsize=7);f.tight_layout();figs.append(('preflight',f))
    f,axes=plt.subplots(1,3,figsize=(12,4));names=list(cases)
    for ax,key,label,scale in [(axes[0],'seconds_per_completed_step','Observed seconds / step',1),(axes[1],'pressure_lu_factor_bytes','Actual pressure LU factor storage / MB',1e-6),(axes[2],'setup_walltime','Mesh/operator/LU setup seconds',1)]:
        ax.bar(names,[cases[n]['performance'][key]*scale for n in names]);ax.set_title(label,fontsize=10);ax.grid(axis='y',alpha=.2)
    f.suptitle('Startup costs under concurrent CPU load; not isolated scaling');f.tight_layout();figs.append(('costs',f))
    f,axes=plt.subplots(1,2,figsize=(10,4))
    for name in names:
        hist=json.loads((r/name/'history.json').read_text());force=json.loads((r/name/'forces.json').read_text())
        axes[0].loglog([x['time']for x in force],[abs(x['Cd'])for x in force],label=name)
        axes[1].semilogy([x['time']for x in hist],[x['continuity']for x in hist],label=name)
    axes[0].set(xlabel='t U/D',ylabel='|Cd| actual startup');axes[1].set(xlabel='t U/D',ylabel='Shared-face divergence');axes[0].legend();axes[1].legend();f.suptitle('Matched 50-step startup at dt=1e-5: NOT converged');f.tight_layout();figs.append(('matched-startup',f))
    z=np.load(r/'medium/checkpoint.npz');c=config['medium'];nodes=z['vertices'];nx,ny,nz=(c[k]for k in ['nx','ny','nz']);k=nz//2;cells=z['connectivity'].reshape(nz,ny,nx,8)[k].reshape(-1,8);polys=nodes.reshape(-1,3)[cells[:,:4],:2]-[c['cylinder_x'],c['cylinder_y']]
    f,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,values,title in zip(axes,[z['pressure'].reshape(nz,ny,nx)[k],np.linalg.norm(z['velocity'].reshape(nz,ny,nx,3)[k],axis=-1)],['Actual instantaneous pressure','Actual instantaneous speed']):
        pc=PolyCollection(polys,array=values.ravel(),cmap='turbo',edgecolors='none');ax.add_collection(pc);ax.autoscale();ax.set(xlim=(-1,4),ylim=(-2,2),xlabel='x/D',ylabel='y/D',title=title);ax.set_aspect('equal');f.colorbar(pc,ax=ax)
    f.suptitle('24576 body-fitted CV, t*=0.0005: startup only');f.tight_layout();figs.append(('medium-fields',f))
    with PdfPages(r/'report.pdf')as pdf:
        for name,f in figs:f.savefig(r/(name+'.png'),dpi=180);f.savefig(r/(name+'.svg'));pdf.savefig(f);plt.close(f)
    lines=['# Re3900 wall refinement and startup development report','','## Geometry and initial-viscosity preflight','',
      'All eight meshes were actually built with the shared 3-D native face metrics, LS gradient and WALE initial viscosity. No global pressure CSR or LU was constructed in the preflight. The startup velocity is not projected; its y+ indicator is not a long-time wall-resolution qualification. Geometry gates use the common `tensorfvm.verification.core.Metric/evaluate` module.','',
      '| Grid | Stretch | Wall distance max / D | Initial y+ max | Delta z / D | Diffusion dt ceiling | Wake h95 / D |','|---|---:|---:|---:|---:|---:|---:|']
    for x in probes:
        c0=x['config'];lines.append(f"| {c0['nx']} x {c0['ny']} x {c0['nz']} | {c0['body_fitted_stretching']} | {x['wall_distance_D']['max']:.3g} | {x['initial_yplus']['max']:.3f} | {x['spanwise_spacing_D']:.4f} | {x['explicit_diffusion_dt_screen_limit']:.3g} | {x['wake_region_volume_length_D']['p95']:.3f} |")
    lines+=['','192 x 96 x 32 / stretch8 gives initial y+max=0.838, but only an initial wall indicator. Its Delta z=0.0982D remains four times the [primary DNS](https://doi.org/10.1063/1.4818641) 128-plane spacing of pi/128=0.02454D; wake volume-length h95=0.085D is also much larger than the paper’s reported near-wake average h about0.018D. A thin first wall cell alone does not establish accurate wake LES.','',
      'The geometry run peak RSS was a cumulative process peak of about2.41 GB; the 589824-CV native stored tensors are recorded separately. These are measured geometry/transport costs, not pressure-solver costs. No large-grid LU timing or memory result is claimed.','',
      '## Matched-time two-grid short runs','',
      'Both real CFD runs use the original DNS domain [-8,16]D x [-10,10]D, span piD, inletU=1, top/bottom slip, cylinder no-slip and periodic z. Both take50 steps at dt=1e-5 to tU/D=0.0005. They ran concurrently with other CPU jobs, so timings are observed under load rather than isolated scaling.','',
      '| Case | CV | Final maxdiv | Final y+max | Seconds/step | LU stored MB | Setup seconds |','|---|---:|---:|---:|---:|---:|---:|']
    for name,x in cases.items():
        perf=x['performance'];lines.append(f"| {name} | {x['mesh_cells']} | {x['final']['continuity']:.3g} | {x['wall_resolution']['yplus_max']:.3f} | {perf['seconds_per_completed_step']:.3f} | {perf['pressure_lu_factor_bytes']/1e6:.2f} | {perf['setup_walltime']:.3f} |")
    sec=cases['medium']['performance']['seconds_per_completed_step'];days=600/1e-5*sec/86400
    lines+=['',f'The medium pressure matrix has170496 entries, while its actual L/U factors have20452054 entries and245.62MB of stored sparse arrays. Four times the CV count produces about9.81 times the factor storage. Measured medium process peak RSS is about1.16GB. At its observed{sec:.3f}s/step, an extrapolation of600D/U at the unchanged dt1e-5 needs60million steps and about{days:.1f}days. **This is an arithmetic cost extrapolation, not a measured long run**; solver iterations, CPU load and future numerical methods will change costs.','',
      'Initial geometry / stretch8 for the589824-CV grid limits diffusion dt to2.38e-6; using20% of that ceiling is about4.75e-7 and implies roughly1.26billion steps over600D/U. No per-step runtime was measured on that grid. Explicit viscosity and full 3-D sparse LU are concrete production bottlenecks; escalating only mesh counts is insufficient.','',
      'The common acceptance audit replays final continuity independently from native checkpoint face fluxes and checks all recorded mass/momentum/CFL/diffusion limits. Both numerical audits pass. Both instantaneous wall-resolution gates fail. Physical accuracy remains false, qualified Cd/St/Cpb/profile errors remain null, and these two startup solutions do not establish a convergence order.','',
      '## Cell/face coupling diagnostic','',
      '| Case | Interior max normal defect / U | p95 normal defect / U | Relative face-flux L2 defect |','|---|---:|---:|---:|']
    for name,x in diag.items():lines.append(f"| {name} | {x['interior_max_normal_velocity_difference']:.6g} | {x['interior_p95_normal_velocity_difference']:.6g} | {x['interior_flux_relative_L2']:.6g} |")
    lines+=['','These compare native projected mass flux to interpolation of the current actual cell velocity, with actual no-slip/slip/inlet/outlet values. They are diagnostic, not new acceptance thresholds. Small face continuity does not prove collocated cell/face physical coupling; that still needs independent pressure/momentum and refinement verification.','',
      '## Spectral-resolution fix','',f'The old4096-sample Welch ceiling at dt=.002 gives DeltaSt=0.12207 and cannot support a3% target aroundSt=.215. The new method selects the actual segment length from the target resolution and available record; no zero padding. A synthetic St=.215 signal with dt=.002 and601D/U returns St={spectral["St"]:.6g}, DeltaSt={spectral["frequency_resolution"]:.6g}, segment samples={spectral["segment_samples"]}, segment time={spectral["segment_duration"]:.6g}. This is an operator test, not a CFD result. Short insufficient-resolution and constant-lift records are blocked. Duration, spectral eligibility and physical accuracy are distinct.','',
      'Latest statistics also use a Student-t block-mean interval with explicit degrees of freedom, block sample counts and durations; independence and stationarity assumptions remain unverified. The CFD producers are preserved exactly in their snapshots; the later statistical postprocessing/doc correction is independently bound by this report source snapshot and is not substituted into older producer manifests.','',
      '## Software commands and next work','',
      '```bash','PYTHONPATH=src python scripts/probe_body_fitted_cylinder3900.py --output results/cylinder3900-wall-preflight',
      'PYTHONPATH=src python scripts/run_body_fitted_cylinder3900.py --output results/cylinder3900-stage2-medium --nx64 --ny32 --nz12 --stretching7 --dt .00001 --steps50'.replace('--nx64','--nx 64').replace('--ny32','--ny 32').replace('--nz12','--nz 12').replace('--stretching7','--stretching 7').replace('--steps50','--steps 50'),
      'PYTHONPATH=src python scripts/audit_body_fitted_cylinder3900.py --input results/cylinder3900-stage2-medium','PYTHONPATH=src python scripts/report_cylinder3900_development.py --input docs/verification-cylinder3900-wall-refinement','```','',
      'Run with OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1. --steps means additional steps, with exact source/configuration restart validation. The large preflight runs do not solve CFD. Next: implicit viscosity with its own response/stability tests; scalable pressure solver (periodic-extrusion Fourier separation or iterative preconditioning); verified collocated pressure/velocity coupling; near-wake/span refinement; then long stationary statistics and actual primary DNS/experimental comparisons <=3%. GPU is untested and is not expanded in this phase.']
    (r/'report.md').write_text('\n'.join(lines)+'\n')
    repo=Path(__file__).resolve().parents[1];sources=[Path(__file__).resolve(),repo/'tests/test_body_fitted_transient3d.py']+[Path(m.__file__).resolve()for name,m in list(sys.modules.items())if(name=='tensorfvm'or name.startswith('tensorfvm.'))and getattr(m,'__file__',None)and Path(m.__file__).suffix=='.py'];bindings={}
    for src in sorted(set(sources)):
        dest=r/'report-source-snapshot'/src.relative_to(repo);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest);bindings[str(src.relative_to(repo))]=sha(dest)
    write_json(r/'manifest.json',dict(schema='tensorfvm.cylinder3900-development/1',report_source_sha256=bindings,artifact_sha256={str(f.relative_to(r)):sha(f)for f in r.rglob('*')if f.is_file()and f!=r/'manifest.json'}))
if __name__=='__main__':main()

#!/usr/bin/env python3
"""Replay raw ABC errors and report actual quadratic FVM refinement evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from tensorfvm.verification.core import Metric,evaluate
from report_native_abc_convergence3d import sections

KEYS=('velocity_vector','velocity_x','velocity_y','velocity_z','pressure_mean_removed')
RESIDUALS=('velocity_change','pressure_change','flux_change','momentum_residual','native_divergence_max','current_reconstructed_divergence_max','face_flux_consistency_max')


def errors(fields):
    v=fields['cell_volumes'];u=fields['velocity'];r=fields['reference_velocity']
    if np.any(v<=0) or not all(np.isfinite(fields[k]).all() for k in ('cell_volumes','velocity','reference_velocity','pressure','reference_pressure')):raise ValueError('invalid raw fields')
    norm=lambda a,b:float(np.sqrt(np.sum(v[:,None]*(a-b)**2)/np.sum(v[:,None]*b**2))) if a.ndim==2 else float(np.sqrt(np.sum(v*(a-b)**2)/np.sum(v*b**2)))
    p=fields['pressure'];q=fields['reference_pressure'];p=p-np.sum(v*p)/v.sum();q=q-np.sum(v*q)/v.sum()
    result=dict(velocity_vector=norm(u,r),pressure_mean_removed=norm(p,q))
    result.update({f'velocity_{axis}':norm(u[:,i],r[:,i]) for i,axis in enumerate('xyz')});return result


def load(directory):
    summary=json.loads((directory/'summary.json').read_text());config=summary.get('config',json.loads((directory/'config.json').read_text()) if (directory/'config.json').exists() else {})
    rows=[];inputs=[directory/'summary.json'];binding_checks=[]
    if (directory/'config.json').exists():inputs.append(directory/'config.json')
    for advertised in sorted(summary['cases'],key=lambda c:c['n']):
        n=advertised['n'];case=directory/f'n-{n}';result=json.loads((case/'result.json').read_text());inputs.append(case/'result.json')
        row=dict(n=n,initialization=None,cells=result.get('cells'),numerically_converged=bool(result.get('numerically_converged',False)),error=result.get('error'),history=result.get('history',[]),relative_errors=None,fields=None)
        if not row['history'] and (case/'failed-history.json').exists():
            row['history']=json.loads((case/'failed-history.json').read_text());inputs.append(case/'failed-history.json')
        if config.get('initialization','zero')=='baseline':
            initial_result=case/'baseline-initialization-result.json';initial_fields=case/'baseline-initialization-fields.npz'
            if initial_result.exists() and initial_fields.exists():
                initial=json.loads(initial_result.read_text())
                with np.load(initial_fields) as data:finite=all(np.isfinite(data[key]).all() for key in ('velocity','pressure','flux'))
                row['initialization']=dict(converged=bool(initial.get('converged',False)),history=initial.get('history',[]),initial_velocity=initial.get('initial_velocity'),finite_raw_fields=finite)
                inputs.extend((initial_result,initial_fields))
        for name,expected in result.get('source_sha256',{}).items():
            path=directory/'source-snapshot'/name;binding_checks.append(dict(source=name,matched=path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest()==expected))
        if (case/'fields.npz').exists():
            inputs.append(case/'fields.npz')
            with np.load(case/'fields.npz') as f:row['fields']={key:f[key].copy() for key in f.files}
            row['relative_errors']=errors(row['fields']);row['cells']=len(row['fields']['cell_volumes'])
            row['reported_error_replay_max']=float(max(abs(row['relative_errors'][k]-result.get('relative_errors',{}).get(k,row['relative_errors'][k])) for k in KEYS))
        rows.append(row)
    return config,rows,inputs,binding_checks


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--directory',type=Path,required=True);parser.add_argument('--baseline',type=Path);parser.add_argument('--output',type=Path);args=parser.parse_args()
    root=args.directory;output=args.output or root/'quadratic-report'
    if output.exists():parser.error('report output must be new; producer files are preserved')
    config,rows,inputs,bindings=load(root);output.mkdir(parents=True)
    metrics=[];tol=float(config.get('solver',{}).get('nonlinear_tolerance',1e-9));valid=[r for r in rows if r['fields'] is not None]
    metrics.append(Metric('producer-source-bindings-missing',int(not bindings),.5,'At least one imported source binding required'))
    metrics.append(Metric('producer-source-hash-mismatches',sum(not r['matched'] for r in bindings),.5,'Imported numerical source snapshot checks'))
    for r in rows:
        prefix=f"n-{r['n']}"
        metrics.append(Metric(prefix+'-raw-field-missing',int(r['fields'] is None),.5,'Raw numerical field availability'))
        metrics.append(Metric(prefix+'-recorded-nonlinear-failure',int(not r['numerically_converged']),.5,'Recorded nonlinear completion'))
        if config.get('initialization','zero')=='baseline':
            initial=r['initialization']
            metrics.append(Metric(prefix+'-baseline-initialization-missing',int(initial is None),.5,'Numerical warm-start raw fields and history required'))
            if initial is not None:
                metrics.append(Metric(prefix+'-baseline-initialization-failure',int(not(initial['converged'] and initial['finite_raw_fields'] and initial['initial_velocity']=='zero')),.5,'Independently converged finite baseline from zero'))
                if initial['history']:
                    for key in RESIDUALS:
                        metrics.append(Metric(prefix+'-baseline-last-'+key,float(initial['history'][-1].get(key,1.)),tol,'Baseline current residual/change before numerical initialization'))
                else:metrics.append(Metric(prefix+'-baseline-missing-history',1.,.5,'Baseline initialization history required'))
        if r['relative_errors'] is not None:
            for k,value in r['relative_errors'].items():metrics.append(Metric(prefix+'-'+k,value,.03,'Replayed pointwise volume-weighted relative L2'))
            metrics.append(Metric(prefix+'-reported-error-replay',r['reported_error_replay_max'],1e-12,'Raw error versus producer ledger'))
        if r['history']:
            for key in RESIDUALS:
                value=r['history'][-1].get(key)
                metrics.append(Metric(prefix+'-last-'+key,float(value) if value is not None else 1.,tol,'Saved current-operator residual/change; not reconstructed by reporter'))
        else:metrics.append(Metric(prefix+'-missing-history',1.,.5,'Current residual history availability'))
    complete=len(valid)==len(rows) and len(rows)>=2
    for left,right in zip(rows,rows[1:]):
        for key in KEYS:
            ratio=right['relative_errors'][key]/left['relative_errors'][key] if left['relative_errors'] is not None and right['relative_errors'] is not None else 2.
            metrics.append(Metric(f"n-{left['n']}-to-{right['n']}-{key}-ratio",ratio,1.,'Strict actual-grid refinement decrease'))
    metric_rows,all_passed=evaluate(metrics)
    monotonic=complete and all(right['relative_errors'][k]<left['relative_errors'][k] for left,right in zip(rows,rows[1:]) for k in KEYS)
    finest=rows[-1] if rows else None
    fine_ok=bool(finest and finest['numerically_converged'] and finest['relative_errors'] is not None and all(v<.03 for v in finest['relative_errors'].values()))
    all_converged=bool(rows and all(r['numerically_converged'] for r in rows))
    current_gates_passed=all(m['passed'] for m in metric_rows if '-last-' in m['name'] or m['name'].endswith('-missing-history') or '-baseline-initialization-' in m['name'])
    acceptance=dict(current_residual_and_change_gates_passed=current_gates_passed,all_metric_gates_passed=all_passed,all_grids_nonlinearly_converged=all_converged,all_error_metrics_strictly_decrease=monotonic,finest_grid_all_errors_below_three_percent=fine_ok,manufactured_spatial_verification_passed=bool(all_converged and current_gates_passed and monotonic and fine_ok and bindings and all(x['matched'] for x in bindings)),suboff_physical_benchmark_passed=False,metrics=metric_rows,source_bindings=bindings)
    (output/'acceptance.json').write_text(json.dumps(acceptance,indent=2)+'\n')
    baseline_rows=[]
    if args.baseline:
        _,baseline_rows,baseline_inputs,_=load(args.baseline);inputs+=baseline_inputs
    fig,ax=plt.subplots(figsize=(8,5))
    for key in KEYS:
        ax.loglog([1/r['n'] for r in valid],[100*r['relative_errors'][key] for r in valid],marker='o',label='Quadratic '+key.replace('_',' '))
    if baseline_rows:
        b=[r for r in baseline_rows if r['relative_errors'] is not None]
        ax.loglog([1/r['n'] for r in b],[100*r['relative_errors']['pressure_mean_removed'] for r in b],marker='s',ls='--',color='black',label='Baseline pressure')
    ax.axhline(3,color='grey',ls=':');ax.set(xlabel='Logical grid spacing 1/n',ylabel='Relative L2 error (%)');ax.legend(fontsize=8);ax.grid(alpha=.25);fig.tight_layout();fig.savefig(output/'spatial-errors.png',dpi=190);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for r in rows:
        history=r['history']
        for ax,key in zip(axes,('momentum_residual','current_reconstructed_divergence_max')):
            pairs=[(h['iteration'],h[key]) for h in history if key in h]
            if pairs:ax.semilogy([p[0] for p in pairs],[max(p[1],1e-18) for p in pairs],label=f"n={r['n']}")
            ax.axhline(tol,ls='--',color='black',lw=.8);ax.set(xlabel='Nonlinear iteration',ylabel=key.replace('_',' '));ax.grid(alpha=.25)
    if any(r['history'] for r in rows):axes[0].legend()
    fig.tight_layout();fig.savefig(output/'saved-current-residuals.png',dpi=190);plt.close(fig)
    for r in valid:
        f=r['fields'];polygons,ids=sections(f);v=f['cell_volumes'];p=f['pressure']-np.sum(v*f['pressure'])/v.sum();ref=f['reference_pressure']-np.sum(v*f['reference_pressure'])/v.sum();scale=max(abs(p).max(),abs(ref).max())
        fig,axes=plt.subplots(2,3,figsize=(13,8),layout='constrained')
        fields=(None,p[ids],ref[ids],np.linalg.norm(f['velocity'][ids],axis=1),(p-ref)[ids],np.linalg.norm(f['velocity'][ids]-f['reference_velocity'][ids],axis=1))
        titles=('Actual native mesh cuts','Computed pressure − mean','Analytic pressure − mean','Computed speed','Pressure error','Vector velocity error')
        allxy=np.concatenate(polygons);lo=allxy.min(axis=0);hi=allxy.max(axis=0)
        for j,(ax,data,title) in enumerate(zip(axes.ravel(),fields,titles)):
            c=PolyCollection(polygons,facecolors='none',edgecolors='black',linewidths=.4) if data is None else PolyCollection(polygons,array=data,cmap='coolwarm' if j in (1,2,4) else 'viridis',edgecolors='black',linewidths=.15)
            if j in (1,2):c.set_clim(-scale,scale)
            ax.add_collection(c);ax.set(xlim=(lo[0]-.02,hi[0]+.02),ylim=(lo[1]-.02,hi[1]+.02),xlabel='x',ylabel='y',title=title);ax.set_aspect('equal')
            if data is not None:fig.colorbar(c,ax=ax)
        fig.suptitle(f"n={r['n']}: actual z=0.5 intersections; native cell values, no smoothing")
        fig.savefig(output/f"n-{r['n']}-mesh-fields.png",dpi=190);plt.close(fig)
    table=''
    for r in rows:
        values=' | '.join(f"{100*r['relative_errors'][k]:.6g}" for k in KEYS) if r['relative_errors'] is not None else '— | — | — | — | —'
        table+=f"| {r['n']} | {r['cells'] or '—'} | {values} | {r['numerically_converged']} |\n"
    failures='\n'.join(f"- n={r['n']}: {r['error']}" for r in rows if r['error']) or 'No producer exceptions recorded.'
    photos='\n\n'.join(f"### n={r['n']}\n\n![Actual mesh, pressure and speed](n-{r['n']}-mesh-fields.png)" for r in valid)
    comparison=''
    if baseline_rows:
        comparison='## Independent baseline replay\n\n| n | Baseline pressure % | Quadratic pressure % |\n|---:|---:|---:|\n'
        for r in valid:
            b=next((b for b in baseline_rows if b['n']==r['n'] and b['relative_errors'] is not None),None)
            if b:comparison+=f"|{r['n']}|{100*b['relative_errors']['pressure_mean_removed']:.6g}|{100*r['relative_errors']['pressure_mean_removed']:.6g}|\n"
        comparison+='\nBoth columns are independently replayed from their own raw NPZ arrays; this reporter does not alter solutions.\n'
    report=f'''# Quadratic native FVM: fixed-parameter three-dimensional ABC verification

Manufactured spatial verification: **{acceptance['manufactured_spatial_verification_passed']}**. All individual numeric gates: **{all_passed}**. SUBOFF physical validation: **False**.

## Problem, discretization and independent data

The logical unit cube is transformed by x=X+.18Y+.12Z, y=Y+.2Z, z=Z. The archived configuration specifies grids and solver tolerances. The ABC velocity is U(sin(kz)+cos(ky), sin(kx)+cos(kz), sin(ky)+cos(kx)); pressure is rho U² sin(kx) cos(ky) sin(kz). Parameters remain U=.5, k=.7, nu=.1 and rho=1. The differential analytic acceleration is independently evaluated at actual cell centroids; it is not obtained by applying or reversing the discrete solver operator. Native hexahedra retain their actual triangular shared facets.

Boundary velocity point values and gradients are independent analytic data. Boundary mass flux and velocity area means use independent triangle quadrature. These manufactured boundary data are not physical SUBOFF no-slip conditions. The current quadratic momentum/continuity operator is solved through matrix-free GMRES; the baseline sparse LU serves as a preconditioner. These are newly solved nonlinear quadratic equations, not postprocessing of the baseline field. Initialization mode: **{config.get('initialization','zero')}**. Recorded initial velocity: **{config.get('initial_velocity','zero')}**. Zero mode starts the quadratic solve from zero. Baseline mode first solves the original numerical FVM equations from zero to convergence, archives its fields and complete history, and then initializes the quadratic solve from that numerical velocity and pressure. Analytic interior truth does not initialize either numerical solve. The arbitrary pressure gauge is supplied analytically.

## Replayed accuracy and saved convergence

Errors are recomputed from raw NPZ numerical and reference fields using actual cell volumes. Each velocity component and vector error is checked separately. Computed and reference pressures have their own volume-weighted means removed, leaving the independent reference denominator. Current momentum, reconstructed mass and flux-consistency histories are copied from the actual producer ledger and tested; the reporter does not pretend to recompute solver residuals without its current operator.

| n | Cells | Vector u % | ux % | uy % | uz % | Pressure % | Nonlinear converged |
|---:|---:|---:|---:|---:|---:|---:|---|
{table}

All grids converged: **{all_converged}**. All metrics strictly decrease: **{monotonic}**. Finest grid every error below 3%: **{fine_ok}**. Shared `verification.core.Metric/evaluate` applies explicit numeric gates in [acceptance.json](acceptance.json). Missing fields, failed nonlinear solves and producer source mismatches remain failed evidence.

{failures}

![Spatial error curves](spatial-errors.png)

![Actual saved current-operator histories](saved-current-residuals.png)

No formal order or asymptotic second-order claim is inferred from this finite refinement series.

## Robustness and standalone FVM development

The earlier [zero-start attempt](../../verification-native-abc3d-quadratic/quadratic-report/report.md) is retained, including its actual failed coarse solve. Changing to a documented numerical baseline initialization improves robustness; its extra solver cost and initialization history remain part of the evidence. Development prioritizes standalone FVM accuracy and physical SUBOFF validation before FVM/LBM coupling. MMS success does not erase initialization sensitivity or replace drag/Cp validation.

{comparison}

## Actual meshes and fields

Plots use true z=.5 intersections and piecewise native cell values. Analytic values correspond to actual cell centroids. Numerical and analytic pressure within each figure share a color scale. There is no smoothing or fabricated flow reference.

{photos}

## Reproduction and provenance

Run `PYTHONPATH=src python scripts/report_native_quadratic_fvm.py --directory RUN_DIRECTORY --baseline BASELINE_DIRECTORY` after the producer finishes. The report is written to a new `quadratic-report` directory; producer artifacts remain intact. Report sources, raw input SHA-256 and output hashes are saved. The numerical source bindings in each producer result are checked against launch snapshots.

This verifies standalone FVM manufactured accuracy. SUBOFF experimental drag, Cp, high-Re turbulence and physical boundary-layer accuracy remain separate targets and are not accepted by these results. No FVM/LBM coupling is executed.
'''
    (output/'report.md').write_text(report)
    snapshot=output/'source-snapshot';snapshot.mkdir()
    repo=Path(__file__).resolve().parents[1]
    for name in ('scripts/report_native_quadratic_fvm.py','scripts/report_native_abc_convergence3d.py','src/tensorfvm/verification/core.py'):
        target=snapshot/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(repo/name,target)
    manifest=dict(input_sha256={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},artifact_sha256={str(p.relative_to(output)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.rglob('*')) if p.is_file()})
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({k:v for k,v in acceptance.items() if k not in ('metrics','source_bindings')},indent=2));return 0 if acceptance['manufactured_spatial_verification_passed'] else 2

if __name__=='__main__':raise SystemExit(main())

"""Supplement external-flow reports with independent pressure sampling and refinement."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from tensorfvm.verification.core import write_json,artifact_manifest,sha
from tensorfvm.verification.external_flow import audit
from tensorfvm.verification.report import plt,figure


def process(output):
    summary=json.loads((output/'summary.json').read_text());rows=summary['runs'];refinement={}
    for row in rows:
        audit(output/row['directory'],row)
        if row['case']=='cylinder':
            with np.load(output/row['directory']/'fields.npz') as f:
                c=row['config'];center=f['surface_centers_m'];angle=np.mod(np.arctan2(center[:,1]-c['cylinder_y'],center[:,0]-c['cylinder_x']),2*np.pi);idx=np.argsort(angle);a=angle[idx];p=f['surface_wall_pressure_pa'][idx]
                aa=np.r_[a[-1]-2*np.pi,a,a[0]+2*np.pi];pp=np.r_[p[-1],p,p[0]];dp=np.interp(np.pi,aa,pp)-np.interp(0,aa,pp)
                np.testing.assert_allclose(dp,row['computed']['pressure_difference_pa'],rtol=0,atol=1e-12)
        linear=row['linear_solver_history']
        if not linear or any(not np.isfinite(q['true_residual']) or q['true_residual']>q['target']*1.05 for q in linear):raise ValueError('Invalid true linear residual')
        history=json.loads((output/row['directory']/'history.json').read_text());fig,ax=plt.subplots(figsize=(6,4))
        for k in ('continuity','momentum','mass_imbalance'):ax.semilogy([q[k] for q in history],label=k)
        ax.set(xlabel='SIMPLE iteration',ylabel='Actual equation residual');ax.legend();ax.grid(alpha=.3);figure(fig,output/row['directory']/'figures','residuals')
    for metric in rows[0]['reference']:
        values=[r['computed'][metric] for r in rows];errors=[next(q['value'] for q in r['metrics'] if q['name']==metric+'_relative_error') for r in rows]
        changes=[abs(b-a)/abs(b) for a,b in zip(values,values[1:])];orders=[float(np.log(a/b)/np.log(rows[i+1]['config']['nx']/rows[i]['config']['nx'])) for i,(a,b) in enumerate(zip(errors,errors[1:]))]
        refinement[metric]=dict(values=values,worst_reference_relative_errors=errors,successive_relative_changes=changes,observed_reference_error_orders=orders)
    write_json(output/'refinement-audit.json',dict(grid_convergence_qualified=False,reason='Three grids alone do not establish an asymptotic regime; physical and iterative failures retained.',metrics=refinement))
    fig,ax=plt.subplots(figsize=(7,4))
    for metric,data in refinement.items():ax.loglog([r['config']['nx'] for r in rows],100*np.array(data['worst_reference_relative_errors']),'o-',label=metric)
    ax.axhline(3,color='red',linestyle='--',label='Strict 3% gate');ax.set(xlabel='Circumferential cells',ylabel='Worst reference error (%)');ax.legend();ax.grid(alpha=.3);figure(fig,output,'refinement')
    with (output/'results.csv').open('w',newline='') as stream:
        w=csv.writer(stream);w.writerow(['case','nx','ny','cells','device','iterations','converged','elapsed_s','drag','lift','physical_passed'])
        for r in rows:w.writerow([r['case'],r['config']['nx'],r['config']['ny'],r['config']['nx']*r['config']['ny'],r['device'],r['iterations'],r['converged'],r['elapsed_s'],r['computed']['drag'],r['computed']['lift'],r['passed']])
    text='\n## 网格误差与残差补充\n\n相邻网格变化和参考误差阶见 `refinement-audit.json`；这些阶基于当前参考，并非无参考的 Richardson 外推。三网格均有实际结果才可讨论趋势，单次耗时只用于复现预算。审计重新检查圆柱前后压力的周期角度插值、所有实际线性求解残差，以及真实压力/黏性力积分。\n\n![网格误差](refinement.png)\n'
    for row in rows:text+=f"\n![{row['resolution']} residuals]({row['directory']}/figures/residuals.png)\n"
    with (output/'report.md').open('a') as f:f.write(text)
    manifest=json.loads((output/'manifest.json').read_text());root=Path(__file__).resolve().parents[1];manifest['source_sha256']['scripts/audit_external_reports.py']=sha(Path(__file__));manifest['source_sha256']['scripts/extract_naca_reference.py']=sha(root/'scripts/extract_naca_reference.py');manifest['artifacts_sha256']=artifact_manifest(output);write_json(output/'manifest.json',manifest)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('outputs',nargs='+',type=Path);a=p.parse_args()
    for output in a.outputs:process(output)

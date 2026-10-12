"""Read OpenFOAM restart histories and apply explicit steady-run gates."""
from pathlib import Path
import re
import numpy as np


def function_history(case, function, filename):
    paths=sorted((Path(case)/'postProcessing'/function).glob('*/'+filename),key=lambda p:float(p.parent.name))
    samples={}
    for path in paths:
        for line in path.read_text().splitlines():
            if not line.strip() or line.lstrip().startswith('#'):continue
            values=np.fromstring(line.replace('(',' ').replace(')',' '),sep=' ')
            if not np.isfinite(values).all():raise ValueError(f'nonfinite history: {path}')
            samples[values[0]]=values
    if not samples:raise ValueError(f'no {function}/{filename} history under {case}')
    return np.array([samples[t] for t in sorted(samples)]),paths


def residual_history(case):
    rows={}
    for path in sorted(Path(case).glob('log.foamRun*')):
        text=path.read_text()
        for match in re.finditer(r'^Time = ([\d.eE+-]+)s?\s*\n(.*?)(?=^Time = |\Z)',text,re.M|re.S):
            t=float(match.group(1));body=match.group(2);r={}
            for field,value in re.findall(r'Solving for (\w+), Initial residual = ([\d.eE+-]+)',body):
                if field not in r:r[field]=float(value) # First pressure correction, not last.
            continuity=re.search(r'continuity errors : sum local = ([\d.eE+-]+), global = ([\d.eE+-]+)',body)
            if continuity:r.update(continuity_local=float(continuity[1]),continuity_global=float(continuity[2]))
            if set(['Ux','Uy','Uz','p','k','omega','continuity_local','continuity_global'])<=r.keys():rows[t]=r
    return rows


def convergence_audit(case,window=200,residual_window=20):
    data,_=function_history(case,'forceCoeffs','forceCoeffs.dat')
    forces,_=function_history(case,'forces','forces.dat')
    history=residual_history(case);last=float(data[-1,0]);threshold=.002
    result=dict(last_iteration=last,window=window,residual_window=residual_window,
        criteria=dict(force_relative_range_max=threshold,force_half_window_drift_max=.001,
            initial_residual_max=1e-5,continuity_local_max=1e-7,continuity_global_max=1e-9),
        physical_benchmark_passed=False,numerically_converged=False)
    enough=len(data)>=window and len(forces)>=window and len(history)>=residual_window
    enough=bool(enough and np.all(np.diff(data[-window:,0])==1) and np.all(np.diff(forces[-window:,0])==1))
    result['sufficient_history']=enough
    if not enough:return result
    if forces[-1,0]!=last:raise ValueError('force coefficient and dimensional force histories end at different iterations')
    summaries={}
    for name,values in [('cd',data[-window:,2]),('pressure_drag',forces[-window:,1]),('viscous_drag',forces[-window:,4])]:
        scale=max(abs(float(values.mean())),1e-15)
        summaries[name]=dict(mean=float(values.mean()),relative_range=float(np.ptp(values)/scale),
            half_window_relative_drift=float(abs(values[window//2:].mean()-values[:window//2].mean())/scale))
    selected=[history.get(t,{}) for t in range(int(last)-residual_window+1,int(last)+1)]
    names=['Ux','Uy','Uz','p','k','omega','continuity_local','continuity_global']
    residuals={name:max(abs(r.get(name,float('inf'))) for r in selected) for name in names}
    result.update(force_windows=summaries,last_window_residual_max={k:(v if np.isfinite(v) else None) for k,v in residuals.items()})
    result['force_stable']=all(v['relative_range']<=threshold and v['half_window_relative_drift']<=.001 for v in summaries.values())
    result['residuals_passed']=all(residuals[k]<=1e-5 for k in names[:6]) and residuals['continuity_local']<=1e-7 and residuals['continuity_global']<=1e-9
    result['numerically_converged']=bool(result['force_stable'] and result['residuals_passed'])
    return result

"""Import original Lee--Moser channel means and sampling standard deviations."""
import hashlib
import io
import re
from pathlib import Path
import numpy as np


def read_profile(path):
    raw=Path(path).read_bytes();text=raw.decode('ascii')
    parameters={}
    for name in ('nu','delta','U_mean','u_tau','Re_tau'):
        matches=re.findall(r'(?m)^%[^\n]*\b'+name+r'\s*=\s*([0-9.eE+\-]+)\s*$',text)
        if len(matches)!=1:raise ValueError('missing or ambiguous DNS parameter '+name)
        parameters[name]=float(matches[0])
    if not np.isfinite(list(parameters.values())).all() or min(parameters.values())<=0:
        raise ValueError('positive finite DNS parameters required')
    data=np.loadtxt(io.StringIO(text),comments='%')
    if data.ndim!=2 or data.shape[1]!=6 or len(data)<4 or not np.isfinite(data).all():
        raise ValueError('expected finite six-column channel profile')
    if np.any(np.diff(data[:,0])<=0) or data[0,0]<0 or data[-1,0]>1:
        raise ValueError('DNS half-channel coordinates must be strictly ordered in [0,1]')
    if not np.allclose(data[:,1],data[:,0]*parameters['Re_tau'],rtol=2e-5,atol=1e-10):
        raise ValueError('inconsistent wall coordinate normalization')
    if not np.isclose(parameters['u_tau']*parameters['delta']/parameters['nu'],parameters['Re_tau'],rtol=2e-5):
        raise ValueError('inconsistent DNS Reynolds normalization')
    return data,parameters,hashlib.sha256(raw).hexdigest()


def import_profiles(mean,stdev,output,*,case):
    import datetime
    from .native_run import write_json
    if case not in ('0180','0550','1000','2000','5200'):raise ValueError('unknown Lee--Moser case')
    data,p,digest=read_profile(mean);unc,q,udigest=read_profile(stdev)
    if p!=q or data.shape!=unc.shape or not np.array_equal(data[:,:2],unc[:,:2]):
        raise ValueError('mean and uncertainty must describe the identical DNS run/grid')
    for path,kind in [(mean,'mean_prof'),(stdev,'mean_stdev')]:
        if 'LM_Channel_'+case+'_'+kind+'.dat' not in Path(path).read_text():
            raise ValueError('DNS filename header does not match requested case')
    if np.any(unc[:,2:]<0):raise ValueError('negative sampling standard deviation')
    output=Path(output)
    if output.exists() and any(output.iterdir()):raise ValueError('reference output must be empty')
    output.mkdir(parents=True,exist_ok=True)
    files={}
    for name,kind,values,raw_hash in [('mean','mean_prof',data,digest),('mean_standard_deviation','mean_stdev',unc,udigest)]:
        path=output/(name+'.csv')
        np.savetxt(path,values,delimiter=',',header='y_over_delta,y_plus,U,dU_dy,W,P',comments='',fmt='%.17g')
        files[name]=dict(upstream_url=f'https://turbulence.oden.utexas.edu/channel2015/data/LM_Channel_{case}_{kind}.dat',
            raw_sha256=raw_hash,csv=path.name,csv_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),rows=len(values))
    meta=dict(authors=['Myoungkyu Lee','Robert D. Moser'],title='Direct numerical simulation of turbulent channel flow up to Re_tau = 5200',
        journal='Journal of Fluid Mechanics 774 (2015), 395-415',doi='10.1017/jfm.2015.268',
        source_type='original authors DNS statistical data; not experiment',
        retrieved_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=files,
        Re_tau_reported=p['Re_tau'],nu=p['nu'],half_width=p['delta'],U_bulk=p['U_mean'],u_tau=p['u_tau'],
        U_bulk_plus=p['U_mean']/p['u_tau'],Cf=2*(p['u_tau']/p['U_mean'])**2,
        Cf_definition='2*tau_wall/(rho*U_bulk^2)',normalization='wall units except y_over_delta',
        uncertainty='separate estimated sampling standard deviations; bulk/friction uncertainty unspecified',
        centerline_missing=bool(data[-1,0]<1),interpolation='PCHIP at all case cell centers; no extrapolation',
        license='no explicit license in source header; numerical values transcribed with author attribution; raw files cached locally',
        metric_profile='sqrt(sum(V*(U_calc-U_DNS)^2)/sum(V*U_DNS^2)) at all cell centers',
        metric_Cf='abs(Cf_calc-Cf_DNS)/abs(Cf_DNS), bulk from actual outlet flux',threshold_strict=.03,source_values_changed=False)
    write_json(output/'provenance.json',meta)
    return meta

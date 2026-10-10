"""Independent NumPy reconstruction for ABC and advected-shear raw evidence."""
import json
import math
from pathlib import Path
import numpy as np
from .audit import close, relative, d, g, mac_acceleration, projected
from .core import sha


def reference(kind,n,nz,t):
    h=2*np.pi/n;hz=2*np.pi/nz
    z,y,x=np.meshgrid((np.arange(nz)+.5)*hz,(np.arange(n)+.5)*h,(np.arange(n)+.5)*h,indexing='ij')
    result=[]
    for a in range(3):
        xyz=[x.copy(),y.copy(),z.copy()];xyz[a]+=[h,h,hz][a]/2;xx,yy,zz=xyz
        v=[np.sin(zz)+np.cos(yy),np.sin(xx)+np.cos(zz),np.sin(yy)+np.cos(xx)] if kind=='abc' else [np.sin(yy-.7*t),np.full_like(xx,.7),np.zeros_like(xx)]
        result.append(v[a]*np.exp(-.1*t) if kind=='abc' or a==0 else v[a])
    if kind=='abc':
        p=-.5*((np.sin(z)+np.cos(y))**2+(np.sin(x)+np.cos(z))**2+(np.sin(y)+np.cos(x))**2)*np.exp(-.2*t);p-=p.mean()
    else:p=np.zeros_like(x)
    return np.array(result),p


def physical(kind,u,eu,p,ep):
    computed=dict(velocity_l2=relative(u,eu),velocity_linf=relative(u,eu,True))
    if kind=='abc':computed.update(pressure_l2=relative(p,ep),pressure_linf=relative(p,ep,True))
    else:computed.update(shear_velocity_l2=relative(u[0],eu[0]),pressure_dynamic_scaled_linf=float(np.max(abs(p)))/.745)
    return computed


def audit(directory):
    directory=Path(directory);s=json.loads((directory/'summary.json').read_text());root=Path(__file__).resolve().parents[3]
    for p,h in s['provenance']['source_sha256'].items():
        if sha(root/p)!=h:raise ValueError('source mismatch: '+p)
    cases=[]
    for r in s['runs']:
        path=directory/r['directory'];f=np.load(path/'fields.npz',allow_pickle=False);hist=json.loads((path/'history.json').read_text());c=r['config'];n=c['nx'];dt=c['time_step_s'];kind=r['case']
        if kind.endswith('-lbm'):
            C=np.array([[0,0],[1,0],[0,1],[-1,0],[0,-1],[1,1],[-1,1],[-1,-1],[1,-1]])
            W=np.array([4/9,*([1/9]*4),*([1/36]*4)],dtype=np.float32).astype(float);scale=c['physical_speed_scale_m_s'];states=f['accepted_populations']
            if len(states)!=len(hist)+1:raise ValueError('missing step evidence')
            yy,xx=np.meshgrid(f['y_m'],f['x_m'],indexing='ij');ui=np.stack((np.sin(yy),np.full_like(yy,.7)))/scale;cu=np.einsum('qa,ayx->qyx',C,ui)
            close(states[0],W[:,None,None]*(1+3*cu+4.5*cu**2-1.5*np.sum(ui**2,axis=0)),1e-12)
            for i,(old,new) in enumerate(zip(states[:-1],states[1:])):
                rho=old.sum(0);u=np.einsum('qyx,qa->ayx',old,C)/rho;cu=np.einsum('qa,ayx->qyx',C,u)
                eq=W[:,None,None]*rho*(1+3*cu+4.5*cu**2-1.5*np.sum(u*u,axis=0));post=old-(old-eq)/c['tau']
                close(new,np.stack([np.roll(post[q],tuple(C[q,::-1]),axis=(0,1)) for q in range(9)]),1e-12)
                close(hist[i]['relative_mass_drift'],abs(new.sum()-states[0].sum())/states[0].sum())
            rho=states[-1].sum(0);u=np.einsum('qyx,qa->ayx',states[-1],C)/rho*scale;u=np.concatenate((u,np.zeros((1,n,n))),axis=0);p=(rho-rho.mean())*scale**2/3
            eu=np.stack((np.sin(yy-.7*r['time_s'])*np.exp(-.1*r['time_s']),np.full_like(yy,.7),np.zeros_like(yy)));ep=np.zeros_like(p)
            close(f['u_m_s'],u[0]);close(f['v_m_s'],u[1]);close(f['p_pa'],p);close(f['reference_velocity_faces_m_s'],eu)
            values=physical('advected-shear',u,eu,p,ep)
            repo=Path(r['provenance']['repository'])
            for name,digest in r['provenance']['source_sha256'].items():
                if sha(repo/name)!=digest:raise ValueError('LBM source changed')
        else:
            nz=c['nz'];h=(2*np.pi/n,2*np.pi/n,2*np.pi/nz);vol=np.prod(h);states=f['accepted_faces_m_s']
            if len(states)!=len(hist)+1:raise ValueError('missing step evidence')
            initial,_=reference(kind,n,nz,0);close(states[0],initial)
            eu,_=reference(kind,n,nz,r['time_s']);_,ep=reference(kind,n,nz,r['pressure_time_s'])
            close(eu,f['reference_velocity_faces_m_s']);close(ep,f['reference_pressure_pa'])
            values=physical(kind,states[-1],eu,f['pressure_pa_3d'],ep)
            diagnostic={k:[] for k in ('scaled_divergence','nonlinear_residual','energy_defect','momentum_change','convective_power','energy_increase','positive_viscous_power')}
            for i,(old,new) in enumerate(zip(states[:-1],states[1:])):
                mid=.5*(old+new);conv,force=mac_acceleration(mid,h,.1,1.);tent=old+dt*(-conv+force);check,p=projected(tent,h,1.,dt)
                residual=float(np.max(abs(check-new)));cp=-vol*np.sum(mid*conv);vp=vol*np.sum(mid*force);pp=-vol*np.sum(mid*g(p,h));k0=.5*vol*np.sum(old**2);k1=.5*vol*np.sum(new**2);defect=k1-k0-dt*(cp+vp+pp)
                vals=dict(scaled_divergence=float(np.max(abs(d(new,h)))),nonlinear_residual=residual,energy_defect=abs(defect),momentum_change=float(np.max(abs(vol*np.sum(new-old,axis=(1,2,3))))),convective_power=abs(cp),energy_increase=max(0.,k1-k0),positive_viscous_power=max(0.,vp))
                for key,value in vals.items():diagnostic[key].append(value)
                for key,value in [('nonlinear_residual_m_s',residual),('energy_balance_defect_J',defect),('kinetic_after_J',k1),('convective_power_W',cp),('viscous_power_W',vp),('pressure_power_W',pp),('max_divergence_s_inv',vals['scaled_divergence'])]:close(hist[i][key],value,2e-10)
                if residual>1.01e-12:raise ValueError('nonlinear residual gate failed')
            close(p,f['pressure_pa_3d']);values.update({k:max(v) for k,v in diagnostic.items()});values['pressure_failure']=0. if all(q['projection']['pressure_converged'] for q in hist) else 1.
        if len(states)!=len(hist)+1 or len(hist)!=r['iterations']:raise ValueError('missing step evidence')
        for i,record in enumerate(hist):close(record['time_s'],(i+1)*dt)
        if set(values)!={q['name'] for q in r['metrics']}:raise ValueError('incomplete metric coverage')
        for q in r['metrics']:
            close(values[q['name']],q['value'],2e-10)
            if q['name'] in ('velocity_l2','velocity_linf','pressure_l2','pressure_linf','shear_velocity_l2','pressure_dynamic_scaled_linf') and q['limit']!=.03:raise ValueError('physical gate altered')
            limits=dict(scaled_divergence=1e-10,nonlinear_residual=1.01e-12,energy_defect=1e-10,momentum_change=1e-10,convective_power=1e-10,energy_increase=1e-12,positive_viscous_power=1e-12,pressure_failure=.5)
            if q['name'] in limits and q['limit']!=limits[q['name']]:raise ValueError('numerical gate altered')
            if q['passed']!=(math.isfinite(values[q['name']]) and values[q['name']]<q['limit']):raise ValueError('false metric badge')
        if r['passed']!=all(q['passed'] for q in r['metrics']):raise ValueError('false run badge')
        cases.append(dict(directory=r['directory'],steps=len(hist),raw_reconstruction_passed=True,benchmark_passed=r['passed']))
    if len([r for r in s['runs'] if r['case']=='abc' and r['role']=='spatial'])!=3 or len([r for r in s['runs'] if r['case']=='advected-shear' and r['role']=='spatial'])!=3:raise ValueError('missing formal refinement family')
    orders={}
    for kind,metric in [('abc','velocity_l2'),('abc','pressure_l2'),('advected-shear','shear_velocity_l2')]:
        rows=[r for r in s['runs'] if r['case']==kind and r['role']=='spatial'];v=[next(q['value'] for q in r['metrics'] if q['name']==metric) for r in rows]
        order=[math.log(a/b)/math.log(rows[i]['spacing_m']/rows[i+1]['spacing_m']) for i,(a,b) in enumerate(zip(v,v[1:]))];orders[kind+'-'+metric]=order;close(order,s['observed_orders'][kind+'-'+metric])
    passed=all(r['passed'] for r in s['runs'] if r['role']=='spatial') and all(not r['passed'] for r in s['runs'] if r['role']=='negative-control') and all(1.8<v<2.2 for row in orders.values() for v in row)
    if passed!=s['passed']:raise ValueError('false suite badge')
    comparison=[r for r in s['runs'] if r['role']=='comparison']
    if s['comparison_passed']!=(all(r['passed'] for r in comparison) if comparison else None):raise ValueError('false comparison badge')
    return dict(raw_reconstruction_passed=True,benchmark_passed=passed,cases=cases,summary_sha256=sha(directory/'summary.json'),auditor_sha256=sha(Path(__file__)),scope='independent NumPy analytic references and every accepted MAC/BGK step')

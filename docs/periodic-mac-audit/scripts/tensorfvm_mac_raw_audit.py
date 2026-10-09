"""Independent NumPy raw periodic MAC audit, including convergence and failures."""
import gzip,json,hashlib,math
from pathlib import Path
import numpy as np
from tensorfvm_mac_independent_review import divergence,gradient,energy,momentum,independent_projection,dual_convection,full_stress
ROOT=Path('/home/jsyc/tensor-suite-development/TensorFVM');OUT=ROOT/'docs/periodic-mac'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(gzip.decompress(p.read_bytes()))
def close(a,b,tol=1e-10):
    error=float(np.max(abs(np.asarray(a)-np.asarray(b))));assert error<tol,error;return error
s=json.loads((OUT/'study.json').read_text());assert s['physical_accuracy_qualified'] is False and s['distributed_supported'] is False
for name,digest in s['source_sha256'].items():assert sha(ROOT/name)==digest
for name,digest in s['artifacts_sha256'].items():assert sha(OUT/name)==digest
rows=[];grids=[];times=[];mms=[]
for name in s['artifacts_sha256']:
    r=load(OUT/name);c=r['config'];sp=tuple(L/n for L,n in zip(c['lengths_m'],(c['nx'],c['ny'],c['nz'])));volume=np.prod(sp);rho=c['density_kg_m3'];dt=c['time_step_s']
    if name.startswith('mms'):
        u=np.asarray(r['velocity']);mu=np.asarray(r['dynamic_viscosity']);force,diss,normals,shears=full_stress(u,mu,sp);error=close(force,r['computed_force_density']);close(normals,r['normal_stress']);tau=np.asarray(r['shear_stress'])
        for (a,b),t in shears.items():close(t,tau[a,b]);close(t,tau[b,a])
        z,y,x=np.indices(u.shape[1:]);xyz=[(x+.5)*sp[0],(y+.5)*sp[1],(z+.5)*sp[2]];exact=[]
        for a in range(3):
            coord=[q.copy() for q in xyz];coord[a]+=sp[a]/2;xx,yy,zz=coord;mf=1+.2*np.sin(xx)+.1*np.cos(zz)
            if a==0:value=-3*mf*np.sin(xx)*np.cos(yy)*np.cos(zz)+.4*np.cos(xx)**2*np.cos(yy)*np.cos(zz)+.1*np.sin(xx)*np.cos(yy)*np.sin(zz)**2
            elif a==1:value=3*mf*np.cos(xx)*np.sin(yy)*np.cos(zz)-.1*np.cos(xx)*np.sin(yy)*np.sin(zz)**2
            else:value=-.2*np.cos(xx)*np.sin(xx)*np.cos(yy)*np.sin(zz)
            exact.append(value)
        close(exact,r['analytic_force_density']);relative=float(np.linalg.norm(np.asarray(force)-exact)/np.linalg.norm(exact));close(relative,r['relative_l2_error']);mms.append(dict(n=c['nx'],relative_l2_error=relative));continue
    assert r['physical_accuracy_qualified'] is False and r['distributed_supported'] is False;initial=np.asarray(r['initial_velocity']);final=np.asarray(r['final_velocity']);q=r['last_step'];old=np.asarray(q['old_velocity']);new=np.asarray(q['candidate_velocity']);mu=np.asarray(q['dynamic_viscosity']);mid=.5*(old+new);close(mid,q['midpoint_velocity']);conv,flux=dual_convection(mid,sp);op,diss,normals,shears=full_stress(mid,mu,sp);close(rho*np.array(flux),q['convective_fluxes']);close(normals,q['normal_stress']);tau=np.asarray(q['shear_stress'])
    for (a,b),t in shears.items():close(t,tau[a,b]);close(t,tau[b,a])
    tentative=old+dt*(-rho*np.array(conv)+np.array(op))/rho;close(tentative,q['tentative_velocity']);check,p=independent_projection(tentative,sp);close(check,q['check_velocity']);close(p*rho/dt,q['pressure']);res=float(np.max(abs(np.array(check)-new)));close(res,q['record']['nonlinear_residual_m_s']);convpower=-rho*volume*np.sum(mid*np.array(conv));viscpower=-volume*diss;defect=energy(new,rho,volume)-energy(old,rho,volume)-dt*(convpower+viscpower);close(defect,q['record']['energy_balance_defect_J']);close(convpower,q['record']['convective_power_W']);close(viscpower,q['record']['viscous_power_W']);pressurepower=-volume*np.sum(mid*np.array(gradient(np.asarray(q['pressure']),sp)));assert abs(pressurepower)<1e-9
    accepted=q['accepted'];record=q['record'];history=r['history']
    if accepted:
        assert res<=c['nonlinear_tolerance']*1.01 and record['nonlinear_converged'] is True;close(final,new);assert np.max(abs(divergence(final,sp)))<1e-10;close(momentum(final,rho,volume),momentum(initial,rho,volume));previous=energy(initial,rho,volume)
        for i,h in enumerate(history):
            assert h['nonlinear_converged'] is True and h['nonlinear_residual_m_s']<=c['nonlinear_tolerance'];close(h['kinetic_before_J'],previous);close(h['energy_balance_defect_J'],h['kinetic_after_J']-h['kinetic_before_J']-dt*(h['convective_power_W']+h['viscous_power_W']));assert h['kinetic_after_J']<=h['kinetic_before_J']+1e-10;close(h['time_s'],(i+1)*dt);previous=h['kinetic_after_J']
        close(previous,energy(final,rho,volume))
        states=[np.asarray(v) for v in r['accepted_face_velocities']];assert len(states)==len(history);previous_field=initial
        for i,(state,h) in enumerate(zip(states,history)):
            midpoint=.5*(previous_field+state);cc,_=dual_convection(midpoint,sp);ss,dd,*_=full_stress(midpoint,mu,sp);cp=-rho*volume*np.sum(midpoint*np.array(cc));vp=-volume*dd;before=energy(previous_field,rho,volume);after=energy(state,rho,volume)
            close(before,h['kinetic_before_J']);close(after,h['kinetic_after_J']);close(cp,h['convective_power_W']);close(vp,h['viscous_power_W']);close(after-before-dt*(cp+vp),h['energy_balance_defect_J']);close(momentum(state,rho,volume)-momentum(previous_field,rho,volume),h['momentum_change_Ns']);assert np.max(abs(divergence(state,sp)))<1e-10
            tentative_i=previous_field+dt*(-rho*np.array(cc)+np.array(ss))/rho;projected_i,_=independent_projection(tentative_i,sp);actual_res=float(np.max(abs(np.asarray(projected_i)-state)));close(actual_res,h['nonlinear_residual_m_s'],1e-12);assert actual_res<=c['nonlinear_tolerance']*1.01;previous_field=state
        close(states[-1],final)
    else:
        assert record['nonlinear_converged'] is False and (res>c['nonlinear_tolerance'] or q['iterations'][-1]['fixed_point_residual_m_s']>c['nonlinear_tolerance']);close(final,initial);assert not history
    rows.append(dict(case=name,accepted=accepted,nonlinear_residual=res,energy_defect_J=float(defect),pressure_power_W=float(pressurepower),steps=len(history),last_picard_increment_m_s=q['iterations'][-1]['fixed_point_residual_m_s']))
    if name.startswith('tg-grid'):
        t=len(history)*dt;exact=initial*math.exp(-2*c['viscosity_pa_s']/rho*t);close(exact,r['analytic_final_velocity']);err=float(np.linalg.norm(final-exact)/np.linalg.norm(exact));close(err,r['relative_l2_error']);grids.append(dict(n=c['nx'],dt=dt,time=t,error=err))
    if name.startswith('tg-time'):
        t=len(history)*dt;eigen=sum(4*math.sin(h/2)**2/h**2 for h in sp[:2]);exact=initial*math.exp(-c['viscosity_pa_s']/rho*eigen*t);close(exact,r['semi_discrete_reference']);err=float(np.linalg.norm(final-exact)/np.linalg.norm(exact));close(err,r['relative_l2_error']);times.append(dict(n=c['nx'],dt=dt,time=t,error=err))
grids.sort(key=lambda a:a['n']);times.sort(key=lambda a:-a['dt']);mms.sort(key=lambda a:a['n'])
for sequence in (grids,times,mms):
    for a,b in zip(sequence,sequence[1:]):assert b['error' if 'error' in b else 'relative_l2_error']<a['error' if 'error' in a else 'relative_l2_error']/3
assert len(set((g['dt'],g['time']) for g in grids))==1;assert len(set((g['n'],g['time']) for g in times))==1
for native,actual,key in [('spatial_refinement',grids,'error'),('temporal_refinement',times,'error'),('variable_mu_mms',mms,'relative_l2_error')]:
    for a,b in zip(s[native],actual):
        close(a['relative_l2_error'],b[key]);assert a['error_3_percent_passed']==(b[key]<.03)
for name,actual,key in [('spatial_order',grids,'error'),('temporal_order',times,'error'),('mms_order',mms,'relative_l2_error')]:
    orders=[math.log(a[key]/b[key],2) for a,b in zip(actual,actual[1:])];close(orders,s[name]);assert all(1.8<v<2.2 for v in orders)
assert s['refinement_order_passed'] is True
assert s['mms_all_grids_3_percent_passed']==all(x['relative_l2_error']<.03 for x in mms)
assert s['mms_finest_grid_3_percent_passed']==(mms[-1]['relative_l2_error']<.03)
report=dict(raw_audit_passed=True,all_accepted_steps_field_reconstructed=True,accepted_steps_verified=sum(row['steps'] for row in rows),cases=rows,spatial_refinement=grids,temporal_refinement=times,variable_mu_MMS=mms,source_sha256=s['source_sha256'],artifacts_sha256=s['artifacts_sha256'],study_sha256=sha(OUT/'study.json'),auditor_sha256=sha(Path(__file__)),helper_sha256=sha(Path('/tmp/tensorfvm_mac_independent_review.py')),physical_accuracy_qualified=False,distributed_supported=False);Path('/tmp/tensorfvm-mac-raw-audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if not k.endswith('sha256')},indent=2))

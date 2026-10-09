"""Independent macroscopic budget diagnostics, not full LBM entropy proof."""
import numpy as np
from .fourth_round_audit import C,close,require,unqualified,sha,load,feedback_case

def energy(state):
    c=state['fluid']['config'];s=state['fluid']['solver'];f=np.asarray(s['f']);rho=f.sum(0);p=np.einsum('iyx,ia->ayx',f,C);y,x=np.indices(rho.shape);nx=s['config']['nx'];ny=s['config']['ny'];cx,cy=s['center'];dx=(x-cx+nx/2)%nx-nx/2;dy=(y-cy+ny/2)%ny-ny/2;solid=dx*dx+dy*dy<=s['config']['radius']**2
    require(np.isfinite(f).all() and (f>=0).all(),'invalid populations');require((rho[~solid]>0).all(),'invalid fluid density')
    kinetic=.5*np.sum(np.sum(p*p,axis=0)/np.maximum(rho,1e-30))*c['density_kg_m3']*c['thickness_m']*c['dx_m']**4/c['dt_s']**2
    v=np.asarray(state['velocity_m_s']);kinetic+=.5*state['body_mass_kg']*(v@v)
    fluidrho=rho[~solid];free=np.sum(fluidrho*np.log(fluidrho)-fluidrho+1)*c['density_kg_m3']*c['thickness_m']*c['dx_m']**4/(3*c['dt_s']**2)
    return dict(kinetic_J=float(kinetic),density_free_energy_J=float(free),total_J=float(kinetic+free))
def case(r):
    unqualified(r);require(r['completed_requested_duration'] is False,'unsupported completion');initial=energy(r['initial']['solver']);accepted=r['accepted'];final=energy(accepted['solver']);candidate=energy(r['rejected_candidate']);budget=accepted['initial_energy_J']*(1+accepted['relative_tolerance'])+accepted['absolute_tolerance_J'];close(initial['total_J'],accepted['initial_energy_J']);require(final['total_J']<=budget,'accepted state exceeds budget');require(candidate['total_J']>budget,'rejection without excess')
    for key in candidate:close(candidate[key],r['rejection'][key]);close(candidate[key],r['independently_replayed_energy'][key])
    close(budget,r['rejection']['allowed_energy_J']);require(r['rejection']['accepted'] is False,'false acceptance')
    rows=accepted['accepted'];hist=accepted['solver']['history'];require(len(rows)==len(hist),'accepted clock count')
    for i,row in enumerate(rows):
        unqualified(row);require(row['accepted'] is True and row['total_J']<=budget,'invalid accepted history');close(row['total_J'],row['kinetic_J']+row['density_free_energy_J']);close(row['allowed_energy_J'],budget);close(row['candidate_time_s'],hist[i]['exchange']['time_end_s']);close(row['kinetic_J'],hist[i]['fluid_kinetic_J']+hist[i]['body_kinetic_J'])
    candidatehist=r['rejected_candidate']['history'];require(candidatehist[:-1]==hist,'rollback prefix mismatch');last=candidatehist[-1]['exchange'];require(last['covered']+last['exposed']>0,'missing crossing');require(all(h['exchange']['covered']+h['exchange']['exposed']==0 for h in hist),'not first crossing');require(r['rollback_exact'] is True and r['restart_exact'] is True,'reported transaction failure')
    wrapper={'physical_accuracy_qualified':False,'flexible_suboff_qualified':False,'energy_accuracy_qualified':False}
    for state in (accepted['solver'],r['rejected_candidate']):feedback_case(dict(wrapper,state=state))
    return dict(initial=initial,accepted=final,rejected=candidate,budget_J=budget,candidate_density_fraction=candidate['density_free_energy_J']/candidate['total_J'],first_conversion_time_s=last['time_end_s'],accepted_time_s=hist[-1]['exchange']['time_end_s'],raw_state_and_budget_checks_passed=True,restart_and_rollback='producer exact replay attestations; accepted/candidate raw histories independently cross-checked',physical_accuracy_qualified=False)
def audit(roots):
    out=roots['TensorFEM']/'docs/assets/passive-rigid-feedback';study=load(out/'study.json');unqualified(study);inputs={str(out/'study.json'):sha(out/'study.json')};cases={}
    for name,digest in study['source_sha256'].items():
        project,rel=name.split('/',1);path=roots[project]/rel;require(sha(path)==digest,'source hash mismatch');inputs[name]=digest
    for name,record in study['cases'].items():
        path=out/(name+'.json');require(sha(path)==record['artifact_sha256'],'raw hash mismatch');inputs[str(path)]=sha(path);cases[name]=case(load(path))
    return dict(schema='tensorfvm.passive-feedback-independent-audit/1',cases=cases,inputs_sha256=inputs,auditor_sha256=sha(__file__),physical_accuracy_qualified=False,scope='diagnostic macroscopic budget; moving-mask pressure defect remains; no full LBM entropy proof')

def audit_balanced(lbm):
    from .fourth_round_audit import local_case
    out=lbm/'docs/assets/balanced-local-moving-boundary';study=load(out/'study.json');unqualified(study);cases={};inputs={str(out/'study.json'):sha(out/'study.json')}
    for item in study['cases']:
        path=out/item['artifact'];require(sha(path)==item['sha256'],'balanced raw hash mismatch');inputs[str(path)]=sha(path);r=load(path)
        for module,field in [('balanced_local_moving_boundary_2d.py','source_sha256'),('moving_boundary_2d.py','base_source_sha256')]:
            source=lbm/'src/tensorlbm'/module;require(sha(source)==r[field],'balanced source mismatch');inputs[str(source)]=sha(source)
        result=local_case(r);conversion=np.sum([h['conversion_impulse_on_solid'] for h in r['final']['history']],axis=0);require(np.max(abs(conversion))<1e-8,'conversion momentum changed');result['conversion_impulse']=conversion.tolist();cases[item['case']]=result
    change=abs(.5*cases['step-half']['impulse'][0]/cases['cross-grid']['impulse'][0]-1);close(change,study['refinement']['relative_impulse_change']);require(study['refinement']['passed']==(change<.03),'balanced false convergence')
    return dict(cases=cases,refinement_percent=change*100,refinement_passed=change<.03,physical_accuracy_qualified=False,co_moving_conversion_not_qualified=True,inputs_sha256=inputs)

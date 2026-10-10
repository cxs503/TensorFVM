"""Reconstruct saved CPU/CUDA fields, timing ratios and every periodic step."""
import json, math, statistics
from pathlib import Path
import numpy as np
from tensorfvm.verification.audit import close, d, mac_acceleration, projected
from tensorfvm.verification.engineering_audit import annular_metrics
from tensorfvm.verification.core import sha, artifact_manifest, write_json

root=Path(__file__).resolve().parents[1]
out=root/'docs/verification-gpu'
s=json.loads((out/'summary.json').read_text())
for name,digest in s['provenance']['source_sha256'].items():
    assert sha(root/name)==digest,name
cases=[];steps=0;failed=[]
for row in s['cases']:
    path=out/row['name'];raw={};fields={}
    for dev in ['cpu','cuda']:
        raw[dev]=json.loads((path/(dev+'-result.json')).read_text());fields[dev]=np.load(path/(dev+'-fields.npz'),allow_pickle=False)
        assert raw[dev]['actual_solver_device'].startswith(dev)
        times=row[dev+'_samples_s'];assert len(times)>=3 and all(math.isfinite(t) and t>0 for t in times)
        close(statistics.median(times),row[dev+'_median_s'])
        f=fields[dev];c=raw[dev]['config'];history=raw[dev]['history']
        if row['name'].startswith('annular'):
            values=annular_metrics(f,c)
            for k,v in values.items():
                close(v,raw[dev]['metrics'][k],2e-10)
                assert v<(2e-10 if k=='true_linear_residual' else 1e-9 if k=='momentum_force_balance' else .03)
            assert f['axial_velocity_m_s'].size==c['ntheta']*c['nr']
        if 'accepted_faces_m_s' in f:
            states=f['accepted_faces_m_s'];h=(2*np.pi/c['nx'],2*np.pi/c['ny'],2*np.pi/c['nz']);dt=c['time_step_s']
            assert len(states)==len(history)+1
            for old,new in zip(states[:-1],states[1:]):
                mid=(old+new)/2;conv,force=mac_acceleration(mid,h,c['viscosity_pa_s'],c['density_kg_m3']);check,p=projected(old+dt*(-conv+force/c['density_kg_m3']),h,c['density_kg_m3'],dt)
                assert np.max(abs(check-new))<1.01e-12
                assert np.max(abs(d(new,h)))<1e-10
                steps+=1
            close(states[-1],f['velocity_m_s']);close(p,f['pressure_pa'])
            for field,ref in [('velocity_m_s','reference_velocity_m_s'),('pressure_pa','reference_pressure_pa')]:
                a,b=f[field],f[ref]
                if np.linalg.norm(b)>1e-12:assert np.linalg.norm(a-b)/np.linalg.norm(b)<.03
                else:assert np.max(abs(a))/.745<.03
        if 'accepted_populations' in f:
            repo=Path(raw[dev]['provenance']['repository'])
            for name,digest in raw[dev]['provenance']['source_sha256'].items():assert sha(repo/name)==digest
            states=f['accepted_populations'];C=np.array([[0,0],[1,0],[0,1],[-1,0],[0,-1],[1,1],[-1,1],[-1,-1],[1,-1]]);W=np.array([4/9,*([1/9]*4),*([1/36]*4)],dtype=np.float32).astype(float)
            assert len(states)==len(history)+1
            for old,new in zip(states[:-1],states[1:]):
                rho=old.sum(0);u=np.einsum('qyx,qa->ayx',old,C)/rho;cu=np.einsum('qa,ayx->qyx',C,u);eq=W[:,None,None]*rho*(1+3*cu+4.5*cu**2-1.5*np.sum(u*u,0));post=old-(old-eq)/c['tau'];close(new,np.stack([np.roll(post[q],tuple(C[q,::-1]),axis=(0,1)) for q in range(9)]),1e-12);steps+=1
            rho=states[-1].sum(0);uv=np.einsum('qyx,qa->ayx',states[-1],C)/rho*c['physical_speed_scale_m_s'];close(uv,f['velocity_m_s'][:2]);close((rho-rho.mean())*c['physical_speed_scale_m_s']**2/3,f['pressure_pa'])
        if row['name']=='wall-law-batch':
            close(f['friction_velocity_m_s'],.05,1e-10);close(f['traction_pa'][:,0],-2.5,1e-8);close(f['traction_pa'][:,1],0.)
        if row['name'].startswith('SA-'):
            assert len(history)==5 and np.isfinite(f['nu_tilde_m2_s']).all() and np.min(f['nu_tilde_m2_s'])>=0 and raw[dev]['physical_passed'] is None
    close(row['cpu_median_s']/row['cuda_median_s'],row['cpu_over_gpu_median'])
    for key in raw['cpu']['primary_fields']:
        a,b=fields['cpu'][key],fields['cuda'][key];scale=row['field_comparison_scales'].get(key);den=np.sqrt(a.size)*scale if scale is not None else max(np.linalg.norm(a.ravel()),1e-14);value=np.linalg.norm((a-b).ravel())/den;close(value,row['relative_field_differences'][key])
    equivalent=all(v<=1e-6 for v in row['relative_field_differences'].values())
    assert row['equivalent']==equivalent
    if not equivalent:failed.append(row['name'])
    cases.append(row['name'])
assert len(cases)==10
assert s['correctness_passed']==(not failed)
write_json(out/'audit.json',dict(raw_evidence_verified=True,all_cases_correctness_passed=s['correctness_passed'],failed_equivalence_cases=failed,cases=cases,independently_reconstructed_periodic_steps=steps,summary_sha256=sha(out/'summary.json'),auditor_sha256=sha(Path(__file__)),scope='timing medians, CPU/CUDA primary fields, full annular flux/geometry, every MAC/BGK accepted step; channel equivalence only; SA five iterations only'))
write_json(out/'manifest.json',dict(source_sha256=s['provenance']['source_sha256'],artifacts_sha256=artifact_manifest(out)))
print('GPU raw evidence verified:',len(cases),'cases,',steps,'periodic steps; equivalence failures:',failed)

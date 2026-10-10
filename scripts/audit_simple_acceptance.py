"""Recompute acceptance from independently reconstructed final physical fields."""
import argparse,json
from pathlib import Path
import numpy as np
from tensorfvm.verification.simple_gpu_audit import reconstruction
from tensorfvm.verification.core import write_json,sha,artifact_manifest
from tensorfvm.verification.audit import close
p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args();directory=a.directory;s=json.loads((directory/'summary.json').read_text());sa=json.loads((directory/'independent-sa-audit.json').read_text());records=[]
for case in s['cases']:
 devices=case.get('devices') or {key:case[key] for key in ['cpu','cuda'] if key in case}
 for device,r in devices.items():
  f=np.load(directory/case['name']/device/'fields.npz');values=reconstruction(f,r['config'],r['case']);res=max(values[k] for k in ['momentum','continuity','mass_imbalance'])
  if r['case']=='sa':
   sr=next(q for q in sa['results'] if q['case']==case['name'] and q['device']==device);res=max(res,sr['independent_sa_residual']);cf=float(f['computed_mean_cf']);ref=.074*r['config']['reynolds']**(-.2);close(ref,f['reference_mean_cf']);error=abs(cf-ref)/ref;m=next(q for q in r['metrics'] if q['name']=='skin_friction_relative_error');close(error,m['value']);assert m['limit']==.03 and m['passed']==(error<.03)
  metric=next(q for q in r['metrics'] if q['name']=='solver_residual');close(res,metric['value'],2e-9);assert metric['limit']==r['config']['tolerance'] and metric['passed']==(res<metric['limit'])
  metric=next(q for q in r['metrics'] if q['name']=='rhie_chow_flux_defect');close(values['rhie_chow_flux_defect'],metric['value'],2e-9);assert metric['limit']==1e-7 and metric['passed']==(values['rhie_chow_flux_defect']<1e-7)
  converged=res<r['config']['tolerance'] and values['rhie_chow_flux_defect']<1e-7;assert r['converged']==converged
  records.append(dict(case=case['name'],device=device,independent_steady_converged=converged,combined_passed=all(q['passed'] for q in r['metrics']),values=values))
write_json(directory/'acceptance-audit.json',dict(raw_gates_recomputed=True,cases=records,auditor_sha256=sha(Path(__file__)),summary_sha256=sha(directory/'summary.json')));m=json.loads((directory/'manifest.json').read_text());m['artifacts_sha256']=artifact_manifest(directory);write_json(directory/'manifest.json',m);print([(r['case'],r['device'],r['combined_passed']) for r in records])

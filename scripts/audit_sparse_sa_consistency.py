"""Verify exact prior five-step setup and raw CPU/CUDA differences independently."""
import json,statistics
from pathlib import Path
import numpy as np
from tensorfvm.verification.core import sha,write_json,artifact_manifest
from tensorfvm.verification.audit import close
root=Path(__file__).resolve().parents[1];p=root/'docs/verification-sparse-sa-consistency';s=json.loads((p/'summary.json').read_text());old=root/'docs/verification-gpu/SA-128x112-five-iterations';cfg=json.loads((old/'cpu-result.json').read_text())['config']
for name,h in s['source_sha256'].items():assert sha(root/name)==h
assert sha(old/'comparison.json')==s['previous_comparison_sha256'];fields={}
for device in ['cpu','cuda']:
 r=json.loads((p/(device+'-result.json')).read_text());assert r['config']==dict(cfg,device=device) and len(r['history'])==5 and r['actual_solver_device'].startswith(device)
 for q in r['linear_history']:
  assert q['actual_device'].startswith(device) and q['true_residual']<=q['target']*1.05
  if q['pressure']:assert q['assembled_operator_relative_defect']<2e-12
 fields[device]=np.load(p/(device+'-fields.npz'));t=s['devices'][device];assert len(t['samples_s'])==1 and t['samples_s'][0]>0;close(statistics.median(t['samples_s']),t['median_s'])
delta={k:float(np.linalg.norm(fields['cpu'][k]-fields['cuda'][k])/np.linalg.norm(fields['cpu'][k])) for k in ['velocity','pressure','nu_tilde']}
for k,v in delta.items():close(v,s['field_differences'][k])
assert s['equivalent']==all(v<=1e-6 for v in delta.values()) and s['physical_passed'] is None
write_json(p/'audit.json',dict(passed=True,equivalent=s['equivalent'],field_differences=delta,auditor_sha256=sha(Path(__file__)),summary_sha256=sha(p/'summary.json'),scope='same original configuration, actual five-step histories, strict true linear residuals, exact pressure operator contract and raw CPU/CUDA field gates; no steady physics or repeated performance claim'));m=json.loads((p/'manifest.json').read_text());m['artifacts_sha256']=artifact_manifest(p);write_json(p/'manifest.json',m);print(delta)

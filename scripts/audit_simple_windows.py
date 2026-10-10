"""Recompute all matched SIMPLE timing medians and saved window field differences."""
import argparse,json,statistics
from pathlib import Path
import numpy as np
from tensorfvm.verification.audit import close
from tensorfvm.verification.core import sha,write_json,artifact_manifest
p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args();s=json.loads((a.directory/'summary.json').read_text());results=[]
for case in s['cases']:
    directory=a.directory/case['name'];fields={};timings={}
    for dev in ['cpu','cuda']:
        fields[dev]=np.load(directory/(dev+'-window-fields.npz'));timings[dev]=json.loads((directory/(dev+'-window.json')).read_text());t=timings[dev];assert len(t['samples_s'])==3 and all(np.isfinite(x) and x>0 for x in t['samples_s']);close(statistics.median(t['samples_s']),t['median_s']);assert t['actual_solver_device'].startswith(dev) and t['five_actual_iterations']
    values={}
    for key,value in case['window_field_differences'].items():
        x,y=fields['cpu'][key],fields['cuda'][key];assert np.isfinite(x).all() and np.isfinite(y).all();values[key]=float(np.linalg.norm(x-y)/max(np.linalg.norm(x),1e-14));close(values[key],value)
    ratio=timings['cpu']['median_s']/timings['cuda']['median_s'];close(ratio,case['window_cpu_over_cuda']);results.append(dict(name=case['name'],fields_equivalent=all(v<=1e-6 for v in values.values()),field_differences=values,cpu_over_cuda=ratio))
write_json(a.directory/'window-audit.json',dict(raw_evidence_verified=True,cases=results,all_windows_equivalent=all(r['fields_equivalent'] for r in results),summary_sha256=sha(a.directory/'summary.json'),auditor_sha256=sha(Path(__file__)),scope='saved final primary window fields and timing medians; no independent replay of every SIMPLE intermediate state'))
m=json.loads((a.directory/'manifest.json').read_text());m['artifacts_sha256']=artifact_manifest(a.directory);write_json(a.directory/'manifest.json',m);print(results)

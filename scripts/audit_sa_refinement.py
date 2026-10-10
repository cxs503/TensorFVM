"""Audit device records, raw reused inputs and actual SA coarse/Krylov residuals."""
import argparse,json
from pathlib import Path
import numpy as np
from tensorfvm.verification.core import sha,write_json,artifact_manifest
from tensorfvm.verification.audit import close
p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args();d=a.directory;root=Path(__file__).resolve().parents[1];s=json.loads((d/'summary.json').read_text());records=[]
for name,h in s['provenance']['source_sha256'].items():assert sha(root/name)==h,name
for case in s['cases']:
 r=case['result'];device=r['config']['device'];path=d/case['name']/device;assert r['actual_solver_device'].startswith(device);assert sha(path/'fields.npz')==case['fields_sha256'];f=np.load(path/'fields.npz');assert len(json.loads((path/'history.json').read_text()))==r['iterations']
 for q in r['linear_history']:
  assert q['actual_device'].startswith(device) and q['true_residual']<=q['target']*1.05
  if q['pressure']:assert q['assembled_operator_relative_defect']<2e-12
 for q in r['coarse_history']:assert q['actual_device'].startswith(device) and q['operator_relative_defect']<2e-11 and q['linear_true_residual']<1e-9
 J,x,b=f['coupled_coarse_matrix'],f['coupled_coarse_last_solution'],f['coupled_coarse_last_rhs'];res=np.linalg.norm(J@x-b)/np.linalg.norm(b);assert res<1e-9;close(res,r['coarse_history'][-1]['linear_true_residual'])
 source=None
 if case['name']=='sa-64x56':source=root/'docs/verification-simple-gpu/sa-64x56/cpu'
 if case['name']=='sa-128x112':source=root/'docs/verification-sa-refinement-cpu-partial/sa-128x112/cpu'
 if source is not None:
  for name in ['fields.npz','history.json','result.json']:assert sha(path/name)==sha(source/name)
 records.append(dict(case=case['name'],device=device,cells=int(f['cell_volumes_m2'].size),coarse_lu_independent_residual=float(res),reused_inputs_identical=source is not None))
assert s['all_converged']==all(case['result']['converged'] for case in s['cases']);write_json(d/'audit.json',dict(passed=True,cases=records,auditor_sha256=sha(Path(__file__)),summary_sha256=sha(d/'summary.json'),scope='source hashes, exact reused CPU fields/history/results, actual devices, all linear/coarse residual gates and independent last coarse J*x-b; final physical residual gates audited separately; no finest-grid CPU/CUDA equivalence'))
m=json.loads((d/'manifest.json').read_text());m['artifacts_sha256']=artifact_manifest(d);write_json(d/'manifest.json',m);print(records)

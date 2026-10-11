"""Compare source-bound producer actions; exact equality on this probe only."""
from pathlib import Path
import json,hashlib
import numpy as np
out=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
old=json.loads((out/'historical.json').read_text());new=json.loads((out/'current.json').read_text())
if old['input_sha256']!=new['input_sha256']:raise ValueError('probe inputs differ')
for name,metadata in [('historical',old),('current',new)]:
 if sha(out/(name+'.npz'))!=metadata['output_sha256']:raise ValueError('probe output hash mismatch')
checks={}
with np.load(out/'historical.npz') as a,np.load(out/'current.npz') as b:
 if set(a.files)!=set(b.files):raise ValueError('probe arrays differ')
 for key in a.files:
  checks[key]=dict(shape=list(a[key].shape),maximum_absolute_difference=float(abs(a[key]-b[key]).max()),exactly_equal=bool(np.array_equal(a[key],b[key])))
result=dict(passed=all(v['exactly_equal'] for v in checks.values()),checks=checks,input_sha256=old['input_sha256'],source_changes={k:dict(historical=old['frozen_import_sha256'].get(k),current=new['frozen_import_sha256'].get(k)) for k in sorted(old['frozen_import_sha256'].keys()|new['frozen_import_sha256'].keys()) if old['frozen_import_sha256'].get(k)!=new['frozen_import_sha256'].get(k)},scope='Same 6,085-cell mesh, actual archived numerical fields and one seeded random full-matrix action. Not a general equivalence proof or new flow solve.')
(out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
(out/'manifest.json').write_text(json.dumps({p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='manifest.json'},indent=2)+'\n')
print(json.dumps(result,indent=2))
if not result['passed']:raise SystemExit(2)

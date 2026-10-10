"""Combine already computed NACA grids with per-run producer/source provenance."""
import json,shutil,subprocess,sys
from pathlib import Path
from tensorfvm.verification.core import write_json,sha,artifact_manifest
from tensorfvm.verification.external_flow import publish

root=Path(__file__).resolve().parents[1]
base=root/'docs/verification-external-naca-simple-control';fine=root/'docs/verification-external-naca-fine';out=root/'docs/verification-external-naca'
b=json.loads((base/'summary.json').read_text());f=json.loads((fine/'summary.json').read_text())
if len(b['runs'])!=2 or len(f['runs'])!=1:raise ValueError('Require two completed baseline grids and one completed fine grid')
if out.exists() and any(out.iterdir()):raise ValueError('Aggregate output must be empty')
out.mkdir(exist_ok=True)
shutil.copytree(base/'source-snapshot',out/'source-snapshot-baseline')
for p,v in f['provenance']['source_sha256'].items():
 if sha(root/p)!=v:raise ValueError('Fine producer source changed: '+p)
 dst=out/'source-snapshot'/p;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(root/p,dst)
rows=[]
for origin,summary,snapshot in [(base,b,'source-snapshot-baseline'),(fine,f,'source-snapshot')]:
 manifest=json.loads((origin/'manifest.json').read_text())
 for row in summary['runs']:
  for name in ('fields.npz','history.json','result.json'):
   key=row['directory']+'/'+name
   if sha(origin/key)!=manifest['artifacts_sha256'][key]:raise ValueError('Raw input artifact changed: '+key)
  for path,value in summary['provenance']['source_sha256'].items():
   if sha(out/snapshot/path)!=value:raise ValueError('Producer source hash mismatch')
  directory=row['directory'];shutil.copytree(origin/directory,out/directory)
  row['producer_provenance']=summary['provenance'];row['source_snapshot_root']=snapshot;row.setdefault('solver','simple');write_json(out/directory/'result.json',row);rows.append(row)
summary=dict(provenance=f['provenance'],runs=rows,passed=all(r['passed'] for r in rows),grid_convergence_qualified=False,assembly='Reuse verified actual 64x26 and 128x52 CPU SuperLU solves; actual 256x104 current-matrix GMRES with cached LU preconditioning and existing Anderson mixing. Per-run producer source snapshots are explicit; no reference state initializes flow.')
write_json(out/'summary.json',summary);publish(out,rows);write_json(out/'manifest.json',dict(source_snapshot_root='source-snapshot',source_sha256=f['provenance']['source_sha256'],artifacts_sha256=artifact_manifest(out)))
for script in ('audit_external_reports.py','audit_external_equations.py','publish_external_manuscript.py','publish_external_frontmatter.py'):
 subprocess.run([sys.executable,str(root/'scripts'/script),str(out)],cwd=root,check=True)
m=json.loads((out/'manifest.json').read_text());m['source_sha256']['scripts/finish_external_benchmarks.py']=sha(Path(__file__));m['artifacts_sha256']=artifact_manifest(out);write_json(out/'manifest.json',m)
print('NACA aggregate complete',[(r['resolution'],r['converged'],r['computed'],r['passed']) for r in rows])

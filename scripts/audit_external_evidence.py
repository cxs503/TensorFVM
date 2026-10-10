"""Check archived and current producer sources, artifacts and per-run provenance."""
import json
from pathlib import Path
from tensorfvm.verification.core import sha

root=Path(__file__).resolve().parents[1]
count=0
for path in sorted((root/'docs').glob('*/manifest.json')):
    m=json.loads(path.read_text());directory=path.parent;count+=1
    for file,value in m.get('source_sha256',{}).items():
        source=directory/m['source_snapshot_root']/file if m.get('source_snapshot_root') else root/file
        if not source.is_file() or sha(source)!=value:raise ValueError('Source mismatch: '+str(source))
    for file,value in m.get('artifacts_sha256',{}).items():
        if not (directory/file).is_file() or sha(directory/file)!=value:raise ValueError('Artifact mismatch: '+str(directory/file))
for name in ('cylinder','naca'):
    directory=root/'docs'/('verification-external-'+name);s=json.loads((directory/'summary.json').read_text())
    if len(s['runs'])!=3:raise ValueError('Incomplete three-grid study')
    for row in s['runs']:
        if row['passed']!=all(q['passed'] for q in row['metrics']):raise ValueError('False case badge')
        for file,value in row.get('producer_provenance',{}).get('source_sha256',{}).items():
            if sha(directory/row['source_snapshot_root']/file)!=value:raise ValueError('Per-run producer mismatch')
        if any(q['limit']!=.03 for q in row['metrics'] if q['name'].endswith('relative_error')):raise ValueError('Relaxed physical gate')
print('Evidence verified:',count,'manifests; six actual external-flow grids; all recorded producer snapshots and artifact hashes')

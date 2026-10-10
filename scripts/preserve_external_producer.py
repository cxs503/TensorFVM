"""Preserve exact hashed producer bytes before attaching newer audits/reports."""
import argparse
import json
from pathlib import Path
import shutil
from tensorfvm.verification.core import sha,write_json,artifact_manifest


def preserve(output,cache=None):
    root=Path(__file__).resolve().parents[1]
    path=output/'manifest.json';manifest=json.loads(path.read_text())
    snapshot_name=manifest.get('source_snapshot_root','source-snapshot')
    snapshot=output/snapshot_name
    for key,value in manifest['source_sha256'].items():
        target=snapshot/key
        if target.is_file() and sha(target)==value:continue
        candidates=[root/key]+([cache/key] if cache else [])
        source=next((p for p in candidates if p.is_file() and sha(p)==value),None)
        if source is None:raise ValueError('Cannot recover producer bytes: '+key)
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    manifest['source_snapshot_root']=snapshot_name;manifest['artifacts_sha256']=artifact_manifest(output)
    write_json(path,manifest);print('Producer preserved:',output)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path);p.add_argument('--cache',type=Path);args=p.parse_args();preserve(args.output,args.cache)

"""Audit terminal runs and qualify the three-grid family; never mark live runs passed."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--suite',type=Path,required=True)
    p.add_argument('--timeout-hours',type=float,default=48.);a=p.parse_args()
    suite=json.loads(a.suite.read_text());root=Path(suite['artifact_root']);repo=Path(__file__).resolve().parents[1]
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONPATH=str(repo/'src'))
    pending={c['name']:c for c in suite['cases']};done={};start=time.monotonic()
    def save():
        path=root/'suite-status.json';temporary=root/'suite-status-next.json'
        temporary.write_text(json.dumps(dict(pending=list(pending),completed=done,qualified=False),indent=2)+'\n');temporary.replace(path)
    save()
    while pending and time.monotonic()-start<a.timeout_hours*3600:
        for name,c in list(pending.items()):
            run=root/name;status=run/'status.json'
            if not status.exists():
                try:os.kill(c['pid'],0)
                except ProcessLookupError:
                    done[name]={'failure':'Producer exited without a terminal status'};del pending[name];save()
                continue
            state=json.loads(status.read_text());report=root/(name+'-final-report')
            if state.get('checkpoint_sha256'):
                with (root/(name+'-audit.log')).open('w') as log:
                    result=subprocess.run([sys.executable,str(repo/'scripts/report_highre_naca.py'),str(run),
                        '--reference',str(root/'reference'),'--output',str(report)],cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT)
                state['audit_exit_code']=result.returncode
                if result.returncode==0:
                    audit=json.loads((report/'audit.json').read_text());state['numerical_passed']=audit['numerical_passed'];state['reference_passed']=audit['reference_passed']
            done[name]=state;del pending[name];save();print(name,json.dumps(state),flush=True)
        if pending:time.sleep(30)
    save()
    family=suite['qualification_family']
    if all(done.get(name,{}).get('audit_exit_code')==0 and (root/(name+'-final-report')/'audit.json').exists() for name in family):
        with (root/'qualification.log').open('w') as log:
            result=subprocess.run([sys.executable,str(repo/'scripts/qualify_highre_naca.py'),
                *[str(root/(name+'-final-report')) for name in family],
                '--output',str(root/'qualification.json')],cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode not in (0,2):print('Qualification script failed; see qualification.log',flush=True)
        if (root/'qualification.json').exists():
            status=json.loads((root/'suite-status.json').read_text());status['qualified']=json.loads((root/'qualification.json').read_text())['qualified']
            (root/'suite-status.json').write_text(json.dumps(status,indent=2)+'\n')
    return 0 if not pending else 2


if __name__=='__main__':raise SystemExit(main())

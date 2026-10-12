#!/usr/bin/env python3
"""Continue an existing decomposed case in bounded blocks and audit each block.

Requires prior mesh gating and decomposition. A stopped run is never a pass.
"""
import argparse,json,re,subprocess
from pathlib import Path
from tensorfvm.openfoam_history import convergence_audit
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--case',type=Path,required=True)
p.add_argument('--foam-wrapper',default='tensorfvm-foam12-mpi')
p.add_argument('--ranks',type=int,required=True)
p.add_argument('--cpu-set',required=True)
p.add_argument('--block',type=int,default=200)
p.add_argument('--max-iteration',type=int,default=2000)
a=p.parse_args();case=a.case.resolve()
if a.block<1 or a.ranks<1 or a.max_iteration<1:p.error('positive block, ranks and max iteration required')
if not (case/'processor0').exists():p.error('case must already be decomposed')
if 'Mesh OK.' not in (case/'log.checkMesh').read_text():p.error('standard mesh gate not passed')
layers=re.findall(r'^wall\s+\d+\s+([\d.eE+-]+)\s+[\d.eE+-]+\s+[\d.eE+-]+\s*$',(case/'log.snappyHexMesh').read_text(),re.M)
if not layers or float(layers[-1])<5:p.error('actual mean layer gate not passed')
stages=[];stagefile=case/'continuation-stages.json'
if stagefile.exists():stages=json.loads(stagefile.read_text())
while True:
    checkpoints=sorted(int(d.name) for d in (case/'processor0').iterdir() if d.is_dir() and d.name.isdigit() and (d/'U').exists())
    start=checkpoints[-1]
    try:audit=convergence_audit(case)
    except ValueError:audit={'numerically_converged':False}
    if audit['numerically_converged'] or start>=a.max_iteration:break
    end=min(start+a.block,a.max_iteration)
    control=case/'system/controlDict';s=control.read_text()
    s=re.sub(r'startFrom\s+\w+;', 'startFrom latestTime;',s)
    s=re.sub(r'endTime\s+\d+;',f'endTime {end};',s);control.write_text(s)
    # Foundation restores the stored time index, which need not equal the
    # selected start time. Choose its endpoint index as the write period so
    # arbitrary restart/block lengths persist the endpoint, including yPlus.
    time_state=case/'processor0'/str(start)/'uniform/time'
    index_match=re.search(r'\bindex\s+(\d+)\s*;',time_state.read_text()) if time_state.exists() else None
    write_interval=(int(index_match[1]) if index_match else 0)+end-start
    s=re.sub(r'writeInterval\s+(\d+);',lambda m:m.group() if int(m[1])==1 else f'writeInterval {write_interval};',s)
    control.write_text(s)
    logfile=case/'log.foamRun'
    if logfile.exists():
        archived=case/f'log.foamRun.before-{start}'
        if archived.exists():raise RuntimeError('log archive collision; preserve evidence')
        logfile.rename(archived)
    stages.append(dict(start=start,end=end,ranks=a.ranks,cpu_set=a.cpu_set,write_interval=write_interval,physical_settings_changed=False))
    stagefile.write_text(json.dumps(stages,indent=2)+'\n')
    with logfile.open('w') as stream:
        subprocess.run([a.foam_wrapper,'mpirun','--mca','btl','self,vader,tcp','--cpu-set',a.cpu_set,'--bind-to','core','-np',str(a.ranks),'foamRun','-parallel','-case',str(case)],stdout=stream,stderr=subprocess.STDOUT,check=True)
    checkpoint_fields=[case/f'processor{rank}'/str(end)/field
                       for rank in range(a.ranks) for field in ['U','p','k','omega','nut','phi']]
    if not all(path.is_file() and path.stat().st_size>0 for path in checkpoint_fields):
        raise RuntimeError(f'block ended without complete checkpoint {end}; refusing to reuse stale fields')
    audit=convergence_audit(case)
    (case/f'convergence-audit-{end}.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(dict(iteration=end,converged=audit['numerically_converged'],force_stable=audit.get('force_stable'),residuals_passed=audit.get('residuals_passed'))),flush=True)
with (case/'log.reconstruct-final').open('w') as stream:
    subprocess.run([a.foam_wrapper,'reconstructPar','-case',str(case),'-latestTime'],stdout=stream,stderr=subprocess.STDOUT,check=True)
(case/'convergence-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
print(json.dumps(audit,indent=2),flush=True)
if not audit['numerically_converged']:
    raise SystemExit('Iteration limit reached without numerical convergence; audit saved.')

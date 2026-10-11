import subprocess,os,time,json,shutil
from pathlib import Path
repo=Path('/home/jsyc/tensor-suite-development/TensorFVM');os.chdir(repo)
python='/data/TensorFEM/.venv/bin/python'
env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONPATH='src')
status=Path('/tmp/suboff-cubic-flow-status.json')
def state(stage,**kw):status.write_text(json.dumps(dict(stage=stage,**kw))+'\n')
run=Path('docs/verification-native-suboff-mms/tail-transition-cubic-viscosity')
seed=Path('docs/verification-native-suboff-mms/tail-transition-quadratic')
cmd=[python,'scripts/run_native_suboff_benchmark.py','--mesh',str(seed/'input.msh'),'--output',str(run),'--reconstruction','quadratic','--quadratic-linear-solver','assembled','--viscous-reconstruction','cubic','--initial-run',str(seed)]
state('full_cubic_viscosity_flow')
with open('/tmp/suboff-cubic-flow.log','w') as log:
 p=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
 observations=[];start=time.monotonic();reason=None
 while p.poll() is None:
  try:
   info=Path(f'/proc/{p.pid}/status').read_text();rss=int(next(x.split()[1] for x in info.splitlines() if x.startswith('VmRSS:')))
   mem=Path('/proc/meminfo').read_text();available=int(next(x.split()[1] for x in mem.splitlines() if x.startswith('MemAvailable:')))
   observations.append(dict(seconds=time.monotonic()-start,rss_kib=rss,system_available_kib=available))
   if available<1536*1024 or time.monotonic()-start>1800:
    reason='available memory or 1800s time limit';p.terminate();state('resource_or_time_limit',available_kib=available);p.wait();break
  except (FileNotFoundError,StopIteration):pass
  time.sleep(2)
 code=p.wait()
(run/'resource-observations.json').write_text(json.dumps(dict(samples=observations,sample_interval_seconds=2,timeout_seconds=1800,minimum_available_MiB=1536,returncode=code,termination_reason=reason,command=cmd),indent=2)+'\n')
shutil.copy2(__file__,run/'executed-pipeline.py');shutil.copy2('/tmp/suboff-cubic-flow.log',run/'run.log')
if code!=0:state('flow_failed',returncode=code);raise SystemExit(code)
for stage,args in [
 ('independent_audit',['scripts/audit_native_suboff_run.py','--run',str(run),'--output',str(run/'audit')]),
 ('local_pressure',['scripts/audit_native_suboff_local_pressure.py','--runs',str(seed),str(run),'--output','docs/verification-native-suboff-mms/cubic-viscosity-local-pressure']),
 ('term_diagnostic',['scripts/audit_native_quadratic_terms.py','--run',str(run),'--output','docs/verification-native-suboff-mms/cubic-viscosity-term-diagnostic']),
 ('field_plots',['scripts/plot_native_suboff_fields.py','--mesh',str(run/'input.msh'),'--fields',str(run/'fields.npz'),'--output',str(run/'plots'),'--case-kind','manufactured','--status-text','Manufactured solution; physical SUBOFF unqualified','--plane-z','.03']),
 ('comparison_plots',['scripts/plot_native_manufactured_comparison.py','--run',str(run),'--output',str(run/'comparison-plots')]),
 ('hotspot_plots',['scripts/plot_native_manufactured_comparison.py','--run',str(run),'--output',str(run/'hotspot-comparison'),'--plane-z','.002857978102883826','--x-limits','.97','1.025','--y-limits','-.015','.015']),
]:
 state(stage)
 with open('/tmp/suboff-cubic-'+stage+'.log','w') as log:
  code=subprocess.call([python]+args,env=env,stdout=log,stderr=subprocess.STDOUT)
 if code:state('postprocessing_failed',step=stage,returncode=code);raise SystemExit(code)
state('complete')

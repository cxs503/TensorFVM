import subprocess,os,time,json
from pathlib import Path
repo=Path('/home/jsyc/tensor-suite-development/TensorFVM');os.chdir(repo)
python='/data/TensorFEM/.venv/bin/python'
env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONPATH='src')
status=Path('/tmp/suboff-tail-flow-status.json')
def state(stage,**kw):status.write_text(json.dumps(dict(stage=stage,**kw))+'\n')
state('waiting_for_numerical_seed')
while Path('/proc/519330').exists():time.sleep(2)
seed=Path('docs/verification-native-suboff-mms/tail-transition-transferred-seed')
if not (seed/'result.json').exists():raise RuntimeError('Numerical seed failed to produce a result')
run=Path('docs/verification-native-suboff-mms/tail-transition-quadratic')
cmd=[python,'scripts/run_native_suboff_benchmark.py','--mesh','docs/verification-suboff-single-domain-mesh/coarse-tail-transition/suboff-hybrid.msh','--output',str(run),'--reconstruction','quadratic','--quadratic-linear-solver','assembled','--initial-run',str(seed)]
state('full_quadratic_flow')
with open('/tmp/suboff-tail-quadratic.log','w') as log:
 p=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
 observations=[];start=time.monotonic()
 while p.poll() is None:
  try:
   info=Path(f'/proc/{p.pid}/status').read_text();rss=int(next(x.split()[1] for x in info.splitlines() if x.startswith('VmRSS:')))
   mem=Path('/proc/meminfo').read_text();available=int(next(x.split()[1] for x in mem.splitlines() if x.startswith('MemAvailable:')))
   observations.append(dict(seconds=time.monotonic()-start,rss_kib=rss,system_available_kib=available))
   if available<768*1024 or time.monotonic()-start>1800:
    p.terminate();state('resource_or_time_limit',available_kib=available);p.wait();break
  except (FileNotFoundError,StopIteration):pass
  time.sleep(2)
 code=p.wait()
(run/'resource-observations.json').write_text(json.dumps(dict(samples=observations,sample_interval_seconds=2,timeout_seconds=1800,minimum_available_MiB=768,returncode=code),indent=2)+'\n')
if code!=0:state('flow_failed',returncode=code);raise SystemExit(code)
for stage,args in [
 ('independent_audit',['scripts/audit_native_suboff_run.py','--run',str(run),'--output',str(run/'audit')]),
 ('field_plots',['scripts/plot_native_suboff_fields.py','--mesh',str(run/'input.msh'),'--fields',str(run/'fields.npz'),'--output',str(run/'plots'),'--case-kind','manufactured','--status-text','Manufactured solution; physical SUBOFF unqualified','--plane-z','.03']),
 ('comparison_plots',['scripts/plot_native_manufactured_comparison.py','--run',str(run),'--output',str(run/'comparison-plots')]),
 ('tail_plots',['scripts/plot_native_manufactured_comparison.py','--run',str(run),'--output',str(run/'tail-comparison'),'--plane-z','0','--x-limits','.9','1.025','--y-limits','-.025','.025']),
]:
 state(stage)
 with open('/tmp/suboff-tail-'+stage+'.log','w') as log:
  code=subprocess.call([python]+args,env=env,stdout=log,stderr=subprocess.STDOUT)
 if code:state('postprocessing_failed',step=stage,returncode=code);raise SystemExit(code)
state('complete')

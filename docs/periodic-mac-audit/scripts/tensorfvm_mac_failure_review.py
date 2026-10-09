import json,hashlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from tensorfvm.periodic_mac import PeriodicMACSolver,PeriodicMACConfig
from tensorfvm_mac_independent_review import independent_projection,divergence
c=PeriodicMACConfig(nx=16,ny=12,nz=8,time_step_s=.5,nonlinear_iterations=4)
s=PeriodicMACSolver(c);r=np.random.default_rng(93);v,_=independent_projection(r.normal(scale=.4,size=(3,*s.shape)),s.spacing);s.velocity=torch.tensor(np.asarray(v),dtype=torch.float64);before=s.velocity.clone();clock=s.time;hist=len(s.history)
try:s.step();raise AssertionError('expected unresolved midpoint candidate')
except RuntimeError as exc:failure=str(exc)
assert torch.equal(before,s.velocity) and s.time==clock and len(s.history)==hist and s.last_raw['accepted'] is False
q=s.last_raw;old=q['old_velocity'];mid=q['midpoint_velocity'];candidate=q['candidate_velocity'];res=float((q['check_velocity']-candidate).abs().max());assert res>c.nonlinear_tolerance
try:PeriodicMACSolver(c,SimpleNamespace(world_size=2));raise AssertionError('unsupported distributed accepted')
except NotImplementedError:distributed_rejected=True
report={'failure':failure,'config':c.__dict__,'nonlinear_residual':res,'rollback_exact_velocity_clock_history':True,'distributed_rejected':distributed_rejected,'physical_accuracy_qualified':False,'source_sha256':hashlib.sha256(Path('src/tensorfvm/periodic_mac.py').read_bytes()).hexdigest(),'auditor_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()};np.savez('/tmp/tensorfvm-mac-failed-candidate.npz',old=old.numpy(),candidate=candidate.numpy(),check=q['check_velocity'].numpy(),midpoint=mid.numpy(),pressure=q['pressure'].numpy());Path('/tmp/tensorfvm-mac-failure-review.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

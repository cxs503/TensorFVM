"""Isolate the rejected 128-square-grid candidate without changing legacy sources."""
from pathlib import Path
import argparse
import numpy as np
import torch
from tensorfvm.periodic_mac import PeriodicMACConfig, PeriodicMACSolver
from tensorfvm.verification.core import provenance,write_json,artifact_manifest,sha

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    c=PeriodicMACConfig(nx=128,ny=128,nz=4,time_step_s=.0125,viscosity_pa_s=.1)
    s=PeriodicMACSolver(c)
    for a in (0,1):
        x,y,z=s.face_coordinates(a)
        s.velocity[a]=torch.sin(x)*torch.cos(y) if a==0 else -torch.cos(x)*torch.sin(y)
    states=[s.velocity.numpy().copy()]
    failure=None
    for k in range(40):
        old=s.velocity.clone();clock=s.time;count=len(s.history)
        try:s.step()
        except RuntimeError as exc:
            failure=str(exc);break
        states.append(s.velocity.numpy().copy())
    if failure is None:raise AssertionError("expected nonlinear rejection not reproduced")
    q=s.last_raw
    np.savez_compressed(out/"failed-candidate.npz",accepted_velocity_states=np.stack(states),
        state_velocity=s.velocity.numpy(),old_velocity=old.numpy(),
        candidate_velocity=q["candidate_velocity"].numpy(),
        check_velocity=q["check_velocity"].numpy(),midpoint_velocity=q["midpoint_velocity"].numpy(),
        pressure=q["pressure"].numpy(),tentative_velocity=q["tentative_velocity"].numpy())
    write_json(out/"accepted-history.json",s.history)
    write_json(out/"nonlinear-failure.json",dict(config=c.__dict__,failure=failure,
       state_unchanged=bool(torch.equal(s.velocity,old)),clock_unchanged=s.time==clock,
       accepted_history_unchanged=len(s.history)==count,clock_s=s.time,
       accepted=q["accepted"],accepted_steps=len(s.history),
       candidate_finite=bool(torch.isfinite(q["candidate_velocity"]).all()),
       reproducer_sha256=sha(Path(__file__)),physical_accuracy_qualified=False,provenance=provenance()))
    (out/"README.md").write_text("""# Retained 128-square-grid nonlinear failure

The original suite stopped at nx=ny=128, nz=4, dt=0.0125 s, mu=0.1 Pa s.
An isolated exact-configuration replay records every accepted velocity state,
then the rejected candidate/check/tentative/pressure arrays. Nonfinite candidate
values are deliberately retained in NPZ. Velocity, clock and history remain at
the last accepted state when this candidate is rejected; this does not mean no
earlier steps were accepted.

Preceding channel and 32/64 vortex raw results from the aborted suite remain.
The final formal spatial study uses 32/48/64 at fixed dt=0.0125. It cannot
overwrite this 128-grid failure. A smaller time step or another nonlinear solver
is required to qualify the 128 grid; this report asserts neither qualification.
Reproduce using scripts/reproduce_periodic_mac_failure.py with an output path.
""")
    write_json(out/"manifest.json",dict(artifacts_sha256=artifact_manifest(out),
        scope="aborted suite and exact nonlinear-failure replay"))
    print("failure retained",s.time,len(s.history),failure)

if __name__=="__main__":main()

"""Bounded 30P30N SIMPLE/SA diagnostic with live residuals and retained failures."""
import argparse,json,signal,time
from pathlib import Path
from dataclasses import replace
import torch
from tensorfvm.solver import SimpleSolver,SolverConfig
from tensorfvm.benchmark_30p30n import export_result
from tensorfvm.verification.core import provenance,sha,write_json,artifact_manifest


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mesh-file',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--iterations',type=int,default=20);p.add_argument('--seconds',type=int,default=120);a=p.parse_args()
    if a.iterations<1 or a.seconds<1:p.error('Positive iteration and time limits required')
    a.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1);root=Path(__file__).resolve().parents[1];prov=provenance()
    for key in ['scripts/run_three_element_flow_diagnostic.py','src/tensorfvm/benchmark_30p30n.py','src/tensorfvm/multi_element.py']:prov['source_sha256'][key]=sha(root/key)
    config=SolverConfig(nx=1,ny=1,length=10,height=10,cylinder_radius=None,mesh_type='three-element',mesh_file=str(a.mesh_file.resolve()),airfoil_chord=1,inlet_velocity=1,density=1,reynolds=5_000_000,angle_of_attack=0,turbulence_model='spalart-allmaras',sa_freestream_ratio=3,velocity_relaxation=.3,pressure_relaxation=.2,turbulence_relaxation=.3,pseudo_time_step=.02,tolerance=1e-5,max_iterations=a.iterations,device='cpu')
    solver=SimpleSolver(config);start=time.perf_counter();failure=None
    def timeout(*_):raise TimeoutError('Wall-clock diagnostic limit reached inside SIMPLE step')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(a.seconds)
    try:
        for _ in range(a.iterations):
            record=solver.step();print(json.dumps(record),flush=True);write_json(a.output/'live-history.json',solver.history)
            if solver.converged:break
    except (TimeoutError,RuntimeError,ValueError) as error:failure=str(error)
    finally:signal.alarm(0)
    # Never export a partially advanced iterate as a completed physical solve.
    if failure is None:
        solver.config=replace(config,max_iterations=len(solver.history));export_result(solver.solve(),a.output)
    write_json(a.output/'diagnostic.json',dict(converged=bool(solver.converged) and failure is None,physics_accepted=False,completed_iterations=len(solver.history),elapsed_s=time.perf_counter()-start,failure=failure,partial_iterate_discarded=failure is not None,cells=solver.count,history=solver.history,mesh_sha256=sha(a.mesh_file),provenance=prov))
    for key in prov['source_sha256']:
        target=a.output/'source-snapshot'/key;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((root/key).read_bytes())
    (a.output/'mesh.msh').write_bytes(a.mesh_file.read_bytes());write_json(a.output/'manifest.json',dict(source_snapshot_root='source-snapshot',source_sha256=prov['source_sha256'],artifacts_sha256=artifact_manifest(a.output)))
    print((a.output/'diagnostic.json').read_text()[:1800]);return 0 if solver.converged and failure is None else 2

if __name__=='__main__':raise SystemExit(main())

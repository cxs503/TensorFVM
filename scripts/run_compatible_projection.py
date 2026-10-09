"""Real native CPU/CUDA face-projection cases; preserve inadequate PCG-budget failure."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from tensorfvm.benchmark_cylinder3d import run_benchmark


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('docs/compatible-projection/native'))
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    cases=[]
    configurations=[('cpu-150-failed','cpu',150),('cpu-250','cpu',250)]
    if torch.cuda.is_available(): configurations.append(('cuda-250','cuda:0',250))
    for name,device,budget in configurations:
        summary=run_benchmark(args.output/name,steps=50,device=device,pressure_iterations=budget)
        hist=json.loads((args.output/name/'history.json').read_text())
        summary.update(case=name,max_continuity=max(h['continuity'] for h in hist),
                       max_mass_imbalance=max(h['boundary_mass_imbalance'] for h in hist),
                       failed_pressure_steps=[h['step'] for h in hist if not h['pressure_converged']],
                       max_pressure_iterations=max(h['pressure_iterations'] for h in hist))
        cases.append(summary)
        print(json.dumps(summary),flush=True)
    root=Path(__file__).resolve().parents[1]
    sources=['src/tensorfvm/solver3d.py','src/tensorfvm/benchmark_cylinder3d.py',
             'src/tensorfvm/runtime.py','src/tensorfvm/mesh3d.py','scripts/run_compatible_projection.py']
    manifest=dict(schema='tensorfvm.native-compatible-projection-study/1',cases=cases,
                  source_sha256={s:hashlib.sha256((root/s).read_bytes()).hexdigest() for s in sources},
                  raw_sha256={str(p.relative_to(args.output)):hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in sorted(args.output.rglob('*')) if p.is_file() and p.name!='study.json'},
                  physical_accuracy_qualified=False,
                  scope='Discrete face projection and mass only; not momentum transport, LES or FSI qualification',
                  gpu_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    (args.output/'study.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()

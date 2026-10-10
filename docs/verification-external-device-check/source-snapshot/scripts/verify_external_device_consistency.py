"""Five actual steps on CPU/CUDA; a consistency check, not a physical qualification."""
import json
from pathlib import Path
import numpy as np
import torch
from tensorfvm.external_linear import ExternalFlowSolver
from tensorfvm.verification.external_flow import configuration
from tensorfvm.verification.core import provenance,sha,write_json,artifact_manifest


def main():
    torch.set_num_threads(1);output=Path('docs/verification-external-device-check')
    if output.exists():raise ValueError('Output must not exist')
    output.mkdir();rows=[]
    for case in ('cylinder','naca'):
        solvers={}
        for device in ('cpu','cuda'):
            s=ExternalFlowSolver(configuration(case,32,12,device,5))
            for _ in range(5):s.step()
            solvers[device]=s
            np.savez_compressed(output/(case+'-'+device+'.npz'),velocity=s.velocity.cpu().numpy(),pressure=s.p.cpu().numpy(),mass_flux=s.mass_flux.cpu().numpy())
            write_json(output/(case+'-'+device+'-history.json'),dict(history=s.history,linear_history=s.linear_history,converged=s.converged))
        errors={}
        for name in ('velocity','p','mass_flux'):
            a=getattr(solvers['cpu'],name).numpy();b=getattr(solvers['cuda'],name).cpu().numpy();errors[name]=float(np.linalg.norm(a-b)/max(np.linalg.norm(a),1e-14))
        rows.append(dict(case=case,grid='32x12',steps=5,relative_l2_errors=errors,consistency_passed=all(x<1e-6 for x in errors.values()),physical_passed=None))
    prov=provenance();root=Path(__file__).resolve().parents[1]
    for path in ('src/tensorfvm/external_linear.py','src/tensorfvm/sparse_simple.py','src/tensorfvm/benchmark_airfoil.py','scripts/verify_external_device_consistency.py'):prov['source_sha256'][path]=sha(root/path)
    write_json(output/'summary.json',dict(provenance=prov,gpu=torch.cuda.get_device_name(0),runs=rows,passed=all(r['consistency_passed'] for r in rows),scope='CPU SuperLU versus CUDA Torch Krylov; five real steps; no steady, physical-accuracy or speedup claim'))
    (output/'report.md').write_text('# 外流 CPU/CUDA 一致性辅助检查\n\n32×12控制体、每例五个实际 SIMPLE 步。CPU SuperLU 与 CUDA Torch Krylov 求解相同离散方程。该检查不能证明稳态、3%物理精度或加速比。\n\n'+json.dumps(rows,indent=2)+'\n')
    write_json(output/'manifest.json',dict(source_sha256=prov['source_sha256'],artifacts_sha256=artifact_manifest(output)))
if __name__=='__main__':main()

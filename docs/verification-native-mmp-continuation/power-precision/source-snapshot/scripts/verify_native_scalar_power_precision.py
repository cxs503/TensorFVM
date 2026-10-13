#!/usr/bin/env python3
"""Record both equivalent power evaluations from the actual low-Re SST solve."""
import argparse
import hashlib
from pathlib import Path
import numpy as np
import tensorfvm.native_hybrid_scalar3d as scalar_module
from tensorfvm.verification.native_hybrid_rans import coupled_shear_solver
from tensorfvm.verification.native_run import NativeRunEvidence,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    out=p.parse_args().output;evidence=NativeRunEvidence(out,__file__);records=[];worst=None
    original=scalar_module.solve_mmp_newton
    def record(*args,**kwargs):
        nonlocal worst
        result=original(*args,**kwargs)
        keys=('diffusion_energy','boundary_diffusive_work','work_balance_direct_float64','work_balance',
              'work_balance_precision_bits','solution_precision_bits','ledger_refinements',
              'current_nonlinear_residual','cell_balance_max','interface_flux_defect_max','global_balance')
        row={k:result[k] for k in keys};row['solve_index']=len(records)+1;records.append(row)
        if worst is None or abs(result['work_balance_direct_float64'])>abs(worst['work_balance_direct_float64']):worst=result
        return result
    scalar_module.solve_mmp_newton=record
    try:
        mesh,solver=coupled_shear_solver(8,'low_re',triangulate_faces=True,scalar_correction='nonlinear_mmp')
        result=solver.solve(max_iterations=60,dt=1.,tolerance=1e-8)
    finally:scalar_module.solve_mmp_newton=original
    assert result['numerically_converged'] and records
    assert all(max(r['cell_balance_max'],r['interface_flux_defect_max'],abs(r['global_balance']),abs(r['work_balance']))<1e-8 for r in records)
    path=out/'largest-direct-power-defect.npz'
    np.savez_compressed(path,vertices=mesh.vertices,cell_volumes=mesh.cell_volumes,
        owner=mesh.owner,neighbor=mesh.neighbor,cell_centers=mesh.cell_centers,face_centers=mesh.face_centers,
        **{k:v for k,v in worst.items() if k!='nonlinear_history'})
    np.testing.assert_allclose(worst['work_balance'],worst['work_balance_cell_terms'].sum()+worst['work_balance_interface_terms'].sum(),atol=1e-14)
    write_json(out/'solves.json',records);write_json(out/'largest-direct-power-history.json',worst['nonlinear_history'])
    evidence.verify_sources()
    summary=dict(source_integrity_verified=True,all_scalar_ledger_gates_passed=True,
        scalar_solves=len(records),fields_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        largest_direct_power_defect=worst['work_balance_direct_float64'],equivalent_stable_power_defect=worst['work_balance'],
        associated_diffusion_energy=worst['diffusion_energy'],physical_suboff_benchmark_passed=False)
    write_json(out/'summary.json',summary);print(summary)


if __name__=='__main__':main()

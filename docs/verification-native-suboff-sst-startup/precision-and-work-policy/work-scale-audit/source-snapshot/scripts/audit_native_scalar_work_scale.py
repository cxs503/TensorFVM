#!/usr/bin/env python3
"""Check work-diagnostic scaling on a bound saved scalar equation/state."""
import argparse,hashlib,json,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorfvm.native_mesh3d import NativeMesh3D
from tensorfvm.native_hybrid_scalar3d import NativeHybridScalar3D
from tensorfvm.verification.native_run import NativeRunEvidence,write_json


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('replay',type=Path);p.add_argument('--mesh',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    summary=json.loads((a.replay/'summary.json').read_text());settings=json.loads((a.replay/'settings.json').read_text())
    assert summary['source_integrity_verified'] and sha(a.replay/'candidate.npz')==summary['candidate_sha256']
    assert sha(a.replay/'inputs.npz')==summary['inputs_sha256'] and sha(a.mesh)==settings['original_bindings']['initial-input.msh']
    for name,digest in json.loads((a.replay/'sources.json').read_text()).items():assert sha(a.replay/'source-snapshot'/name)==digest
    evidence=NativeRunEvidence(a.output,__file__);out=a.output
    write_json(out/'settings.json',dict(replay=str(a.replay),candidate_sha256=summary['candidate_sha256'],inputs_sha256=summary['inputs_sha256'],mesh_sha256=sha(a.mesh),
        scope='homogeneous rescaling of a single frozen transport equation; not a new SST flow calculation',
        work_relative_threshold=1e-12,absolute_flux_threshold_unchanged=1e-8))
    m=NativeMesh3D.from_gmsh(a.mesh,triangulate_faces=True);x=np.load(a.replay/'inputs.npz');c=np.load(a.replay/'candidate.npz');rows=[]
    # Derive dt and inflow from the original verified startup settings, preserving
    # their input values. This tool is scoped to this explicitly bound startup.
    original=Path(settings['original_run']);old_settings=json.loads((original/'settings.json').read_text())
    assert sha(original/'settings.json')==settings['original_bindings']['settings.json']
    invdt=1/old_settings['dt'];inflow=old_settings['omega_inflow']
    for exponent in (-20,0,20):
        factor=2.**exponent;op=NativeHybridScalar3D(m,x['coefficient'],dirichlet_mask=x['dirichlet'],correction='nonlinear_mmp')
        state=np.r_[c['cell_values'],c['face_values']]*factor;op.diffusion,_=op.mmp.operator(state)
        r=op._solution_result(state,x['volume_flux'],x['source']*factor,x['sink'],x['old']*factor,invdt,inflow*factor,0.)
        rows.append(dict(exponent=exponent,scale=factor,work=r['work_balance'],work_scale=r['work_balance_scale'],relative_work=r['relative_work_balance'],
            absolute_work_gate_passed=r['absolute_work_gate_passed'],cell_balance=r['cell_balance_max']))
    np.testing.assert_allclose([r['relative_work'] for r in rows],rows[1]['relative_work'],rtol=1e-14)
    evidence.verify_sources();write_json(out/'audit.json',dict(source_integrity_verified=True,rows=rows,relative_work_invariant=True,
        legacy_absolute_work_classification_changes=len({r['absolute_work_gate_passed'] for r in rows})>1,
        scalar_flux_absolute_gate_not_claimed_unit_invariant=True,physical_suboff_benchmark_passed=False))
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    exponents=[r['exponent'] for r in rows]
    axes[0].semilogy(exponents,[abs(r['work']) for r in rows],'o-');axes[0].axhline(1e-8,ls='--',c='k',label='Legacy absolute gate')
    axes[0].set(xlabel='Scalar rescaling exponent: phi -> 2^n phi',ylabel='Absolute quadratic-work defect',title='Legacy work gate depends on numeric scale')
    axes[1].semilogy(exponents,[r['relative_work'] for r in rows],'o-');axes[1].axhline(1e-12,ls='--',c='k',label='Explicit relative work gate')
    axes[1].set(xlabel='Scalar rescaling exponent',ylabel='Defect / gross work terms',title='Same dimensionless work defect')
    for ax in axes:ax.grid(True);ax.legend(fontsize=8)
    fig.savefig(out/'work-scale.png',dpi=170);fig.savefig(out/'work-scale.pdf');plt.close(fig)
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()

#!/usr/bin/env python3
"""Compare finite Navier-slip states with an independently converged no-slip run."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def read(root):
    cfg=json.loads((root/'settings.json').read_text());summary=json.loads((root/'summary.json').read_text())
    audit=json.loads((root/'artifact-audit.json').read_text())
    if not summary['numerically_converged'] or not audit['source_and_fields_verified']:raise ValueError('verified converged flow required')
    if sha(root/'fields.npz')!=summary['fields_sha256'] or sha(root/'input.msh')!=cfg['mesh_sha256']:raise ValueError('changed numerical input/output')
    for name,digest in json.loads((root/'sources.json').read_text()).items():
        if sha(root/'source-snapshot'/name)!=digest:raise ValueError('changed producer source')
    with np.load(root/'fields.npz',allow_pickle=False) as f:fields={k:f[k].copy() for k in f.files}
    wall=fields['patch_labels']=='wall';force=-fields['total_traction_integrated'][wall].sum(0)
    cd=2*force[0]/(cfg['rho']*cfg['U']**2*cfg['L']**2)
    np.testing.assert_allclose(cd,summary['Cd_L2'],rtol=1e-12)
    return cfg,summary,fields,float(cd)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--no-slip',type=Path,required=True)
    p.add_argument('--slip',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists() and any(a.output.iterdir()):p.error('output must be empty')
    cfg,base,f0,cd0=read(a.no_slip)
    if cfg.get('wall_friction') is not None:raise ValueError('no-slip baseline required')
    rows=[]
    for root in a.slip:
        c,s,f,cd=read(root)
        keys=('backend','Re_L','rho','nu','L','U','mesh_sha256','body_force','convection_order','outlet','inlet_and_farfield_velocity')
        if any(c[k]!=cfg[k] for k in keys):raise ValueError('same geometry and physical parameters required')
        beta=c.get('wall_friction')
        if beta is None or beta<=0:raise ValueError('positive finite slip coefficient required')
        np.testing.assert_array_equal(f['owner'],f0['owner']);np.testing.assert_allclose(f['cell_centers'],f0['cell_centers'],atol=0,rtol=0)
        volume=f['cell_volumes'];u_error=float(np.sqrt(np.sum(volume[:,None]*(f['velocity']-f0['velocity'])**2)/np.sum(volume[:,None]*f0['velocity']**2)))
        rows.append(dict(run=str(root),friction=beta,Cd_L2=cd,Cd_relative_difference_from_no_slip=abs(cd-cd0)/abs(cd0),
            velocity_relative_difference_from_no_slip=u_error,normal_velocity_error=s['wall_normal_velocity_error'],
            robin_tangential_defect=s['wall_robin_tangential_defect'],source_bindings={name:sha(root/name) for name in ('settings.json','summary.json','fields.npz','sources.json','artifact-audit.json')}))
    rows.sort(key=lambda r:r['friction']);a.output.mkdir(parents=True,exist_ok=True)
    result=dict(no_slip_Cd_L2=cd0,no_slip_source=str(a.no_slip),no_slip_bindings={name:sha(a.no_slip/name) for name in ('settings.json','summary.json','fields.npz','sources.json','artifact-audit.json')},
        runs=rows,physical_suboff_benchmark_passed=False,grid_convergence_certified=False,
        scope='Same-grid boundary limit consistency; differences from no-slip are not experimental errors')
    (a.output/'comparison.json').write_text(json.dumps(result,indent=2)+'\n');shutil.copy2(__file__,a.output/Path(__file__).name)
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    axes[0].plot([0]+[1/r['friction'] for r in rows],[cd0]+[r['Cd_L2'] for r in rows],'o-')
    axes[0].set(xlabel='Inverse wall friction (0 = no-slip)',ylabel='Cd, reference area L²',title='Boundary-condition limit, not experimental validation')
    axes[1].loglog([r['friction'] for r in rows],[r['Cd_relative_difference_from_no_slip'] for r in rows],'o-',label='Drag difference')
    axes[1].loglog([r['friction'] for r in rows],[r['velocity_relative_difference_from_no_slip'] for r in rows],'s-',label='Velocity L2 difference')
    axes[1].set(xlabel='Wall friction',ylabel='Relative difference from no-slip');axes[1].legend()
    for ax in axes:ax.grid(True)
    fig.savefig(a.output/'boundary-limit.png',dpi=180);fig.savefig(a.output/'boundary-limit.pdf');plt.close(fig);print(result)


if __name__=='__main__':main()

#!/usr/bin/env python3
"""Compare converged native SUBOFF startup grids; no experimental certification."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs',type=Path,nargs='+')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if len(args.runs)<2:parser.error('at least two independently converged grids required')
    if args.output.exists() and any(args.output.iterdir()):parser.error('output must be empty')
    rows=[];settings0=None;curves=[]
    same=('backend','flow','Re_L','L','U','rho','nu','convection_order','body_force',
          'inlet_and_farfield_velocity','wall_velocity','outlet','outlet_backflow_velocity','reference_area')
    for root in args.runs:
        settings=json.loads((root/'settings.json').read_text())
        summary=json.loads((root/'summary.json').read_text())
        audit=json.loads((root/'artifact-audit.json').read_text())
        if not summary['numerically_converged'] or not audit['numerically_converged']:
            raise ValueError(f'nonconverged run cannot enter grid comparison: {root}')
        if settings['backend']!='native_hybrid_cell_and_shared_face_velocity':raise ValueError('native hybrid run required')
        if settings0 is None:settings0=settings
        if any(settings[key]!=settings0[key] for key in same) or settings.get('wall_friction')!=settings0.get('wall_friction'):
            raise ValueError('different physical problems or wall boundary conditions')
        if digest(root/'fields.npz')!=summary['fields_sha256']:raise ValueError('fields changed')
        if digest(root/'input.msh')!=settings['mesh_sha256']:raise ValueError('mesh changed')
        if digest(root/'wall-pressure-traces.npz')!=audit['pressure_trace']['fields_sha256']:
            raise ValueError('pressure traces changed')
        for name,sha in json.loads((root/'sources.json').read_text()).items():
            if digest(root/'source-snapshot'/name)!=sha:raise ValueError('numerical source changed')
        for name,sha in json.loads((root/'postprocessing-sources.json').read_text()).items():
            if digest(root/'postprocessing-source'/name)!=sha:raise ValueError('postprocessor changed')
        with np.load(root/'fields.npz',allow_pickle=False) as fields,np.load(root/'wall-pressure-traces.npz',allow_pickle=False) as trace:
            if str(trace['input_fields_sha256'])!=summary['fields_sha256']:raise ValueError('trace input mismatch')
            wall=fields['patch_labels']=='wall'
            force=-fields['total_traction_integrated'][wall].sum(0)
            cd=float(2*force[0]/(settings['rho']*settings['U']**2*settings['L']**2))
            np.testing.assert_allclose(cd,summary['Cd_L2'],rtol=1e-12)
            np.testing.assert_array_equal(trace['face_indices'],np.flatnonzero(wall))
            volume=float(-np.einsum('ij,ij->',fields['face_centers'][wall],fields['area_vectors'][wall])/3)
            domain=np.stack((fields['vertices'].min(0),fields['vertices'].max(0)))
            if rows:np.testing.assert_allclose(domain,rows[0]['domain_bounds'],rtol=0,atol=1e-12)
            curves.append((fields['face_centers'][wall,0].copy(),2*trace['pressure'].copy()/(settings['rho']*settings['U']**2)))
            rows.append(dict(run=str(root),cells=len(fields['cell_volumes']),faces=len(wall),Cd_L2=cd,
                body_polyhedral_volume=volume,domain_bounds=domain.tolist(),
                minimum_cell_volume=float(fields['cell_volumes'].min()),
                mass_divergence_max=summary['mass_divergence_max'],
                source_bindings={name:digest(root/name) for name in ('settings.json','summary.json','artifact-audit.json',
                    'sources.json','postprocessing-sources.json','input.msh','fields.npz','wall-pressure-traces.npz')}))
    if any(rows[i]['cells']>=rows[i+1]['cells'] for i in range(len(rows)-1)):
        raise ValueError('runs must be ordered by increasing cell count')
    for previous,row in zip(rows,rows[1:]):
        row['Cd_relative_change_from_previous_using_finer_denominator']=abs(row['Cd_L2']-previous['Cd_L2'])/abs(row['Cd_L2'])
    args.output.mkdir(parents=True,exist_ok=True)
    result=dict(Re_L=settings0['Re_L'],runs=rows,wall_Cp_certified=False,grid_convergence_certified=False,
        physical_suboff_benchmark_passed=False,
        interpretation='Changes include geometry and spatial discretization. Two grids or small drag changes do not establish grid convergence. No independent physical reference is used.')
    (args.output/'comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    shutil.copy2(__file__,args.output/Path(__file__).name)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    for row,(x,cp) in zip(rows,curves):axes[0].scatter(x,cp,s=9,label=f"{row['cells']} cells")
    axes[0].set(xlabel='x/L',ylabel='Connected-cell pressure trace Cp',title='All actual wall facets; wall Cp NOT certified')
    axes[0].legend();axes[0].grid(True)
    axes[1].plot([row['cells'] for row in rows],[row['Cd_L2'] for row in rows],'o-')
    axes[1].set(xlabel='Cells',ylabel='Cd with reference area L^2',title=f"Native laminar Re_L={settings0['Re_L']:g}; no reference data")
    axes[1].grid(True)
    fig.savefig(args.output/'grid-comparison.png',dpi=180);fig.savefig(args.output/'grid-comparison.pdf');plt.close(fig)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()

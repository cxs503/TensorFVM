#!/usr/bin/env python3
"""Combine independently audited native SST grids without selecting a favorable grid."""
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


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('runs',nargs='+',type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args()
    if a.output.exists() and any(a.output.iterdir()):p.error('output must be empty')
    rows=[];profiles={};first=None
    keys=('Re_tau','density','half_width','length','span','wall_normal_stretch','body_force','pressure_drop',
          'pressure_variable','total_shear','wall','wall_function','advection','inflow','scalar_correction')
    for root in a.runs:
        cfg=json.loads((root/'settings.json').read_text());audit=json.loads((root/'artifact-audit.json').read_text())
        if not audit['source_snapshot_complete'] or not audit['source_and_field_integrity_verified']:raise ValueError('unverified run')
        if first is None:first=cfg
        if any(cfg[k]!=first[k] for k in keys):raise ValueError('different physical settings')
        if cfg['reference_provenance']['files']!=first['reference_provenance']['files']:raise ValueError('different DNS data')
        for name,digest in json.loads((root/'sources.json').read_text()).items():
            if sha(root/'source-snapshot'/name)!=digest:raise ValueError('modified numerical source')
        for row in json.loads((root/'cases.json').read_text()):
            if row['kind']!='sst':continue
            if not row['numerical_convergence']:raise ValueError('unconverged SST grid')
            file=root/(row['case']+'-fields.npz')
            if sha(file)!=row['fields_sha256']:raise ValueError('modified numerical fields')
            with np.load(file,allow_pickle=False) as f:
                area=np.linalg.norm(f['area_vectors'],axis=1)
                bulk=float(f['flux'][f['outlet']].sum()/area[f['outlet']].sum())
                tau=-float(f['total_traction_integrated'][f['wall'],0].sum()/area[f['wall']].sum())
                cf=2*tau/bulk**2
                np.testing.assert_allclose(cf,row['Cf'],rtol=1e-12)
                y=f['cell_centers'][:,1];unique=np.unique(y)
                u=np.array([np.average(f['velocity'][y==v,0],weights=f['cell_volumes'][y==v]) for v in unique])
                if row['n'] in profiles:raise ValueError('duplicate grid')
                profiles[row['n']]=(unique,u)
            rows.append(dict(n=row['n'],cells=row['cells'],Cf=cf,Cf_relative_error=abs(cf/first['reference_provenance']['Cf']-1),
                velocity_relative_l2=row['velocity_relative_l2'],first_cell_y_plus=row['first_cell_y_plus'],
                run=str(root),fields_sha256=row['fields_sha256'],audit_sha256=sha(root/'artifact-audit.json'),
                settings_sha256=sha(root/'settings.json'),sources_sha256=sha(root/'sources.json')))
    rows.sort(key=lambda r:r['n'])
    for previous,row in zip(rows,rows[1:]):row['Cf_change_from_previous_using_finer_denominator']=abs(row['Cf']-previous['Cf'])/abs(row['Cf'])
    refroot=Path(__file__).resolve().parents[1]/'docs/verification-native-channel-sst/reference'
    reference=first['reference_provenance'];ref=reference['files']['mean']
    if sha(refroot/ref['csv'])!=ref['csv_sha256']:raise ValueError('DNS reference modified')
    dns=np.loadtxt(refroot/ref['csv'],delimiter=',',skiprows=1)
    a.output.mkdir(parents=True,exist_ok=True)
    result=dict(cases=rows,Cf_DNS=reference['Cf'],finest_both_errors_below_3_percent=bool(max(rows[-1]['Cf_relative_error'],rows[-1]['velocity_relative_l2'])<.03),
        grid_convergence_certified=False,model_error_separated_from_discretization=False,physical_suboff_benchmark_passed=False)
    (a.output/'comparison.json').write_text(json.dumps(result,indent=2)+'\n');shutil.copy2(__file__,a.output/Path(__file__).name)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    axes[0].plot(dns[:,1],dns[:,2],'k-',label='Lee–Moser DNS')
    for row in rows:
        y,u=profiles[row['n']];axes[0].plot(y*first['Re_tau'],u,'o-',ms=2,label=f"n={row['n']}")
    axes[0].set(xscale='log',xlabel='y+',ylabel='U+',title='All audited native SST grids');axes[0].grid(True);axes[0].legend()
    axes[1].plot([r['n'] for r in rows],[r['Cf'] for r in rows],'o-',label='Native SST')
    axes[1].axhline(reference['Cf'],color='k',label='DNS')
    axes[1].axhspan(reference['Cf']*.97,reference['Cf']*1.03,color='g',alpha=.1,label='±3% nominal band')
    axes[1].set(xlabel='Wall-normal cells',ylabel='Cf',title='Finest grid FAILS friction accuracy');axes[1].grid(True);axes[1].legend()
    fig.savefig(a.output/'four-grid-comparison.png',dpi=180);fig.savefig(a.output/'four-grid-comparison.pdf');plt.close(fig)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()

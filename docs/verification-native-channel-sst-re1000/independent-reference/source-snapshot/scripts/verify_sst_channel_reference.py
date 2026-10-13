#!/usr/bin/env python3
"""Independent 1D SST diagnostic for native-channel model/discretization separation."""
import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path
import numpy as np
from scipy.interpolate import PchipInterpolator
from tensorfvm.verification.sst_channel_reference import solve_reference
from tensorfvm.verification.native_run import NativeRunEvidence,write_json
from tensorfvm.verification.sst_channel_report import plot_reference


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--levels',type=int,nargs='+',default=[16,32,64,128,512,2048,8192])
    p.add_argument('--reference',type=Path,help='DNS reference directory; frozen with the diagnostic')
    p.add_argument('--model-source',type=Path,required=True)
    p.add_argument('--native-runs',type=Path,nargs='*',default=[Path('docs/verification-native-channel-sst/refinement-v2-interrupted'),Path('docs/verification-native-channel-sst/refinement-n128')])
    a=p.parse_args()
    if any(n<4 for n in a.levels) or a.levels!=sorted(set(a.levels)):p.error('strictly increasing unique levels >=4 required')
    out=a.output;evidence=NativeRunEvidence(out,__file__)
    refroot=a.reference or Path(__file__).resolve().parents[1]/'docs/verification-native-channel-sst/reference'
    meta=json.loads((refroot/'provenance.json').read_text());re_tau=meta['Re_tau_reported']
    for record in meta['files'].values():
        if sha(refroot/record['csv'])!=record['csv_sha256']:raise ValueError('modified DNS reference')
    shutil.copytree(refroot,out/'dns-reference');shutil.copy2(a.model_source,out/'model-source.html')
    write_json(out/'settings.json',dict(levels=a.levels,Re_tau=re_tau,nu=1/re_tau,half_width=1.,density=1.,wall_shear=1.,
        pressure_gradient=-1.,body_force=0.,wall_k=0.,wall_omega='60*nu/(.075*first_cell_distance^2)',
        centerline='zero k/omega derivative; zero total shear',scalar_diffusion='harmonic face conductance; independent tridiagonal solver',
        momentum='exact total-shear identity; eddy-viscosity limiter solved algebraically',
        residual_gate='max per-cell equation defect / local flux+source magnitudes <1e-10',
        update_gate='two consecutive full unrelaxed scalar updates <1e-10',
        dt=.01,relaxation=.5,stretch=3.,production_modules_used=False,DNS_enters_solve=False,
        native_benchmark_gates_replaced=False,physical_suboff_benchmark_passed=False,
        model_source_url='https://tmbwg.github.io/turbmodels/sst.html#sst-2003',model_source_sha256=sha(out/'model-source.html')))
    rows=[];solutions={}
    for n in a.levels:
        tick=time.perf_counter();r=solve_reference(n,re_tau=re_tau)
        row={k:r[k] for k in ('n','converged','Cf','bulk_velocity','first_cell_y_plus','wall_omega')}
        row.update(runtime_seconds=time.perf_counter()-tick,iterations=len(r['history']),last=r['history'][-1],
            Cf_relative_error_DNS=abs(r['Cf']/meta['Cf']-1),
            scalar_ledgers={name:{k:v for k,v in values.items() if not isinstance(v,np.ndarray)} for name,values in r['scalar_residuals'].items()})
        fields={k:v for k,v in r.items() if isinstance(v,np.ndarray)}
        for name,values in r['scalar_residuals'].items():
            fields.update({name+'_'+k:v for k,v in values.items() if isinstance(v,np.ndarray)})
        file=out/f'reference-n{n}-fields.npz';np.savez_compressed(file,**fields);row['fields_sha256']=sha(file)
        write_json(out/f'reference-n{n}-history.json',r['history']);rows.append(row);solutions[n]=r
        write_json(out/'cases.json',rows);print(row,flush=True)
    native=[];finest=solutions[max(solutions)]
    interpolation=PchipInterpolator(np.r_[0,finest['y'],1],np.r_[0,finest['velocity'],finest['face_velocity'][-1]],extrapolate=False)
    for root in a.native_runs:
        cfg=json.loads((root/'settings.json').read_text());audit=json.loads((root/'artifact-audit.json').read_text())
        if not audit['source_snapshot_complete'] or not audit['source_and_field_integrity_verified']:raise ValueError('unverified native run')
        if cfg['Re_tau']!=re_tau or cfg['wall_function'] or cfg['scalar_correction']!='none':raise ValueError('different native model conditions')
        for file,digest in json.loads((root/'sources.json').read_text()).items():
            if sha(root/'source-snapshot'/file)!=digest:raise ValueError('changed native source')
        for case in json.loads((root/'cases.json').read_text()):
            if case['kind']!='sst':continue
            if not case['numerical_convergence']:raise ValueError('unconverged native field')
            path=root/(case['case']+'-fields.npz')
            if sha(path)!=case['fields_sha256']:raise ValueError('changed native field')
            with np.load(path,allow_pickle=False) as f:
                target=interpolation(f['cell_centers'][:,1]);volume=f['cell_volumes']
                error=float(np.sqrt(np.sum(volume*(f['velocity'][:,0]-target)**2)/np.sum(volume*target**2)))
                area=np.linalg.norm(f['area_vectors'],axis=1)
                bulk=float(f['flux'][f['outlet']].sum()/area[f['outlet']].sum())
                shear=-float(f['total_traction_integrated'][f['wall'],0].sum()/area[f['wall']].sum())
                cf=2*shear/bulk**2;np.testing.assert_allclose(cf,case['Cf'],rtol=1e-12)
            native.append(dict(n=case['n'],run=str(root),fields_sha256=sha(path),settings_sha256=sha(root/'settings.json'),
                audit_sha256=sha(root/'artifact-audit.json'),Cf=cf,Cf_relative_difference_from_finest_reference=abs(cf/finest['Cf']-1),
                Cf_relative_difference_from_same_n_reference=abs(cf/solutions[case['n']]['Cf']-1) if case['n'] in solutions else None,
                velocity_relative_l2_from_finest_reference=error))
    for previous,row in zip(rows,rows[1:]):row['Cf_change_from_previous']=abs(row['Cf']-previous['Cf'])/row['Cf']
    write_json(out/'cases.json',rows);write_json(out/'native-comparison.json',native)
    plot_reference(out)
    evidence.verify_sources()
    summary=dict(source_integrity_verified=True,all_reference_numerically_converged=all(r['converged'] for r in rows),
        finest_reference_Cf=finest['Cf'],Cf_DNS=meta['Cf'],finest_Cf_relative_error_DNS=abs(finest['Cf']/meta['Cf']-1),
        last_reference_grid_change=rows[-1].get('Cf_change_from_previous'),physical_suboff_benchmark_passed=False,
        scope='independent SST model diagnostic; does not replace native physical validation or establish universal model accuracy')
    write_json(out/'summary.json',summary);print(summary,flush=True)
    raise SystemExit(0 if summary['all_reference_numerically_converged'] else 2)


if __name__=='__main__':main()

#!/usr/bin/env python3
"""Postprocess native channel fields against independently solved SST references."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
from scipy.interpolate import PchipInterpolator


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def verify_sources(root):
    for name,digest in json.loads((root/'sources.json').read_text()).items():
        if sha(root/'source-snapshot'/name)!=digest:raise ValueError('changed producer source')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--native',type=Path,required=True)
    p.add_argument('--references',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists() and any(a.output.iterdir()):p.error('output must be empty')
    root=a.native;cfg=json.loads((root/'settings.json').read_text());audit=json.loads((root/'artifact-audit.json').read_text())
    if not audit['source_snapshot_complete'] or not audit['source_and_field_integrity_verified']:raise ValueError('verified native run required')
    if cfg['wall_function'] or cfg['scalar_correction']!='none':raise ValueError('same wall-resolved SST formulation required')
    verify_sources(root);references={};excluded=[];bindings={}
    for directory in a.references:
        settings=json.loads((directory/'settings.json').read_text());s=json.loads((directory/'summary.json').read_text())
        check=json.loads((directory/'artifact-audit.json').read_text())
        if not s['source_integrity_verified'] or not check['source_and_field_hashes_verified']:raise ValueError('unverified independent run')
        if settings['Re_tau']!=cfg['Re_tau']:raise ValueError('different Reynolds numbers')
        if sha(root/'source-snapshot/src/tensorfvm/native_sst3d.py')!=check['audit_sources']['src/tensorfvm/native_sst3d.py']:
            raise ValueError('native closure differs from the independently audited closure')
        verify_sources(directory)
        bindings[str(directory)]={name:sha(directory/name) for name in ('settings.json','summary.json','cases.json','artifact-audit.json','sources.json')}
        for case in json.loads((directory/'cases.json').read_text()):
            if not case['converged']:
                excluded.append(dict(n=case['n'],run=str(directory),reason='failed declared relative reference criterion'));continue
            field=directory/f"reference-n{case['n']}-fields.npz"
            if sha(field)!=case['fields_sha256']:raise ValueError('changed reference field')
            with np.load(field,allow_pickle=False) as f:references[case['n']]=(case,{k:f[k].copy() for k in f.files})
    if not references:raise ValueError('no converged independent reference')
    fine,fields=references[max(references)]
    interpolator=PchipInterpolator(np.r_[0,fields['y'],1],np.r_[0,fields['velocity'],fields['face_velocity'][-1]],extrapolate=False)
    rows=[]
    for case in json.loads((root/'cases.json').read_text()):
        if case['kind']!='sst':continue
        if not case['numerical_convergence']:raise ValueError('native flow not converged')
        path=root/(case['case']+'-fields.npz')
        if sha(path)!=case['fields_sha256']:raise ValueError('changed native field')
        with np.load(path,allow_pickle=False) as f:
            volume=f['cell_volumes'];area=np.linalg.norm(f['area_vectors'],axis=1)
            bulk=float(f['flux'][f['outlet']].sum()/area[f['outlet']].sum())
            shear=-float(f['total_traction_integrated'][f['wall'],0].sum()/area[f['wall']].sum())
            cf=2*shear/bulk**2;np.testing.assert_allclose(cf,case['Cf'],rtol=1e-12)
            target=interpolator(f['cell_centers'][:,1])
            if not np.isfinite(target).all():raise ValueError('reference interpolation would extrapolate')
            error=float(np.sqrt(np.sum(volume*(f['velocity'][:,0]-target)**2)/np.sum(volume*target**2)))
        rows.append(dict(n=case['n'],Cf=cf,fields_sha256=sha(path),Cf_relative_error_DNS=case['Cf_relative_error'],
            native_accuracy_passed=case['accuracy_passed'],Cf_relative_difference_same_n=abs(cf/references[case['n']][0]['Cf']-1) if case['n'] in references else None,
            Cf_relative_difference_finest_accepted_reference=abs(cf/fine['Cf']-1),velocity_relative_l2_finest_accepted_reference=error))
    result=dict(Re_tau=cfg['Re_tau'],native_bindings={name:sha(root/name) for name in ('settings.json','summary.json','cases.json','sources.json','artifact-audit.json')},
        reference_bindings=bindings,finest_accepted_reference_n=fine['n'],finest_accepted_reference_Cf=fine['Cf'],excluded_reference_cases=excluded,cases=rows,
        native_absolute_gates_replaced=False,physical_suboff_benchmark_passed=False,
        scope='Independent model/discretization comparison, not physical benchmark qualification; reference absolute residual limits remain applicable')
    a.output.mkdir(parents=True,exist_ok=True);(a.output/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    shutil.copy2(__file__,a.output/Path(__file__).name);print(result)


if __name__=='__main__':main()

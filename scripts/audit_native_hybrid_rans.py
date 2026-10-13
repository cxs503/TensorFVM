#!/usr/bin/env python3
"""Audit saved coupling fields, pressure/stress conventions and scalar ledgers."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path)
    root=p.parse_args().directory
    for name,digest in json.loads((root/'sources.json').read_text()).items():
        assert hashlib.sha256((root/'source-snapshot'/name).read_bytes()).hexdigest()==digest
    summary=json.loads((root/'summary.json').read_text());assert summary['source_integrity_verified']
    cases=json.loads((root/'cases.json').read_text());audit=[];comparison={}
    for row in cases:
        path=root/(row['case']+'-fields.npz')
        assert hashlib.sha256(path.read_bytes()).hexdigest()==row['fields_sha256']
        with np.load(path,allow_pickle=False) as f:
            owner=f['owner'];neighbor=f['neighbor'];interior=neighbor>=0;volume=f['cell_volumes']
            def divergence(flux):
                result=np.zeros((len(volume),)+flux.shape[1:])
                np.add.at(result,owner,flux);np.add.at(result,neighbor[interior],-flux[interior])
                return result
            mass=float(np.max(np.abs(divergence(f['flux'])/volume)))
            momentum=float(np.max(np.abs(f['momentum_flux_integrated'][~interior].sum(0))))
            np.testing.assert_allclose(f['physical_pressure'],f['modified_pressure']-2*f['k']/3,atol=1e-14)
            np.testing.assert_allclose(f['pressure_traction_integrated'],-f['physical_pressure'][owner,None]*f['area_vectors'],atol=1e-14)
            np.testing.assert_allclose(f['isotropic_turbulent_traction_integrated'],-2*f['k'][owner,None]*f['area_vectors']/3,atol=1e-14)
            parts=sum(f[k] for k in ('pressure_traction_integrated','isotropic_turbulent_traction_integrated','deviatoric_traction_integrated','numerical_traction_correction'))
            np.testing.assert_allclose(parts,f['total_traction_integrated'],atol=1e-14)
            residuals={};balances={}
            for name in ('k','omega'):
                assert np.min(f[name])>=0 and np.min(f[name+'_faces'])>=0
                if name=='omega':assert min(f[name].min(),f[name+'_faces'].min())>0
                residuals[name]=float(np.linalg.norm(f[name+'_steady_free_residual'])/f[name+'_residual_scale'])
                np.testing.assert_allclose(residuals[name],row['current'][name],rtol=1e-12,atol=1e-15)
                balance=divergence(f[name+'_total_flux'])-volume*f[name+'_source']
                balances[name]=float(np.linalg.norm(balance)/f[name+'_residual_scale'])
                assert balances[name]<1e-8 and residuals[name]<1e-8
            assert mass<1e-8 and momentum<1e-7
            if row['kind']=='laminar_limit':
                np.testing.assert_allclose(f['velocity'][:,0],.3*f['cell_centers'][:,1],atol=1e-10)
                wall=(neighbor<0)&(f['face_centers'][:,1]>1-1e-10)
                np.testing.assert_allclose(f['total_traction_integrated'][wall,0].sum(),.03,atol=1e-10)
            if row['kind']=='reservoir' and row['n']==cases[0]['n']:
                comparison[row['case'].split('-')[-1]]={k:f[k].copy() for k in ('velocity','modified_pressure','k','omega','nu_t')}
        assert row['converged'] and max(row['current'].values())<1e-8 and max(row['final_changes'].values())<1e-8
        audit.append(dict(case=row['case'],divergence=mass,global_momentum=momentum,current_scalar_residual=residuals,scalar_shared_balance=balances))
    difference={k:float(np.max(np.abs(comparison['newton'][k]-comparison['simplec'][k]))) for k in comparison['newton']}
    assert max(difference.values())<1e-7
    failure=json.loads((root/'rejected-triangle-wall.json').read_text())
    with np.load(root/'rejected-triangle-wall-candidate.npz') as f:
        assert f['face_values'].min()<0 and failure['rejected']
        np.testing.assert_allclose(f['face_values'].min(),failure['face_min'])
    output=dict(source_and_field_integrity_verified=True,cases=audit,
        simplec_newton_difference=difference,triangulated_wall_counterexample_confirmed=True,
        physical_suboff_benchmark_passed=False,turbulent_accuracy_verified=False,
        audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (root/'artifact-audit.json').write_text(json.dumps(output,indent=2)+'\n')
    print('Verified',len(cases),'coupling controls and rejected wall candidate; physical benchmark remains unverified.')


if __name__=='__main__':main()

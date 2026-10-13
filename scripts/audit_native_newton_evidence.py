#!/usr/bin/env python3
"""Independently audit saved original fields, source hashes, and refinement gates."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('report',type=Path)
    root=parser.parse_args().report;all_rows={};comparison=[]
    names=['potential','direct-abc','abc-amg','kovasznay','kovasznay-amg']
    if (root/'kovasznay-fine'/'summary.json').exists():names.append('kovasznay-fine')
    for name in names:
        directory=root/name
        for path,expected in json.loads((directory/'sources.json').read_text()).items():
            assert hashlib.sha256((directory/'source-snapshot'/path).read_bytes()).hexdigest()==expected
        if (directory/'termination.json').exists():
            assert json.loads((directory/'termination.json').read_text())['source_integrity_verified_before_stop']
        else:assert json.loads((directory/'summary.json').read_text())['source_integrity_verified']
        rows=json.loads((directory/'cases.json').read_text());all_rows[name]=rows
        for row in rows:
            path=directory/(row['case']+'-fields.npz')
            with np.load(path,allow_pickle=False) as f:
                v=f['cell_volumes'];u=f['velocity'];ue=f['exact_velocity'];p=f['pressure'];pe=f['exact_pressure']
                velocity=np.sqrt(np.sum(v[:,None]*(u-ue)**2)/np.sum(v[:,None]*ue**2))
                fixed=np.sqrt(np.sum(v*(p-pe)**2)/np.sum(v*pe**2))
                p=p-np.average(p,weights=v);pe=pe-np.average(pe,weights=v)
                centered=np.sqrt(np.sum(v*(p-pe)**2)/np.sum(v*pe**2))
                balance=np.max(np.abs(f['momentum_flux_integrated'][f['neighbor']<0].sum(0)-np.sum(v[:,None]*f['body_force'],axis=0)))
            for actual,key in ((velocity,'velocity_relative_l2'),(fixed,'pressure_fixed_gauge_relative_l2'),
                               (centered,'pressure_mean_free_relative_l2'),(balance,'global_momentum_balance_max')):
                np.testing.assert_allclose(actual,row[key],atol=1e-12,rtol=1e-10)
            ledger=max(row[k] for k in ('energy_balance_error','convection_energy_identity_error',
                'interface_traction_defect_max','local_momentum_balance_max','global_momentum_balance_max'))
            pressure=fixed if name.startswith('kovasznay') else centered
            assert row['analytic_accuracy_and_ledger_passed']==bool(row['numerically_converged'] and max(velocity,pressure)<.03 and ledger<1e-7)
            row['fields_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
            for iteration in json.loads((directory/(row['case']+'-history.json')).read_text()):
                if iteration.get('accepted') and iteration['iteration']>0:
                    at_tolerance=max(iteration[k] for k in ('current_momentum_residual','divergence_max',
                        'gauge_error','velocity_change','pressure_change'))<1e-8
                    armijo=iteration['merit_after']<=(1-1e-4*iteration['step_length'])*iteration['merit_before']
                    assert armijo or at_tolerance,iteration

    for old,new,names in (('direct-abc','abc-amg',['abc-n8-mu0.1']),
                          ('kovasznay','kovasznay-amg',['kovasznay-n64-mu0.025'])):
        for name in names:
            with np.load(root/old/(name+'-fields.npz')) as a,np.load(root/new/(name+'-fields.npz')) as b:
                errors={k:float(np.max(np.abs(a[k]-b[k]))) for k in ('velocity','pressure','face_velocity','flux')}
            assert max(errors.values())<1e-7,errors
            comparison.append(dict(case=name,compared_runs=[old,new],maximum_absolute_differences=errors,tolerance=1e-7,passed=True))
    abc=all_rows['abc-amg'];kova=all_rows['kovasznay']+all_rows['kovasznay-amg']+all_rows.get('kovasznay-fine',[])
    def gate(rows,pkey):
        unique={row['n']:row for row in rows};rows=[unique[n] for n in sorted(unique)]
        return dict(levels=[r['n'] for r in rows],cells=[r['cells'] for r in rows],
            finest_accuracy_and_ledger_passed=rows[-1]['analytic_accuracy_and_ledger_passed'],
            all_grids_numerically_converged=all(r['numerically_converged'] for r in rows),
            pressure_metric=pkey,velocity_errors=[r['velocity_relative_l2'] for r in rows],pressure_errors=[r[pkey] for r in rows],
            errors_decrease_with_refinement=all(a[k]>b[k] for a,b in zip(rows[:-1],rows[1:]) for k in ('velocity_relative_l2',pkey)))
    output=dict(artifact_consistency_verified=True,physical_suboff_benchmark_passed=False,
        abc_3d_refinement=gate(abc,'pressure_mean_free_relative_l2'),
        unforced_kovasznay_2d_refinement=gate(kova,'pressure_fixed_gauge_relative_l2'),
        low_viscosity_potential_all_grids_passed=all(r['analytic_accuracy_and_ledger_passed'] for r in all_rows['potential']),
        linear_algorithm_comparisons=comparison,cases=all_rows,
        audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (root/'artifact-audit.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print('Verified',sum(map(len,all_rows.values())),'saved cases; physical SUBOFF remains unverified.')


if __name__=='__main__':main()

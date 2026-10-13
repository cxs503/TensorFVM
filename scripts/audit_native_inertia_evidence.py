#!/usr/bin/env python3
"""Recompute reported errors and global momentum from saved original fields."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('report',type=Path);args=p.parse_args()
    root=args.report;records=[]
    for name in ('abc','potential','viscosity','simplec'):
        directory=root/name
        hashes=json.loads((directory/'sources.json').read_text())
        for path,expected in hashes.items():
            assert hashlib.sha256((directory/'source-snapshot'/path).read_bytes()).hexdigest()==expected,path
        summary=json.loads((directory/'summary.json').read_text())
        assert summary['source_integrity_verified']
        assert not summary['physical_suboff_benchmark_passed']
        for row in json.loads((directory/'cases.json').read_text()):
            path=directory/(row['case']+'-fields.npz')
            with np.load(path,allow_pickle=False) as f:
                v=f['cell_volumes'];u=f['velocity'];ue=f['exact_velocity']
                pressure=f['pressure'];exact=f['exact_pressure']
                pressure=pressure-np.sum(v*pressure)/v.sum()
                exact=exact-np.sum(v*exact)/v.sum()
                velocity_error=np.sqrt(np.sum(v[:,None]*(u-ue)**2)/np.sum(v[:,None]*ue**2))
                pressure_error=np.sqrt(np.sum(v*(pressure-exact)**2)/np.sum(v*exact**2))
                force=np.sum(v[:,None]*f['body_force'],axis=0)
                balance=np.max(np.abs(f['momentum_flux_integrated'][f['neighbor']<0].sum(0)-force))
                for actual,key in ((velocity_error,'velocity_relative_l2'),(pressure_error,'pressure_mean_free_relative_l2'),(balance,'global_momentum_balance_max')):
                    np.testing.assert_allclose(actual,row[key],rtol=1e-10,atol=1e-12)
                assert all(np.isfinite(f[k]).all() for k in ('velocity','pressure','flux','momentum_flux_integrated'))
            assert row['analytic_accuracy_and_ledger_passed']==bool(row['numerically_converged'] and max(velocity_error,pressure_error)<.03 and max(row[k] for k in ('energy_balance_error','convection_energy_identity_error','convection_global_balance_error','interface_traction_defect_max','local_momentum_balance_max','global_momentum_balance_max'))<1e-7)
            records.append(dict(run=name,case=row['case'],fields_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                velocity_relative_l2=float(velocity_error),pressure_mean_free_relative_l2=float(pressure_error),
                global_momentum_balance_max=float(balance),numerically_converged=row['numerically_converged']))
    output=dict(artifact_consistency_verified=True,physical_suboff_benchmark_passed=False,
                audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),cases=records)
    (root/'artifact-audit.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print(f'Verified {len(records)} saved cases; accuracy failures retained, no physical SUBOFF pass.')


if __name__=='__main__':main()

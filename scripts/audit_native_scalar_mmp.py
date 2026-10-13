#!/usr/bin/env python3
"""Audit completed and rejected scalar correction artifacts without relabeling failures."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path)
    root=p.parse_args().directory;output={}
    for name in ('scalar','coupling-partial','controls-partial','failures'):
        d=root/name
        for source,digest in json.loads((d/'sources.json').read_text()).items():
            assert hashlib.sha256((d/'source-snapshot'/source).read_bytes()).hexdigest()==digest
        assert json.loads((d/'summary.json').read_text())['source_integrity_verified']
        rows=json.loads((d/'cases.json').read_text())
        for row in rows:
            if name=='scalar':path=d/f"scalar-n{row['n']}-fields.npz"
            elif name=='failures':path=d/(row['case']+'-candidate.npz')
            else:path=d/(row['case']+'-fields.npz')
            if 'fields_sha256' in row:assert hashlib.sha256(path.read_bytes()).hexdigest()==row['fields_sha256']
            with np.load(path,allow_pickle=False) as f:
                internal=f['neighbor']>=0
                if name=='coupling-partial':
                    np.testing.assert_allclose(f['physical_pressure'],f['modified_pressure']-2*f['k']/3,atol=1e-14)
                    for scalar in ('k','omega'):
                        value=np.linalg.norm(f[scalar+'_steady_free_residual'])/f[scalar+'_residual_scale']
                        np.testing.assert_allclose(value,row['current'][scalar],atol=1e-14)
                        assert value<1e-8 and min(f[scalar].min(),f[scalar+'_faces'].min())>=0
                    assert row['converged']
                else:
                    balance=(f['storage']+f['destruction']-f['production']).copy()
                    np.add.at(balance,f['owner'],f['total_flux'])
                    np.add.at(balance,f['neighbor'][internal],-f['total_flux'][internal])
                    np.testing.assert_allclose(balance,f['cell_balance'],atol=1e-12)
                    assert f['correction_energy']>=0 if 'correction_energy' in f else True
                    if name=='failures':
                        assert row['rejected'] and not bool(f['converged'])
                        assert max(row['current_nonlinear_residual'],row['change'])>=1e-12
                        history=json.loads((d/(row['case']+'-history.json')).read_text())
                        assert len(history)==600
                    elif name=='controls-partial' and row['case']=='harmonic-nonlinear_mmp':
                        assert min(f['cell_values'].min(),f['face_values'].min())>=0
                        assert max(f['cell_values'].max(),f['face_values'].max())<=1
        output[name]=dict(source_hashes_verified=True,saved_cases=len(rows))
    output.update(physical_suboff_benchmark_passed=False,all_coupling_cases_passed=False,
                  actual_suboff_layer_correction_converged=False,
                  audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (root/'artifact-audit.json').write_text(json.dumps(output,indent=2)+'\n')
    print('Verified saved fields and unaccepted candidates; SUBOFF and full coupling remain unverified.')


if __name__=='__main__':main()

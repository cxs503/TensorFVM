#!/usr/bin/env python3
"""Recompute scalar errors and finite-volume balances from frozen raw fields."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path)
    root=p.parse_args().directory
    hashes=json.loads((root/'sources.json').read_text())
    for name,expected in hashes.items():
        assert hashlib.sha256((root/'source-snapshot'/name).read_bytes()).hexdigest()==expected
    assert json.loads((root/'summary.json').read_text())['source_integrity_verified']
    cases=json.loads((root/'cases.json').read_text());rows=[]
    for case in cases:
        path=root/f"scalar-n{case['n']}-fields.npz"
        assert hashlib.sha256(path.read_bytes()).hexdigest()==case['fields_sha256']
        with np.load(path,allow_pickle=False) as f:
            v=f['cell_volumes'];x=f['cell_centers'];phi=f['cell_values']
            exact=1+np.sum(x*x,axis=1)
            source=2*x @ np.array([.6,-.3,.2])-(6+8*x[:,0])+.7*exact
            np.testing.assert_allclose(f['exact'],exact,atol=1e-14)
            np.testing.assert_allclose(f['source'],source,atol=1e-14)
            err=float(np.sqrt(np.sum(v*(phi-exact)**2)/np.sum(v*exact**2)))
            total=f['advective_flux']-f['diffusive_flux']
            np.testing.assert_allclose(f['total_flux'],total,atol=1e-14)
            local=v*(.7*phi-source);internal=f['neighbor']>=0
            np.add.at(local,f['owner'],total)
            np.add.at(local,f['neighbor'][internal],-total[internal])
            interface=total[internal]+f['neighbor_total_flux'][internal]
            balance=float(total[~internal].sum()+np.sum(v*(.7*phi-source)))
            np.testing.assert_allclose(local,f['cell_balance'],atol=1e-13)
            np.testing.assert_allclose(err,case['relative_l2'],rtol=1e-12)
            assert max(np.max(np.abs(local)),np.max(np.abs(interface)),abs(balance))<1e-8
            rows.append(dict(n=case['n'],relative_l2=err,global_balance=balance,
                             local_max=float(np.max(np.abs(local))),interface_max=float(np.max(np.abs(interface)))))
    assert all(a['relative_l2']>b['relative_l2'] for a,b in zip(rows[:-1],rows[1:]))
    assert rows[-1]['relative_l2']<.03
    output=dict(raw_fields_verified=True,source_hashes_verified=True,cases=rows,
        coupled_rans_verified=False,physical_suboff_benchmark_passed=False,
        audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (root/'artifact-audit.json').write_text(json.dumps(output,indent=2)+'\n')
    print(f'Verified {len(rows)} scalar grids; integrated RANS and physical SUBOFF unverified.')


if __name__=='__main__':main()

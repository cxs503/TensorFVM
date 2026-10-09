"""Audit stored LBM profiles, declared SI scales, file hashes and pass decisions."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from compare_lbm_channel_reference import map_case


def audit(directory):
    summary = json.loads((directory/'comparison.json').read_text())
    if summary['thresholds'] != {'velocity_l2_relative':.03, 'pressure_gradient_relative':.03, 'require_runner_steady':True}:
        raise ValueError('Reference acceptance thresholds changed')
    for case in summary['cases']:
        path = directory/f"H{case['ny']}-raw.json"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != summary['artifact_sha256'][path.name]:
            raise ValueError(f'Raw artifact changed: {path.name}')
        raw = json.loads(path.read_text())
        if not raw['finite'] or not np.isclose(raw['Re'], 1., atol=1e-14, rtol=0):
            raise ValueError('Invalid physical reference')
        expected = map_case(raw)
        for key, value in expected.items():
            if isinstance(value, (bool,int)):
                if case[key] != value:
                    raise ValueError(f'Decision mismatch: {key}')
            elif not np.allclose(case[key],value,rtol=1e-13,atol=1e-15):
                raise ValueError(f'SI reconstruction mismatch: {key}')
    if summary['passed'] != all(c['passed'] for c in summary['cases']):
        raise ValueError('Summary pass mismatch')
    return {'raw_reference_audit_passed':True,'reference_passed':summary['passed'],
            'heights':[c['ny'] for c in summary['cases']],
            'scope':'stored profiles and pressure diagnostics, not full fluid fields'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,default=Path('docs/suite-channel/lbm-extended-reference'))
    args=parser.parse_args()
    print(json.dumps(audit(args.directory)))


if __name__=='__main__':
    main()

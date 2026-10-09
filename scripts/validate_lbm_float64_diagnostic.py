"""Recompute the original monitor gate and SI diagnostics from float64 evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from compare_lbm_channel_reference import map_case


def audit(directory):
    summary=json.loads((directory/'comparison.json').read_text())
    checks=[]
    for case in summary['cases']:
        path=directory/f"H{case['ny']}-raw.json"
        if hashlib.sha256(path.read_bytes()).hexdigest()!=summary['artifact_sha256'][path.name]:
            raise ValueError('Raw diagnostic artifact changed')
        raw=json.loads(path.read_text());hist=raw['umax_history']
        if raw['storage_dtype']!='torch.float64' or len(hist)*200!=raw['n_steps']:
            raise ValueError('Invalid monitor coverage/dtype')
        drift=[]
        for i in range(9,len(hist)):
            recent=hist[i-9:i+1]
            drift.append((max(recent)-min(recent))/max(abs(sum(recent)/10),1e-12))
            if (i+1)*200>=4000 and drift[-1]<1e-5 and i!=len(hist)-1:
                raise ValueError('Runner should have stopped earlier')
        expected_steady=bool(raw['n_steps']>=4000 and drift[-1]<1e-5)
        if raw['steady']!=expected_steady or not np.isclose(drift[-1],raw['last_monitor_relative_drift'],atol=1e-16,rtol=1e-13):
            raise ValueError('Monitor convergence decision mismatch')
        rho=np.asarray(raw['final_rho']);ux=np.asarray(raw['final_ux']);uy=np.asarray(raw['final_uy'])
        if rho.shape!=(raw['ny'],raw['nx']) or ux.shape!=rho.shape or uy.shape!=rho.shape:
            raise ValueError('Final fluid field shape mismatch')
        if not np.isfinite(np.stack([rho,ux,uy])).all() or (rho<=0).any():
            raise ValueError('Invalid final fluid fields')
        density_profile=rho[raw['ny']//2]
        if not np.array_equal(density_profile,np.asarray(raw['rho_profile_x'])):
            raise ValueError('Density field profile mismatch')
        slope=(density_profile[-1]-density_profile[0])/(raw['nx']-1)
        if not np.isclose(slope,raw['rho_slope_meas'],rtol=1e-13,atol=1e-16):
            raise ValueError('Pressure slope mismatch')
        for key,value in map_case(raw).items():
            if not np.allclose(case[key],value,rtol=1e-13,atol=1e-15):
                raise ValueError(f'SI summary mismatch: {key}')
        checks.append({'height':case['ny'],'steady':expected_steady,'last_drift':drift[-1],'field_and_si_audit':True})
    if summary['passed']!=all(c['passed'] for c in summary['cases']):
        raise ValueError('Summary decision mismatch')
    return {'audit_passed':True,'reference_passed':summary['passed'],'cases':checks,
            'scope':'actual history steady criterion and stored final macroscopic fields; populations/time-average samples not stored'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,default=Path('docs/suite-channel/lbm-float64-diagnostic'))
    print(json.dumps(audit(parser.parse_args().directory),indent=2))


if __name__=='__main__':
    main()

#!/usr/bin/env python3
"""Compare actual cold/warm scalar starts with identical RANS inputs and gates."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from tensorfvm.verification.native_run import NativeRunEvidence,write_json


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('cold',type=Path);p.add_argument('warm',type=Path)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    runs=[]
    for root in (a.cold,a.warm):
        cfg=json.loads((root/'settings.json').read_text());summary=json.loads((root/'summary.json').read_text())
        if not summary['startup_steps_completed'] or not summary['source_integrity_verified']:
            raise ValueError('completed source-verified run required')
        if digest(root/'fields.npz')!=summary['fields_sha256']:raise ValueError('changed fields')
        for name,sha in json.loads((root/'sources.json').read_text()).items():
            if digest(root/'source-snapshot'/name)!=sha:raise ValueError('changed producer source')
        for prefix,key in (('initial-','initial_bindings'),('restart-','restart_bindings')):
            for name,sha in cfg[key].items():
                if digest(root/(prefix+name))!=sha:raise ValueError('changed input')
        runs.append((cfg,summary))
    cold,warm=(r[0] for r in runs)
    ignored={'initial_laminar_run','restart_run','scalar_warm_start'}
    different={key for key in cold.keys()|warm.keys() if key not in ignored and cold.get(key)!=warm.get(key)}
    if different:raise ValueError('incomparable settings: '+str(sorted(different)))
    if cold.get('scalar_warm_start',False) or not warm['scalar_warm_start']:
        raise ValueError('expected cold then warm run')
    fields={}
    with np.load(a.cold/'fields.npz') as x,np.load(a.warm/'fields.npz') as y:
        for key in ('velocity','face_velocity','modified_pressure','physical_pressure','k','omega','k_faces','omega_faces','nu_t'):
            left=np.asarray(x[key],np.longdouble);right=np.asarray(y[key],np.longdouble)
            if left.shape!=right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
                raise ValueError('invalid comparison fields')
            delta=left-right
            fields[key]=dict(max_abs=float(np.max(abs(delta))),relative_l2=float(np.linalg.norm(delta)/max(np.linalg.norm(left),1e-30)))
    with np.load(a.cold/'initial-turbulence.npz') as x,np.load(a.warm/'initial-turbulence.npz') as y:
        for key in ('k','omega','k_faces','omega_faces'):np.testing.assert_array_equal(x[key],y[key])
    evidence=NativeRunEvidence(a.output,__file__)
    output=dict(scope=__doc__,same_settings_except_scalar_initialization=True,initial_turbulence_exactly_equal=True,
        runs=[dict(path=str(root),bindings={name:digest(root/name) for name in ('settings.json','summary.json','sources.json','fields.npz','initial-turbulence.npz')},summary=run[1])
              for root,run in zip((a.cold,a.warm),runs)],fields=fields,
        elapsed_time_ratio=runs[0][1]['runtime_seconds']/runs[1][1]['runtime_seconds'],
        timing_scope='observed elapsed times; concurrent system load was not controlled; not a hardware speedup benchmark',
        numerical_qualification=False,physical_suboff_benchmark_passed=False)
    evidence.verify_sources();write_json(a.output/'comparison.json',output);print(json.dumps(output,indent=2))


if __name__=='__main__':main()

"""Run unmodified TensorLBM verified channel; map its independent result to SI."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lbm-repo', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('docs/suite-channel/lbm-reference'))
    args = parser.parse_args()
    repo = args.lbm_repo.resolve()
    source = repo/'benchmarks/verified/poiseuille_2d/run.py'
    sys.path.insert(0, str(repo/'src'))
    spec = importlib.util.spec_from_file_location('suite_lbm_reference', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    cases = []
    for height in (12,24,48):
        raw = module.run_case(height, .8, .15/height, 'bgk', 4000, 30000,
                              str(args.output/f'H{height}-raw.json'), compile_mode=None)
        dx = .02/height
        dt = .1*dx**2/1e-5
        velocity_scale = dx/dt
        reference_density = 1000.
        pressure_scale = reference_density*velocity_scale**2/3
        slope = raw['rho_slope_meas']*pressure_scale/dx
        profile = np.asarray(raw['u_profile'])*velocity_scale
        y = (np.arange(height)+.5)*dx
        exact = 6*.0005*(y/.02)*(1-y/.02)
        error = float(np.linalg.norm(profile-exact)/np.linalg.norm(exact))
        cases.append({'ny':height, 'nx':3*height, 'reference_length_m':.06,
                      'dx_m':dx,'dt_s':dt,'bulk_velocity_reference_m_s':.0005,
                      'profile_y_m':y.tolist(),'profile_u_m_s':profile.tolist(),
                      'velocity_l2_relative_error':error, 'pressure_gradient_pa_m':slope,
                      'pressure_gradient_relative_error':abs(slope+.15)/.15,
                      'steady':raw['steady'],'n_steps':raw['n_steps'],
                      'passed':bool(raw['finite'] and raw['steady'] and error<.03
                                    and abs(slope+.15)/.15<.03)})
    summary = {'schema':'tensor-suite.channel-reference-comparison/1',
               'lbm_git_base':subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),
               'lbm_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
               'scope':'independent common SI analytic verification only: FVM L/H=6 uniform velocity inlet; LBM L/H=3 Zou-He pressure inlet/outlet, weakly compressible BGK; NOT same boundary problem or FSI validation',
               'pressure_note':'upstream defines pressure length nx, but boundary node distance nx-1; measured SI gradient retained without correction',
               'physical_reference':{'height_m':.02,'density_kg_m3':1000.,'nu_m2_s':1e-5,'bulk_velocity_m_s':.0005},
               'lbm_fields_scope':'upstream raw profiles and pressure diagnostics; full 2-D LBM field not exported by this runner',
               'cases':cases,'passed':all(c['passed'] for c in cases)}
    (args.output/'comparison.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'passed':summary['passed'],'cases':[{k:v for k,v in c.items() if not k.startswith('profile')} for c in cases]},indent=2))
    return 0 if summary['passed'] else 2


if __name__=='__main__':
    raise SystemExit(main())

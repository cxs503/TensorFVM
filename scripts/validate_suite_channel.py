"""Reconstruct SI channel diagnostics from stored staggered fields without solving."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from tensorfvm.benchmark_suite_channel import channel_metrics
from tensorfvm.solver import SolverConfig, SolverResult


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=Path('docs/suite-channel'))
    args = parser.parse_args()
    summary = json.loads((args.directory/'benchmark.json').read_text())
    for path, digest in summary['artifact_sha256'].items():
        if hashlib.sha256((args.directory/path).read_bytes()).hexdigest() != digest:
            raise ValueError(f'artifact changed: {path}')
    verified = []
    for case in summary['cases']:
        directory = args.directory/f"{case['nx']}x{case['ny']}"
        fields = json.loads((directory/'si-fields.json').read_text())
        def tensor(name):
            return torch.tensor(fields[name], dtype=torch.float64)
        c = SolverConfig(**fields['config'])
        r = SolverResult(c,tensor('u_faces_m_s'),tensor('v_faces_m_s'),tensor('p_pa'),
                         torch.ones((c.ny,c.nx),dtype=torch.bool),tensor('x_m'),tensor('y_m'),
                         json.loads((directory/'history.json').read_text()),case['converged'])
        metrics, lower, upper = channel_metrics(r)
        for key in ('mass_inlet_kg_s_per_m_depth','mass_outlet_kg_s_per_m_depth',
                    'max_cell_divergence_s_inv','pressure_gradient',
                    'developed_wall_force_on_fluid_n_per_m_depth','wall_shear_relative_linf_error'):
            if not abs(metrics[key]-case[key]) <= 1e-12*max(1,abs(case[key])):
                raise ValueError(f'metric mismatch {directory}: {key}')
        for k, value in zip(('u_m_s','v_m_s'),r.cell_center_velocity()):
            torch.testing.assert_close(tensor(k),value,rtol=1e-12,atol=1e-14)
        torch.testing.assert_close(tensor('lower_wall_traction_on_fluid_pa'),lower)
        torch.testing.assert_close(tensor('upper_wall_traction_on_fluid_pa'),upper)
        verified.append(f'{c.nx}x{c.ny}')
    print(json.dumps({'raw_field_checks_passed':True,'grids':verified}))


if __name__=='__main__':
    main()

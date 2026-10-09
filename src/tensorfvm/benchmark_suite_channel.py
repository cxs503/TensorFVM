"""SI channel export for cross-solver analytic verification, not an FSI backend."""
from dataclasses import asdict
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import torch

from .benchmark import compare_poiseuille
from .cylinder import export_result
from .solver import SimpleSolver, SolverConfig


REFERENCE = {
    'height_m': .02, 'length_m': .12, 'density_kg_m3': 1000.,
    'kinematic_viscosity_m2_s': 1e-5, 'bulk_velocity_m_s': .0005,
    'reynolds_height': 1., 'pressure_gradient_pa_m': -.15,
    'wall_shear_magnitude_pa': .0015,
}


def channel_metrics(result):
    """Measure physical face fluxes and wall diffusion from the computed fields."""
    c = result.config
    dx, dy = c.length/c.nx, c.height/c.ny
    u, v = result.cell_center_velocity()
    region = (result.x >= 3*c.height) & (result.x <= 5*c.height)
    # Same half-cell no-slip stencil as momentum assembly, with force ON fluid.
    lower = -c.viscosity * u[0] / (.5*dy)
    upper = -c.viscosity * u[-1] / (.5*dy)
    report = compare_poiseuille(result)
    inlet = float(c.density*dy*result.u[:, 0].sum())
    outlet = float(c.density*dy*result.u[:, -1].sum())
    divergence = (result.u[:, 1:]-result.u[:, :-1])/dx + (result.v[1:]-result.v[:-1])/dy
    length = int(region.sum())*dx
    pressure_force = -report['pressure_gradient']*c.height*length
    wall_force = float((lower[region]+upper[region]).sum()*dx)
    shear_error = float(torch.max(torch.abs(torch.cat((lower[region], upper[region])) + .0015))/.0015)
    report.update({
        'mass_inlet_kg_s_per_m_depth': inlet,
        'mass_outlet_kg_s_per_m_depth': outlet,
        'relative_mass_imbalance': abs(outlet-inlet)/abs(inlet),
        'max_cell_divergence_s_inv': float(divergence.abs().max()),
        'developed_region_length_m': length,
        'developed_wall_force_on_fluid_n_per_m_depth': wall_force,
        'developed_pressure_force_on_fluid_n_per_m_depth': pressure_force,
        'developed_force_balance_relative': abs(wall_force+pressure_force)/abs(pressure_force),
        'wall_shear_relative_linf_error': shear_error,
        'errors_scope': 'velocity and pressure compare developed section; wall shear uses half-cell stencil',
    })
    report['passed'] = report['passed'] and shear_error < .03 and report['relative_mass_imbalance'] < c.tolerance
    return report, lower, upper


def run(output, grids=(12, 24, 48), max_iterations=1000):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    cases = []
    for ny in grids:
        c = SolverConfig(nx=3*ny, ny=ny, length=.12, height=.02,
                         inlet_velocity=.0005, density=1000., reynolds=1.,
                         cylinder_radius=None, tolerance=1e-6, max_iterations=max_iterations)
        result = SimpleSolver(c).solve()
        case, lower, upper = channel_metrics(result)
        directory = output / f'{c.nx}x{ny}'
        export_result(result, directory)
        fields = {'schema': 'tensor-suite.channel-fields/1', 'units': 'SI',
                  'layout': 'cell arrays [y][x]; u_faces [ny][nx+1]; v_faces [ny+1][nx]',
                  'config': asdict(c), 'x_m': result.x.tolist(), 'y_m': result.y.tolist(),
                  'u_m_s': result.cell_center_velocity()[0].tolist(),
                  'v_m_s': result.cell_center_velocity()[1].tolist(),
                  'p_pa': result.p.tolist(), 'u_faces_m_s': result.u.tolist(),
                  'v_faces_m_s': result.v.tolist(),
                  'lower_wall_traction_on_fluid_pa': lower.tolist(),
                  'upper_wall_traction_on_fluid_pa': upper.tolist()}
        (directory/'si-fields.json').write_text(json.dumps(fields, indent=2, allow_nan=False)+'\n')
        cases.append(case)
    manifest = {str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(output.rglob('*')) if p.is_file() and p.name != 'benchmark.json'}
    summary = {'schema': 'tensor-suite.channel-verification/1', 'solver': 'TensorFVM SIMPLE',
               'reference': REFERENCE, 'boundary': {'inlet': 'uniform prescribed velocity',
               'outlet': 'zero gauge pressure, predicted velocity zero normal gradient',
               'walls': 'no slip, half-cell diffusion'},
               'comparison_scope': 'common analytic fully developed channel reference; boundary conditions differ from periodic LBM; no joint FSI validation',
               'moving_ice_backend_qualified': False,
               'cases': cases, 'passed': all(c['passed'] for c in cases),
               'git_base': subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
               'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in (Path(__file__), Path(__file__).with_name('solver.py'))},
               'artifact_sha256': manifest}
    (output/'benchmark.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('docs/suite-channel'))
    parser.add_argument('--max-iterations', type=int, default=1000)
    args = parser.parse_args()
    torch.set_num_threads(1)
    summary = run(args.output, max_iterations=args.max_iterations)
    print(json.dumps({'passed': summary['passed'], 'cases': summary['cases']}, indent=2))
    return 0 if summary['passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())

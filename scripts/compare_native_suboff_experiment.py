#!/usr/bin/env python3
"""Compare a frozen physical SST run with the appropriate SUBOFF reference."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
from tensorfvm.native_mesh3d import NativeMesh3D
from tensorfvm.suboff_validation import require_reynolds, drag_comparison, sample_upper_meridian_pressure, cp_comparison
from run_native_suboff_benchmark import write, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--case', choices=['drag', 'cp'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); run = args.run; out = args.output
    if out.exists() and any(out.iterdir()):parser.error('output must be empty')
    result = json.loads((run / 'result.json').read_text())
    config = json.loads((run / 'config.json').read_text())
    if 'numerically_converged' not in result or 'Re_L' not in result or config.get('kind') != 'native SST SUBOFF diagnostic':
        parser.error('Physical native SST run required; manufactured solutions cannot be compared as SUBOFF flow')
    repo = Path(__file__).resolve().parents[1]
    spec_path = repo / 'benchmarks/suboff_aff1_validation.json'; spec = json.loads(spec_path.read_text())
    case = spec['cases'][0 if args.case == 'drag' else 1]
    require_reynolds(result['Re_L'], case['operating_conditions']['Re_L'])
    if digest(run / 'input.msh') != config['mesh_sha256']:
        raise ValueError('Frozen mesh hash differs from producer configuration')
    out.mkdir(parents=True, exist_ok=True); shutil.copy2(spec_path, out / 'reference-specification.json')
    bindings = {str(p):digest(p) for p in [run / 'input.msh', run / 'fields.npz', run / 'config.json', run / 'result.json']}
    mesh = NativeMesh3D.from_gmsh(run / 'input.msh', triangulate_faces=True)
    fields = np.load(run / 'fields.npz', allow_pickle=False)
    wall = mesh.patch_labels == 'wall'
    # Replay the same total surface force independently from raw face fields.
    force_replayed = (fields['pressure_faces'][wall, None] * mesh.area_vectors[wall]
        - fields['anisotropic_stress_traction'][wall]
        - fields['isotropic_reynolds_stress_traction'][wall]).sum(axis=0)
    reported = np.asarray(result['forces_dimensionless']['total_force'])
    if not np.allclose(force_replayed, reported, rtol=1e-10, atol=1e-12):
        raise ValueError('Reported force does not agree with native face traction replay')
    if args.case == 'drag':
        comparison = drag_comparison(force_replayed, case)
        metric_passed = comparison['nominal_error_below_3_percent']
    else:
        points = case['reference']['corrected_candidate_points']
        sample = sample_upper_meridian_pressure(mesh, fields['pressure_faces'], [row['x_L'] for row in points])
        comparison = {**sample, **cp_comparison(sample['Cp_calculated'], points)}
        metric_passed = comparison['provisional_nominal_error_below_3_percent'] and comparison['correction_sensitivity_error_below_3_percent']
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.errorbar(sample['x_L'], comparison['candidate_Cp_listed'], yerr=.015, fmt='o', label='Secondary experimental candidate; 95% ±0.015')
        ax.plot(sample['x_L'], sample['Cp_calculated'], 's-', label='Actual native physical face pressure')
        ax.plot(sample['x_L'], comparison['candidate_Cp_arithmetic'], '.', label='Correction arithmetic sensitivity')
        ax.set(xlabel='x/L', ylabel='Cp', title=f'SUBOFF Cp diagnostic — numerical convergence: {result["numerically_converged"]}')
        ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(out / 'cp-comparison.png', dpi=220); plt.close(fig)
    summary = dict(case_id=case['case_id'], metrics=comparison, source_sha256=bindings,
        native_force_replay_max_error=float(np.max(np.abs(force_replayed - reported))),
        numerically_converged=result['numerically_converged'], nominal_metric_passed=bool(metric_passed),
        wall_resolution_passed=result['low_Re_wall_resolution']['maximum_yplus_below_one'],
        spatial_convergence_qualified=result['spatial_convergence_qualified'],
        physical_validation_passed=False,
        note='Single-case nominal metrics cannot establish all-case spatial/physical qualification; Cp reference is provisional.')
    write(out / 'comparison.json', summary)
    table = '\n'.join(f'| {key} | {value:.8g} |' for key, value in comparison.items()
        if isinstance(value, (float, int)) and not isinstance(value, bool))
    (out / 'report.md').write_text(f'''# SUBOFF {args.case} experimental comparison

Case: {case['case_id']}. Nominal metric passed: **{metric_passed}**.
Numerically converged: **{result['numerically_converged']}**.
All-case physical validation: **false**; a refinement study remains required.

| Computed/reference metric | Value |
|---|---:|
{table}

See `comparison.json` for actual raw metrics, input hashes, reference qualification
and uncertainty. The reference specification is frozen alongside this report.
''' + ('\n![Actual calculated Cp and reference candidate](cp-comparison.png)\n' if args.case == 'cp' else ''))
    print(json.dumps(summary, indent=2, allow_nan=False)); return 0


if __name__ == '__main__':raise SystemExit(main())

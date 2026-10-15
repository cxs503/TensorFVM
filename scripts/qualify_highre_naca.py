"""Combine at least three independently audited NACA grids without relaxing gates."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from report_highre_naca import curve_error


def qualify(reports):
    if len(reports)<3:raise ValueError('At least three grids required')
    rows=sorted(reports,key=lambda r:r['cells'])
    if len({r['mesh_sha256'] for r in rows})!=len(rows):raise ValueError('Distinct grids required')
    for key in ['alpha','reynolds','sources_sha256','reference_manifest']:
        if any(r[key]!=rows[0][key] for r in rows):raise ValueError('Inconsistent '+key)
    drag=np.array([r['coefficients']['drag'] for r in rows]);delta=np.diff(drag)
    monotonic=bool(np.all(delta>=0) or np.all(delta<=0))
    decreasing=bool(abs(delta[-1])<abs(delta[-2]))
    fine_change=float(abs(delta[-1])/abs(rows[-1]['reference_coefficients']['drag']))
    ready=all(r['numerical_passed'] and r['wall_resolution_passed'] and r['source_integrity_verified'] for r in rows)
    return dict(cells=[r['cells'] for r in rows],numerical_and_wall_passed=ready,
        reference_passed=bool(rows[-1]['reference_passed']),drag_monotonic=monotonic,
        drag_differences_decreasing=decreasing,drag_fine_medium_relative_change=fine_change,
        drag_grid_passed=monotonic and decreasing and fine_change<.01,
        qualified=False,scope='Native incompressible SIMPLE/SA NACA0012 only; does not qualify SST, SUBOFF, overset or GPU.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('reports',nargs='+',type=Path)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    directories=sorted(a.reports,key=lambda d:json.loads((d/'audit.json').read_text())['cells'])
    rows=[json.loads((d/'audit.json').read_text()) for d in directories];result=qualify(rows)
    changes={}
    for side in ['upper','lower']:
        curves=[np.loadtxt(d/(side+'-cp.csv'),delimiter=',',skiprows=1) for d in directories]
        errors=[curve_error(fine[:,0],fine[:,1],coarse[:,:2])[0] for coarse,fine in zip(curves,curves[1:])]
        changes[side]=dict(relative_l2_changes=[e['relative_l2'] for e in errors],
            passed=bool(errors[-1]['relative_l2']<.01 and errors[-1]['relative_l2']<errors[-2]['relative_l2'] and all(e['coverage_chord']>.999 for e in errors)))
    result['cp_grid_changes']=changes
    result['grid_convergence_passed']=result['drag_grid_passed'] and all(e['passed'] for e in changes.values())
    result['qualified']=result['numerical_and_wall_passed'] and result['reference_passed'] and result['grid_convergence_passed']
    result['inputs']=[dict(directory=str(d.resolve()),audit_sha256=hashlib.sha256((d/'audit.json').read_bytes()).hexdigest()) for d in directories]
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result,indent=2))
    return 0 if result['qualified'] else 2


if __name__=='__main__':raise SystemExit(main())

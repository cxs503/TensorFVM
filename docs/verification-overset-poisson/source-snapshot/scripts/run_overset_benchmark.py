"""Shared-output static overset diffusion benchmark; no NS qualification."""
import argparse
from pathlib import Path
import time
import numpy as np
import torch
from tensorfvm.overset import cartesian_grid,annular_grid,circle_polygon,build_overset,ACTIVE
from tensorfvm.overset_poisson import solve_overset_poisson
from tensorfvm.verification.core import Metric,evaluate,save_run,provenance,sha,write_json,artifact_manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--levels',type=int,nargs='+',default=[32,64,128])
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists() and any(args.output.iterdir()):p.error('Output must be empty')
    if any(n<16 or n%4 for n in args.levels) or len(set(args.levels))!=len(args.levels):p.error('Unique levels >=16 divisible by four required')
    torch.set_num_threads(1);args.output.mkdir(parents=True,exist_ok=True)
    root=Path(__file__).resolve().parents[1];prov=provenance()
    for key in ('src/tensorfvm/overset.py','src/tensorfvm/overset_poisson.py','scripts/run_overset_benchmark.py'):
        prov['source_sha256'][key]=sha(root/key)
    summary=dict(case='static-overset-poisson',provenance=prov,runs=[],passed=False,navier_stokes_verified=False,strict_local_conservation=False)
    for n in sorted(args.levels):
        label=f'background-{n}';directory=args.output/label;directory.mkdir()
        start=time.perf_counter()
        conn=build_overset(cartesian_grid((-2,2,-2,2),n,n),annular_grid((0,0),.3,1.,2*n,n//4),circle_polygon((0,0),.3,points=2*n),circle_polygon((0,0),.65))
        result=solve_overset_poisson(conn);m=result.metrics;conn.save(directory/'connectivity.npz')
        quantities=[Metric('grid_0_relative_l2_error',m['grid_0_relative_l2_error'],.03,'Cartesian ACTIVE scalar error'),Metric('grid_1_relative_l2_error',m['grid_1_relative_l2_error'],.03,'Body-fitted ACTIVE scalar error'),Metric('linear_relative_residual',m['relative_algebraic_residual'],1e-10,'Actual coupled sparse equations'),Metric('donor_constraint_residual',m['maximum_donor_constraint_residual'],1e-10,'Simultaneous donor constraints'),Metric('global_diffusion_balance_relative_error',m['physical_global_conservation_defect_relative'],.03,'Unique physical rectangle minus polygonal body; not a local flux certificate')]
        records,passed=evaluate(quantities);fields={}
        for k,g in enumerate(conn.grids):
            fields.update({f'g{k}_polygons':np.array(g.polygons),f'g{k}_centers':g.centers,f'g{k}_volumes':g.volumes,f'g{k}_state':conn.states[k],f'g{k}_solution':result.fields[k],f'g{k}_exact':result.exact_fields[k]})
        row=save_run(directory,dict(case=summary['case'],directory=label,resolution=n,config=dict(bounds=[-2,2,-2,2],body_radius=.3,component_outer_radius=1.,blanking_radius=.65,background_shape=[n,n],component_shape=[n//4,2*n]),computed=m,connectivity=conn.diagnostics(),metrics=records,passed=passed,elapsed_s=time.perf_counter()-start,history=[],fields=fields))
        summary['runs'].append(row);summary['passed']=len(summary['runs'])>=3 and all(r['passed'] for r in summary['runs'])
        write_json(args.output/'summary.json',summary);write_json(args.output/'manifest.json',dict(source_sha256=prov['source_sha256'],artifacts_sha256=artifact_manifest(args.output)))
        print(n,'component_max_error',max(m['grid_0_relative_l2_error'],m['grid_1_relative_l2_error']),'global_flux_error',m['physical_global_conservation_defect_relative'],'passed',passed,flush=True)
    return 0 if summary['passed'] else 2


if __name__=='__main__':raise SystemExit(main())

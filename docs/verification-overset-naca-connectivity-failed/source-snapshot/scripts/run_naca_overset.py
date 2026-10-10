"""Actual NACA0012 overset connectivity and nonorthogonal scalar diffusion."""
import argparse,importlib.util,time
from pathlib import Path
from dataclasses import replace
import numpy as np
import torch
from tensorfvm.gmsh_external import GmshExternalMesh
from tensorfvm.overset import OversetGrid,cartesian_grid,build_overset
from tensorfvm.overset_scalar import solve_manufactured_diffusion
from tensorfvm.verification.external_flow import configuration
from tensorfvm.verification.core import Metric,evaluate,save_run,provenance,sha,write_json,artifact_manifest


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--levels',type=int,nargs='+',default=[32,64,128]);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists() and any(args.output.iterdir()):p.error('Output must be empty')
    if any(n<32 or n%32 for n in args.levels):p.error('Levels must be >=32 and divisible by 32')
    root=Path(__file__).resolve().parents[1];out=args.output.resolve();out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
    generator=root/'scripts/generate_external_gmsh.py';spec=importlib.util.spec_from_file_location('generator',generator);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    prov=provenance()
    for key in ('src/tensorfvm/overset.py','src/tensorfvm/overset_scalar.py','src/tensorfvm/gmsh_external.py','src/tensorfvm/multi_element.py','scripts/generate_external_gmsh.py','scripts/run_naca_overset.py'):prov['source_sha256'][key]=sha(root/key)
    summary=dict(case='naca0012-static-overset-scalar',runs=[],provenance=prov,passed=False,navier_stokes_verified=False,strict_local_conservation=False)
    for n in args.levels:
        start=time.perf_counter();scale=128/n;label=f'background-{n}';directory=out/label;directory.mkdir()
        meshfile=directory/'component.msh';metadata=module.generate('naca',meshfile,scale,length=4,height=3,leading_x=1.25,leading_y=1.5,geometric_incidence_deg=4,hybrid=True)
        config=replace(configuration('naca',64,26,'cpu',1),mesh_type='gmsh-airfoil',mesh_file=str(meshfile));mesh=GmshExternalMesh(config)
        border=mesh.boundary&~mesh.masks['airfoil'];outer=np.zeros(mesh.volumes.numel(),bool);outer[mesh.owner[border].numpy()]=True
        component=OversetGrid.from_mesh(mesh,'NACA0012 component',outer_boundary_mask=outer)
        edges=mesh.face_vertices[mesh.masks['airfoil']].numpy();links={tuple(a):tuple(b) for a,b in edges};body=[tuple(edges[0,0])]
        while len(body)<len(edges):body.append(links[body[-1]])
        if links[body[-1]]!=body[0]:raise ValueError('Unclosed airfoil boundary')
        blank=np.array([[.5,.5],[3.5,.5],[3.5,2.5],[.5,2.5]])
        conn=build_overset(cartesian_grid((-2,6,-2,5),n,7*n//8),component,np.array(body),blank)
        print(n,'connected',conn.diagnostics(),flush=True)
        result=solve_manufactured_diffusion(conn);m=result.metrics;conn.save(directory/'connectivity.npz')
        quantities=[Metric(f'grid_{k}_relative_l2_error',m[f'grid_{k}_relative_l2_error'],.03,'Actual ACTIVE scalar field vs analytic quadratic') for k in (0,1)]
        quantities.extend([Metric('linear_relative_residual',m['relative_algebraic_residual'],1e-10,'Full nonorthogonal coupled diffusion'),Metric('donor_constraint_residual',m['maximum_donor_constraint_residual'],1e-10,'Actual two-way constraints'),Metric('global_diffusion_balance_relative_error',m['physical_global_conservation_defect_relative'],.03,'Unique physical rectangle minus exact wall polygon')])
        records,passed=evaluate(quantities);fields={}
        for k,g in enumerate(conn.grids):
            polygons=np.array([np.r_[poly,poly[-1:]] if len(poly)==3 else poly for poly in g.polygons])
            fields.update({f'g{k}_polygons':polygons,f'g{k}_centers':g.centers,f'g{k}_volumes':g.volumes,f'g{k}_state':conn.states[k],f'g{k}_solution':result.fields[k],f'g{k}_exact':result.exact_fields[k]})
        row=save_run(directory,dict(case=summary['case'],directory=label,resolution=n,config=dict(bounds=[-2,6,-2,5],airfoil_code='0012',chord=1,incidence_deg=4,leading_edge=[1.25,1.5],component_domain=[0,4,0,3],mesh_scale=scale),mesh_generation=metadata,computed=m,connectivity=conn.diagnostics(),metrics=records,passed=passed,elapsed_s=time.perf_counter()-start,history=[],fields=fields))
        summary['runs'].append(row);summary['passed']=len(summary['runs'])>=3 and all(r['passed'] for r in summary['runs']);write_json(out/'summary.json',summary);write_json(out/'manifest.json',dict(source_sha256=prov['source_sha256'],artifacts_sha256=artifact_manifest(out)))
        print(n,m,'passed',passed,flush=True)
    return 0 if summary['passed'] else 2


if __name__=='__main__':raise SystemExit(main())

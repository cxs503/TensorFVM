"""30P30N: three distinct solids on one fitted component, Cartesian background.
Static manufactured diffusion; this does not solve overset Navier–Stokes.
"""
import argparse, json, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import gmsh
from tensorfvm.benchmark_30p30n import write_gmsh_geo, PROFILE_DIRECTORY, _read_profile, _minimum_profile_gap
from tensorfvm.multi_element import ThreeElementMesh
from tensorfvm.overset import OversetGrid, cartesian_grid, build_overset, ACTIVE
from tensorfvm.overset_scalar import solve_manufactured_diffusion
from tensorfvm.verification.core import Metric, evaluate, save_run, provenance, sha, write_json, artifact_manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--levels',nargs='+',type=int,default=[32,64,128])
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):parser.error('Output must be empty')
    if any(n<32 or n%32 for n in args.levels):parser.error('Levels must be positive multiples of 32')
    torch.set_num_threads(1);root=Path(__file__).resolve().parents[1]
    sources=['scripts/run_three_element_overset.py','src/tensorfvm/overset.py','src/tensorfvm/overset_scalar.py','src/tensorfvm/benchmark_30p30n.py','src/tensorfvm/multi_element.py']
    sources += [f'src/tensorfvm/data/30p30n/{name}.dat' for name in ('slat','main','flap')]
    prov=provenance()
    for key in sources:
        prov['source_sha256'][key]=sha(root/key)
    for key in prov['source_sha256']:
        target=out/'source-snapshot'/key;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((root/key).read_bytes())
    profiles={name:_read_profile(PROFILE_DIRECTORY/f'{name}.dat') for name in ('slat','main','flap')}
    gaps={a+'-'+b:_minimum_profile_gap(profiles[a],profiles[b]) for a,b in (('slat','main'),('main','flap'),('slat','flap'))}
    summary=dict(case='30p30n-static-overset-diffusion',runs=[],provenance=prov,minimum_gaps=gaps,passed=False,navier_stokes_verified=False,strict_local_conservation=False,component_grids=1,physical_solids=3)
    for n in args.levels:
        start=time.perf_counter();directory=out/f'background-{n}';directory.mkdir();scale=128/n
        geo=write_gmsh_geo(directory/'component.geo',domain=(-1,3,-1.5,1.5),near_size=.001*scale,far_size=.03*scale,first_layer=.000025*scale,layer_thickness=.00025*scale)
        gmsh.initialize();gmsh.option.setNumber('General.NumThreads',1);gmsh.option.setNumber('General.Terminal',0)
        try:
            gmsh.open(str(geo));gmsh.model.mesh.generate(2);gmsh.option.setNumber('Mesh.MshFileVersion',2.2);gmsh.write(str(directory/'component.msh'))
        finally:gmsh.finalize()
        mesh=ThreeElementMesh(SimpleNamespace(mesh_file=directory/'component.msh',device='cpu'))
        outer=np.zeros(mesh.volumes.numel(),bool);outer[mesh.owner[mesh.boundary&~mesh.masks['wall']].numpy()]=True
        component=OversetGrid.from_mesh(mesh,'30P30N fitted component',outer_boundary_mask=outer)
        bodies=[]
        for name in profiles:
            edges=mesh.face_vertices[mesh.masks[name]].numpy();links={tuple(a):tuple(b) for a,b in edges};body=[tuple(edges[0,0])]
            while len(body)<len(edges):body.append(links[body[-1]])
            if links[body[-1]]!=body[0]:raise ValueError('Unclosed '+name)
            bodies.append(np.array(body))
        blank=np.array([[-.5,-1],[2.5,-1],[2.5,1],[-.5,1]])
        conn=build_overset(cartesian_grid((-3,5,-3.5,3.5),n,7*n//8),component,bodies[0],blank,body_polygons=bodies)
        conn.save(directory/'connectivity.npz');print(n,'connected',conn.diagnostics(),flush=True)
        result=solve_manufactured_diffusion(conn);m=result.metrics
        records,passed=evaluate([Metric(f'grid_{k}_relative_l2_error',m[f'grid_{k}_relative_l2_error'],.03,'Actual active scalar field') for k in (0,1)]+[Metric('algebraic_residual',m['relative_algebraic_residual'],1e-10,'Full coupled nonorthogonal equation'),Metric('donor_residual',m['maximum_donor_constraint_residual'],1e-10,'Two-way interpolation constraints'),Metric('global_balance',m['physical_global_conservation_defect_relative'],.03,'Unique domain minus three separate solids')])
        fields={}
        for k,g in enumerate(conn.grids):
            fields.update({f'g{k}_polygons':np.array([np.r_[p,p[-1:]] if len(p)==3 else p for p in g.polygons]),f'g{k}_centers':g.centers,f'g{k}_state':conn.states[k],f'g{k}_volumes':g.volumes,f'g{k}_solution':result.fields[k],f'g{k}_exact':result.exact_fields[k]})
        row=save_run(directory,dict(case=summary['case'],directory=directory.name,resolution=n,computed=m,connectivity=conn.diagnostics(),metrics=records,passed=passed,elapsed_s=time.perf_counter()-start,history=[],fields=fields))
        summary['runs'].append(row);write_json(out/'summary.json',summary);print(n,m,'passed',passed,flush=True)
    summary['passed']=len(summary['runs'])>=3 and all(r['passed'] for r in summary['runs']);write_json(out/'summary.json',summary)
    write_json(out/'manifest.json',dict(source_snapshot_root='source-snapshot',source_sha256=prov['source_sha256'],artifacts_sha256=artifact_manifest(out)))
    return 0 if summary['passed'] else 2

if __name__=='__main__':raise SystemExit(main())

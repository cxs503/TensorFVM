#!/usr/bin/env python3
"""Nonzero native 3D NS/SST integration pilot, explicitly not physical validation."""
import argparse,json,hashlib,sys,time,shutil
from pathlib import Path
from dataclasses import replace
import numpy as np
from tensorfvm.native_mesh3d import NativeMesh3D
from tensorfvm.native_collocated3d import NativeSteadySettings
from tensorfvm.native_rans3d import NativeSSTRansSolver3D,NativeRansSettings


def channel(n):
    vertices=np.array([[x,y,z] for z in np.linspace(0,1,n+1) for y in np.linspace(0,1,n+1) for x in np.linspace(0,1,n+1)])
    ix=lambda x,y,z:z*(n+1)**2+y*(n+1)+x
    cells=[]
    for z in range(n):
        for y in range(n):
            for x in range(n):cells.append([ix(x,y,z),ix(x+1,y,z),ix(x+1,y+1,z),ix(x,y+1,z),ix(x,y,z+1),ix(x+1,y,z+1),ix(x+1,y+1,z+1),ix(x,y+1,z+1)])
    m=NativeMesh3D.from_cells(vertices,cells,['hexahedron']*len(cells),triangulate_faces=True)
    b=m.neighbor<0;fc=m.face_centers;labels=np.full(len(m.owner),'internal',dtype='<U10');labels[b]='farfield'
    labels[b&((np.abs(fc[:,1])<1e-12)|(np.abs(fc[:,1]-1)<1e-12))]='wall'
    labels[b&(np.abs(fc[:,0])<1e-12)]='inlet';labels[b&(np.abs(fc[:,0]-1)<1e-12)]='outlet'
    return replace(m,patch_labels=labels)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--n',type=int,default=3);parser.add_argument('--outer',type=int,default=2);args=parser.parse_args()
    out=args.output
    if out.exists() and any(out.iterdir()):parser.error('output must be empty to preserve numerical evidence')
    out.mkdir(parents=True,exist_ok=True);repo=Path(__file__).resolve().parents[1]
    files={Path(__file__).resolve()}|{Path(mod.__file__).resolve() for name,mod in list(sys.modules.items()) if name.startswith('tensorfvm') and getattr(mod,'__file__',None) and str(mod.__file__).endswith('.py')}
    source_hashes={}
    for p in sorted(files):
        rel=p.relative_to(repo);dest=out/'source-snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest);source_hashes[str(rel)]=hashlib.sha256(dest.read_bytes()).hexdigest()
    mesh=channel(args.n);labels=mesh.patch_labels;fixed=(mesh.neighbor<0)&(labels!='outlet')
    boundary=np.zeros((len(mesh.owner),3));boundary[fixed&(labels!='wall'),0]=1.
    solver=NativeSSTRansSolver3D(mesh,fixed,boundary,labels=='outlet',settings=NativeSteadySettings(viscosity=.05,max_iterations=60,convection_order=1,nonlinear_tolerance=1e-8),rans_settings=NativeRansSettings(max_outer_iterations=args.outer,pseudo_time_step=1e-4,turbulence_relaxation=.4))
    started=time.perf_counter();state=solver.solve_rans(.001,1.,inlet_k=.001,inlet_omega=1.);elapsed=time.perf_counter()-started;forces=solver.forces()
    summary=dict(kind='Nonzero 3D channel NS/SST integration pilot; not physical validation',cells=solver.N,faces=solver.F,Re=20.,density=1.,dynamic_molecular_viscosity=.05,inlet_speed=1.,initial_k=.001,initial_omega=1.,wall_turbulent_viscosity_zero=True,geometry=mesh.audit(),wall_distance_error=float(np.max(np.abs(solver.wall_distance-np.minimum(mesh.cell_centers[:,1],1-mesh.cell_centers[:,1])))),velocity_x_range=[float(state['velocity'][:,0].min()),float(state['velocity'][:,0].max())],wall_normal_flux_max=float(np.abs(state['flux'][solver.wall]).max()),total_wall_force=forces['total_force'].tolist(),pressure_wall_force=forces['physical_pressure_force'].tolist(),anisotropic_wall_force=forces['anisotropic_viscous_force'].tolist(),isotropic_wall_force=forces['isotropic_turbulent_force'].tolist(),force_decomposition_error=float(np.max(np.abs(forces['total_force']-forces['physical_pressure_force']-forces['anisotropic_viscous_force']-forces['isotropic_turbulent_force']))),physical_pressure_conversion_error=float(np.max(np.abs(state['physical_pressure']-(state['modified_pressure']-2/3*state['k'])))),numerically_converged=state['numerically_converged'],physical_benchmark_passed=False,elapsed_seconds=elapsed,history=state['history'],source_sha256=source_hashes)
    (out/'result.json').write_text(json.dumps(summary,indent=2)+'\n')
    np.savez_compressed(out/'fields.npz',velocity=state['velocity'],pressure=state['physical_pressure'],modified_pressure=state['modified_pressure'],k=state['k'],omega=state['omega'],flux=state['flux'],wall_distance=state['wall_distance'],cell_centers=mesh.cell_centers,cell_volumes=mesh.cell_volumes,vertices=mesh.vertices,cells=np.asarray(mesh.cells),owner=mesh.owner,neighbor=mesh.neighbor,area_vectors=mesh.area_vectors,face_centers=mesh.face_centers,patch_labels=mesh.patch_labels)
    last=state['history'][-1]
    report=f'''# Native 3D NS/SST integration pilot

**This is a nonzero inlet/no-slip integration test, not physical turbulence validation or a SUBOFF benchmark.**

The actual unit-box native mesh has {solver.N} hexahedral cells with planar native triangle facets. Inlet speed is 1, density 1, molecular viscosity .05 (Re=20). Two y walls have prescribed zero velocity, x=1 has physical pressure zero, and z boundaries have prescribed uniform streamwise velocity. Initial/inflow k=.001 and omega=1. These are test inputs, not measured turbulence. This low-Re setup is not a turbulent reference benchmark.

The solver assembles complete momentum/continuity with variable molecular/eddy stress and SST-2003 transport. Eddy viscosity is exactly zero at low-Re walls. The nearest actual wall-triangle distance is verified against the independent box-plane distance: maximum error {summary['wall_distance_error']:.6g}.

## Actual result

- Cell streamwise velocity range: {summary['velocity_x_range']}.
- Wall native normal flux maximum: {summary['wall_normal_flux_max']:.6g}.
- Native face divergence maximum: {last['stored_native_divergence_max']:.6g}.
- Current reconstructed divergence maximum after updating SST viscosity: {last['current_reconstructed_divergence_max']:.6g}.
- Current nonlinear SST k residual: {last['k_current_nonlinear_residual']:.6g}.
- Current nonlinear SST omega residual: {last['omega_current_nonlinear_residual']:.6g}.
- Total wall force: {summary['total_wall_force']} in normalized rho U^2 L^2 units.
- Physical pressure/isotropic/anisotropic force decomposition error: {summary['force_decomposition_error']:.6g}.
- Numerical convergence: **{state['numerically_converged']}**. Physical accuracy qualification: **False**.

Two outer updates exercise the coupling and expose unconverged nonlinear turbulence. The current sources are recomputed from the current fields; the small frozen transport ledger is separately reported and is not used as a nonlinear convergence claim. There is no known drag/Cp target and no below-3% qualification.

## Pressure and boundary forces

The solved pressure is pi=p+2rho k/3. Physical pressure is p=pi−2rho k/3. Physical outlet pressure is converted with current boundary k, and all pressure source vectors are refreshed. Total force is invariant: pi S−tau_anis S = p S−(tau_anis−2rho k I/3)S. The same shared face quantities supply both momentum and reported boundary forces.

## Reproduction and evidence

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src python scripts/verify_native_rans_integration3d.py --n {args.n} --outer {args.outer} --output NEW_EMPTY_DIRECTORY
```

[result.json](result.json) contains actual fields' metrics and complete outer histories, including current nonlinear and frozen linearized ledgers. Raw fields and native geometry are in fields.npz; imported producer sources are archived at launch. This small integration test does not establish engineering mesh cost, turbulence accuracy, wall y+, spatial convergence or GPU performance.
'''
    (out/'report.md').write_text(report);print(json.dumps(summary,indent=2));return 0

if __name__=='__main__':raise SystemExit(main())

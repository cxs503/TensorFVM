#!/usr/bin/env python3
"""Fixed-parameter nonlinear 3D ABC manufactured NS spatial convergence."""
import argparse,json,hashlib,sys,shutil,time,traceback
from pathlib import Path
import numpy as np
from tensorfvm.native_mesh3d import NativeMesh3D
from tensorfvm.native_collocated3d import NativeSteadySolver3D,NativeSteadySettings
from tensorfvm.native_manufactured3d import abc_fields,exact_boundary_flux,exact_boundary_velocity_means,volume_weighted_error
from tensorfvm.native_quadratic_flow3d import NativeQuadraticSteadySolver3D

SHEAR=np.array([[1,.18,.12],[0,1,.2],[0,0,1]])


def skew_box(n):
    points=np.array([[x,y,z] for z in np.linspace(0,1,n+1) for y in np.linspace(0,1,n+1) for x in np.linspace(0,1,n+1)])@SHEAR.T
    ix=lambda x,y,z:z*(n+1)**2+y*(n+1)+x
    cells=[]
    for z in range(n):
        for y in range(n):
            for x in range(n):cells.append([ix(x,y,z),ix(x+1,y,z),ix(x+1,y+1,z),ix(x,y+1,z),ix(x,y,z+1),ix(x+1,y,z+1),ix(x+1,y+1,z+1),ix(x,y+1,z+1)])
    return NativeMesh3D.from_cells(points,cells,['hexahedron']*len(cells),triangulate_faces=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--grids',type=int,nargs='+',default=[3,6,12]);p.add_argument('--reconstruction',choices=['linear','quadratic'],default='linear');args=p.parse_args();out=args.output
    if out.exists() and any(out.iterdir()):p.error('empty output required to preserve evidence')
    out.mkdir(parents=True,exist_ok=True);repo=Path(__file__).resolve().parents[1]
    files={Path(__file__).resolve()}|{Path(mod.__file__).resolve() for name,mod in list(sys.modules.items()) if name.startswith('tensorfvm') and getattr(mod,'__file__',None) and str(mod.__file__).endswith('.py')}
    bindings={}
    for source in sorted(files):
        rel=source.relative_to(repo);dest=out/'source-snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest);bindings[str(rel)]=hashlib.sha256(dest.read_bytes()).hexdigest()
    config=dict(reconstruction=args.reconstruction,kind='Independent fixed-parameter 3D nonlinear ABC manufactured NS',amplitude=.5,wavenumber=.7,kinematic_viscosity=.1,density=1.,logical_domain=[[0,1],[0,1],[0,1]],coordinate_transform=SHEAR.tolist(),grids=args.grids,velocity_boundary='analytic Dirichlet on all physical facets',native_boundary_flux='order-8 Duffy triangle Gauss quadrature of the analytic velocity',body_force='independent differential analytic acceleration at actual cell centroids',initial_velocity='zero',initial_pressure='constant gauge p_exact(first cell)',solver=dict(convection_order=2,bounded_convection=False,nonlinear_tolerance=1e-9,linear_tolerance=1e-11,max_iterations=100,relaxation=.8),error_norm='volume-weighted relative L2 at actual cell centroids; pressure means removed independently',mms_error_limit=.03,suboff_physics_passed=False)
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n');records=[]
    for n in args.grids:
        case=out/f'n-{n}';case.mkdir();started=time.perf_counter()
        try:
            mesh=skew_box(n);exact=abc_fields(mesh.cell_centers);boundary=abc_fields(mesh.face_centers)['velocity'];flux=exact_boundary_flux(mesh,abc_fields)
            settings=NativeSteadySettings(viscosity=.1,density=1.,convection=True,convection_order=2,bounded_convection=False,nonlinear_tolerance=1e-9,linear_tolerance=1e-11,max_iterations=100,relaxation=.8)
            solver_type=NativeQuadraticSteadySolver3D if args.reconstruction=='quadratic' else NativeSteadySolver3D
            extra={} if args.reconstruction=='linear' else dict(boundary_velocity_means=exact_boundary_velocity_means(mesh,abc_fields),boundary_velocity_gradients=abc_fields(mesh.face_centers)['velocity_gradient'])
            solver=solver_type(mesh,mesh.neighbor<0,boundary,**extra,body_force=exact['acceleration'],boundary_flux=flux,gauge_value=exact['pressure'][0],settings=settings)
            def cb(row):
                if row['iteration']%10==0:print(json.dumps(dict(n=n,**row)),flush=True)
            state=solver.solve(callback=cb)
            errors=dict(velocity_vector=volume_weighted_error(state['velocity'],exact['velocity'],mesh.cell_volumes),pressure_mean_removed=volume_weighted_error(state['pressure'],exact['pressure'],mesh.cell_volumes,remove_mean=True))
            errors.update({f'velocity_{axis}':volume_weighted_error(state['velocity'][:,j],exact['velocity'][:,j],mesh.cell_volumes) for j,axis in enumerate('xyz')})
            result=dict(n=n,cells=len(mesh.cells),faces=len(mesh.owner),geometry=mesh.audit(),numerically_converged=state['converged'],relative_errors=errors,all_error_metrics_below_3_percent=all(v<.03 for v in errors.values()),final=state['history'][-1],history=state['history'],elapsed_seconds=time.perf_counter()-started,original_analytic_boundary_flux_sum=float(flux.sum()),mms_spatial_accuracy_passed=state['converged'] and all(v<.03 for v in errors.values()),suboff_physical_benchmark_passed=False,source_sha256=bindings)
            np.savez_compressed(case/'fields.npz',velocity=state['velocity'],pressure=state['pressure'],reference_velocity=exact['velocity'],reference_pressure=exact['pressure'],analytic_acceleration=exact['acceleration'],flux=state['flux'],exact_boundary_flux=flux,vertices=mesh.vertices,cells=np.asarray(mesh.cells),cell_centers=mesh.cell_centers,cell_volumes=mesh.cell_volumes,owner=mesh.owner,neighbor=mesh.neighbor,face_centers=mesh.face_centers,area_vectors=mesh.area_vectors)
        except Exception as exc:
            result=dict(n=n,error=str(exc),traceback=traceback.format_exc(),numerically_converged=False,mms_spatial_accuracy_passed=False,suboff_physical_benchmark_passed=False,source_sha256=bindings)
        (case/'result.json').write_text(json.dumps(result,indent=2)+'\n');records.append(result);print(json.dumps({k:v for k,v in result.items() if k not in ['history','source_sha256','traceback']}),flush=True)
    (out/'summary.json').write_text(json.dumps(dict(config=config,cases=records),indent=2)+'\n')
    return 0 if all(r['mms_spatial_accuracy_passed'] for r in records) else 2
if __name__=='__main__':raise SystemExit(main())

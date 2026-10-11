#!/usr/bin/env python3
"""Native SUBOFF Hodge initialization diagnostic, not a viscous flow run."""
import argparse,json,shutil,time,sys,platform,subprocess
from pathlib import Path
import numpy as np
import scipy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.backends.backend_pdf import PdfPages
from tensorfvm.native_mesh3d import NativeMesh3D
from tensorfvm.native_projection3d import NativeProjection3D
from tensorfvm.collocated_diagnostics import flux_coupling
from tensorfvm.verification.core import Metric,evaluate,write_json,sha


def sections(mesh):
    polygons=[];indices=[]
    edges=((0,1),(0,2),(0,3),(1,2),(1,3),(2,3))
    for j,cell in enumerate(mesh.cells):
        if mesh.cell_types[j]!='tetra':raise ValueError('section plot requires actual tetrahedral cells')
        v=mesh.vertices[list(cell)]
        if v[:,2].min()>0 or v[:,2].max()<0:continue
        points=[p[:2]for p in v if abs(p[2])<1e-13]
        for a,b in edges:
            if v[a,2]*v[b,2]<0:
                t=-v[a,2]/(v[b,2]-v[a,2]);points.append((v[a]+t*(v[b]-v[a]))[:2])
        if len(points)<3:continue
        points=np.unique(np.round(points,12),axis=0)
        if len(points)<3:continue
        angle=np.arctan2(*(points-points.mean(0))[:,::-1].T)
        polygons.append(points[np.argsort(angle)]);indices.append(j)
    return polygons,np.array(indices,dtype=int)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mesh',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output
    if out.exists()and any(out.iterdir()):p.error('output must be empty to preserve evidence')
    out.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter();mesh=NativeMesh3D.from_gmsh(a.mesh)
    labels=mesh.patch_labels;required={'wall','inlet','outlet','farfield'}
    if set(labels)!=(required|{'internal'}):raise ValueError('SUBOFF physical patches wall/inlet/outlet/farfield required')
    # Snapshot at launch before factorization/projection. Physical pressure,
    # Reynolds number, viscosity and an evolution time are deliberately absent.
    repo=Path(__file__).resolve().parents[1]
    sources={Path(__file__).resolve()}|{Path(mod.__file__).resolve()for name,mod in list(sys.modules.items())if(name=='tensorfvm'or name.startswith('tensorfvm.'))and getattr(mod,'__file__',None)and Path(mod.__file__).suffix=='.py'}
    bindings={}
    for source in sorted(sources):
        target=out/'source-snapshot'/source.relative_to(repo);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target);bindings[str(source.relative_to(repo))]=sha(target)
    shutil.copy2(a.mesh,out/'input.msh')
    config=dict(kind='native-face Hodge initialization diagnostic',mesh_input=str(a.mesh),mesh_sha256=sha(a.mesh),coordinates='x/L,y/L,z/L; inlet U=1',velocity_potential_bc=dict(wall='dphi/dn=U dot n; corrected native normal flux zero',inlet='dphi/dn=0; prescribed inflow unchanged',farfield='dphi/dn=0; prescribed outer normal flux unchanged',outlet='phi=0'),physical_pressure_computed=False,time_advanced=False,physical_accuracy_qualified=False)
    write_json(out/'config.json',config)
    op=NativeProjection3D(mesh,labels=='outlet')
    assembly=time.perf_counter()-start
    uniform=np.array([1.,0.,0.]);tentative=mesh.area_vectors@uniform
    neumann=np.zeros(len(labels));wall=labels=='wall';neumann[wall]=op.normal[wall]@uniform
    solve_start=time.perf_counter();potential,flux=op.project(tentative,neumann);solve_time=time.perf_counter()-solve_start
    faces=op.scalar_face_values(potential,neumann)
    gradient=op.op.gauss_gradient(faces);velocity=uniform-gradient
    normal_velocity=np.broadcast_to(uniform,velocity.shape).copy()
    safe=mesh.neighbor.clip(min=0);face_velocity=.5*(velocity[mesh.owner]+velocity[safe]);boundary=mesh.neighbor<0
    face_velocity[labels=='inlet']=uniform;face_velocity[labels=='farfield']=uniform;face_velocity[labels=='outlet']=velocity[mesh.owner[labels=='outlet']]
    face_velocity[wall]=velocity[mesh.owner[wall]]-op.normal[wall]*(velocity[mesh.owner[wall]]*op.normal[wall]).sum(1)[:,None]
    coupling=flux_coupling(mesh.owner,mesh.neighbor,mesh.area_vectors,mesh.cell_volumes,flux,face_velocity,1.)
    traction=faces[:,None]*mesh.area_vectors
    impulse_ledger=((velocity-normal_velocity)*mesh.cell_volumes[:,None]).sum(0)+traction[boundary].sum(0)
    wall_normal_max=float(np.max(np.abs(flux[wall])/np.linalg.norm(mesh.area_vectors[wall],axis=1)))
    other=boundary&~wall&~(labels=='outlet')
    fixed_flux_max=float(np.max(np.abs(flux[other]-tentative[other])))
    gates,passed=evaluate([Metric('native_projected_divergence',coupling['native_face_divergence_max'],1e-7,'replayed native face flux / actual cell volumes','U/L'),Metric('wall_native_normal_speed',wall_normal_max,1e-12,'wall q/area after prescribed potential Neumann correction','U'),Metric('fixed_outer_flux_change',fixed_flux_max,1e-12,'inlet and farfield normal flux unchanged','U L^2'),Metric('conservative_potential_impulse_ledger',float(np.abs(impulse_ledger).max()),1e-10,'volume velocity change balanced by shared-face potential traction; not physical impulse','U L^3')])
    # A separate analytic affine operator response on this exact arbitrary mesh.
    affine_gradient=np.array([.7,-.3,.2]);exact=1.2+mesh.cell_centers@affine_gradient;boundary_scalar=1.2+mesh.face_centers@affine_gradient;boundary_derivative=op.normal@affine_gradient
    reconstructed,remainder=op.project(mesh.area_vectors@affine_gradient,boundary_derivative,boundary_scalar)
    manufactured=dict(scalar_max_error=float(np.abs(reconstructed-exact).max()),gradient_max_error=float(np.abs(op.gradient(exact,boundary_derivative,boundary_scalar)-affine_gradient).max()),flux_remainder_max=float(np.abs(remainder).max()),kind='independent analytic affine field, not CFD or experiment',units=dict(scalar='U L',gradient='U',flux='U L^2'),error_normalization='absolute errors in normalized U=L=1 coordinates')
    lu=op.lu
    def sparse_bytes(a):return int(a.data.nbytes+a.indices.nbytes+a.indptr.nbytes)
    result=dict(kind=config['kind'],geometry=mesh.audit(),projection_metrics=gates,projection_contract_passed=passed,coupling=coupling,manufactured_affine=manufactured,physical_pressure_computed=False,physical_accuracy_qualified=False,numerical_flow_qualified=False,time_advanced=False,
                performance=dict(setup_walltime=assembly,solve_walltime=solve_time,matrix_nnz=int(op.matrix.nnz),lu_nnz=int(lu.L.nnz+lu.U.nnz),lu_factor_bytes=sparse_bytes(lu.L)+sparse_bytes(lu.U)),
                limitations=['velocity potential is not pressure; no Reynolds number or physical time evolution','cell/face compatibility and spatial convergence remain unqualified','slip potential initialization does not enforce no-slip viscous evolution','no boundary-layer or turbulence resolution qualification','CPU sparse LU reference; GPU untested'])
    write_json(out/'result.json',result)
    write_json(out/'environment.json',dict(python=sys.version,python_executable=sys.executable,platform=platform.platform(),numpy=np.__version__,scipy=scipy.__version__,git_base=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()))
    np.savez_compressed(out/'fields.npz',potential=potential,velocity=velocity,flux=flux,tentative_flux=tentative,wall_neumann=neumann,potential_faces=faces,vertices=mesh.vertices,cells=np.array(mesh.cells),owner=mesh.owner,neighbor=mesh.neighbor,area_vectors=mesh.area_vectors,face_centers=mesh.face_centers,cell_centers=mesh.cell_centers,cell_volumes=mesh.cell_volumes,patch_labels=labels)
    polygons,indices=sections(mesh)
    figure,axes=plt.subplots(3,1,figsize=(12,9))
    for ax,values,title in zip(axes,[None,potential[indices],np.linalg.norm(velocity[indices],axis=1)],['Actual tetrahedral intersections with z=0','Velocity potential phi / (U L), NOT pressure','Cell speed / U, initialization diagnostic only']):
        pc=PolyCollection(polygons,facecolors='none',edgecolors='black',linewidths=.25)if values is None else PolyCollection(polygons,array=values,cmap='coolwarm'if ax is axes[1]else'turbo',edgecolors='none')
        ax.add_collection(pc);ax.set(xlim=(-.15,1.3),ylim=(-.18,.18),xlabel='x/L',ylabel='y/L',title=title);ax.set_aspect('equal')
        if values is not None:figure.colorbar(pc,ax=ax)
    figure.tight_layout();figure.savefig(out/'initialization.png',dpi=220);figure.savefig(out/'initialization.svg')
    with PdfPages(out/'report.pdf')as pdf:
        notice=plt.figure(figsize=(8.3,11.7));notice.text(.08,.92,'SUBOFF native-mesh initialization diagnostic',fontsize=16)
        notice.text(.08,.83,f"Actual body-fitted cells: {len(mesh.cells):,}\nNative divergence: {coupling['native_face_divergence_max']:.4g} U/L\nWall normal speed: {wall_normal_max:.4g} U\nInterior cell/face normal discrepancy: {coupling['interior_max_normal_velocity_difference']:.4g} U\n\nThe computed scalar is velocity potential, NOT physical pressure.\nNo physical time, viscosity, turbulence or drag has been computed.\nNo Reynolds-number or 3% accuracy qualification is implied.\n\nThis tests SUBOFF topology, boundary normals, native projection,\nconservative initialization and exposes cell/face discrepancy.\nSee report.md and result.json for exact scope and reproduction.",fontsize=12,va='top',linespacing=1.8)
        pdf.savefig(notice);pdf.savefig(figure);plt.close(notice)
    plt.close(figure)
    rows='\n'.join(f"| {m['name']} | {m['value']:.6g} | {m['limit']:.6g} | {m['passed']} |"for m in gates)
    report=f'''# SUBOFF native-mesh Hodge initialization diagnostic

**This is an inviscid initialization/mesh-operator diagnostic, not a viscous SUBOFF benchmark. The scalar is velocity potential, not physical pressure.**

## Problem and software

Actual bare SUBOFF tetrahedra are read from `{a.mesh}` (SHA256 `{sha(a.mesh)}`). The native 3D mesh has {len(mesh.cells):,} cells and {len(mesh.owner):,} unique shared faces. No extrusion, cylinder topology or Cartesian mask is used. Starting from uniform U=(1,0,0), solve the mixed-boundary velocity-potential Poisson problem, then correct native face flux. At the wall dphi/dn=U dot n removes penetration; inlet/farfield derivative zero preserves their normal flux, and outlet phi=0 anchors the system. The whole skew/nonorthogonal face response is assembled with mixed LS boundary rows and CPU sparse LU. One scalar face traction supplies a conservative Gauss cell correction.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src python scripts/initialize_suboff_native.py --mesh {a.mesh} --output NEW_EMPTY_DIRECTORY
```

## Actual mesh and fields

![True z=0 tetrahedral cuts and computed velocity-potential / speed fields](initialization.png)

These polygons are true tetrahedral intersections, not projected edges or interpolated raster cells. Raw native fields and topology are in fields.npz. No pressure or Cp/drag curve is inferred from velocity potential. No physical time has advanced; the velocity represents a slip potential initialization, not a no-slip viscous solution.

## Native projection checks

| Metric | Actual | Limit | Passed |
| --- | ---: | ---: | --- |
{rows}

Independent mixed-BC affine manufacture phi=1.2+0.7x−0.3y+0.2z on this exact mesh: scalar error {manufactured['scalar_max_error']:.6g}, gradient error {manufactured['gradient_max_error']:.6g}, final flux remainder {manufactured['flux_remainder_max']:.6g}. Analytic boundary data are supplied on all physical faces.

## Unresolved cell/face consistency

Native projected divergence is {coupling['native_face_divergence_max']:.6g} U/L. Re-interpolating the corrected conservative cell velocity gives divergence {coupling['interpolated_cell_velocity_divergence_max']:.6g} U/L and maximum interior normal speed difference {coupling['interior_max_normal_velocity_difference']:.6g} U. These are exposed separately: successful face projection does not certify a coupled incompressible Navier--Stokes solver. Numerical flow and physical qualification remain false.

## Performance and next stage

Setup {assembly:.4f} s; one solve {solve_time:.4f} s. The actual full nonorthogonal matrix has {op.matrix.nnz:,} entries and L/U factors have {lu.L.nnz+lu.U.nnz:,} entries ({(sparse_bytes(lu.L)+sparse_bytes(lu.U))/1e6:.3f} MB sparse arrays). These are measured CPU reference costs, not GPU or engineering-scale performance.

Next: conservative low-Re viscous momentum on the native mesh with matching wall pressure response; independent cell/face and stress verification; boundary-layer/wake refinement; then matched experimental Reynolds and original reference data. The present tetrahedral meshes have no prism boundary layer. geometry/operator/projection checks do not imply drag accuracy or mesh independence.
'''
    (out/'report.md').write_text(report)
    write_json(out/'manifest.json',dict(schema='tensorfvm.suboff-native-initialization/1',source_sha256=bindings,input_mesh_sha256=sha(a.mesh),artifact_sha256={str(f.relative_to(out)):sha(f)for f in out.rglob('*')if f.is_file()and f.name!='manifest.json'}))
    print(json.dumps(result,indent=2));return 0 if passed else 2

if __name__=='__main__':raise SystemExit(main())

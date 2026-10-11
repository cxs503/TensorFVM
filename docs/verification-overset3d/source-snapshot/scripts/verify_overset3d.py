#!/usr/bin/env python3
"""Verify static Cartesian / radial-wedge overset connectivity and point transfer."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import time
import numpy as np
from tensorfvm.overset3d import (ACTIVE, HOLE, FRINGE, CartesianGrid3D,
    SphereSolid, sphere_shell_mesh, build_spherical_overset)
from plot_native_suboff_fields import cell_section_polygons


def analytic(points):
    x,y,z=np.asarray(points).T
    return np.column_stack((1+.3*np.sin(.7*x)*np.cos(.6*y)+.2*np.cos(.8*z),
        .8+.2*np.cos(.5*x+.4*y)+.15*np.sin(.9*z),
        1.2+.25*np.sin(.6*y+.3*z)+.1*np.cos(.7*x)))


def packed(rows):
    lengths=np.array([len(v) for v in rows],dtype=np.int64)
    return np.concatenate(rows).astype(np.int64),np.r_[0,np.cumsum(lengths)]


def save_mesh(path,mesh,status):
    cells,cp=packed(mesh.cells); faces,fp=packed(mesh.face_vertices)
    np.savez_compressed(path,vertices=mesh.vertices,cells=cells,cell_offsets=cp,
        cell_types=np.asarray(mesh.cell_types),face_vertices=faces,face_offsets=fp,
        owner=mesh.owner,neighbor=mesh.neighbor,face_centers=mesh.face_centers,
        area_vectors=mesh.area_vectors,cell_centers=mesh.cell_centers,
        cell_volumes=mesh.cell_volumes,patch_labels=mesh.patch_labels,status=status)


def transfer_metrics(donor,receiver,status,mapping):
    points=receiver.cell_centers[mapping.receiver_indices]
    source=analytic(donor.cell_centers); exact=analytic(points); mapped=mapping.interpolate(source)
    weight=receiver.cell_volumes[mapping.receiver_indices]
    vector=float(np.sqrt(np.sum(weight[:,None]*(mapped-exact)**2)/np.sum(weight[:,None]*exact**2)))
    components=np.sqrt(np.sum(weight[:,None]*(mapped-exact)**2,axis=0)/np.sum(weight[:,None]*exact**2,axis=0))
    affine=lambda p:np.column_stack((1+2*p[:,0]-3*p[:,1]+.7*p[:,2], -.2+p[:,0]+.4*p[:,1]-p[:,2]))
    const=float(np.max(np.abs(mapping.interpolate(np.ones(len(donor.cells)))-1)))
    aff=float(np.max(np.abs(mapping.interpolate(affine(donor.cell_centers))-affine(points))))
    return dict(receivers=len(points),constant_max_error=const,affine_max_error=aff,
        nonlinear_relative_l2=vector,nonlinear_component_relative_l2=components.tolist(),
        donors_only_active=bool(np.all(status[mapping.donor_indices]==ACTIVE)),
        containing_cells_only_active=bool(np.all(status[mapping.containing_cells]==ACTIVE)),
        minimum_weight=float(mapping.weights.min()),weight_sum_max_error=float(np.max(abs(mapping.weights.sum(axis=1)-1)))),dict(
        receiver_indices=mapping.receiver_indices,donor_indices=mapping.donor_indices,
        weights=mapping.weights,containing_cells=mapping.containing_cells,
        receiver_points=points,donor_field=source,reference_receiver_field=exact,
        transferred_receiver_field=mapped,receiver_volume_weights=weight)


def plot_case(folder,background,body,connection):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import ListedColormap,BoundaryNorm
    colors=ListedColormap(['#c4e6c3','#d5d5d5','#edaf66']);norm=BoundaryNorm([-.5,.5,1.5,2.5],3)
    fig,axes=plt.subplots(1,2,figsize=(12,6))
    for ax,mesh,status,title in zip(axes,(background,body),(connection.background_status,connection.body_status),('Cartesian background','Radially structured wedge shell')):
        polygons,ids=cell_section_polygons(mesh,axis=2,value=0)
        coll=PolyCollection(polygons,array=status[ids].astype(float),cmap=colors,norm=norm,edgecolors='#444444',linewidths=.22)
        ax.add_collection(coll);ax.autoscale_view();ax.set_aspect('equal');ax.set_xlabel('x');ax.set_ylabel('y');ax.set_title(title+'; actual z=0 intersections')
        np.savez_compressed(folder/(title.split()[0].lower()+'-section.npz'),vertices=np.concatenate(polygons),offsets=np.r_[0,np.cumsum([len(p) for p in polygons])],native_cell_ids=ids,status=status[ids])
    fig.colorbar(coll,ax=axes,ticks=[0,1,2],label='0 active; 1 hole; 2 fringe',shrink=.75)
    fig.suptitle('Static overset geometry; no fluid equations solved')
    fig.savefig(folder/'mesh-classification.png',dpi=180,bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,mapping,donor,receiver,title in zip(axes,(connection.background_from_body,connection.body_from_background),(body,background),(background,body),('Shell → Cartesian','Cartesian → shell')):
        p=receiver.cell_centers[mapping.receiver_indices];error=np.linalg.norm(mapping.interpolate(analytic(donor.cell_centers))-analytic(p),axis=1)
        image=ax.scatter(p[:,0],p[:,1],c=error,s=8,cmap='magma');ax.set_aspect('equal');ax.set_title(title+' fringe point error');ax.set_xlabel('x (projection)');ax.set_ylabel('y (projection)');fig.colorbar(image,ax=ax,label='Absolute vector interpolation error')
    fig.suptitle('All 3D fringe centres projected to x-y; prescribed analytic field, not flow')
    fig.tight_layout();fig.savefig(folder/'fringe-interpolation-error.png',dpi=180);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output
    if out.exists() and any(out.iterdir()):parser.error('output must be empty')
    out.mkdir(parents=True,exist_ok=True);root=Path(__file__).resolve().parents[1]
    sources=('scripts/verify_overset3d.py','scripts/plot_native_suboff_fields.py','src/tensorfvm/overset3d.py','src/tensorfvm/native_mesh3d.py')
    for name in sources:
        dest=out/'source-snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(root/name,dest)
    config=dict(background_bounds=[[-3,-3,-3],[3,3,3]],inner_radius=1,outer_radius=2.6,solid_radius=1,split_radius=1.5,fringe_width=.3,
        levels=[dict(background_n=n,radial_layers=r,angular_subdivision=a) for n,r,a in ((14,10,1),(18,12,2),(22,14,2))],
        field='[1+.3*sin(.7x)*cos(.6y)+.2*cos(.8z), .8+.2*cos(.5x+.4y)+.15*sin(.9z), 1.2+.25*sin(.6y+.3z)+.1*cos(.7x)]',
        value_semantics='cell-centroid point values; not cell volume averages',norm='receiver-volume-weighted relative L2 on fringe centres')
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n');(out/'environment.json').write_text(json.dumps(dict(python=platform.python_version(),numpy=np.__version__,platform=platform.platform()),indent=2)+'\n')
    rows=[]
    for level,settings in enumerate(config['levels']):
        folder=out/f'level-{level+1}';folder.mkdir();start=time.monotonic()
        try:
            n=settings['background_n'];background=CartesianGrid3D.from_bounds((-3,)*3,(3,)*3,(n,)*3)
            body=sphere_shell_mesh(1,2.6,settings['radial_layers'],settings['angular_subdivision'])
            connection=build_spherical_overset(background,body,SphereSolid(np.zeros(3),1),1.5,.3)
            save_mesh(folder/'background-native.npz',background,connection.background_status);save_mesh(folder/'body-native.npz',body,connection.body_status)
            bm,br=transfer_metrics(body,background,connection.body_status,connection.background_from_body)
            rm,rr=transfer_metrics(background,body,connection.background_status,connection.body_from_background)
            np.savez_compressed(folder/'background-from-body.npz',**br);np.savez_compressed(folder/'body-from-background.npz',**rr)
            plot_case(folder,background,body,connection)
            row=dict(level=level+1,**settings,background_cells=len(background.cells),body_cells=len(body.cells),background_from_body=bm,body_from_background=rm,
                background_status_counts=np.bincount(connection.background_status,minlength=3).tolist(),body_status_counts=np.bincount(connection.body_status,minlength=3).tolist(),
                minimum_volume=min(float(background.cell_volumes.min()),float(body.cell_volumes.min())),elapsed_seconds=time.monotonic()-start,completed=True)
        except Exception as exc:
            row=dict(level=level+1,**settings,completed=False,error_type=type(exc).__name__,error=str(exc),elapsed_seconds=time.monotonic()-start)
        rows.append(row);(folder/'summary.json').write_text(json.dumps(row,indent=2)+'\n');print(json.dumps(row),flush=True)
    complete=all(r['completed'] for r in rows)
    decreasing=complete and all(all(rows[i+1][direction]['nonlinear_relative_l2']<rows[i][direction]['nonlinear_relative_l2'] for i in range(2)) for direction in ('background_from_body','body_from_background'))
    invariants=complete and all(all(r[d]['constant_max_error']<1e-12 and r[d]['affine_max_error']<1e-10 and r[d]['donors_only_active'] and r[d]['containing_cells_only_active'] and r[d]['minimum_weight']>=-1e-11 for d in ('background_from_body','body_from_background')) for r in rows)
    fine3=complete and all(rows[-1][d]['nonlinear_relative_l2']<.03 for d in ('background_from_body','body_from_background'))
    summary=dict(rows=rows,three_level_error_decrease=decreasing,invariants_passed=invariants,fine_interpolation_below_three_percent=fine3,connectivity_interpolation_qualified=bool(decreasing and invariants and fine3),flow_solved=False,navier_stokes_qualified=False,conservative_flux_transfer=False,fvm_lbm_coupling_executed=False)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    report='# Three-dimensional Cartesian / body-fitted overset verification\n\n## Scope and method\n\nThe background is a native Cartesian hexahedral grid on [-3,3]^3. The component grid has radially structured wedge layers between radii 1 and 2.6, using an icosphere angular triangulation. Planar facets approximate the spherical wall. A radius-1 solid is excluded. Background centroid classification is hole below radius 1.5, fringe below 1.8, otherwise active; shell centres beyond radius 2.1 are fringe. This is centroid classification, not a cut-cell method or a moving-grid calculation.\n\nBoth transfer directions require a containing active native polyhedron and a convex tetrahedron of active donor centres. A tangent separating half-space protects the entire interpolation tetrahedron from the sphere. No nearest-neighbour fallback or orphan suppression is used. Weights reproduce constants and affine fields. The independently prescribed nonlinear vector field is fixed across all levels; formulas and coordinates are in config.json. Errors use receiver-cell-volume-weighted relative L2 on fringe point values. These weights define a diagnostic norm, not conservative remapping of volume averages.\n\n## Results\n\n|Level|Background cells|Shell cells|Shell → background L2|Background → shell L2|Time s|\n|---|---:|---:|---:|---:|---:|\n'
    for r in rows:
        if r['completed']:report+=f"|{r['level']}|{r['background_cells']}|{r['body_cells']}|{100*r['background_from_body']['nonlinear_relative_l2']:.6f}%|{100*r['body_from_background']['nonlinear_relative_l2']:.6f}%|{r['elapsed_seconds']:.2f}|\n"
        else:report+=f"|{r['level']}|failed|failed|{r['error_type']}: {r['error']}|—|{r['elapsed_seconds']:.2f}|\n"
    report+=f'\nThree-level error decrease: **{decreasing}**. Constants/affine reproduction and active-donor invariants: **{invariants}**. Fine-level two-way interpolation errors below 3%: **{fine3}**. Full raw metrics, both donor maps, native geometry, classifications and fields are archived for independent replay.\n'
    for r in rows:
        if r['completed']:report+=f"\n### Level {r['level']} real geometry and interpolation diagnostic\n\n![Actual mesh classification](level-{r['level']}/mesh-classification.png)\n\n![Fringe interpolation errors](level-{r['level']}/fringe-interpolation-error.png)\n"
    report+='\n## Limits and reproduction\n\nNo Navier–Stokes equations, pressure correction, turbulence, LBM populations, time stepping, moving-solid hole cutting or conservative shared interface flux are solved in this verification. The error plots show point interpolation of a prescribed analytic vector field; they are not velocity or pressure flow contours. The generic native mesh adapter uses a sphere-specific solid-clearance contract; a SUBOFF solid-clearance adapter remains necessary. FVM/LBM coupling requires common units and physical coordinates, shared time advancement, consistent pressure/density conversion, population reconstruction and conservative interface flux. Those are separate acceptance gates.\n\nRun `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src python scripts/verify_overset3d.py --output EMPTY_DIRECTORY`. Producer sources were copied before execution. Every archived output except the manifest itself has a SHA-256 binding in manifest.json.\n'
    (out/'report.md').write_text(report)
    manifest={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.rglob('*')) if p.is_file()}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(summary,indent=2))
    return 0 if summary['connectivity_interpolation_qualified'] else 2

if __name__=='__main__':raise SystemExit(main())

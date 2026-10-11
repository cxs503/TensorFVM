#!/usr/bin/env python3
"""Compare actual numerical/reference centroid values on identical native cuts."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
from plot_native_suboff_fields import load_plot_mesh,cell_section_polygons


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--plane-z',type=float,default=.03)
    parser.add_argument('--x-limits',type=float,nargs=2,default=[-.05,1.08]);parser.add_argument('--y-limits',type=float,nargs=2,default=[-.13,.13])
    args=parser.parse_args();run=args.run.resolve();out=args.output.resolve()
    if out.exists():parser.error('output must be new to preserve prior evidence')
    if not np.isfinite(args.plane_z):parser.error('finite section plane required')
    for limits in (args.x_limits,args.y_limits):
        if not np.isfinite(limits).all() or limits[0]>=limits[1]:parser.error('finite increasing display limits required')
    fields_path=run/'fields.npz';audit_path=run/'audit/audit.json';audit=json.loads(audit_path.read_text())
    bound=audit.get('producer_input_sha256',{})
    for path in (fields_path,run/'config.json',run/'input.msh'):
        # Archived runs can move with a repository checkout. Require one
        # unambiguous producer input of this name and the exact same bytes.
        matches=[value for name,value in bound.items() if Path(name).name==path.name]
        if len(matches)!=1 or matches[0]!=sha(path):raise ValueError('audit does not bind these exact plotting inputs: '+str(path))
    mesh,policy,merge=load_plot_mesh(run/'input.msh',fields_path)
    with np.load(fields_path,allow_pickle=False) as fields:
        data={name:fields[name].copy() for name in ('velocity','reference_velocity','pressure','reference_pressure','flux')}
    n=len(mesh.cells);f=len(mesh.owner)
    for name in ('velocity','reference_velocity'):
        if data[name].shape!=(n,3):raise ValueError('invalid cell velocity shape')
    for name in ('pressure','reference_pressure'):
        if data[name].shape!=(n,):raise ValueError('invalid cell pressure shape')
    if data['flux'].shape!=(f,) or not all(np.isfinite(a).all() for a in data.values()):raise ValueError('invalid field topology or nonfinite values')
    polygons,ids=cell_section_polygons(mesh,value=args.plane_z)
    # Colour scales must describe the displayed window, not distant cells
    # outside the axes. Keep true polygons and actual centroid field samples.
    visible=np.array([poly[:,0].max()>=args.x_limits[0] and poly[:,0].min()<=args.x_limits[1]
                      and poly[:,1].max()>=args.y_limits[0] and poly[:,1].min()<=args.y_limits[1] for poly in polygons],dtype=bool)
    polygons=[poly for poly,keep in zip(polygons,visible) if keep];ids=ids[visible]
    if not len(ids):raise ValueError('empty native section')
    volumes=mesh.cell_volumes;mean=lambda a:float(np.sum(volumes*a)/volumes.sum())
    numerical_mean=mean(data['pressure']);reference_mean=mean(data['reference_pressure'])
    numerical=data['pressure']-numerical_mean;reference=data['reference_pressure']-reference_mean
    speed=np.linalg.norm(data['velocity'],axis=1);ref_speed=np.linalg.norm(data['reference_velocity'],axis=1)
    p_error=np.abs(numerical-reference);u_error=np.linalg.norm(data['velocity']-data['reference_velocity'],axis=1)
    pressure_limits=(float(min(numerical[ids].min(),reference[ids].min())),float(max(numerical[ids].max(),reference[ids].max())))
    speed_limits=(float(min(speed[ids].min(),ref_speed[ids].min())),float(max(speed[ids].max(),ref_speed[ids].max())))
    out.mkdir(parents=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    fig,axes=plt.subplots(3,2,figsize=(15,11),layout='constrained')
    arrays=(numerical[ids],speed[ids],reference[ids],ref_speed[ids],p_error[ids],u_error[ids])
    titles=('Numerical pressure − global volume mean','Numerical speed |u|','Reference pressure − global volume mean','Reference speed |u|','Absolute gauge-aligned pressure error','Vector velocity error magnitude')
    for i,(ax,array,title) in enumerate(zip(axes.ravel(),arrays,titles)):
        col=PolyCollection(polygons,array=array,cmap='coolwarm' if i in (0,2) else 'viridis',edgecolors='#777777',linewidths=.08)
        limits=pressure_limits if i in (0,2) else speed_limits if i in (1,3) else (0,float(array.max()))
        col.set_clim(*limits);ax.add_collection(col);ax.set(xlim=args.x_limits,ylim=args.y_limits,xlabel='x/L',ylabel='y/L',title=title);ax.set_aspect('equal');fig.colorbar(col,ax=ax,label='Solver units')
    status='Independent manufactured audit passed' if audit.get('passed') is True else 'Independent manufactured audit FAILED'
    fig.suptitle(f'Actual z/L={args.plane_z:g} cell cuts; identical centroid sampling, no smoothing\n{status}; physical SUBOFF unqualified')
    fig.savefig(out/'comparison.png',dpi=190,bbox_inches='tight');fig.savefig(out/'comparison.pdf',bbox_inches='tight');plt.close(fig)
    offsets=np.r_[0,np.cumsum([len(p) for p in polygons])]
    np.savez_compressed(out/'section-comparison.npz',polygon_vertices=np.concatenate(polygons),polygon_offsets=offsets,native_cell_indices=ids,cell_centers=mesh.cell_centers[ids],numerical_pressure_mean_removed=numerical[ids],reference_pressure_mean_removed=reference[ids],numerical_speed=speed[ids],reference_speed=ref_speed[ids],absolute_pressure_error=p_error[ids],vector_velocity_error=u_error[ids])
    summary=dict(plane_z_L=args.plane_z,face_policy=policy,section_polygons=len(polygons),pressure_common_limits=list(pressure_limits),speed_common_limits=list(speed_limits),pressure_alignment=dict(kind='subtract each field own full-domain cell-volume-weighted mean',numerical_mean=numerical_mean,reference_mean=reference_mean,numerical_to_reference_gauge_offset=reference_mean-numerical_mean),audit_passed=audit.get('passed') is True,audit_sha256=sha(audit_path),audit_current_equation_residuals=audit.get('current_equation_residuals'),audit_relative_L2_errors=audit.get('relative_L2_errors'),experimental_accuracy_qualified=False,representation='identical true native section polygons; actual numerical/reference values evaluated at same corresponding native cell centroid; no smoothing, plane interpolation or geometric projection')
    summary['display_window_L']=dict(x=args.x_limits,y=args.y_limits)
    summary['colour_scale_population']='actual native section polygons intersecting the displayed rectangular window; cells are not volume-clipped'
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    (out/'README.md').write_text(f'''# Manufactured SUBOFF: numerical/reference comparison

![Actual centroid field comparison](comparison.png)

The section is **z/L={args.plane_z:g}**, displayed over x/L={args.x_limits}, y/L={args.y_limits}. Rows 1 and 2 show numerical and independently prescribed reference pressure/speed on the **same actual native cell polygons**, using values at the **same actual cell centroids**. Each column shares one colour range across numerical/reference rows, computed from cells whose section polygons intersect this window. Pressure means are removed separately using **all cell volumes in the complete domain**; no fitted amplitude or local offset is applied. Row 3 shows absolute gauge-aligned pressure error and vector velocity error magnitude.

Intersected tetrahedron centroids may lie away from the plane. Similar polygon patterns in the reference and numerical plots reveal this centroid sampling effect; the error row separates their discrepancy. No smoothing, reconstructed plane values, off-plane geometric projection or fake experimental contours are used. Native polygons are exact planar facet intersections.

The source NPZ and mesh/config hashes match the independent audit bindings. Audit passed: **{summary['audit_passed']}**. Full-field errors and actual current-equation residuals are copied from that bound audit and remain in summary.json. This plot cannot create a numerical or physical qualification. **Physical SUBOFF drag/Cp validation remains false.** Polygon coordinates, native cell IDs and displayed arrays are saved in section-comparison.npz.

Run `PYTHONPATH=src python scripts/plot_native_manufactured_comparison.py --run RUN_DIRECTORY --output NEW_DIRECTORY --plane-z .03` from the repository root. The helper import uses the executable script directory in Python's normal script search path.
''')
    sources=(Path(__file__).resolve(),Path(__file__).resolve().parent/'plot_native_suboff_fields.py')
    for source in sources:shutil.copy2(source,out/source.name)
    manifest=dict(input_sha256={str(p):sha(p) for p in (fields_path,run/'config.json',run/'input.msh',audit_path)},artifact_sha256={str(p.relative_to(out)):sha(p) for p in sorted(out.iterdir()) if p.is_file()})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(summary,indent=2))

if __name__=='__main__':main()

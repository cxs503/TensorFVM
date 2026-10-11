#!/usr/bin/env python3
"""Plot real native mixed-cell sections and actual SUBOFF solution fields.

No center interpolation, geometric projection, or fabricated experimental Cp.
Default labels deliberately identify unqualified diagnostic computations.
"""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np


def cell_section_polygons(mesh, *, axis=2, value=0., tolerance=1e-11):
    """Intersect actual planar native facets, then stitch each cell boundary.

    Coplanar facet edges cancel their internal triangulation diagonals. Normal
    intersections are retained as segments. Returns polygons and native cell
    indices; zero-area tangencies are omitted. Nonclosed finite-area sections
    fail rather than silently substituting a convex hull.
    """
    if axis not in (0,1,2) or not np.isfinite(value) or tolerance<=0:
        raise ValueError('finite plane and positive tolerance required')
    dimensions=[i for i in range(3) if i!=axis]
    normal=[set() for _ in mesh.cells];coplanar=[{} for _ in mesh.cells]
    coordinates={}
    def key(point):
        projected=point[dimensions]
        k=tuple(np.rint(projected/tolerance).astype(np.int64))
        coordinates.setdefault(k,projected.copy())
        return k
    for fi,ids in enumerate(mesh.face_vertices):
        pts=mesh.vertices[list(ids)];d=pts[:,axis]-value
        d[np.abs(d)<=tolerance]=0.
        if d.min()>0 or d.max()<0:continue
        owners=[int(mesh.owner[fi])]
        if mesh.neighbor[fi]>=0:owners.append(int(mesh.neighbor[fi]))
        if np.all(d==0):
            edges=[tuple(sorted((key(pts[j]),key(pts[(j+1)%len(pts)])))) for j in range(len(pts))]
            for ci in owners:
                for edge in edges:
                    if edge[0]!=edge[1]:coplanar[ci][edge]=coplanar[ci].get(edge,0)+1
            continue
        hits=[]
        for j in range(len(pts)):
            k=(j+1)%len(pts)
            if d[j]==0:hits.append(key(pts[j]))
            if d[j]*d[k]<0:hits.append(key(pts[j]+d[j]/(d[j]-d[k])*(pts[k]-pts[j])))
        hits=list(dict.fromkeys(hits))
        if len(hits)==2:
            edge=tuple(sorted(hits))
            for ci in owners:normal[ci].add(edge)
        elif len(hits)>2:raise ValueError('nonplanar facet plane intersection')
    polygons=[];cell_ids=[]
    for ci in range(len(mesh.cells)):
        edges=normal[ci]|{e for e,count in coplanar[ci].items() if count%2}
        if not edges:continue
        graph={}
        for a,b in edges:
            graph.setdefault(a,set()).add(b);graph.setdefault(b,set()).add(a)
        if any(len(v)!=2 for v in graph.values()):
            # Edge/point tangencies have no finite area and are not cells on
            # the plane. A genuine broken finite-area ring is an error.
            points=np.array([coordinates[k] for k in graph])
            if len(points)<3 or np.linalg.matrix_rank(points-points[0],tol=tolerance)<=1:continue
            raise ValueError(f'nonclosed section boundary in native cell {ci}')
        unused=set(graph)
        while unused:
            start=min(unused);ring=[start];previous=None;current=start
            for _ in range(len(graph)+1):
                following=min(graph[current]-({previous} if previous is not None else set()))
                if following==start:break
                if following in ring:raise ValueError(f'non-simple section in native cell {ci}')
                ring.append(following);previous,current=current,following
            else:raise ValueError(f'open section in native cell {ci}')
            unused.difference_update(ring)
            polygon=np.array([coordinates[k] for k in ring])
            area=.5*np.sum(polygon[:,0]*np.roll(polygon[:,1],-1)-polygon[:,1]*np.roll(polygon[:,0],-1))
            if abs(area)>tolerance*tolerance:
                if area<0:polygon=polygon[::-1]
                polygons.append(polygon);cell_ids.append(ci)
    return polygons,np.asarray(cell_ids,dtype=int)



def load_plot_mesh(mesh_path, fields_path):
    """Reproduce the producer face policy from the adjacent run config.

    Legacy runs without this option retain the original triangulated topology.
    No flux or face field is interpolated to accommodate a topology mismatch.
    """
    from tensorfvm.native_mesh3d import NativeMesh3D
    config_path=Path(fields_path).parent/'config.json'
    config=json.loads(config_path.read_text()) if config_path.exists() else {}
    if not isinstance(config,dict):raise ValueError('run config must be a JSON object')
    if 'mesh_sha256' in config and hashlib.sha256(Path(mesh_path).read_bytes()).hexdigest()!=config['mesh_sha256']:
        raise ValueError('input mesh SHA256 does not match the field producer config')
    merged=config.get('merge_coplanar_faces',False)
    if not isinstance(merged,bool):raise ValueError('merge_coplanar_faces must be a JSON boolean')
    mesh=NativeMesh3D.from_gmsh(mesh_path,triangulate_faces=True)
    original_faces=len(mesh.owner);merge_metadata=None
    if merged:
        from tensorfvm.native_face_geometry3d import merge_coplanar_faces
        mesh,merge_metadata=merge_coplanar_faces(mesh)
    policy=dict(face_policy='merged_coplanar_polygon_faces' if merged else 'triangulated_native_faces',
        merge_coplanar_faces=merged,original_faces=original_faces,reconstructed_faces=len(mesh.owner),
        merged_pairs=merge_metadata['merged_pairs'] if merge_metadata else 0,
        config_path=str(config_path) if config_path.exists() else None,
        config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest() if config_path.exists() else None)
    return mesh,policy,merge_metadata


def plot_fields(mesh_path,fields_path,out,*,status_text='Diagnostic / unqualified',case_kind='diagnostic',plane=0.):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    out=Path(out)
    if out.exists() and any(out.iterdir()):raise ValueError('output must be empty to preserve evidence')
    out.mkdir(parents=True,exist_ok=True)
    mesh,face_policy,merge_metadata=load_plot_mesh(mesh_path,fields_path)
    fields=np.load(fields_path,allow_pickle=False);velocity=fields['velocity'];pressure=fields['pressure'];flux=fields['flux']
    if velocity.shape!=(len(mesh.cells),3) or pressure.shape!=(len(mesh.cells),) or flux.shape!=(len(mesh.owner),):
        raise ValueError(f'field shapes do not match reconstructed native mesh cells/faces ({face_policy["face_policy"]}, {len(mesh.cells)} cells, {len(mesh.owner)} faces; flux shape {flux.shape})')
    if not all(np.isfinite(v).all() for v in (velocity,pressure,flux)):raise ValueError('nonfinite solution cannot be plotted')
    if 'pressure_faces' in fields and (fields['pressure_faces'].shape!=(len(mesh.owner),) or not np.isfinite(fields['pressure_faces']).all()):
        raise ValueError('invalid reconstructed face pressure array for configured face policy')
    polygons,ids=cell_section_polygons(mesh,value=plane)
    if not len(ids):raise ValueError('no finite area cells intersect the requested plane')
    speed=np.linalg.norm(velocity,axis=1)
    limits=(-.05,1.08,-.13,.13)
    fig,axes=plt.subplots(3,1,figsize=(13,9))
    col=PolyCollection(polygons,facecolors='none',edgecolors='#555555',linewidths=.25);axes[0].add_collection(col);axes[0].set_title(f'Actual native cell polygons: z/L={plane:g}')
    for ax,data,title,label in zip(axes[1:],[pressure,speed],['Cell pressure','Velocity magnitude'],['Pressure (solver units)','Speed (solver units)']):
        col=PolyCollection(polygons,array=data[ids],cmap='coolwarm' if title=='Cell pressure' else 'viridis',edgecolors='none')
        ax.add_collection(col);fig.colorbar(col,ax=ax,label=label,pad=.02);ax.set_title(title+'; piecewise constant actual cell values')
    for ax in axes:
        ax.set_xlim(limits[:2]);ax.set_ylim(limits[2:]);ax.set_aspect('equal');ax.set_xlabel('x/L');ax.set_ylabel('y/L');ax.text(.01,.96,status_text,transform=ax.transAxes,va='top',color='#8b0000',fontsize=10,bbox={'facecolor':'white','alpha':.85,'edgecolor':'none'})
    fig.suptitle('SUBOFF native mixed-cell section — '+case_kind)
    fig.tight_layout();fig.savefig(out/'fields.png',dpi=180);fig.savefig(out/'fields.pdf');plt.close(fig)
    wall=np.flatnonzero(mesh.patch_labels=='wall');x=mesh.face_centers[wall,0];adjacent_pressure=pressure[mesh.owner[wall]];area=np.linalg.norm(mesh.area_vectors[wall],axis=1)
    if 'pressure_faces' in fields:
        face_pressure=fields['pressure_faces']
        if face_pressure.shape!=(len(mesh.owner),) or not np.isfinite(face_pressure).all():
            raise ValueError('invalid reconstructed face pressure array')
        p=face_pressure[wall];wall_definition='Reconstructed native wall-face pressure supplied by the solver'
    else:
        p=adjacent_pressure;wall_definition='Actual cell pressure adjacent to native wall face; no reconstructed wall pressure'

    with (out/'wall-pressure.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['native_face_index','x_L','y_L','z_L','plotted_wall_pressure','adjacent_cell_pressure','face_area_L2'])
        for fi,xx,pp,aa in zip(wall,x,p,area):writer.writerow([fi,*mesh.face_centers[fi],pp,pressure[mesh.owner[fi]],aa])
    # Actual face samples are primary. Area-weighted axial bins are explicitly
    # labeled summaries, not experimental pressure coefficients or tap data.
    bins=np.linspace(0,1,51);centers=[];means=[]
    for lo,hi in zip(bins[:-1],bins[1:]):
        selected=(x>=lo)&(x<hi if hi<1 else x<=hi)
        if selected.any():centers.append(float(np.average(x[selected],weights=area[selected])));means.append(float(np.average(p[selected],weights=area[selected])))
    fig,ax=plt.subplots(figsize=(10,4));ax.scatter(x,p,s=3,alpha=.2,label=wall_definition);ax.plot(centers,means,'o-',ms=3,label='Area-weighted axial bin summary')
    ax.set_xlabel('x/L');ax.set_ylabel('Pressure (solver units)');ax.set_title('Manufactured solution: computed wall pressure; not experimental Cp' if case_kind=='manufactured' else 'Computed wall pressure; not experimental Cp')
    ax.text(.01,.98,status_text,transform=ax.transAxes,va='top',color='#8b0000');ax.legend(loc='lower right');fig.tight_layout();fig.savefig(out/'wall-pressure.png',dpi=180);fig.savefig(out/'wall-pressure.pdf');plt.close(fig)
    types=np.asarray(mesh.cell_types)
    summary={'mesh_path':str(mesh_path),'mesh_sha256':hashlib.sha256(Path(mesh_path).read_bytes()).hexdigest(),'fields_path':str(fields_path),'fields_sha256':hashlib.sha256(Path(fields_path).read_bytes()).hexdigest(),'case_kind':case_kind,'status_text':status_text,'plane_z_L':plane,'section_polygons':len(polygons),'section_cells_by_type':{str(k):int(v) for k,v in zip(*np.unique(types[ids],return_counts=True))},'pressure_range':[float(pressure.min()),float(pressure.max())],'speed_range':[float(speed.min()),float(speed.max())],'section_pressure_range':[float(pressure[ids].min()),float(pressure[ids].max())],'section_speed_range':[float(speed[ids].min()),float(speed[ids].max())],'wall_pressure_definition':wall_definition+'; not experimental Cp','representation':'Native facet plane intersection polygons, piecewise constant cell fields','qualification_inferred_from_figures':False}
    summary.update(face_policy)
    if face_policy['config_path'] is not None:(out/'run-config.json').write_bytes(Path(face_policy['config_path']).read_bytes())
    if merge_metadata is not None:
        lengths=np.array([len(g) for g in merge_metadata['original_face_groups']],dtype=int)
        np.savez_compressed(out/'face-topology-map.npz',original_to_merged=merge_metadata['original_to_merged'],group_indices=np.concatenate(merge_metadata['original_face_groups']),group_offsets=np.r_[0,np.cumsum(lengths)])
        from tensorfvm import native_face_geometry3d
        geometry_source=Path(native_face_geometry3d.__file__)
        (out/'face-geometry-source.py').write_bytes(geometry_source.read_bytes())
        summary['face_geometry_source_sha256']=hashlib.sha256(geometry_source.read_bytes()).hexdigest()
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(out/'plot-source.py').write_bytes(Path(__file__).read_bytes())
    (out/'manifest.json').write_text(json.dumps({f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in out.iterdir() if f.is_file() and f.name!='manifest.json'},indent=2)+'\n')
    return summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--mesh',type=Path,required=True);p.add_argument('--fields',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--case-kind',choices=('diagnostic','manufactured','physical'),default='diagnostic');p.add_argument('--status-text',default='Diagnostic / unqualified');p.add_argument('--plane-z',type=float,default=0.)
    a=p.parse_args();print(json.dumps(plot_fields(a.mesh,a.fields,a.output,status_text=a.status_text,case_kind=a.case_kind,plane=a.plane_z),indent=2))

if __name__=='__main__':main()

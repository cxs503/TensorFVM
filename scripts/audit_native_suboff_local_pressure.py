"""Audit local manufactured pressure errors without regional gauge fitting."""
import sys,json,time,hashlib,argparse,shutil
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from tensorfvm.native_mesh3d import NativeMesh3D
from tensorfvm.native_face_geometry3d import merge_coplanar_faces
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--runs',type=Path,nargs='+',required=True);parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
if args.output.exists():parser.error('output must be new to preserve evidence')
if len({p.name for p in args.runs})!=len(args.runs):parser.error('run names must be unique')
args.output.mkdir(parents=True);out={};start=time.perf_counter()
for case in args.runs:
 case=case.resolve();name=case.name;
 audit=json.loads((case/'audit/audit.json').read_text());bindings=audit['producer_input_sha256']
 for path in (case/'config.json',case/'fields.npz',case/'input.msh'):
  hashes=[v for k,v in bindings.items() if Path(k).name==path.name]
  if len(hashes)!=1 or hashes[0]!=hashlib.sha256(path.read_bytes()).hexdigest():raise ValueError('input does not match independent audit: '+str(path))
 config=json.loads((case/'config.json').read_text());
 if config.get('mode')!='manufactured':raise ValueError('this fixed ABC pressure scale applies to manufactured runs only')
 fields=np.load(case/'fields.npz');m=NativeMesh3D.from_gmsh(case/'input.msh',triangulate_faces=True)
 if config.get('merge_coplanar_faces',False):m=merge_coplanar_faces(m)[0]
 p=fields['pressure'];ref=fields['reference_pressure'];V=m.cell_volumes
 pa=p-np.dot(p,V)/V.sum();pr=ref-np.dot(ref,V)/V.sum();e=pa-pr
 wall=np.unique(m.owner[m.patch_labels=='wall']);wallmask=np.zeros(len(p),bool);wallmask[wall]=True;types=np.asarray(m.cell_types);x=m.cell_centers[:,0]
 radial=np.linalg.norm(m.cell_centers[:,1:],axis=1)
 # Fixed spatial windows keep their definition across mesh refinements. Wall
 # owners alone cover a progressively thinner region as the first layer shrinks.
 fixed_windows={'near_body_fixed_window':{'x_min':-.025,'x_max':1.025,'radius_max':.08},
                'near_tail_fixed_window':{'x_min':.9,'x_max':1.025,'radius_max':.025}}
 regions={'all':np.ones(len(p),bool),'wall_adjacent':wallmask,'nonwall':~wallmask,'fore_x_less_0233':x<3.333333/14.291667,'midbody':(x>=3.333333/14.291667)&(x<=10.645833/14.291667),'aftbody':x>10.645833/14.291667,'tail_x_greater_09':x>.9}
 for name_window,window in fixed_windows.items():
  regions[name_window]=(x>=window['x_min'])&(x<=window['x_max'])&(radial<=window['radius_max'])
 for t in sorted(set(types)):regions[t]=types==t;regions['wall_adjacent_'+t]=wallmask&(types==t)
 stats={}
 for region,mask in regions.items():
  ids=np.flatnonzero(mask)
  if not len(ids):continue
  normref=np.dot(pr[ids]**2,V[ids]);error=np.dot(e[ids]**2,V[ids]);j=ids[np.argmax(abs(e[ids]))]
  stats[region]={'cells':len(ids),'volume':float(V[ids].sum()),'volume_fraction':float(V[ids].sum()/V.sum()),'pressure_relative_L2_global_gauge_aligned':float(np.sqrt(error/normref)) if normref>0 else None,'error_energy_fraction':float(error/np.dot(e*e,V)) if np.dot(e*e,V)>0 else 0.,'maximum_absolute_error':float(abs(e[j])),'maximum_error_over_rho_U2':float(abs(e[j])/.25),'maximum_cell':int(j),'maximum_center':m.cell_centers[j].tolist(),'maximum_cell_type':str(types[j]),'maximum_cell_volume':float(V[j]),'maximum_reference_pressure_global_aligned':float(pr[j]),'maximum_computed_pressure_global_aligned':float(pa[j])}
  exceeds=abs(e[ids])>.03*.25
  stats[region].update(rms_error_over_rho_U2=float(np.sqrt(error/V[ids].sum())/.25),
      cells_above_3percent_rho_U2=int(exceeds.sum()),
      region_volume_fraction_above_3percent_rho_U2=float(V[ids][exceeds].sum()/V[ids].sum()))
 ids=np.argsort(abs(e))[-10:][::-1];hotspots=[]
 for j in ids:
  faces=np.flatnonzero((m.owner==j)|(m.neighbor==j));neighbors=np.unique(np.r_[m.neighbor[faces[m.owner[faces]==j]],m.owner[faces[m.neighbor[faces]==j]]]);neighbors=neighbors[neighbors>=0]
  d=m.cell_centers[np.where(m.neighbor[faces]>=0,m.neighbor[faces],m.owner[faces])]-m.cell_centers[m.owner[faces]];boundary=m.neighbor[faces]<0;d[boundary]=m.face_centers[faces[boundary]]-m.cell_centers[m.owner[faces[boundary]]];S=m.area_vectors[faces];angles=np.rad2deg(np.arccos(np.clip(np.einsum('ij,ij->i',S,d)/(np.linalg.norm(S,axis=1)*np.linalg.norm(d,axis=1)),-1,1)))
  hotspots.append({'cell':int(j),'type':str(types[j]),'center':m.cell_centers[j].tolist(),'volume':float(V[j]),'signed_error_global_gauge':float(e[j]),'error_over_rho_U2':float(e[j]/.25),'wall_adjacent':bool(wallmask[j]),'face_nonorthogonality_max':float(angles.max()),'neighbor_cells':neighbors.tolist(),'neighbor_errors_global_gauge':e[neighbors].tolist(),'neighbor_volumes':V[neighbors].tolist(),'cell_neighbor_volume_ratio_max':float(max(np.max(V[neighbors]/V[j]),np.max(V[j]/V[neighbors]))) if len(neighbors) else None})
 out[name]={'mesh_sha256':hashlib.sha256((case/'input.msh').read_bytes()).hexdigest(),'fields_sha256':hashlib.sha256((case/'fields.npz').read_bytes()).hexdigest(),'cells':len(p),'faces':len(m.owner),'gauge_definition':'remove one actual and one reference volume-weighted mean over ALL cells, no regional gauge removal','actual_removed_global_mean':float(np.dot(p,V)/V.sum()),'reference_removed_global_mean':float(np.dot(ref,V)/V.sum()),'pressure_fixed_scale_rho_U2':.25,'regions':stats,'top10_hotspots':hotspots,'scope':'raw field/local spatial accuracy audit, no new solve or changed acceptance threshold'}
 out[name]['fixed_spatial_windows_L']=fixed_windows
 out[name]['window_selection']='cell centroids inside fixed cylinders; no partial-cell clipping, hull-distance estimate, or regional gauge fit'
 out[name]['fixed_scale_diagnostic']='RMS and maximum error divided by rho*U^2=0.25; not reference-relative L2, physical Cp, or a replacement acceptance gate'
 (args.output/'result.json').write_text(json.dumps(out,indent=2));print(json.dumps({'case':name,'regions':{key:stats[key] for key in ['all','wall_adjacent','hexahedron','wedge','tetra','tail_x_greater_09'] if key in stats},'worst':hotspots[0]},indent=2),flush=True)
print(json.dumps({'seconds':time.perf_counter()-start}))

shutil.copy2(Path(__file__).resolve(),args.output/'executed-audit.py')
repo=Path(__file__).resolve().parents[1]
sources={}
for relative in ('src/tensorfvm/native_mesh3d.py','src/tensorfvm/native_face_geometry3d.py'):
 source=repo/relative;target=args.output/'source-snapshot'/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target);sources[relative]=hashlib.sha256(target.read_bytes()).hexdigest()
(args.output/'source-bindings.json').write_text(json.dumps(sources,indent=2)+'\n')
manifest={str(p.relative_to(args.output)):hashlib.sha256(p.read_bytes()).hexdigest() for p in args.output.rglob('*') if p.is_file()}
(args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')

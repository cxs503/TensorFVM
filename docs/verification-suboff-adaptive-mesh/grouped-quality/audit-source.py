#!/usr/bin/env python3
"""Supplemental grouped nonorthogonality audit of an actual native mesh NPZ."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--geometry',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
if args.output.exists() and any(args.output.iterdir()):raise SystemExit('Output must be empty')
args.output.mkdir(parents=True,exist_ok=True)
a=np.load(args.geometry);owner=a['owner'];neighbor=a['neighbor'];area=a['area_vectors'];inside=neighbor>=0
vectors=a['face_centers']-a['cell_centers'][owner];vectors[inside]=a['cell_centers'][neighbor[inside]]-a['cell_centers'][owner[inside]]
angles=np.degrees(np.arccos(np.clip(np.einsum('ij,ij->i',area,vectors)/(np.linalg.norm(area,axis=1)*np.linalg.norm(vectors,axis=1)),-1,1)))
k=a['cell_types'];left=k[owner];right=k[np.maximum(neighbor,0)]
groups={'all_internal':inside,'tet_to_tet':inside&(left=='tetra')&(right=='tetra'),'layer_to_layer':inside&(left!='tetra')&(right!='tetra'),'interface':inside&((left=='tetra')!=(right=='tetra')),'wall':a['patch_labels']=='wall'}
result={}
for name,mask in groups.items():
 ids=np.flatnonzero(mask);maximum=int(ids[np.argmax(angles[mask])])
 result[name]={'faces':len(ids),'maximum_degrees':float(angles[maximum]),'p95_degrees':float(np.percentile(angles[mask],95)),'maximum_face_index':maximum,'maximum_face_center_L':a['face_centers'][maximum].tolist()}
s={'geometry_sha256':hashlib.sha256(args.geometry.read_bytes()).hexdigest(),'geometry_path':str(args.geometry),'nonorthogonality':result,'flow_qualified':False}
(args.output/'quality.json').write_text(json.dumps(s,indent=2)+'\n')
(args.output/'audit-source.py').write_bytes(Path(__file__).read_bytes())
fig,ax=plt.subplots(figsize=(9,4));names=list(result);x=np.arange(len(names));ax.bar(x-.18,[result[n]['maximum_degrees'] for n in names],.36,label='Maximum');ax.bar(x+.18,[result[n]['p95_degrees'] for n in names],.36,label='P95');ax.set_xticks(x,names);ax.set_ylabel('Nonorthogonality (degrees)');ax.legend();ax.set_ylim(0,95);fig.tight_layout();fig.savefig(args.output/'quality.png',dpi=160);fig.savefig(args.output/'quality.pdf');plt.close(fig)
(args.output/'manifest.json').write_text(json.dumps({f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in args.output.iterdir() if f.is_file() and f.name!='manifest.json'},indent=2)+'\n')
print(json.dumps(s,indent=2))

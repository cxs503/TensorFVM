from pathlib import Path
from types import SimpleNamespace
import json, numpy as np,gmsh,torch,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from tensorfvm.multi_element import ThreeElementMesh
from tensorfvm.hlpw30p30n import CAD_FILE,CAD_CURVE_TAGS,CAD_SCALE_MM_PER_CHORD
out=Path('results/30p-hlpw4-cad-triangles-final');torch.set_num_threads(1)
m=ThreeElementMesh(SimpleNamespace(mesh_file=out/'mesh.msh',device='cpu'))
gmsh.initialize();gmsh.option.setNumber('General.Terminal',0);gmsh.option.setString('Geometry.OCCTargetUnit','MM');gmsh.model.occ.importShapes(str(CAD_FILE.resolve()),highestDimOnly=False);gmsh.model.occ.synchronize()
errors={}
for name,tags in CAD_CURVE_TAGS.items():
    ids=np.flatnonzero(m.masks[name].numpy());p=np.unique(m.face_vertices[ids].numpy().reshape(-1,2),axis=0);dist=[]
    for x,y in p:
        xyz=[float(x*CAD_SCALE_MM_PER_CHORD),float(y*CAD_SCALE_MM_PER_CHORD),0.]
        dist.append(min(np.linalg.norm(np.asarray(gmsh.model.getClosestPoint(1,t,xyz)[0])-xyz) for t in tags)/CAD_SCALE_MM_PER_CHORD)
    errors[name]=dict(wall_nodes=len(p),maximum_distance_C=float(max(dist)))
gmsh.finalize()
(out/'cad-wall-audit.json').write_text(json.dumps(errors,indent=2)+'\n')
fig,axs=plt.subplots(1,2,figsize=(13,5));vertices=m.vertices.numpy();edges=m.face_vertices.numpy()
for ax,lims in zip(axs,[(-.14,1.18,-.27,.24),(-.055,-.01,-.127,-.09)]):
    centers=edges.mean(1);keep=(centers[:,0]>lims[0])&(centers[:,0]<lims[1])&(centers[:,1]>lims[2])&(centers[:,1]<lims[3]);ax.add_collection(LineCollection(edges[keep],linewidths=.3,color='gray'))
    for name,color in [('slat','tab:blue'),('main','tab:orange'),('flap','tab:green')]:ax.add_collection(LineCollection(edges[m.masks[name].numpy()],linewidths=1,color=color,label=name))
    ax.set(xlim=lims[:2],ylim=lims[2:],xlabel='x/C',ylabel='y/C',aspect='equal');ax.legend()
axs[0].set_title('Official CAD triangles: geometry control, no boundary layers');axs[1].set_title('True slat cove');fig.tight_layout();fig.savefig(out/'cad-mesh-geometry.png',dpi=180);plt.close(fig)
print(errors)

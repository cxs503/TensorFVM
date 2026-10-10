"""Measure generated wall layers; requested Gmsh options alone are not evidence."""
import numpy as np
from .mesh_quality import mesh_quality


def hybrid_mesh_quality(mesh):
    a=lambda x:x.detach().cpu().numpy()
    cells=mesh.cell_nodes
    kinds=np.array([len(c) for c in cells])
    owner,neighbor=a(mesh.owner),a(mesh.neighbor)
    faces=a(mesh.face_vertices);centers=a(mesh.centers).reshape(-1,2)
    normals=a(mesh.face_normals);fc=a(mesh.face_centers)
    wall=np.flatnonzero(a(mesh.masks[mesh.body_label]))
    distance=np.einsum('ij,ij->i',fc[wall]-centers[owner[wall]],normals[wall])
    incidence=[[] for _ in cells]
    for fi,(o,n) in enumerate(zip(owner,neighbor)):
        incidence[o].append(fi)
        if n>=0:incidence[n].append(fi)
    lengths=[];ratios=[];depths=[]
    for fi in wall:
        current=owner[fi];entry=fi;steps=[];visited=set()
        while kinds[current]==4 and current not in visited:
            visited.add(current)
            # Opposite quad face shares no vertex with the entry face.
            endpoints=faces[entry]
            opposite=[f for f in incidence[current] if not np.any(np.all(faces[f][:,None,:]==endpoints[None,:,:],axis=-1))]
            if len(opposite)!=1:break
            exit_face=opposite[0]
            steps.append(float(np.linalg.norm(fc[exit_face]-fc[entry])))
            next_cell=neighbor[exit_face] if owner[exit_face]==current else owner[exit_face]
            if next_cell<0:break
            current=next_cell;entry=exit_face
        depths.append(len(steps));lengths.extend(steps)
        ratios.extend(np.array(steps[1:])/np.array(steps[:-1]))
    report=mesh_quality(mesh)
    report.update(triangle_cells=int(sum(kinds==3)),quadrilateral_cells=int(sum(kinds==4)),
                  wall_faces=int(len(wall)),wall_quad_coverage=float(np.mean(kinds[owner[wall]]==4)),
                  wall_cell_normal_distance_m=dict(min=float(distance.min()),median=float(np.median(distance)),max=float(distance.max())),
                  quad_layers=dict(min=int(min(depths)),median=float(np.median(depths)),max=int(max(depths))),
                  measured_layer_growth=dict(p05=float(np.percentile(ratios,5)),median=float(np.median(ratios)),p95=float(np.percentile(ratios,95))) if ratios else None,
                  conforming_interface=True)
    # The reader rejects duplicate edges, hanging interfaces and missing boundary tags.
    report['hybrid_gate_passed']=bool(report['quality_gate_passed'] and report['triangle_cells']>0 and report['wall_quad_coverage']==1 and min(depths)>=3)
    return report

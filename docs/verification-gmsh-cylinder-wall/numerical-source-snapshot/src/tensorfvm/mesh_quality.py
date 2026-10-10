"""Physical face/cell quality metrics independent of logical grid dimensions."""
import numpy as np


def mesh_quality(mesh):
    array=lambda x:x.detach().cpu().numpy()
    o,n,inside=array(mesh.owner),array(mesh.neighbor),array(mesh.interior)
    centers=array(mesh.centers).reshape(-1,2);fc=array(mesh.face_centers);S=array(mesh.face_area_vectors)
    d=fc-centers[o];d[inside]=centers[n[inside]]-centers[o[inside]]
    projected=np.einsum('ij,ij->i',S,d)
    angles=np.degrees(np.arccos(np.clip(projected/np.linalg.norm(S,axis=1)/np.linalg.norm(d,axis=1),-1,1)))
    # Face center distance from intersection of centroid connector with face line,
    # normalized by centroid distance; boundary connector intersects at face center.
    fractions=np.einsum('ij,ij->i',S,fc-centers[o])/projected
    crossings=centers[o]+fractions[:,None]*d
    skew=np.linalg.norm(fc-crossings,axis=1)/np.linalg.norm(d,axis=1)
    volumes=array(mesh.volumes).ravel()
    return dict(cells=int(len(volumes)),faces=int(len(o)),minimum_cell_area_m2=float(volumes.min()),minimum_face_projected_distance_m=float(np.min(projected/np.linalg.norm(S,axis=1))),nonorthogonality_max_deg=float(angles.max()),nonorthogonality_p95_deg=float(np.percentile(angles,95)),internal_skewness_max=float(skew[inside].max()),positive_cell_areas=bool(np.all(volumes>0)),positive_projected_distances=bool(np.all(projected>0)),quality_gate_passed=bool(np.all(volumes>0) and np.all(projected>0) and angles.max()<70))

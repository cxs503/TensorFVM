"""Exact planar polygon face moments and topology-preserving facet merging."""
from dataclasses import replace
import numpy as np


def polygon_triangles(points, *, planarity_rtol=1e-12):
    """Validate a finite, ordered, strictly convex planar polygon; return fan.

    Orientation is preserved. Folded, concave, degenerate and nonplanar faces
    raise ValueError. Tolerance is relative to local edge extent plus coordinate
    rounding, rather than the full domain diameter.
    """
    p = np.asarray(points, float)
    if p.ndim != 2 or p.shape[1] != 3 or len(p) < 3 or not np.isfinite(p).all():
        raise ValueError('Finite polygon vertices (n>=3,3) required')
    if not np.isfinite(planarity_rtol) or planarity_rtol <= 0:
        raise ValueError('Positive finite planarity tolerance required')
    scale = float(np.max(np.linalg.norm(p-p[0], axis=1)))
    if scale <= 0:
        raise ValueError('Degenerate polygon')
    triangles = np.array([[p[0], p[j], p[j+1]] for j in range(1, len(p)-1)])
    vectors = .5*np.cross(triangles[:, 1]-triangles[:, 0],
                            triangles[:, 2]-triangles[:, 0])
    area = vectors.sum(axis=0)
    norm = np.linalg.norm(area)
    if norm <= 64*np.finfo(float).eps*scale**2:
        raise ValueError('Degenerate polygon area')
    normal = area/norm
    tolerance = planarity_rtol*scale+16*np.finfo(float).eps*np.max(np.abs(p))
    if np.max(np.abs((p-p[0])@normal)) > tolerance:
        raise ValueError('Nonplanar polygon')
    edges = np.roll(p, -1, axis=0)-p
    turns = np.einsum('ij,j->i', np.cross(edges, np.roll(edges, -1, axis=0)), normal)
    if np.any(turns <= 64*np.finfo(float).eps*scale**2) or np.any(vectors@normal <= 0):
        raise ValueError('Polygon must be strictly convex and consistently oriented')
    return triangles


def polygon_face_covariance(points, face_center=None, *, planarity_rtol=1e-12):
    """Exact area mean of (x-center)(x-center)^T by real triangle fan.

    Default center is the polygon area centroid. A supplied center is retained
    so callers can exactly reproduce their mesh's stored geometric convention.
    """
    p = np.asarray(points, float)
    triangles = polygon_triangles(p, planarity_rtol=planarity_rtol)
    area = np.linalg.norm(.5*np.cross(triangles[:, 1]-triangles[:, 0],
                                       triangles[:, 2]-triangles[:, 0]), axis=1)
    centers = triangles.mean(axis=1)
    center = (np.sum(area[:, None]*centers, axis=0)/area.sum()
              if face_center is None else np.asarray(face_center, float))
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError('Finite face center (3,) required')
    offset = triangles-centers[:, None]
    covariance = np.einsum('tai,taj->tij', offset, offset)/12
    displacement = centers-center
    covariance += displacement[:, :, None]*displacement[:, None, :]
    return np.einsum('t,tij->ij', area/area.sum(), covariance)


def merge_coplanar_faces(mesh, *, planarity_rtol=1e-12):
    """Merge compatible pairs of triangular facets into convex planar quads.

    Returns (new_mesh, metadata). metadata['original_face_groups'][new_face]
    lists original indices; metadata['original_to_merged'] maps old to new.
    Only identical owner, neighbor and physical patch may merge. Cell topology,
    centers and volumes are retained without geometric re-integration.
    """
    if not np.isfinite(planarity_rtol) or planarity_rtol <= 0:
        raise ValueError('Positive finite planarity tolerance required')
    candidates = {}
    for face, ids in enumerate(mesh.face_vertices):
        if len(ids) != 3:
            continue
        group = (int(mesh.owner[face]), int(mesh.neighbor[face]), str(mesh.patch_labels[face]))
        for a, b in zip(ids, ids[1:]+ids[:1]):
            candidates.setdefault((group, tuple(sorted((a, b)))), []).append(face)
    pairs = sorted({tuple(sorted(faces)) for faces in candidates.values() if len(faces) == 2})
    merged = {}
    used = set()
    for first, second in pairs:
        if first in used or second in used:
            continue
        ids1, ids2 = mesh.face_vertices[first], mesh.face_vertices[second]
        if len(set(ids1)|set(ids2)) != 4 or len(set(ids1)&set(ids2)) != 2:
            continue
        s1, s2 = mesh.area_vectors[[first, second]]
        n1, n2 = s1/np.linalg.norm(s1), s2/np.linalg.norm(s2)
        if np.linalg.norm(n1-n2) > 4*planarity_rtol:
            continue
        edges = {}
        for ids in (ids1, ids2):
            for a, b in zip(ids, ids[1:]+ids[:1]):
                key = tuple(sorted((a, b)))
                edges[key] = edges.get(key, 0)+1
        perimeter = [edge for edge, count in edges.items() if count == 1]
        adjacency = {}
        for a, b in perimeter:
            adjacency.setdefault(a, []).append(b)
            adjacency.setdefault(b, []).append(a)
        if len(perimeter) != 4 or any(len(v) != 2 for v in adjacency.values()):
            continue
        order = [min(adjacency)]
        while len(order) < 4:
            options = [v for v in adjacency[order[-1]] if v not in order]
            if not options:
                break
            order.append(min(options))
        if len(order) != 4 or order[0] not in adjacency[order[-1]]:
            continue
        points = mesh.vertices[order]
        try:
            triangles = polygon_triangles(points, planarity_rtol=planarity_rtol)
        except ValueError:
            continue
        area = .5*np.cross(triangles[:, 1]-triangles[:, 0],
                             triangles[:, 2]-triangles[:, 0]).sum(axis=0)
        if np.dot(area, s1+s2) < 0:
            order.reverse()
        merged[first] = (second, tuple(order))
        used.update((first, second))
    faces, groups, owners, neighbors, centers, vectors, patches = [], [], [], [], [], [], []
    old_to_new = np.empty(len(mesh.owner), int)
    consumed = {value[0] for value in merged.values()}
    for face, original in enumerate(mesh.face_vertices):
        if face in consumed:
            continue
        group = (face, merged[face][0]) if face in merged else (face,)
        faces.append(merged[face][1] if face in merged else original)
        groups.append(group)
        old_to_new[list(group)] = len(faces)-1
        owners.append(mesh.owner[face]); neighbors.append(mesh.neighbor[face])
        patches.append(mesh.patch_labels[face])
        areas = np.linalg.norm(mesh.area_vectors[list(group)], axis=1)
        centers.append(np.sum(areas[:, None]*mesh.face_centers[list(group)], axis=0)/areas.sum()
                       if len(group) > 1 else mesh.face_centers[face])
        vectors.append(mesh.area_vectors[list(group)].sum(axis=0)
                       if len(group) > 1 else mesh.area_vectors[face])
    result = replace(mesh, face_vertices=tuple(faces), owner=np.asarray(owners),
                     neighbor=np.asarray(neighbors), face_centers=np.asarray(centers),
                     area_vectors=np.asarray(vectors), patch_labels=np.asarray(patches))
    audit = result.audit()
    if audit['maximum_relative_closure'] > 1e-10 or audit['minimum_face_projection'] <= 0:
        raise ValueError('Merged mesh fails cell closure or face projection audit')
    return result, dict(original_face_groups=tuple(groups), original_to_merged=old_to_new,
                       merged_pairs=len(merged), original_faces=len(mesh.owner),
                       resulting_faces=len(result.owner), audit=audit)

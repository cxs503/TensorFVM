"""Exact planar sections of native cell facets for verification plots."""
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

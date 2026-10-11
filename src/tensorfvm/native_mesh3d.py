"""Static, non-periodic polyhedral finite-volume geometry from linear cells.

Supports convex tetrahedra, hexahedra, wedges and pyramids with planar faces.
Opt-in triangulate_faces uses deterministic vertex-ID diagonals for all faces;
warped quads then represent two planar facets, not a bilinear curved surface.
This is a geometry contract, not evidence of a validated flow discretization.
"""
from dataclasses import dataclass
import numpy as np

_FACES = {
 'tetra': ((0,2,1),(0,1,3),(1,2,3),(2,0,3)),
 'hexahedron': ((0,3,2,1),(4,5,6,7),(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7)),
 'wedge': ((0,2,1),(3,4,5),(0,1,4,3),(1,2,5,4),(2,0,3,5)),
 'pyramid': ((0,3,2,1),(0,1,4),(1,2,4),(2,3,4),(3,0,4)),
}
_COUNTS = {'tetra':4,'hexahedron':8,'wedge':6,'pyramid':5}

@dataclass(frozen=True)
class NativeMesh3D:
    vertices: np.ndarray
    cells: tuple
    cell_types: tuple
    face_vertices: tuple
    owner: np.ndarray
    neighbor: np.ndarray
    face_centers: np.ndarray
    area_vectors: np.ndarray
    cell_centers: np.ndarray
    cell_volumes: np.ndarray
    patch_labels: np.ndarray

    @classmethod
    def from_cells(cls, vertices, cells, cell_types, boundary_patches=None, *, triangulate_faces=False):
        vertices=np.asarray(vertices,dtype=float)
        if vertices.ndim!=2 or vertices.shape[1]!=3 or not np.isfinite(vertices).all():
            raise ValueError('vertices must be finite (n,3) coordinates')
        cells=tuple(tuple(map(int,c)) for c in cells); types=tuple(cell_types)
        if not cells or len(cells)!=len(types):raise ValueError('nonempty cells and matching types required')
        patches={}
        if boundary_patches is not None:
            for face,label in boundary_patches.items():
                face=tuple(face)
                if triangulate_faces:
                    pivot=face.index(min(face));face=face[pivot:]+face[:pivot]
                    pieces=[(face[0],face[j],face[j+1]) for j in range(1,len(face)-1)]
                else: pieces=[face]
                for piece in pieces:
                    key=tuple(sorted(piece))
                    if key in patches and patches[key]!=str(label): raise ValueError('ambiguous physical patch')
                    patches[key]=str(label)
        faces=[]; centers=[]; areas=[]; own=[]; nei=[]; lookup={}; volumes=[]; centroids=[]
        scale=max(float(np.ptp(vertices,axis=0).max()),np.finfo(float).tiny)
        for ci,(cell,kind) in enumerate(zip(cells,types)):
            if kind not in _FACES:raise ValueError(f'unsupported cell type: {kind}')
            if len(cell)!=_COUNTS[kind] or len(set(cell))!=len(cell) or min(cell)<0 or max(cell)>=len(vertices):
                raise ValueError(f'invalid {kind} connectivity at cell {ci}')
            ref=vertices[list(cell)].mean(axis=0); volume=0.; moment=np.zeros(3)
            local_faces = _FACES[kind]
            if triangulate_faces:
                local_faces = []
                for face in _FACES[kind]:
                    pivot = min(range(len(face)), key=lambda j: cell[face[j]])
                    ordered = face[pivot:] + face[:pivot]
                    local_faces.extend((ordered[0], ordered[j], ordered[j+1]) for j in range(1,len(ordered)-1))
            for local in local_faces:
                ids=tuple(cell[j] for j in local); pts=vertices[list(ids)]
                tris=np.stack([np.stack((pts[0],pts[j],pts[j+1])) for j in range(1,len(pts)-1)])
                sv=.5*np.cross(tris[:,1]-tris[:,0],tris[:,2]-tris[:,0]); area=sv.sum(axis=0)
                if np.dot(area,pts.mean(axis=0)-ref)<0:
                    ids=tuple(reversed(ids));pts=vertices[list(ids)]
                    tris=np.stack([np.stack((pts[0],pts[j],pts[j+1])) for j in range(1,len(pts)-1)])
                    sv=.5*np.cross(tris[:,1]-tris[:,0],tris[:,2]-tris[:,0]);area=sv.sum(axis=0)
                norm=np.linalg.norm(area)
                if norm<=np.finfo(float).eps*scale**2:raise ValueError('degenerate face')
                normal=area/norm
                if np.max(np.abs((pts-pts[0])@normal))>1e-10*scale:raise ValueError('nonplanar face unsupported')
                if np.any(sv@normal<=0):raise ValueError('nonconvex or folded face unsupported')
                weights=np.linalg.norm(sv,axis=1)
                fc=(tris.mean(axis=1)*weights[:,None]).sum(axis=0)/weights.sum()
                if np.dot(area,fc-ref)<=0:raise ValueError('nonconvex cell unsupported')
                tetra_v=np.einsum('ij,ij->i',tris[:,0]-ref,np.cross(tris[:,1]-ref,tris[:,2]-ref))/6
                if np.any(tetra_v<=0):raise ValueError('invalid cell tetrahedral decomposition')
                volume+=tetra_v.sum();moment+=(tetra_v[:,None]*(tris.sum(axis=1)+ref)/4).sum(axis=0)
                key=tuple(sorted(ids))
                if key in lookup:
                    fi=lookup[key]
                    if nei[fi]!=-1:raise ValueError('nonmanifold face shared by more than two cells')
                    if not np.allclose(areas[fi],-area,rtol=1e-10,atol=1e-14*scale**2):raise ValueError('shared face orientations inconsistent')
                    nei[fi]=ci
                else:
                    lookup[key]=len(faces);faces.append(ids);centers.append(fc);areas.append(area);own.append(ci);nei.append(-1)
            if volume<=0:raise ValueError('nonpositive cell volume')
            volumes.append(volume);centroids.append(moment/volume)
        labels=[]
        for face,n in zip(faces,nei):
            key=tuple(sorted(face))
            if n>=0:
                if key in patches:raise ValueError('physical boundary tag found on internal face')
                labels.append('internal')
            else:
                if boundary_patches is not None and key not in patches:raise ValueError('untagged boundary face')
                labels.append(patches.get(key,'boundary'))
        if set(patches)-set(lookup):raise ValueError('surface elements absent from volume topology')
        mesh=cls(vertices,cells,types,tuple(faces),np.array(own),np.array(nei),np.array(centers),np.array(areas),np.array(centroids),np.array(volumes),np.array(labels))
        metrics=mesh.audit()
        if metrics['maximum_relative_closure']>1e-10:raise ValueError('cell face areas do not close')
        if metrics['minimum_face_projection']<=0:raise ValueError('nonpositive owner-neighbor face projection')
        return mesh

    @classmethod
    def from_gmsh(cls,path, *, triangulate_faces=False):
        """Read Gmsh via its official API; reject high-order/unsupported elements."""
        import gmsh
        if gmsh.isInitialized():raise RuntimeError('from_gmsh requires an uninitialized Gmsh session')
        gmsh.initialize();gmsh.option.setNumber('General.Terminal',0)
        try:
            gmsh.open(str(path)); tags,xyz,_=gmsh.model.mesh.getNodes()
            if triangulate_faces:
                order=np.argsort(tags);tags=tags[order];xyz=np.asarray(xyz).reshape(-1,3)[order].ravel()
            index={int(t):i for i,t in enumerate(tags)};vertices=np.asarray(xyz).reshape(-1,3)
            cells=[];types=[]; supported={4:'tetra',5:'hexahedron',6:'wedge',7:'pyramid'}
            ets,_,nodes=gmsh.model.mesh.getElements(3)
            for et,conn in zip(ets,nodes):
                if et not in supported:raise ValueError(f'unsupported Gmsh volume element type {et}')
                kind=supported[et];arr=np.asarray(conn).reshape(-1,_COUNTS[kind])
                cells.extend(tuple(index[int(v)] for v in c) for c in arr);types.extend([kind]*len(arr))
            patches={};patch_names={}
            for dim,tag in gmsh.model.getPhysicalGroups(2):
                name=gmsh.model.getPhysicalName(dim,tag) or f'physical_{tag}'
                for entity in gmsh.model.getEntitiesForPhysicalGroup(dim,tag):
                    ets,_,nodes=gmsh.model.mesh.getElements(2,int(entity))
                    for et,conn in zip(ets,nodes):
                        if et not in (2,3):raise ValueError(f'unsupported Gmsh surface element type {et}')
                        for face in np.asarray(conn).reshape(-1,3 if et==2 else 4):
                            cyclic=tuple(index[int(v)] for v in face);key=tuple(sorted(cyclic))
                            if key in patch_names and patch_names[key]!=name:raise ValueError('ambiguous physical patch')
                            patch_names[key]=name;patches[cyclic]=name
            return cls.from_cells(vertices,cells,types,patches,triangulate_faces=triangulate_faces)
        finally:gmsh.finalize()

    def audit(self):
        closure=np.zeros_like(self.cell_centers); area=np.zeros(len(self.cells))
        np.add.at(closure,self.owner,self.area_vectors);np.add.at(area,self.owner,np.linalg.norm(self.area_vectors,axis=1))
        mask=self.neighbor>=0
        np.add.at(closure,self.neighbor[mask],-self.area_vectors[mask]);np.add.at(area,self.neighbor[mask],np.linalg.norm(self.area_vectors[mask],axis=1))
        d=self.face_centers-self.cell_centers[self.owner]
        d[mask]=self.cell_centers[self.neighbor[mask]]-self.cell_centers[self.owner[mask]]
        projection=np.einsum('ij,ij->i',self.area_vectors,d)
        return dict(cells=len(self.cells),faces=len(self.owner),minimum_cell_volume=float(self.cell_volumes.min()),maximum_relative_closure=float((np.linalg.norm(closure,axis=1)/area).max()),minimum_face_projection=float(projection.min()),maximum_nonorthogonality_degrees=float(np.degrees(np.arccos(np.clip(projection/(np.linalg.norm(self.area_vectors,axis=1)*np.linalg.norm(d,axis=1)),-1,1))).max()),patch_face_counts={str(k):int(v) for k,v in zip(*np.unique(self.patch_labels,return_counts=True))})

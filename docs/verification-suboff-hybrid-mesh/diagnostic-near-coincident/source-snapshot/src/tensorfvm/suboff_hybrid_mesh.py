"""Conforming structured SUBOFF layers and genuine Gmsh exterior tetrahedra.

Hexahedral quads are deterministically faceted in the FV importer. The exterior
uses exactly the same outer-layer triangles, with no interpolation interface.
Nose/tail caps use wedges. No y+ or viscous-flow qualification is implied.
"""
from pathlib import Path
import numpy as np
from .suboff_geometry import radius, LENGTH_FT
from .native_mesh3d import NativeMesh3D


def facet(face):
    face=tuple(face); k=face.index(min(face)); f=face[k:]+face[:k]
    return [(f[0],f[j],f[j+1]) for j in range(1,len(f)-1)]


def structured_layers(*, axial=48, azimuthal=24, layers=4, first_height=.0004, growth=1.25):
    if axial<12 or azimuthal<8 or layers<1 or not np.isfinite([first_height,growth]).all() or first_height<=0 or growth<1:
        raise ValueError('require axial>=12, azimuthal>=8, layers>=1, positive height and growth>=1')
    breaks=np.array([3.333333,10.645833,13.979167])/LENGTH_FT
    x=np.unique(np.r_[.5*(1-np.cos(np.linspace(0,np.pi,axial+1))),breaks])
    r=radius(x);r[[0,-1]]=0
    eps=1e-7
    slope=(radius(np.minimum(x+eps,1))-radius(np.maximum(x-eps,0)))/(np.minimum(x+eps,1)-np.maximum(x-eps,0))
    nx=-slope/np.sqrt(1+slope*slope);nr=1/np.sqrt(1+slope*slope)
    nx[[0,-1]]=[-1,1];nr[[0,-1]]=0
    heights=first_height*growth**np.arange(layers)
    offsets=np.r_[0,np.cumsum(heights)]
    if offsets[-1]>.02:raise ValueError('development layer thickness must be <=0.02 L; larger offsets need collision controls')
    points=[]; ids=[]
    for d in offsets:
        rows=[]
        for i in range(len(x)):
            if i in (0,len(x)-1):
                tag=len(points);points.append([x[i]+d*nx[i],0,0]);rows.append([tag]*azimuthal)
            else:
                row=[]
                for j in range(azimuthal):
                    a=2*np.pi*j/azimuthal;row.append(len(points))
                    points.append([x[i]+d*nx[i],(r[i]+d*nr[i])*np.cos(a),(r[i]+d*nr[i])*np.sin(a)])
                rows.append(row)
        ids.append(rows)
    cells=[];types=[]
    def surface(k,i,j):
        a=ids[k][i][j];b=ids[k][i][(j+1)%azimuthal];c=ids[k][i+1][(j+1)%azimuthal];d=ids[k][i+1][j]
        return tuple(dict.fromkeys((a,b,c,d)))
    for k in range(layers):
        for i in range(len(x)-1):
            for j in range(azimuthal):
                lower=surface(k,i,j);upper=surface(k+1,i,j)
                cells.append(lower+upper);types.append('wedge' if len(lower)==3 else 'hexahedron')
    wall=[surface(0,i,j) for i in range(len(x)-1) for j in range(azimuthal)]
    interface=[surface(layers,i,j) for i in range(len(x)-1) for j in range(azimuthal)]
    # Fail before Gmsh if offsets fold or facet projections become nonpositive.
    NativeMesh3D.from_cells(points,cells,types,triangulate_faces=True)
    return np.asarray(points),cells,types,wall,interface,dict(axial_stations=len(x),azimuthal=azimuthal,layers=layers,first_height_L=first_height,growth=growth,layer_heights_L=heights.tolist(),total_thickness_L=float(offsets[-1]))


def generate_suboff_hybrid_mesh(path, *, far_size=.35, **settings):
    import gmsh
    if not np.isfinite(far_size) or far_size<=0:raise ValueError('far_size must be positive')
    points,cells,types,wall,interface,summary=structured_layers(**settings)
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if gmsh.isInitialized():raise RuntimeError('requires an uninitialized Gmsh session')
    gmsh.initialize();gmsh.option.setNumber('General.Terminal',0)
    try:
        gmsh.option.setNumber('General.NumThreads',1);gmsh.option.setNumber('Mesh.RandomSeed',1);gmsh.option.setNumber('Mesh.Renumber',0)
        box=gmsh.model.occ.addBox(-1,-1,-1,4,2,2);gmsh.model.occ.synchronize()
        outer=[tag for dim,tag in gmsh.model.getBoundary([(3,box)],oriented=False)]
        patches={'inlet':[],'outlet':[],'farfield':[]}
        for tag in outer:
            bb=gmsh.model.getBoundingBox(2,tag)
            name='inlet' if abs(bb[0]+1)<1e-6 and abs(bb[3]+1)<1e-6 else 'outlet' if abs(bb[0]-3)<1e-6 and abs(bb[3]-3)<1e-6 else 'farfield'
            patches[name].append(tag)
        gmsh.option.setNumber('Mesh.MeshSizeMax',far_size);gmsh.option.setNumber('Mesh.MeshSizeMin',far_size)
        gmsh.model.mesh.generate(2);gmsh.model.removeEntities([(3,box)])
        old,_,_=gmsh.model.mesh.getNodes();base=int(max(old))+1
        skin=gmsh.model.addDiscreteEntity(2)
        triangles=np.array([t for f in interface for t in facet(f)],dtype=int)
        outer_ids=np.unique(triangles)
        gmsh.model.mesh.addNodes(2,skin,outer_ids+base,points[outer_ids].ravel())
        gmsh.model.mesh.addElementsByType(skin,2,[],(triangles+base).ravel())
        loop1=gmsh.model.geo.addSurfaceLoop(outer);loop2=gmsh.model.geo.addSurfaceLoop([skin])
        volume=gmsh.model.geo.addVolume([loop1,loop2]);gmsh.model.geo.synchronize()
        gmsh.model.mesh.generate(3)
        # Exterior mesher must preserve the imposed interface exactly.
        et,_,conn=gmsh.model.mesh.getElements(2,skin)
        actual={tuple(sorted(map(int,f))) for typ,c in zip(et,conn) for f in np.asarray(c).reshape(-1,3) if typ==2}
        expected={tuple(sorted(map(int,f))) for f in triangles+base}
        if actual!=expected:raise RuntimeError(f'Gmsh altered the structured/exterior interface: {len(actual)} vs {len(expected)}, missing {list(expected-actual)[:2]}, added {list(actual-expected)[:2]}')
        shell=gmsh.model.addDiscreteEntity(3)
        inner_ids=np.setdiff1d(np.arange(len(points)),outer_ids)
        gmsh.model.mesh.addNodes(3,shell,inner_ids+base,points[inner_ids].ravel())
        for kind,typ in [('hexahedron',5),('wedge',6)]:
            group=np.array([c for c,t in zip(cells,types) if t==kind],dtype=int)
            if len(group):gmsh.model.mesh.addElementsByType(shell,typ,[],(group+base).ravel())
        wall_entity=gmsh.model.addDiscreteEntity(2)
        for width,typ in [(3,2),(4,3)]:
            group=np.array([f for f in wall if len(f)==width],dtype=int)
            if len(group):gmsh.model.mesh.addElementsByType(wall_entity,typ,[],(group+base).ravel())
        gmsh.model.addPhysicalGroup(2,[wall_entity],name='wall')
        for name,tags in patches.items():gmsh.model.addPhysicalGroup(2,tags,name=name)
        gmsh.model.addPhysicalGroup(3,[volume],name='exterior');gmsh.model.addPhysicalGroup(3,[shell],name='structured_layer')
        gmsh.option.setNumber('Mesh.SaveAll',0);gmsh.write(str(path))
        summary.update(gmsh_version=gmsh.__version__,interface_triangles=len(triangles),structured_hexes=types.count('hexahedron'),cap_wedges=types.count('wedge'),conforming_interface=True,physics_qualified=False,domain_L=[[-1,3],[-1,1],[-1,1]])
    finally:gmsh.finalize()
    mesh=NativeMesh3D.from_gmsh(path,triangulate_faces=True)
    summary['mesh_audit']=mesh.audit();summary['exterior_tetrahedra']=mesh.cell_types.count('tetra')
    # Every imposed triangle must appear exactly once as an internal shared face.
    from scipy.spatial import cKDTree
    distance,mapping=cKDTree(mesh.vertices).query(points)
    if distance.max()>1e-13 or len(np.unique(mapping))!=len(points):
        raise RuntimeError('structured vertices changed or merged during mesh round-trip')
    lookup={tuple(sorted(f)):i for i,f in enumerate(mesh.face_vertices)}
    shared=[lookup[tuple(sorted(mapping[list(f)]))] for f in triangles]
    if not all(mesh.neighbor[i]>=0 and {mesh.cell_types[mesh.owner[i]],mesh.cell_types[mesh.neighbor[i]]} & {'hexahedron','wedge'} and 'tetra' in {mesh.cell_types[mesh.owner[i]],mesh.cell_types[mesh.neighbor[i]]} for i in shared):
        raise RuntimeError('interface has a gap or unexpected adjacency')
    summary['verified_shared_interface_faces']=len(shared)
    return mesh,summary

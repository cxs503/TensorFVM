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


def adaptive_meridian_stations(chord_tolerance, *, minimum_spacing=1e-5, maximum_spacing=.05, max_stations=512):
    """Sample each mother segment by its distance from a meridian chord.

    Dense cosine reference samples resolve tip curvature. Recursive farthest
    point splitting retains exact polynomial junctions. Tolerance is in L and
    measures Euclidean meridian distance, not radial discrepancy at fixed x.
    A spacing conflict fails explicitly rather than introducing needle cells.
    The tolerance is checked on dense samples; it is not a continuous proof.
    """
    if not np.isfinite([chord_tolerance,minimum_spacing,maximum_spacing]).all() or chord_tolerance<=0 or minimum_spacing<=0 or maximum_spacing<minimum_spacing or max_stations<5:
        raise ValueError('positive chord tolerance / minimum spacing and >=5 stations required')
    breaks=np.array([0,3.333333,10.645833,13.979167,LENGTH_FT])/LENGTH_FT
    result=[];errors=[]
    for a,b in zip(breaks[:-1],breaks[1:]):
        xx=a+(b-a)*.5*(1-np.cos(np.linspace(0,np.pi,8193)))
        xx[[0,-1]]=[a,b]
        rr=radius(xx);rr[xx==0]=0;rr[xx==1]=0
        points=np.column_stack((xx,rr));chosen={0,len(xx)-1};stack=[(0,len(xx)-1)]
        while stack:
            lo,hi=stack.pop()
            v=points[hi]-points[lo];t=np.clip((points[lo:hi+1]-points[lo])@v/(v@v),0,1)
            distances=np.linalg.norm(points[lo:hi+1]-points[lo]-t[:,None]*v,axis=1)
            index=lo+int(np.argmax(distances));maximum=float(distances.max())
            if maximum<=chord_tolerance and xx[hi]-xx[lo]<=maximum_spacing:
                errors.append(maximum);continue
            if maximum<=chord_tolerance:
                index=lo+int(np.argmin(abs(xx[lo:hi+1]-.5*(xx[lo]+xx[hi]))))
            if xx[index]-xx[lo]<minimum_spacing or xx[hi]-xx[index]<minimum_spacing:
                raise ValueError('chord tolerance conflicts with minimum axial spacing; refine spacing policy explicitly')
            chosen.add(index)
            if len(result)+len(chosen)>max_stations:raise ValueError('adaptive station limit exceeded')
            stack.extend([(lo,index),(index,hi)])
        stations=xx[sorted(chosen)]
        result.extend(stations if not result else stations[1:])
    x=np.asarray(result)
    if np.any(np.diff(x)<minimum_spacing*(1-1e-12)):
        raise ValueError('adaptive stations violate minimum spacing')
    return x,dict(chord_tolerance_L=float(chord_tolerance),minimum_axial_spacing_L=float(minimum_spacing),
                  actual_minimum_axial_spacing_L=float(np.diff(x).min()),maximum_axial_spacing_L=float(maximum_spacing),
                  actual_maximum_axial_spacing_L=float(np.diff(x).max()),reference_points_per_segment=8193,
                  maximum_dense_meridian_chord_error_L=max(errors),continuous_geometry_certified=False)


def wall_facet_geometry_audit(mesh):
    """Distances of actual triangular wall interiors to continuous mother hull.

    Axisymmetry reduces the 3D closest surface distance to a 2D meridian
    minimization. Includes vertices, edge midpoints, centroid and interior
    quarter points. Reported maxima are sampled, not a global Hausdorff bound.
    """
    from scipy.spatial import cKDTree
    from scipy.optimize import minimize_scalar
    xx=.5*(1-np.cos(np.linspace(0,np.pi,32769)))
    reference=np.column_stack((xx,radius(xx)));reference[[0,-1],1]=0
    tree=cKDTree(reference)
    bary=np.array([[1,0,0],[0,1,0],[0,0,1],[.5,.5,0],[0,.5,.5],[.5,0,.5],
                   [1/3,1/3,1/3],[.5,.25,.25],[.25,.5,.25],[.25,.25,.5]])
    faces=np.flatnonzero(mesh.patch_labels=='wall'); samples=[]
    for fi in faces:
        vertices=mesh.vertices[list(mesh.face_vertices[fi])]
        if len(vertices)!=3:raise ValueError('audit requires actual triangular wall facets')
        q=bary@vertices;samples.extend(np.column_stack((q[:,0],np.linalg.norm(q[:,1:],axis=1))))
    samples=np.asarray(samples);_,nearest=tree.query(samples);distances=[]
    for q,k in zip(samples,nearest):
        lo=xx[max(int(k)-3,0)];hi=xx[min(int(k)+3,len(xx)-1)]
        def distance2(x):
            r=0. if x in (0.,1.) else float(radius(x))
            return (x-q[0])**2+(r-q[1])**2
        solve=minimize_scalar(distance2,bounds=(lo,hi),method='bounded',options={'xatol':1e-14})
        distances.append(np.sqrt(min(solve.fun,distance2(lo),distance2(hi))))
    distances=np.asarray(distances).reshape(len(faces),len(bary))
    return dict(wall_triangles=len(faces),samples_per_triangle=len(bary),
                maximum_sampled_surface_distance_L=float(distances.max()),
                maximum_sampled_interior_distance_L=float(distances[:,3:].max()),
                maximum_sampled_vertex_distance_L=float(distances[:,:3].max()),
                p95_sampled_surface_distance_L=float(np.percentile(distances,95)),
                continuous_geometry_certified=False)


def structured_layers(*, axial=48, azimuthal=24, layers=4, first_height=.0004, growth=1.25, chord_tolerance=None, minimum_spacing=1e-5, maximum_spacing=.05, max_stations=512):
    if axial<12 or azimuthal<8 or layers<1 or not np.isfinite([first_height,growth]).all() or first_height<=0 or growth<1:
        raise ValueError('require axial>=12, azimuthal>=8, layers>=1, positive height and growth>=1')
    adaptive={}
    if chord_tolerance is not None:
        x,adaptive=adaptive_meridian_stations(chord_tolerance,minimum_spacing=minimum_spacing,maximum_spacing=maximum_spacing,max_stations=max_stations)
    else:
        breaks=np.array([3.333333,10.645833,13.979167])/LENGTH_FT
        x=.5*(1-np.cos(np.linspace(0,np.pi,axial+1)))
        # Replace nearby cosine samples with exact junctions; appending them creates
        # nearly coincident stations and needle-like exterior boundary triangles.
        for junction in breaks:
            nearest=1+np.argmin(np.abs(x[1:-1]-junction));x[nearest]=junction
        x=np.sort(x)
        if np.any(np.diff(x)<=0):raise ValueError('colliding axial stations')
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
    return np.asarray(points),cells,types,wall,interface,dict(axial_stations=len(x),azimuthal=azimuthal,layers=layers,first_height_L=first_height,growth=growth,layer_heights_L=heights.tolist(),total_thickness_L=float(offsets[-1]),adaptive_sampling=adaptive)


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

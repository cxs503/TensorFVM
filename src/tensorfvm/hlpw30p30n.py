"""Primary HLPW4 30P30N geometry and LTPT pressure data adapters.

Experimental zones remain ordered around the complete contour, including coves.
No branch averaging, extrapolation, or fitted experimental forcing is used.
"""
import json
import re
from pathlib import Path
import numpy as np
from .benchmark_30p30n import write_gmsh_geo

DATA = Path(__file__).with_name('data') / '30p30n-hlpw4'


def reference_zones(alpha):
    """Return all three original, ordered Cp zones at a supported incidence."""
    lines = (DATA/'original/Exp_Data_LTPT_Re9million/cp_exp.dat').read_text().splitlines()
    zones = {}; current = None
    for line in lines:
        if line.lower().startswith('zone'):
            match = re.search(r't="(slat|main|flap), Re = ([\de.+-]+), alpha = ([\d.]+) deg"', line)
            if not match:
                raise ValueError('Unrecognized experimental zone')
            name, reynolds, angle = match.groups()
            current = name if abs(float(angle)-alpha)<1e-8 else None
            if current:
                if float(reynolds)!=9e6 or current in zones:
                    raise ValueError('Invalid/duplicate experimental zone')
                zones[current] = []
        elif current and line.strip():
            row = list(map(float,line.split()))
            if len(row)!=2 or not np.isfinite(row).all():
                raise ValueError('Invalid original Cp datum')
            zones[current].append(row)
    if set(zones)!={'slat','main','flap'}:
        raise ValueError('No complete Cp reference for requested alpha')
    return {name:np.asarray(rows) for name,rows in zones.items()}


def metadata():
    return json.loads((DATA/'metadata.json').read_text())


def resample_profile(points, max_turn=1.5, max_length=None, corner_turn=30.):
    """Curvature-adaptive resampling of a closed polygon along cubic splines.

    The official 30P30N polylines turn by 11-16 degrees per facet around the
    leading edges (3.3e-3 c facets on a 0.015 c nose radius).  Each convex
    polygon corner is a local expansion/recompression for the wall-resolved
    boundary layer (first cells 2e-6 c): the computed Cp and Cf alternate by
    +-0.3 and a factor 2-3 between neighbouring facets, which thickens the
    main-element layer before its suction peak and triggers a separation
    bubble there.  Smooth segments between genuine corners (turn above
    ``corner_turn`` degrees: blunt trailing edges, slat cusp, cove lips) are
    therefore interpolated by chord-length parametrised cubic splines through
    the original vertices; every original facet whose spline arc turns by more
    than ``max_turn`` degrees (or is longer than ``max_length``) is subdivided
    on the spline into equal-turning pieces.  All original vertices, the
    corners and the orientation are preserved exactly.
    """
    from scipy.interpolate import splprep, splev
    P=np.asarray(points,dtype=float)
    if np.allclose(P[0],P[-1]):P=P[:-1]
    n=len(P)
    if n<4 or not 0<max_turn<corner_turn:
        raise ValueError('resample_profile needs a closed polygon and 0<max_turn<corner_turn')
    previous=P-np.roll(P,1,axis=0);following=np.roll(P,-1,axis=0)-P
    turn=np.degrees(np.arctan2(previous[:,0]*following[:,1]-previous[:,1]*following[:,0],(previous*following).sum(1)))
    corners=np.flatnonzero(np.abs(turn)>corner_turn)
    if len(corners)==0:
        segments=[np.vstack([P,P[:1]])];periodic=True
    else:
        periodic=False;order=np.roll(np.arange(n),-corners[0]);Q=P[order];cs=sorted(set((c-corners[0])%n for c in corners))
        bounds=cs+[n];segments=[Q[a:b+1] if b<n else np.vstack([Q[a:],Q[:1]]) for a,b in zip(bounds[:-1],bounds[1:])]
    if max_length is not None and not max_length>0:
        raise ValueError('max_length must be positive')
    out=[]
    for seg in segments:
        m=len(seg)
        if m<5:
            # too short for a cubic spline (blunt trailing edges): straight facets, length cap only
            for k in range(m-1):
                pieces=1 if max_length is None else int(np.ceil(np.linalg.norm(seg[k+1]-seg[k])/max_length))
                out.extend(np.linspace(seg[k],seg[k+1],pieces+1)[:-1].tolist())
            continue
        tck,u=splprep([seg[:,0],seg[:,1]],s=0,k=3,per=periodic)
        limit=np.radians(max_turn)
        for k in range(m-1):
            out.append(seg[k].tolist())
            # turning and arc length of the spline across this original facet
            t=np.linspace(u[k],u[k+1],64);dx,dy=splev(t,tck,der=1)
            theta=np.unwrap(np.arctan2(dy,dx));dtheta=np.abs(np.diff(theta))
            x,y=splev(t,tck);ds=np.hypot(np.diff(x),np.diff(y))
            def subdivide(cost):
                # equal-cost split of the facet; cost = turning/limit (+ length/max_length)
                cum=np.concatenate([[0],np.cumsum(cost)]);pieces=int(np.ceil(cum[-1]-1e-9))
                if pieces<=1:return np.zeros((0,2))
                targets=np.linspace(0,cum[-1],pieces+1)[1:-1]
                tt=np.interp(targets,cum,t) if cum[-1]>0 else np.linspace(u[k],u[k+1],pieces+1)[1:-1]
                return np.column_stack(splev(tt,tck))
            new=subdivide(dtheta/limit)  # equal-turning subdivision (unchanged when no length cap binds)
            if max_length is not None:
                chain=np.vstack([seg[k],new,seg[k+1]])
                if np.linalg.norm(np.diff(chain,axis=0),axis=1).max()>max_length:
                    # equal-cost subdivision bounds both the turning and the length of every piece
                    new=subdivide(dtheta/limit+ds/max_length)
            out.extend(new.tolist())
    out=np.asarray(out)
    return np.vstack([out,out[:1]])


def per_element(value,name):
    """One value for all elements or a mapping element -> value (None stays None)."""
    if value is None:return None
    if isinstance(value,dict):
        if set(value)!={'slat','main','flap'}:raise ValueError('Per-element values need slat, main and flap entries')
        return value[name]
    return value


def profile_directory(max_turn=None,destination=None,max_length=None):
    """Official polylines (``max_turn`` None) or a resampled copy next to the mesh.

    ``max_turn`` is one angle for all elements or a mapping element -> angle
    (Gmsh's boundary-layer edge recovery is not robust to every node
    distribution in the 0.0127 c main-flap gap, so the flap may need a
    slightly different angle than slat and main).  ``max_length`` (chords, one
    value or per element) additionally subdivides long facets: 3.2e-3 c flap
    facets turning 1-2 degrees still imprint a Cp sawtooth of +-0.05 on the
    flap suction side (the polygon sagitta l*theta/8 is 2-3 wall units).
    """
    if max_turn is None:
        if max_length is not None:raise ValueError('max_length resampling needs max_turn')
        return DATA
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    for name in ['slat','main','flap']:
        np.savetxt(destination/(name+'.dat'),resample_profile(np.loadtxt(DATA/(name+'.dat')),max_turn=float(per_element(max_turn,name)),
                                                              max_length=per_element(max_length,name)),fmt='%.17g')
    return destination


def mesh_quality(mesh,alpha):
    """Independent centroid/face-angle and freestream boundary checks."""
    import torch
    owner,neighbor,interior=mesh.owner,mesh.neighbor,mesh.interior
    centers=mesh.centers.reshape(-1,2)
    delta=mesh.face_centers-centers[owner]
    delta[interior]=centers[neighbor[interior]]-centers[owner[interior]]
    S=mesh.face_area_vectors
    cosine=(S*delta).sum(1)/torch.linalg.vector_norm(S,dim=1)/torch.linalg.vector_norm(delta,dim=1)
    angle=torch.rad2deg(torch.acos(cosine.clamp(-1,1)))
    theta=np.radians(alpha);direction=torch.tensor([np.cos(theta),np.sin(theta)],dtype=S.dtype,device=S.device)
    flow=S@direction;wall=mesh.masks['wall']
    record=dict(wall_nonorthogonality_max_deg=float(angle[wall].max()),wall_angle_over_75_count=int((angle[wall]>75).sum()),interior_nonorthogonality_max_deg=float(angle[interior].max()) if interior.any() else 0.,nonpositive_centroid_projection_faces=int((cosine<=0).sum()),outlet_backflow_faces=int(((flow<0)&mesh.masks['outlet']).sum()))
    record['accepted']=record['wall_angle_over_75_count']==0 and record['nonpositive_centroid_projection_faces']==0 and record['outlet_backflow_faces']==0
    return record



def layer_zones(profiles, thin, thick):
    """Map every wall facet (Gmsh curve id) of the polygon layout to a quad-layer thickness.

    The shared writer limits one uniform layer to half the main/flap gap (0.0064 C), which
    leaves the aft main boundary layer, the slat/main wakes and the flap confluent layer
    (0.01-0.05 C thick) to isotropic triangles.  Gmsh 2D accepts several BoundaryLayer
    fields, so facets that have free fluid above them get ``thick`` while facets that face
    another element, and both facets of every fan corner, keep ``thin``:

    * main upper surface, 0.06 < x < 0.855 (clear of the slat trailing edge at x 0.019 and of
      the blunt main trailing edge whose fan must stay inside the 0.013 C main/flap gap);
    * flap upper surface, x > 0.905 (downstream of the main trailing edge at x 0.874);
    * both flap surfaces for 0.96 < x < 1.115; the blunt flap trailing edge (x 1.128-1.131)
      keeps the thin layer because a 0.04 C fan on its 0.003 C base produced 88-degree
      wall non-orthogonality.

    Facet ids equal the ids of their first point; contours are clockwise (outward normal of
    the edge d is (-d_y, d_x)).  Returns ``{curve_id: thickness}``.
    """
    zones = {}; offset = 1
    for name in ['slat', 'main', 'flap']:
        q = np.loadtxt(profiles / (name + '.dat'))
        if np.allclose(q[0], q[-1]): q = q[:-1]
        d = np.roll(q, -1, axis=0) - q; mid = q + d / 2
        n = np.c_[-d[:, 1], d[:, 0]] / np.linalg.norm(d, axis=1)[:, None]
        for i, ((x, y), (nx, ny)) in enumerate(zip(mid, n)):
            t = thin
            if name == 'main' and ny > 0.2 and 0.06 < x < 0.855: t = thick
            if name == 'flap' and ((ny > -0.2 and x > 0.905) or x > 0.96) and x < 1.115: t = thick
            zones[offset + i] = t
        offset += len(q)
    return zones

def write_mesh_geo(path, *, near_size=.003, far_size=25, first_layer=2e-6,
                   layer_thickness=.0005, layer_ratio=1.18, grading_distance=100,
                   alpha=8.1,fan_elements=12,fan_angle=45,wake_size=None,wake_distance=.5,max_turn=None,max_length=None,
                   thick_layer=None):
    """Shared Gmsh wall layers; official circular 500C farfield and nominal C=1.

    Left semicircle uses prescribed freestream velocity; right uses zero gauge
    pressure. This is an incompressible approximation to a compressible farfield.

    ``fan_angle`` is the convex polygon turn (degrees) above which a corner gets
    a boundary-layer fan.  Only genuinely sharp corners (trailing edges, slat
    cusp, main cove lip; turns of 60-162 degrees) should get one: the 11-16
    degree facets of the sampled leading edges used to receive two-sector fans
    whose first ring is a sliver with no wall face, which produced spurious
    near-wall jets at every suction peak.  Smooth kinks are extruded with the
    averaged normal instead.

    ``max_turn`` (optional, degrees) resamples the polylines with
    :func:`resample_profile` so that no smooth facet turns by more than this;
    ``max_length`` (optional, chords; one value or per element) also caps the
    facet length (see :func:`profile_directory`).

    ``wake_size`` (optional) caps the triangle size at ``wake_distance`` chords
    from the walls, so that the slat/main wakes, the gap flows and the outer
    boundary layers are not left to the 500C far-field grading, which otherwise
    reaches 0.13C already half a chord behind the flap.  The size is piecewise
    linear in wall distance: near_size -> wake_size -> far_size.

    ``thick_layer`` (optional, chords) gives the facets selected by :func:`layer_zones` a
    second, thicker quad layer (same first layer and ratio); ``layer_thickness`` remains
    the thin layer of the gap regions and of every fan corner.
    """
    if thick_layer is not None and not layer_thickness<thick_layer<.2:
        raise ValueError('thick_layer must exceed layer_thickness and stay below 0.2 C')
    if not 0<grading_distance<=500:
        raise ValueError('Grading distance must be in (0,500]')
    if not 0<fan_angle<180:
        raise ValueError('Fan angle must be in (0,180) degrees')
    if wake_size is not None and not (near_size<wake_size<far_size and layer_thickness<wake_distance<grading_distance):
        raise ValueError('Wake refinement must satisfy near_size<wake_size<far_size and layer_thickness<wake_distance<grading_distance')
    profiles = profile_directory(max_turn, Path(path).parent/'profiles', max_length)
    path = write_gmsh_geo(path,domain=(-500,500,-500,500),near_size=near_size,
                         far_size=far_size,first_layer=first_layer,
                         layer_thickness=layer_thickness,layer_ratio=layer_ratio,
                         profile_directory=profiles)
    text = path.read_text()
    points = list(map(int,re.findall(r'Point\((\d+)\)',text)))
    corners = points[-4:]; center = max(points)+1
    theta=np.radians(alpha);rotation=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    for tag,xy in zip(corners,np.array([(0,-500),(500,0),(0,500),(-500,0)])@rotation.T):
        text = re.sub(rf'Point\({tag}\) = .*?;',f'Point({tag}) = {{{xy[0]}, {xy[1]}, 0, {far_size}}};',text)
    arcs = []
    for j in range(4):
        pattern = rf'Line\((\d+)\) = \{{{corners[j]}, {corners[(j+1)%4]}\}};'
        match = re.search(pattern,text)
        if not match:
            raise ValueError('Shared mesh writer outer topology changed')
        tag = int(match.group(1)); arcs.append(tag)
        replacement = f'Circle({tag}) = {{{corners[j]}, {center}, {corners[(j+1)%4]}}};'
        if j==0:
            replacement = f'Point({center}) = {{0,0,0,{far_size}}};\n'+replacement
        text = re.sub(pattern,replacement,text)
    for name,ids in [('outlet',arcs[:2]),('inlet',[arcs[3]]),('far-field',[arcs[2]])]:
        text = re.sub(rf'Physical Curve\("{name}"\).*?;',
                      f'Physical Curve("{name}") = {{{", ".join(map(str,ids))}}};',text)
    text = text.replace('Field[2].DistMax = 0.25;',f'Field[2].DistMax = {grading_distance};')
    if wake_size is not None:
        text = text.replace('Background Field = 2;','\n'.join([
            'Field[4] = Threshold;','Field[4].InField = 1;',
            f'Field[4].SizeMin = {near_size:.12g};',f'Field[4].SizeMax = {wake_size:.12g};',
            f'Field[4].DistMin = {layer_thickness:.12g};',f'Field[4].DistMax = {wake_distance:.12g};',
            'Field[5] = Threshold;','Field[5].InField = 1;',
            f'Field[5].SizeMin = {wake_size:.12g};',f'Field[5].SizeMax = {far_size:.12g};',
            f'Field[5].DistMin = {wake_distance:.12g};',f'Field[5].DistMax = {grading_distance:.12g};',
            'Field[6] = MathEval;',f'Field[6].F = "F4+F5-{wake_size:.12g}";',
            'Background Field = 6;']))
    fans=[];offset=1
    for name in ['slat','main','flap']:
        profile=np.loadtxt(profiles/(name+'.dat'))[:-1]
        previous=profile-np.roll(profile,1,axis=0);following=np.roll(profile,-1,axis=0)-profile
        turn=np.degrees(np.arctan2(previous[:,0]*following[:,1]-previous[:,1]*following[:,0],(previous*following).sum(1)))
        # Canonical contours are clockwise. Convex body corners need a fan
        # in the external fluid; smooth curvature remains normally extruded.
        fans.extend((offset+np.flatnonzero(turn < -fan_angle)).tolist());offset+=len(profile)
    extra=''
    if thick_layer is not None:
        zones=layer_zones(profiles,layer_thickness,thick_layer)
        match=re.search(r'Field\[3\]\.CurvesList = \{(.*?)\};',text)
        if not match:raise ValueError('Shared mesh writer boundary-layer field changed')
        curves=[int(v) for v in match.group(1).split(',')]
        thin=[c for c in curves if zones.get(c,layer_thickness)==layer_thickness]
        thick=[c for c in curves if zones.get(c,layer_thickness)!=layer_thickness]
        if not thick:raise ValueError('thick_layer selected no facets')
        text=text.replace(match.group(0),'Field[3].CurvesList = {'+', '.join(map(str,thin))+'};')
        extra='\n'.join(['Field[7] = BoundaryLayer;','Field[7].CurvesList = {'+', '.join(map(str,thick))+'};',
                         f'Field[7].hwall_n = {first_layer:.12g};',f'Field[7].thickness = {thick_layer:.12g};',
                         f'Field[7].ratio = {layer_ratio:.12g};','Field[7].Quads = 1;','BoundaryLayer Field = 7;'])+'\n'
    text=text.replace('BoundaryLayer Field = 3;',
                      'Field[3].FanPointsList = {'+', '.join(map(str,fans))+'};\n'+
                      'Field[3].FanPointsSizesList = {'+', '.join([str(fan_elements)]*len(fans))+'};\nBoundaryLayer Field = 3;\n'+extra)
    path.write_text(text)
    return path


CAD_FILE = DATA/'original/Geometry/2010_30p30n_thik_te_18inches.stp'
CAD_SCALE_MM_PER_CHORD = 457.2
CAD_CURVE_TAGS = {'main':[1,2,3,4,5], 'flap':[6,7,8], 'slat':[9,10,11]}
# Only convex corners of the real STEP curves. Internal sampling points of
# smooth BSplines never become geometry vertices or artificial fan centres.
CAD_FAN_INCOMING_CURVES = [2,4,5,7,8,9,11]


def write_cad_mesh_geo(path, *, near_size=.003, far_size=25, first_layer=2e-6,
                       layer_thickness=.0005, layer_ratio=1.18,
                       grading_distance=100, alpha=8.1, fan_elements=12,
                       curvature_points=48, terminate_cove=True, intersect_metrics=True,
                       cove_trim_radius=None):
    """Mesh the original STEP BSplines, retaining the declared nominal chord.

    STEP declares INCH, while OCC is explicitly instructed to import in MM;
    18 inches therefore becomes 457.2 MM and is scaled to C=1. Coherence welds
    geometrically coincident curve endpoints before constructing closed wires.
    The original geometric curves are retained, without profile interpolation.
    """
    import hashlib
    import shutil
    parameters=[near_size,far_size,first_layer,layer_thickness,layer_ratio,grading_distance,alpha,fan_elements,curvature_points]
    if not np.isfinite(parameters).all() or min(near_size,far_size,first_layer,layer_thickness)<=0:
        raise ValueError('CAD mesh parameters must be finite and sizes positive')
    if not 0<grading_distance<=500 or layer_ratio<=1 or fan_elements<2 or curvature_points<16:
        raise ValueError('Invalid CAD grading, layer growth, fan or curvature resolution')
    if not first_layer<layer_thickness<.006:
        raise ValueError('CAD first layer must be below layer thickness, below .006C')
    if not CAD_FILE.is_file():raise FileNotFoundError('Original official STEP geometry is missing')
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    local_cad=path.parent/'geometry-original.step'
    shutil.copy2(CAD_FILE,local_cad)
    # A true acute fluid cove cannot accommodate the full BL on both walls.
    # Split the actual OCC curves, retaining their exact geometric support.
    from scipy.optimize import brentq
    import gmsh
    if cove_trim_radius is None:cove_trim_radius=max(.004,8*layer_thickness)
    if not 0<cove_trim_radius<.05:raise ValueError('CAD cove trim radius must be in (0,.05C)')
    gmsh.initialize()
    try:
        gmsh.option.setNumber('General.Terminal',0)
        gmsh.option.setString('Geometry.OCCTargetUnit','MM')
        gmsh.model.occ.importShapes(str(CAD_FILE.resolve()),highestDimOnly=False)
        gmsh.model.occ.synchronize()
        tip=np.asarray(gmsh.model.getValue(1,10,[float(gmsh.model.getParametrizationBounds(1,10)[1][0])]))
        cuts={}
        for tag in (10,11):
            lo,hi=gmsh.model.getParametrizationBounds(1,tag);t=np.linspace(float(lo[0]),float(hi[0]),2001)
            coords=np.asarray(gmsh.model.getValue(1,tag,t.tolist())).reshape(-1,3)
            distance=np.linalg.norm(coords-tip,axis=1)/CAD_SCALE_MM_PER_CHORD-cove_trim_radius
            crossings=np.flatnonzero(distance[:-1]*distance[1:]<=0)
            if not len(crossings):raise ValueError('CAD cove split radius does not intersect source curve')
            j=crossings[-1] if tag==10 else crossings[0]
            value=lambda u:np.asarray(gmsh.model.getValue(1,tag,[float(u)]))
            parameter=brentq(lambda u:np.linalg.norm(value(u)-tip)/CAD_SCALE_MM_PER_CHORD-cove_trim_radius,t[j],t[j+1],xtol=1e-14)
            tangent=np.asarray(gmsh.model.getDerivative(1,tag,[float(parameter)]))[:2]
            normal=np.array([-tangent[1],tangent[0]])/np.linalg.norm(tangent)
            point=value(parameter)/CAD_SCALE_MM_PER_CHORD
            cuts[tag]=dict(parameter=float(parameter),coordinates_C=point.tolist(),
                normal_connector_end_C=(point[:2]+2*layer_thickness*normal).tolist())
    finally:gmsh.finalize()
    (path.parent/'cad-mesh-construction.json').write_text(json.dumps(dict(cove_trim_radius_C=cove_trim_radius,
        original_cove_C=(tip/CAD_SCALE_MM_PER_CHORD).tolist(),cuts=cuts,
        local_cove_triangular_wall_target_C=first_layer,
        boundary_layer_exception='Two exact CAD cove tip trims use refined triangles; remaining curves use wall quad layers'),indent=2)+'\n')
    theta=np.radians(alpha);rot=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    xy=np.array([(0,-500),(500,0),(0,500),(-500,0)])@rot.T
    lines=[
        '// Original HLPW4 STEP BSplines; no sampled-polyline reconstruction.',
        '// STEP SHA256: '+hashlib.sha256(CAD_FILE.read_bytes()).hexdigest(),
        'SetFactory("OpenCASCADE");',
        'Geometry.OCCTargetUnit = "MM";',
        'Merge "geometry-original.step";',
        'Dilate {{0,0,0}, 1/457.2} { Curve{1:11}; }',
        'Coherence;',
        f'Point(2000) = {{{cuts[10]["coordinates_C"][0]:.17g},{cuts[10]["coordinates_C"][1]:.17g},0,{first_layer:.17g}}};',
        f'Point(2001) = {{{cuts[11]["coordinates_C"][0]:.17g},{cuts[11]["coordinates_C"][1]:.17g},0,{first_layer:.17g}}};',
        'f10[] = BooleanFragments { Curve{10}; Delete; } { Point{2000}; Delete; };',
        'f11[] = BooleanFragments { Curve{11}; Delete; } { Point{2001}; Delete; };',
        'Curve Loop(1) = {1,2,3,4,5};',
        'Curve Loop(2) = {6,7,8};',
        'Curve Loop(3) = {9,f10[0],f10[1],f11[0],f11[1]};',
        'Point(1000) = {0,0,0,25};',
    ]
    for j,(x,y) in enumerate(xy,1001):lines.append(f'Point({j}) = {{{x:.17g},{y:.17g},0,{far_size:.17g}}};')
    for j in range(4):lines.append(f'Circle({1001+j}) = {{{1001+j},1000,{1001+(j+1)%4}}};')
    lines += [
        'Curve Loop(1000) = {1001,1002,1003,1004};',
        'Plane Surface(1) = {1000,1,2,3};',
        f'Point(2002) = {{{cuts[10]["normal_connector_end_C"][0]:.17g},{cuts[10]["normal_connector_end_C"][1]:.17g},0,{near_size:.17g}}};',
        f'Point(2003) = {{{cuts[11]["normal_connector_end_C"][0]:.17g},{cuts[11]["normal_connector_end_C"][1]:.17g},0,{near_size:.17g}}};',
        'Line(2000) = {2000,2002};',
        'Line(2001) = {2001,2003};',
        'Curve{2000,2001} In Surface{1};',
        'Physical Curve("main") = {1,2,3,4,5};',
        'Physical Curve("flap") = {6,7,8};',
        'Physical Curve("slat") = {9,f10[0],f10[1],f11[0],f11[1]};',
        'Physical Curve("outlet") = {1001,1002};',
        'Physical Curve("far-field") = {1003};',
        'Physical Curve("inlet") = {1004};',
        'Physical Surface("fluid") = {1};',
        'Mesh.Algorithm = 6;', 'Mesh.ElementOrder = 1;', 'Mesh.SaveAll = 0;',
        f'Mesh.MeshSizeMin = {first_layer:.17g};',
        f'Mesh.MeshSizeMax = {far_size:.17g};',
        'Mesh.MeshSizeFromPoints = 0;',
        f'Mesh.MeshSizeFromCurvature = {curvature_points};',
        'Mesh.MeshSizeExtendFromBoundary = 0;',
        'Field[1] = Distance;', 'Field[1].CurvesList = {1:9,f10[0],f10[1],f11[0],f11[1]};',
        'Field[1].Sampling = 2000;',
        'Field[2] = Threshold;', 'Field[2].InField = 1;',
        f'Field[2].SizeMin = {near_size:.17g};',
        f'Field[2].SizeMax = {far_size:.17g};',
        f'Field[2].DistMin = {layer_thickness:.17g};',
        f'Field[2].DistMax = {grading_distance:.17g};',
        'Field[4] = Distance;', 'Field[4].CurvesList = {f10[1],f11[0]};',
        'Field[4].Sampling = 2000;',
        'Field[5] = Threshold;', 'Field[5].InField = 4;',
        f'Field[5].SizeMin = {first_layer:.17g};',
        f'Field[5].SizeMax = {near_size:.17g};',
        f'Field[5].DistMin = {first_layer:.17g};',
        f'Field[5].DistMax = {cove_trim_radius:.17g};',
        'Field[5].StopAtDistMax = 1;',
        'Field[6] = Min;', 'Field[6].FieldsList = {2,5};',
        'Background Field = 6;',
        'Field[3] = BoundaryLayer;', 'Field[3].CurvesList = {1:9,f10[0],f11[1]};',
        f'Field[3].hwall_n = {first_layer:.17g};',
        f'Field[3].thickness = {layer_thickness:.17g};',
        f'Field[3].ratio = {layer_ratio:.17g};', 'Field[3].Quads = 1;',
    ]
    for curve in CAD_FAN_INCOMING_CURVES:
        reference='f11[1]' if curve==11 else str(curve)
        lines.append(f'fan{curve}[] = Boundary {{ Curve{{{reference}}}; }};')
    lines += [
        'Field[3].FanPointsList = {'+', '.join(f'Abs(fan{t}[1])' for t in CAD_FAN_INCOMING_CURVES)+'};',
        'Field[3].FanPointsSizesList = {'+', '.join([str(fan_elements)]*len(CAD_FAN_INCOMING_CURVES))+'};',
        'BoundaryLayer Field = 3;',
    ]
    if intersect_metrics:lines.insert(-1,'Field[3].IntersectMetrics = 1;')
    if terminate_cove:
        lines.insert(-1,'end10[] = Boundary { Curve{f10[0]}; };')
        lines.insert(-1,'end11[] = Boundary { Curve{f11[1]}; };')
        lines.insert(-1,'Field[3].PointsList = {Abs(end10[1]),Abs(end11[0])};')
    path.write_text('\n'.join(lines)+'\n')
    return path


def write_cad_triangle_mesh_geo(path, *, near_size=.003, far_size=25,
                                grading_distance=100, alpha=8.1,
                                curvature_points=48, **unused):
    """Exact official CAD geometry control; no boundary layers or accuracy claim."""
    import shutil
    if not np.isfinite([near_size,far_size,grading_distance,alpha]).all() or min(near_size,far_size,grading_distance)<=0:
        raise ValueError('Triangle mesh sizes must be positive and finite')
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(CAD_FILE,path.parent/'geometry-original.step')
    theta=np.radians(alpha);rot=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    xy=np.array([(0,-500),(500,0),(0,500),(-500,0)])@rot.T
    lines=['SetFactory("OpenCASCADE");','Geometry.OCCTargetUnit = "MM";',
        'Merge "geometry-original.step";','Dilate {{0,0,0}, 1/457.2} { Curve{1:11}; }','Coherence;',
        'Curve Loop(1) = {1:5};','Curve Loop(2) = {6:8};','Curve Loop(3) = {9:11};',
        'Point(1000) = {0,0,0,25};']
    for j,(x,y) in enumerate(xy,1001):lines.append(f'Point({j}) = {{{x:.17g},{y:.17g},0,{far_size:.17g}}};')
    for j in range(4):lines.append(f'Circle({1001+j}) = {{{1001+j},1000,{1001+(j+1)%4}}};')
    lines+=['Curve Loop(1000) = {1001:1004};','Plane Surface(1) = {1000,1,2,3};',
        'Physical Curve("main") = {1:5};','Physical Curve("flap") = {6:8};','Physical Curve("slat") = {9:11};',
        'Physical Curve("outlet") = {1001,1002};','Physical Curve("far-field") = {1003};',
        'Physical Curve("inlet") = {1004};','Physical Surface("fluid") = {1};',
        'Mesh.Algorithm = 6;','Mesh.ElementOrder = 1;','Mesh.SaveAll = 0;',
        f'Mesh.MeshSizeMin = {near_size:.17g};',f'Mesh.MeshSizeMax = {far_size:.17g};',
        'Mesh.MeshSizeFromPoints = 0;',f'Mesh.MeshSizeFromCurvature = {curvature_points};',
        'Mesh.MeshSizeExtendFromBoundary = 0;', 'Field[1] = Distance;',
        'Field[1].CurvesList = {1:11};','Field[1].Sampling = 2000;',
        'Field[2] = Threshold;','Field[2].InField = 1;',f'Field[2].SizeMin = {near_size:.17g};',
        f'Field[2].SizeMax = {far_size:.17g};','Field[2].DistMin = 0;',
        f'Field[2].DistMax = {grading_distance:.17g};','Background Field = 2;']
    path.write_text('\n'.join(lines)+'\n')
    (path.parent/'cad-mesh-construction.json').write_text(json.dumps(dict(
        mesh_mode='cad_triangle',boundary_layers=False,accuracy_qualified=False,
        purpose='Exact CAD geometry control only; not a Re9M wall-resolved benchmark mesh'),indent=2)+'\n')
    return path

"""Static three-dimensional overset geometry and point-value interpolation.

This is not a conservative Navier--Stokes or FVM/LBM coupling solver. Donors
are active cell-centre point values, never volume averages. A native containing
polyhedron is required before a local convex tetrahedral interpolation is
accepted. Sphere clearance uses a separating tangent half-space, which protects
all points of the donor tetrahedron, not only its edges.
"""
from dataclasses import dataclass
import numpy as np
from scipy.spatial import cKDTree, Delaunay, ConvexHull, QhullError
from .native_mesh3d import NativeMesh3D

ACTIVE, HOLE, FRINGE = 0, 1, 2

@dataclass(frozen=True)
class SphereSolid:
    center: np.ndarray
    radius: float

    def __post_init__(self):
        c = np.asarray(self.center, dtype=float)
        if c.shape != (3,) or not np.isfinite(c).all() or not np.isfinite(self.radius) or self.radius <= 0:
            raise ValueError('finite sphere centre and positive radius required')
        object.__setattr__(self, 'center', c)

    def fluid_halfspace(self, points, receiver):
        direction = np.asarray(receiver) - self.center
        distance = np.linalg.norm(direction)
        if distance <= self.radius:
            raise ValueError('receiver lies in the solid')
        return (np.asarray(points) - self.center) @ (direction / distance) >= self.radius + 1e-12 * self.radius

class CartesianGrid3D:
    @staticmethod
    def from_bounds(lower, upper, shape):
        lo, hi = np.asarray(lower, float), np.asarray(upper, float)
        sh = np.asarray(shape)
        if lo.shape != (3,) or hi.shape != (3,) or sh.shape != (3,) or not np.isfinite(lo).all() or not np.isfinite(hi).all() or np.any(hi <= lo) or np.any(sh < 1) or np.any(sh != sh.astype(int)):
            raise ValueError('valid Cartesian bounds and positive integer shape required')
        sh = sh.astype(int)
        axes = [np.linspace(lo[i], hi[i], sh[i]+1) for i in range(3)]
        xyz = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1)
        ids = np.arange(np.prod(sh+1)).reshape(tuple(sh+1))
        cells = []
        for i in range(sh[0]):
            for j in range(sh[1]):
                for k in range(sh[2]):
                    cells.append((ids[i,j,k],ids[i+1,j,k],ids[i+1,j+1,k],ids[i,j+1,k],ids[i,j,k+1],ids[i+1,j,k+1],ids[i+1,j+1,k+1],ids[i,j+1,k+1]))
        return NativeMesh3D.from_cells(xyz.reshape(-1,3), cells, ('hexahedron',)*len(cells))

def shell_radial_coordinates(inner_radius, outer_radius, layers, first_height=None):
    """Uniform or geometric normal layers with exact requested shell thickness.

    A specified first height determines the geometric growth, rather than
    silently changing that height to fit the outer radius. No y+ is inferred.
    """
    if not np.isfinite([inner_radius,outer_radius]).all() or not 0 < inner_radius < outer_radius or int(layers)!=layers or layers<1:
        raise ValueError('finite ordered radii and positive integer layers required')
    layers=int(layers); thickness=outer_radius-inner_radius
    if first_height is None:
        return np.linspace(inner_radius,outer_radius,layers+1)
    if not np.isfinite(first_height) or first_height<=0 or first_height>thickness/layers*(1+1e-12):
        raise ValueError('first height must be positive and no greater than uniform spacing')
    if layers==1:
        if not np.isclose(first_height,thickness,rtol=1e-12,atol=0):
            raise ValueError('one layer must span the entire shell')
        return np.array([inner_radius,outer_radius])
    from scipy.optimize import brentq
    # Solve in log growth; summing normalized positive heights avoids overflow.
    exponents=np.arange(layers,dtype=float)
    target=np.log(thickness/first_height)
    from scipy.special import logsumexp
    if np.isclose(first_height,thickness/layers,rtol=1e-12,atol=0):
        growth_log=0.
    else:
        growth_log=brentq(lambda x:logsumexp(exponents*x)-target,0,target/(layers-1))
    radii=inner_radius+np.r_[0,np.cumsum(first_height*np.exp(exponents*growth_log))]
    radii[-1]=outer_radius
    if not np.all(np.diff(radii)>0):
        raise ValueError('layer spacing is below coordinate floating-point resolution')
    return radii


def sphere_shell_mesh(inner_radius, outer_radius, radial_layers, angular_resolution, center=(0,0,0), *, first_height=None, wall_geometry='inscribed'):
    """Radially structured wedge layers on an icosphere angular surface.

    Angular triangles are refined by midpoint subdivision. Surface facets are
    planar approximations to a sphere; geometry is not an exact curved wall.
    """
    if not (0 < inner_radius < outer_radius) or int(radial_layers) != radial_layers or radial_layers < 1 or int(angular_resolution) != angular_resolution or angular_resolution < 0:
        raise ValueError('positive ordered radii, layers and nonnegative subdivision required')
    if wall_geometry not in ('inscribed','circumscribed'):
        raise ValueError('wall geometry must be inscribed or circumscribed')
    c = np.asarray(center,float)
    if c.shape != (3,) or not np.isfinite(c).all():
        raise ValueError('finite centre required')
    phi=(1+np.sqrt(5))/2
    vertices=np.array([(0,a,b*phi) for a in (-1,1) for b in (-1,1)]+[(a,b*phi,0) for a in (-1,1) for b in (-1,1)]+[(b*phi,0,a) for a in (-1,1) for b in (-1,1)],float)
    vertices /= np.linalg.norm(vertices,axis=1)[:,None]
    triangles=ConvexHull(vertices).simplices
    for _ in range(int(angular_resolution)):
        points=list(vertices); cache={}; refined=[]
        def midpoint(a,b):
            key=tuple(sorted((int(a),int(b))))
            if key not in cache:
                p=vertices[a]+vertices[b];p/=np.linalg.norm(p)
                cache[key]=len(points);points.append(p)
            return cache[key]
        for a,b,d in triangles:
            ab,bd,da=midpoint(a,b),midpoint(b,d),midpoint(d,a)
            refined.extend(((a,ab,da),(ab,b,bd),(da,bd,d),(ab,bd,da)))
        vertices=np.asarray(points);triangles=np.asarray(refined)
    if wall_geometry=='circumscribed':
        # Thin layers outside an inscribed planar wall can still lie inside
        # the analytic solid. Move the wall out until every facet plane
        # separates the entire sphere, then fit normal layers to that wall.
        a,b,d=vertices[triangles[:,0]],vertices[triangles[:,1]],vertices[triangles[:,2]]
        normals=np.cross(b-a,d-a)
        distances=np.abs(np.einsum('ij,ij->i',normals,a))/np.linalg.norm(normals,axis=1)
        inner_radius=inner_radius/distances.min()*(1+8*np.finfo(float).eps)
    radii=shell_radial_coordinates(inner_radius,outer_radius,radial_layers,first_height)
    count=len(vertices);xyz=np.concatenate([c+r*vertices for r in radii]);cells=[]
    for layer in range(int(radial_layers)):
        for tri in triangles:
            cells.append(tuple(layer*count+tri)+tuple((layer+1)*count+tri))
    return NativeMesh3D.from_cells(xyz,cells,('wedge',)*len(cells),triangulate_faces=True)

@dataclass(frozen=True)
class DonorMap3D:
    receiver_indices: np.ndarray
    donor_indices: np.ndarray
    weights: np.ndarray
    containing_cells: np.ndarray

    def interpolate(self, values):
        a=np.asarray(values)
        if a.ndim < 1 or not np.isfinite(a).all():
            raise ValueError('finite donor values required')
        if self.donor_indices.size and a.shape[0] <= self.donor_indices.max():
            raise ValueError('donor values have wrong length')
        return np.einsum('ij,ij...->i...',self.weights,a[self.donor_indices])

@dataclass(frozen=True)
class OversetConnectivity3D:
    grids: tuple
    background_status: np.ndarray
    body_status: np.ndarray
    background_from_body: DonorMap3D
    body_from_background: DonorMap3D

    def interpolate(self, fields):
        """Simultaneous exchange; replace only fringe cells in copied arrays."""
        if len(fields) != 2:
            raise ValueError("background and body fields required")
        arrays = [np.asarray(field) for field in fields]
        for grid, field in zip(self.grids, arrays):
            if field.ndim < 1 or field.shape[0] != len(grid.cells) or not np.isfinite(field).all():
                raise ValueError("one finite value per cell required")
        if arrays[0].shape[1:] != arrays[1].shape[1:]:
            raise ValueError("matching component dimensions required")
        result = [a.astype(np.result_type(a.dtype, float), copy=True) for a in arrays]
        result[0][self.background_from_body.receiver_indices] = self.transfer_background(arrays[1])
        result[1][self.body_from_background.receiver_indices] = self.transfer_body(arrays[0])
        return tuple(result)

    def transfer_background(self, body_values):
        return self.background_from_body.interpolate(body_values)

    def transfer_body(self, background_values):
        return self.body_from_background.interpolate(background_values)

class _Locator:
    def __init__(self, mesh, active):
        self.mesh=mesh; self.active=np.asarray(active,int)
        self.tree=cKDTree(mesh.cell_centers[self.active])
        self.faces=[[] for _ in mesh.cells]
        for f,(o,n) in enumerate(zip(mesh.owner,mesh.neighbor)):
            self.faces[o].append((f,1))
            if n>=0:self.faces[n].append((f,-1))
        # NativeMesh3D construction checks orientation and positive fans;
        # point location additionally requires every vertex inside every facet.
        for j in self.active:
            vertices = mesh.vertices[list(mesh.cells[j])]
            scale = max(1., np.ptp(vertices, axis=0).max())
            for f, sign in self.faces[j]:
                normal = sign * mesh.area_vectors[f]
                if np.max((vertices - mesh.face_centers[f]) @ normal) > 1e-10 * scale * np.linalg.norm(normal):
                    raise ValueError(f'nonconvex donor cell {j}: facet containment unsupported')
        self.radius=np.array([np.linalg.norm(mesh.vertices[list(mesh.cells[j])]-mesh.cell_centers[j],axis=1).max() for j in self.active])
        self.max_radius=self.radius.max()

    def containing(self, point):
        candidates=self.tree.query_ball_point(point,self.max_radius*(1+1e-12))
        for k in candidates:
            j=self.active[k]
            if np.linalg.norm(point-self.mesh.cell_centers[j])>self.radius[k]*(1+1e-12):continue
            if all(sign*np.dot(point-self.mesh.face_centers[f],self.mesh.area_vectors[f]) <= 1e-11*np.linalg.norm(self.mesh.area_vectors[f])*max(1,self.max_radius) for f,sign in self.faces[j]):
                return j
        return -1

def build_donor_map(donor_mesh, donor_status, receiver_points, receiver_indices, solid, max_candidates=256):
    """Generic NativeMesh3D donor adapter with exact convex facet containment.

    Solid currently must be SphereSolid; a SUBOFF adapter must implement an
    equally strong convex-hull/solid separation check before this is generalised.
    No inactive donor, nearest-neighbour extrapolation or orphan suppression.
    """
    status=np.asarray(donor_status)
    if status.shape != (len(donor_mesh.cells),) or not np.isin(status,(ACTIVE,HOLE,FRINGE)).all():
        raise ValueError('one valid status per donor cell required')
    indices=np.asarray(receiver_indices,int);points=np.asarray(receiver_points,float)
    if points.shape != (len(indices),3) or not np.isfinite(points).all():
        raise ValueError('finite receiver coordinates required')
    if not isinstance(solid,SphereSolid):raise TypeError('a validated SphereSolid clearance contract is required')
    active=np.flatnonzero(status==ACTIVE)
    if len(active)<4:raise ValueError('fewer than four active donors')
    locator=_Locator(donor_mesh,active); donors=[]; weights=[]; containing=[]
    for receiver,point in zip(indices,points):
        cell=locator.containing(point)
        if cell<0:raise ValueError(f'orphan receiver {receiver}: no containing active polyhedron')
        accepted=False
        for count in sorted(set((min(32,len(active)),min(64,len(active)),min(128,len(active)),min(max_candidates,len(active))))):
            _,local=locator.tree.query(point,k=count); candidates=active[np.atleast_1d(local)]
            candidates=candidates[solid.fluid_halfspace(donor_mesh.cell_centers[candidates],point)]
            if len(candidates)<4:continue
            try:
                tri=Delaunay(donor_mesh.cell_centers[candidates]-point)
                simplex=tri.find_simplex(np.zeros(3),tol=1e-12)
            except QhullError:continue
            if simplex<0:continue
            transform=tri.transform[simplex]; w=transform[:3]@(-transform[3]);w=np.r_[w,1-w.sum()]
            if w.min() < -1e-11:continue
            chosen=candidates[tri.simplices[simplex]]
            if np.linalg.norm(w@donor_mesh.cell_centers[chosen]-point)>1e-10:continue
            donors.append(chosen);weights.append(w);containing.append(cell);accepted=True;break
        if not accepted:raise ValueError(f'orphan receiver {receiver}: no solid-safe convex donor tetrahedron')
    return DonorMap3D(indices,np.asarray(donors,int).reshape(-1,4),np.asarray(weights,float).reshape(-1,4),np.asarray(containing,int))

def build_spherical_overset(background,body,solid,split_radius,fringe_width):
    if split_radius<=solid.radius or fringe_width<=0:raise ValueError('fluid split radius and positive fringe width required')
    rb=np.linalg.norm(background.cell_centers-solid.center,axis=1)
    rr=np.linalg.norm(body.cell_centers-solid.center,axis=1)
    bs=np.where(rb<split_radius,HOLE,np.where(rb<split_radius+fringe_width,FRINGE,ACTIVE))
    # Conservative Cartesian/general-cell AABB blanking. For Cartesian cells
    # this is an exact sphere intersection test; for other cells it may blank
    # extra cells but cannot leave a solid-intersecting active background cell.
    lows=np.array([background.vertices[list(cell)].min(axis=0) for cell in background.cells])
    highs=np.array([background.vertices[list(cell)].max(axis=0) for cell in background.cells])
    closest=np.minimum(np.maximum(solid.center,lows),highs)
    intersects=np.linalg.norm(closest-solid.center,axis=1)<=solid.radius*(1+1e-12)
    bs[intersects]=HOLE
    rs=np.where(rr<=solid.radius,HOLE,np.where(rr>split_radius+2*fringe_width,FRINGE,ACTIVE))
    bi=np.flatnonzero(bs==FRINGE);ri=np.flatnonzero(rs==FRINGE)
    if not len(bi) or not len(ri):raise ValueError('both grids require fringe receivers')
    bmap=build_donor_map(body,rs,background.cell_centers[bi],bi,solid)
    rmap=build_donor_map(background,bs,body.cell_centers[ri],ri,solid)
    return OversetConnectivity3D((background,body),bs,rs,bmap,rmap)

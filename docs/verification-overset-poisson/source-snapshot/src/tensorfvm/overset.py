"""Static two-dimensional overset connectivity for finite-volume cell fields.

A body-fitted component overlaps a Cartesian background. Hole cells are excluded,
fringe values use barycentric interpolation from the other grid's ACTIVE cells,
and orphan receivers are an error. This is point-value coupling: it does NOT
provide conservative face fluxes, pressure coupling, moving-grid GCL, or an NS
solver. SciPy is imported only when connectivity is constructed.
"""
from dataclasses import dataclass
import numpy as np

ACTIVE, HOLE, FRINGE = 0, 1, 2


def _polygon_geometry(p):
    q = np.roll(p, -1, axis=0)
    cross = p[:, 0] * q[:, 1] - q[:, 0] * p[:, 1]
    area = .5 * cross.sum()
    if not np.isfinite(area) or area <= 0:
        raise ValueError('Overset cell polygons must have positive CCW area')
    return ((p + q) * cross[:, None]).sum(axis=0) / (6 * area), area


def points_in_polygon(points, polygon):
    """Ray-casting inclusion; points on polygon segments count as inside."""
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    poly = np.asarray(polygon, dtype=float)
    if poly.ndim != 2 or poly.shape[1] != 2 or len(poly) < 3 or not np.isfinite(poly).all():
        raise ValueError('Expected a finite polygon with at least three vertices')
    inside = np.zeros(len(pts), dtype=bool)
    boundary = inside.copy()
    tolerance = 1e-12 * max(1., np.ptp(poly, axis=0).max())
    for a, b in zip(poly, np.roll(poly, -1, axis=0)):
        edge = b - a
        rel = pts - a
        cross = rel[:, 0] * edge[1] - rel[:, 1] * edge[0]
        dot = rel @ edge
        boundary |= (np.abs(cross) <= tolerance * np.linalg.norm(edge)) & (dot >= -tolerance) & (dot <= edge @ edge + tolerance)
        crossing = (a[1] > pts[:, 1]) != (b[1] > pts[:, 1])
        if abs(edge[1]) > tolerance:
            xhit = a[0] + (pts[:, 1] - a[1]) * edge[0] / edge[1]
            inside ^= crossing & (pts[:, 0] < xhit)
    return inside | boundary


@dataclass(frozen=True)
class OversetGrid:
    name: str
    centers: np.ndarray
    polygons: tuple
    neighbors: tuple
    volumes: np.ndarray
    shape: tuple
    boundaries: dict

    @classmethod
    def from_polygons(cls, name, polygons, *, shape=None, boundaries=None):
        """Use actual FV polygons, including existing Mesh2D triangle/quad cells."""
        polygons = tuple(np.asarray(p, dtype=float).copy() for p in polygons)
        if not polygons:
            raise ValueError('Overset grid cannot be empty')
        geometry = [_polygon_geometry(p) for p in polygons]
        centers = np.array([g[0] for g in geometry])
        volumes = np.array([g[1] for g in geometry])
        edges, neighbors = {}, [set() for _ in polygons]
        # Keys use exact floating coordinates. Imported Mesh2D shared vertices
        # must be shared exactly, rather than silently welding unrelated faces.
        for i, p in enumerate(polygons):
            if p.ndim != 2 or p.shape[1] != 2 or len(p) < 3 or not np.isfinite(p).all():
                raise ValueError('Invalid finite-volume polygon')
            for a, b in zip(p, np.roll(p, -1, axis=0)):
                key = tuple(sorted((tuple(a), tuple(b))))
                if key in edges:
                    j, count = edges[key]
                    if count != 1:
                        raise ValueError('Non-manifold overset grid edge')
                    edges[key] = (j, 2)
                    neighbors[i].add(j); neighbors[j].add(i)
                else:
                    edges[key] = (i, 1)
        shape = (len(polygons),) if shape is None else tuple(shape)
        if np.prod(shape) != len(polygons):
            raise ValueError('Grid shape does not match cell count')
        boundaries = {} if boundaries is None else {k:np.asarray(v, dtype=bool).copy() for k,v in boundaries.items()}
        if any(v.shape != (len(polygons),) for v in boundaries.values()):
            raise ValueError('Boundary cell mask shape mismatch')
        return cls(name, centers, polygons, tuple(np.array(sorted(n), dtype=int) for n in neighbors), volumes, shape, boundaries)

    @classmethod
    def from_mesh(cls, mesh, name='mesh', *, outer_boundary_mask=None):
        """Adapt an existing TensorFVM Mesh2D using its real cell vertices."""
        def numpy(value):
            return value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)
        vertices = numpy(mesh.vertices)
        cells = getattr(mesh, 'cell_nodes', getattr(mesh, 'cells', None))
        if cells is None:
            raise ValueError('Mesh2D requires explicit cell vertex connectivity')
        boundaries = {}
        if outer_boundary_mask is not None:
            boundaries['outer'] = np.asarray(outer_boundary_mask, dtype=bool)
        return cls.from_polygons(name, [vertices[np.asarray(c, dtype=int)] for c in cells], boundaries=boundaries)


def cartesian_grid(bounds, nx, ny, name='background'):
    """Exact rectangular FV polygons; cell order is (ny, nx)."""
    xmin, xmax, ymin, ymax = map(float, bounds)
    if not (xmax > xmin and ymax > ymin and nx >= 2 and ny >= 2):
        raise ValueError('Invalid Cartesian bounds or resolution')
    x, y = np.linspace(xmin, xmax, nx+1), np.linspace(ymin, ymax, ny+1)
    polygons = [np.array([[x[i],y[j]],[x[i+1],y[j]],[x[i+1],y[j+1]],[x[i],y[j+1]]]) for j in range(ny) for i in range(nx)]
    outer = np.zeros((ny,nx), dtype=bool)
    outer[[0,-1],:] = True; outer[:,[0,-1]] = True
    return OversetGrid.from_polygons(name, polygons, shape=(ny,nx), boundaries={'outer':outer.ravel()})


def annular_grid(center, inner_radius, outer_radius, ntheta, nradial, name='body-fitted', radial_growth=1.):
    """Straight-edge annular quads, periodic in theta, shape (nradial, ntheta)."""
    center = np.asarray(center, dtype=float)
    if center.shape != (2,) or not np.isfinite(center).all() or not (0 < inner_radius < outer_radius and ntheta >= 8 and nradial >= 3 and radial_growth >= 1.):
        raise ValueError('Invalid annular mesh parameters')
    theta = np.arange(ntheta) * (2*np.pi/ntheta)
    widths = radial_growth ** np.arange(nradial, dtype=float)
    widths *= (outer_radius-inner_radius)/widths.sum()
    radii = np.r_[inner_radius,inner_radius+np.cumsum(widths)]
    vertices = center + radii[:,None,None]*np.stack((np.cos(theta),np.sin(theta)),axis=-1)[None]
    polygons = []
    for j in range(nradial):
        for i in range(ntheta):
            k = (i+1)%ntheta
            polygons.append(vertices[[j,j+1,j+1,j],[i,i,k,k]])
    inner, outer = np.zeros((nradial,ntheta),bool), np.zeros((nradial,ntheta),bool)
    inner[0,:] = True; outer[-1,:] = True
    return OversetGrid.from_polygons(name,polygons,shape=(nradial,ntheta),boundaries={'inner':inner.ravel(),'outer':outer.ravel()})


def circle_polygon(center, radius, points=256):
    t = np.arange(points)*2*np.pi/points
    return np.asarray(center)[None] + radius*np.stack((np.cos(t),np.sin(t)),axis=-1)


@dataclass(frozen=True)
class ReceiverStencil:
    receiver_grid: int
    receiver_cells: np.ndarray
    donor_grid: int
    donor_cells: np.ndarray
    weights: np.ndarray


@dataclass(frozen=True)
class OversetConnectivity:
    grids: tuple
    states: tuple
    stencils: tuple
    body_polygon: np.ndarray
    blanking_polygon: np.ndarray

    def interpolate(self, fields):
        """Simultaneous exchange; donor values are never fringe values.

        Input and output contain all cells (including unused HOLE entries).
        Scalar fields have shape (N,), vector/tensor fields (N, ...).
        """
        if len(fields) != len(self.grids):
            raise ValueError('One field array per overset component is required')
        arrays = tuple(np.asarray(f) for f in fields)
        if any(a.shape[0] != len(g.centers) for a,g in zip(arrays,self.grids)):
            raise ValueError('Field cell count mismatch')
        if any(a.shape[1:] != arrays[0].shape[1:] for a in arrays):
            raise ValueError('Component field shape mismatch')
        result = [np.array(a,dtype=np.result_type(a.dtype,float),copy=True) for a in arrays]
        for s in self.stencils:
            values = arrays[s.donor_grid][s.donor_cells]
            extra = (None,)*(values.ndim-2)
            result[s.receiver_grid][s.receiver_cells] = (values*s.weights[(slice(None),slice(None))+extra]).sum(axis=1)
        return tuple(result)

    def diagnostics(self):
        reports = []
        for i,(g,state) in enumerate(zip(self.grids,self.states)):
            reports.append(dict(name=g.name,cells=len(state),active=int((state==ACTIVE).sum()),hole=int((state==HOLE).sum()),fringe=int((state==FRINGE).sum()),volume=float(g.volumes.sum())))
        row_error = max(float(np.abs(s.weights.sum(axis=1)-1).max()) for s in self.stencils)
        affine_error = max(float(np.abs(np.einsum('nk,nkj->nj',s.weights,self.grids[s.donor_grid].centers[s.donor_cells])-self.grids[s.receiver_grid].centers[s.receiver_cells]).max()) for s in self.stencils)
        return dict(grids=reports,orphan_receivers=0,max_row_sum_error=row_error,max_affine_coordinate_error=affine_error,point_value_coupling=True,conservative_flux_coupling=False)

    def save(self, path):
        """Save geometry, classifications and donor stencils without pickles."""
        arrays = {'body_polygon':self.body_polygon,'blanking_polygon':self.blanking_polygon}
        for i,(g,state) in enumerate(zip(self.grids,self.states)):
            nodes = np.concatenate(g.polygons)
            arrays.update({f'g{i}_centers':g.centers,f'g{i}_volumes':g.volumes,f'g{i}_cell_vertices':nodes,f'g{i}_cell_offsets':np.r_[0,np.cumsum([len(p) for p in g.polygons])],f'g{i}_state':state,f'g{i}_shape':np.array(g.shape)})
        for i,s in enumerate(self.stencils):
            arrays.update({f's{i}_receiver_grid':np.array(s.receiver_grid),f's{i}_receiver_cells':s.receiver_cells,f's{i}_donor_grid':np.array(s.donor_grid),f's{i}_donor_cells':s.donor_cells,f's{i}_weights':s.weights})
        np.savez_compressed(path,**arrays)


def _expand(mask, neighbors, layers, forbidden):
    selected = mask.copy()
    for _ in range(layers):
        expanded = selected.copy()
        for i in np.flatnonzero(selected):
            expanded[neighbors[i]] = True
        selected = expanded & ~forbidden
    return selected


def _triangle_intersects_polygon(triangle, polygon):
    """Reject donor triangles entering holes, including thin corner crossings."""
    if points_in_polygon(triangle,polygon).any():
        return True
    # A hole wholly enclosed by the triangle also invalidates a stencil.
    if points_in_polygon(polygon,triangle).any():
        return True
    a, b = polygon, np.roll(polygon,-1,axis=0)
    def cross(u,v):
        return u[...,0]*v[...,1]-u[...,1]*v[...,0]
    for p,q in zip(triangle,np.roll(triangle,-1,axis=0)):
        c1,c2 = cross(q-p,a-p),cross(q-p,b-p)
        c3,c4 = cross(b-a,p-a),cross(b-a,q-a)
        if ((c1*c2 < 0)&(c3*c4 < 0)).any():
            return True
    return False



def background_hole_mask(background, body_polygon, blanking_polygon):
    """Center-based artificial cut plus conservative physical-body exclusion.

    Cells touching/intersecting the solid are holes even if their centers lie
    outside both the body and artificial cut. This does not turn those Cartesian
    faces into physical wall faces; the component alone represents the wall.
    """
    body = np.asarray(body_polygon,dtype=float)
    hole = points_in_polygon(background.centers,blanking_polygon)
    body_min, body_max = body.min(axis=0), body.max(axis=0)
    for i, polygon in enumerate(background.polygons):
        if not hole[i] and (polygon.max(axis=0)>=body_min).all() and (polygon.min(axis=0)<=body_max).all():
            if _triangle_intersects_polygon(polygon,body):
                hole[i] = True
    return hole


def build_overset(background, component, body_polygon, blanking_polygon, *, fringe_layers=1):
    """Classify and connect a static body-fitted/Cartesian two-grid overlap.

    The blanking contour must enclose the solid body and remain inside the
    component. Background cells whose CENTERS lie in that contour, and all cells
    intersecting the physical solid, are HOLE;
    adjacent cell layers become FRINGE. Component outer layers become FRINGE.
    Both receiver groups interpolate exclusively from the other grid's ACTIVE
    centers. Insufficient overlap, crossed holes, or absent donors are errors.
    """
    from scipy.spatial import Delaunay, QhullError
    if not isinstance(fringe_layers,int) or fringe_layers < 1:
        raise ValueError('fringe_layers must be a positive integer')
    body, blanking = np.asarray(body_polygon,dtype=float),np.asarray(blanking_polygon,dtype=float)
    if not points_in_polygon(body,blanking).all():
        raise ValueError('Blanking contour must enclose the physical body')
    grids = (background,component)
    if 'outer' not in component.boundaries or not component.boundaries['outer'].any():
        raise ValueError('Body-fitted component requires an outer boundary cell mask')
    bg_hole = background_hole_mask(background,body,blanking)
    cp_hole = points_in_polygon(component.centers,body)
    if not bg_hole.any():
        raise ValueError('Background resolution does not resolve the blanking hole')
    bg_fringe = _expand(bg_hole,background.neighbors,fringe_layers,np.zeros(len(bg_hole),bool)) & ~bg_hole
    cp_fringe = _expand(component.boundaries['outer'],component.neighbors,fringe_layers-1,cp_hole)
    states = []
    for hole,fringe in ((bg_hole,bg_fringe),(cp_hole,cp_fringe)):
        state = np.full(len(hole),ACTIVE,dtype=np.uint8)
        state[hole]=HOLE; state[fringe]=FRINGE
        if not fringe.any():
            raise ValueError('Each component must have fringe receiver cells')
        states.append(state)
    stencils = []
    for receiver,donor in ((0,1),(1,0)):
        rcells = np.flatnonzero(states[receiver]==FRINGE)
        dcells = np.flatnonzero(states[donor]==ACTIVE)
        if len(dcells)<3:
            raise ValueError('Insufficient active donor cells')
        try:
            triangulation = Delaunay(grids[donor].centers[dcells])
        except QhullError as exc:
            raise ValueError('Active donor centers do not span a two-dimensional region') from exc
        points = grids[receiver].centers[rcells]
        simplex = triangulation.find_simplex(points,tol=1e-12)
        missing = np.flatnonzero(simplex<0)
        if len(missing):
            raise ValueError(f'Orphan overset receivers in {grids[receiver].name}: {len(missing)}, first cell {int(rcells[missing[0]])}; increase overlap or refine the donor grid')
        donor_cells = dcells[triangulation.simplices[simplex]]
        excluded_polygon = body if donor==1 else blanking
        for k in np.unique(simplex):
            triangle = grids[donor].centers[dcells[triangulation.simplices[k]]]
            if _triangle_intersects_polygon(triangle,excluded_polygon):
                raise ValueError(f'Donor triangle crosses a hole for receiver grid {grids[receiver].name}; increase overlap')
        transforms = triangulation.transform[simplex]
        first = np.einsum('nij,nj->ni',transforms[:,:2],points-transforms[:,2])
        weights = np.column_stack((first,1-first.sum(axis=1)))
        if not np.isfinite(weights).all() or weights.min() < -1e-10:
            raise ValueError('Invalid or extrapolating donor stencil')
        weights = np.maximum(weights,0.)
        weights /= weights.sum(axis=1,keepdims=True)
        error = np.abs(np.einsum('nk,nkj->nj',weights,grids[donor].centers[donor_cells])-points).max()
        if error > 1e-10*max(1.,np.abs(points).max()):
            raise ValueError('Donor interpolation failed affine coordinate reproduction')
        stencils.append(ReceiverStencil(receiver,rcells,donor,donor_cells,weights))
    return OversetConnectivity(grids,tuple(states),tuple(stencils),body.copy(),blanking.copy())

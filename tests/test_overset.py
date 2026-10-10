"""Geometry and interpolation contracts, with actual overlapping FV grids."""
import numpy as np
import pytest
from tensorfvm.overset import (
    ACTIVE, HOLE, FRINGE, OversetGrid, annular_grid, cartesian_grid,
    circle_polygon, build_overset, points_in_polygon, _triangle_intersects_polygon,
    background_hole_mask,
)


def connection(n=32):
    bg = cartesian_grid((-2,2,-2,2),n,n)
    cp = annular_grid((0,0),.3,1.,3*n, max(6,n//2))
    return build_overset(bg,cp,circle_polygon((0,0),.3),circle_polygon((0,0),.65))


def test_geometry_classification_and_active_only_donors(tmp_path):
    conn = connection()
    bg,cp = conn.grids
    assert bg.shape == (32,32)
    assert np.all(bg.volumes>0) and np.all(cp.volumes>0)
    assert np.isclose(bg.volumes.sum(),16.)
    assert np.isclose(cp.volumes.sum(),.5*96*np.sin(2*np.pi/96)*(1-.3**2))
    for i,p in enumerate(bg.polygons):
        if _triangle_intersects_polygon(p,conn.body_polygon):
            assert conn.states[0][i] == HOLE
    for i in np.flatnonzero(conn.states[0]==HOLE):
        assert not np.any(conn.states[0][bg.neighbors[i]]==ACTIVE)
    assert np.all(conn.states[1][cp.boundaries['outer']]==FRINGE)
    assert np.all(conn.states[1][cp.boundaries['inner']]==ACTIVE)
    for stencil in conn.stencils:
        assert stencil.donor_grid != stencil.receiver_grid
        assert np.all(conn.states[stencil.donor_grid][stencil.donor_cells]==ACTIVE)
        assert np.all(stencil.weights>=0)
        assert np.max(np.abs(stencil.weights.sum(axis=1)-1))<1e-14
        for tri in conn.grids[stencil.donor_grid].centers[stencil.donor_cells]:
            forbidden = conn.body_polygon if stencil.donor_grid==1 else conn.blanking_polygon
            assert not _triangle_intersects_polygon(tri,forbidden)
    path = tmp_path/'connectivity.npz'
    conn.save(path)
    with np.load(path,allow_pickle=False) as raw:
        assert np.array_equal(raw['g0_state'],conn.states[0])
        assert np.array_equal(raw['s0_donor_cells'],conn.stencils[0].donor_cells)
        assert np.allclose(raw['g1_volumes'],cp.volumes)
        assert raw['g1_cell_offsets'][-1] == len(raw['g1_cell_vertices'])
    report = conn.diagnostics()
    assert report['orphan_receivers']==0
    assert report['max_affine_coordinate_error']<1e-14
    assert report['conservative_flux_coupling'] is False


def test_constant_and_affine_vector_reproduction_and_no_fringe_circularity():
    conn = connection()
    def exact(points):
        x,y = points.T
        return np.column_stack((2+3*x-2*y,-4+.2*x+.7*y))
    fields = tuple(exact(g.centers) for g in conn.grids)
    poisoned = [f.copy() for f in fields]
    for f,s in zip(poisoned,conn.states):
        f[s==FRINGE] = np.nan
        f[s==HOLE] = np.nan
    exchanged = conn.interpolate(poisoned)
    for i,s in enumerate(conn.states):
        assert np.allclose(exchanged[i][s!=HOLE],fields[i][s!=HOLE],atol=2e-14,rtol=0)
        assert np.array_equal(exchanged[i][s==ACTIVE],poisoned[i][s==ACTIVE])
        assert np.isnan(poisoned[i][s==FRINGE]).all()
    scalar = conn.interpolate([np.full(len(g.centers),7.) for g in conn.grids])
    assert all(np.allclose(f,7.,rtol=0,atol=2e-15) for f in scalar)


def test_quadratic_interpolation_refines():
    errors = []
    for n in (24,48,96):
        conn = connection(n)
        fields = tuple((g.centers**2).sum(axis=1) for g in conn.grids)
        exchanged = conn.interpolate(fields)
        errors.append(max(np.max(np.abs(f[s==FRINGE]-a[s==FRINGE])) for f,a,s in zip(exchanged,fields,conn.states)))
    assert errors[1]<.4*errors[0] and errors[2]<.4*errors[1]
    assert min(np.log2(np.array(errors[:-1])/np.array(errors[1:])))>1.5


def test_orphans_and_hole_crossing_are_rejected():
    bg = cartesian_grid((-2,2,-2,2),24,24)
    cp = annular_grid((0,0),.3,.67,72,12)
    with pytest.raises(ValueError,match='Orphan|crosses a hole'):
        build_overset(bg,cp,circle_polygon((0,0),.3),circle_polygon((0,0),.65))
    # All three vertices and edges lie outside the circle; a test that only
    # checks edge distance misses the solid entirely enclosed by this triangle.
    tri = np.array([[-2.,-2.],[2.,-2.],[0.,2.]])
    body = circle_polygon((0,0),.3)
    assert not points_in_polygon(tri,body).any()
    assert _triangle_intersects_polygon(tri,body)
    # Thin solid cut at a triangle corner, even with no included vertices.
    polygon = np.array([[.9,-2.],[1.1,-2.],[1.1,2.],[.9,2.]])
    triangle = np.array([[0.,0.],[2.,0.],[0.,1.]])
    assert _triangle_intersects_polygon(triangle,polygon)


def test_existing_mesh2d_adapter_and_invalid_inputs():
    from types import SimpleNamespace
    original = cartesian_grid((-1,1,-1,1),8,8)
    vertices = np.concatenate(original.polygons)
    mesh = SimpleNamespace(vertices=vertices,cell_nodes=np.arange(len(vertices)).reshape(-1,4))
    adapted = OversetGrid.from_mesh(mesh,outer_boundary_mask=original.boundaries['outer'])
    assert np.allclose(adapted.centers,original.centers)
    assert np.allclose(adapted.volumes,original.volumes)
    assert all(np.array_equal(a,b) for a,b in zip(adapted.neighbors,original.neighbors))
    with pytest.raises(ValueError,match='shape'):
        OversetGrid.from_polygons('bad',original.polygons,shape=(2,))
    with pytest.raises(ValueError,match='positive'):
        OversetGrid.from_polygons('bad',[original.polygons[0][::-1]])
    conn = connection(24)
    with pytest.raises(ValueError,match='cell count'):
        conn.interpolate([np.zeros(1),np.zeros(1)])
    with pytest.raises(ValueError,match='enclose'):
        build_overset(*conn.grids,circle_polygon((0,0),.8),circle_polygon((0,0),.65))


def test_physical_holes_exclude_cells_with_centers_outside_the_blanking_cut():
    coarse = cartesian_grid((-2,2,-2,2),8,8)
    body = circle_polygon((0,0),.3)
    cut = circle_polygon((0,0),.31)
    assert not points_in_polygon(coarse.centers,cut).any()
    holes = background_hole_mask(coarse,body,cut)
    assert holes.sum()==4
    assert np.all(np.linalg.norm(coarse.centers[holes],axis=1)>.31)
    for polygon in np.array(coarse.polygons)[holes]:
        assert _triangle_intersects_polygon(polygon,body)

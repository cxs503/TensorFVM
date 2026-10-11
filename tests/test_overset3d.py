import numpy as np
import pytest
from tensorfvm.overset3d import (ACTIVE, HOLE, FRINGE, SphereSolid,
    CartesianGrid3D, sphere_shell_mesh, build_spherical_overset, build_donor_map)

@pytest.fixture(scope='module')
def problem():
    background=CartesianGrid3D.from_bounds((-3,)*3,(3,)*3,(14,)*3)
    body=sphere_shell_mesh(1,2.6,10,2)
    solid=SphereSolid((0,0,0),1)
    connectivity=build_spherical_overset(background,body,solid,1.5,.3)
    return background,body,solid,connectivity

def test_cartesian_geometry():
    mesh=CartesianGrid3D.from_bounds((-1,-2,-3),(1,2,3),(2,3,4))
    np.testing.assert_allclose(mesh.cell_volumes,2)
    assert len(mesh.cells)==24
    assert mesh.audit()['minimum_face_projection']>0

def test_bidirectional_no_orphans_or_inactive_donors(problem):
    background,body,solid,c=problem
    for receiver,donor,status,donormap in ((background,body,c.body_status,c.background_from_body),(body,background,c.background_status,c.body_from_background)):
        assert len(donormap.receiver_indices)>0
        assert np.all(status[donormap.donor_indices]==ACTIVE)
        assert np.all(status[donormap.containing_cells]==ACTIVE)
        assert np.all(donormap.weights>=-1e-11)
        np.testing.assert_allclose(donormap.weights.sum(axis=1),1,atol=2e-14)
        for index,ids in zip(donormap.receiver_indices,donormap.donor_indices):
            assert solid.fluid_halfspace(donor.cell_centers[ids],receiver.cell_centers[index]).all()
        np.testing.assert_allclose(np.einsum('ij,ijk->ik',donormap.weights,donor.cell_centers[donormap.donor_indices]),receiver.cell_centers[donormap.receiver_indices],atol=1e-12)

def test_constant_affine_and_simultaneous_exchange(problem):
    background,body,solid,c=problem
    constant=[np.full(len(m.cells),4.25) for m in c.grids]
    for actual in c.interpolate(constant):np.testing.assert_allclose(actual,4.25,atol=1e-13)
    fields=[np.column_stack((2+m.cell_centers@np.array((.2,-.4,.7)),m.cell_centers@np.array((-.8,.3,.1)))) for m in c.grids]
    result=c.interpolate(fields)
    for actual,expected in zip(result,fields):np.testing.assert_allclose(actual,expected,atol=3e-13)
    different=[np.full(len(background.cells),1.),np.full(len(body.cells),2.)]
    result=c.interpolate(different)
    assert np.all(result[0][c.background_status!=FRINGE]==1)
    assert np.all(result[1][c.body_status!=FRINGE]==2)
    np.testing.assert_allclose(result[0][c.background_status==FRINGE],2)
    np.testing.assert_allclose(result[1][c.body_status==FRINGE],1)
    assert np.all(different[0]==1) and np.all(different[1]==2)

def test_actual_containing_polyhedra(problem):
    background,body,solid,c=problem
    for receiver,donor,dm in ((background,body,c.background_from_body),(body,background,c.body_from_background)):
        for receiver_id,cell in zip(dm.receiver_indices,dm.containing_cells):
            point=receiver.cell_centers[receiver_id]
            faces=np.flatnonzero((donor.owner==cell)|(donor.neighbor==cell))
            sign=np.where(donor.owner[faces]==cell,1,-1)
            projection=np.einsum('ij,ij->i',point-donor.face_centers[faces],donor.area_vectors[faces])*sign
            assert projection.max()<1e-10

def test_orphan_and_solid_rejection(problem):
    background,body,solid,c=problem
    with pytest.raises(ValueError,match='orphan'):
        build_donor_map(background,c.background_status,np.array([[9,9,9]]),[77],solid)
    with pytest.raises(ValueError,match='solid'):
        solid.fluid_halfspace(np.array([[2,0,0]]),np.zeros(3))
    with pytest.raises(ValueError,match='fewer'):
        build_donor_map(background,np.full(len(background.cells),HOLE),np.array([[2,0,0]]),[0],solid)

def test_near_wall_centroid_inside_polyhedron_still_rejects_unsafe_tetrahedron():
    # A broad active polyhedron contains the point, but centres surrounding it
    # lie across the solid. Containment alone must not authorize the transfer.
    background=CartesianGrid3D.from_bounds((-2,)*3,(2,)*3,(4,)*3)
    with pytest.raises(ValueError,match='solid-safe'):
        build_donor_map(background,np.zeros(len(background.cells),int),np.array([[1.01,0,0]]),[9],SphereSolid((0,0,0),1))

def test_invalid_parameters():
    with pytest.raises(ValueError):CartesianGrid3D.from_bounds((0,0,0),(1,1,1),(1.5,2,2))
    with pytest.raises(ValueError):sphere_shell_mesh(2,1,2,1)
    with pytest.raises(ValueError):SphereSolid((0,0,0),-1)

def test_background_solid_intersection_is_never_active(problem):
    background,body,solid,c=problem
    for cell,status in zip(background.cells,c.background_status):
        xyz=background.vertices[list(cell)]
        closest=np.clip(solid.center,xyz.min(axis=0),xyz.max(axis=0))
        if np.linalg.norm(closest-solid.center)<=solid.radius:
            assert status==HOLE

def test_nonconvex_native_donor_is_rejected():
    from dataclasses import replace
    mesh=CartesianGrid3D.from_bounds((-2,)*3,(2,)*3,(2,)*3)
    vertices=mesh.vertices.copy()
    vertices[mesh.cells[0][0]]=np.array((3.,3.,3.))
    inconsistent=replace(mesh,vertices=vertices)
    with pytest.raises(ValueError,match='nonconvex'):
        build_donor_map(inconsistent,np.zeros(len(mesh.cells),int),np.array([[1.5,1.5,1.5]]),[0],SphereSolid((0,0,0),.1))

def test_shared_macroscopic_exchange_contract(problem):
    from tensorfvm.overset_exchange import MacroscopicState,exchange_macroscopic
    background,body,solid,c=problem
    states=[]; references=[]
    for grid,status in zip(c.grids,(c.background_status,c.body_status)):
        points=grid.cell_centers
        reference=MacroscopicState(np.ones(len(points))*998.2,.01*points,101325+points@np.array((2.,3.,-1.)),.5)
        references.append(reference)
        rho,u,p=reference.density.copy(),reference.velocity.copy(),reference.pressure.copy()
        rho[status==FRINGE]+=100
        u[status==FRINGE]+=2
        p[status==FRINGE]+=1000
        states.append(MacroscopicState(rho,u,p,.5))
    result=exchange_macroscopic(c,states)
    for actual,expected in zip(result,references):
        np.testing.assert_allclose(actual.density,expected.density,atol=1e-12)
        np.testing.assert_allclose(actual.velocity,expected.velocity,atol=1e-13)
        np.testing.assert_allclose(actual.pressure,expected.pressure,atol=3e-10)
        assert actual.time==.5


def test_graded_layers_preserve_first_height_and_total_thickness():
    from tensorfvm.overset3d import shell_radial_coordinates
    radii=shell_radial_coordinates(1,2.6,10,.02)
    h=np.diff(radii)
    assert radii[0]==1 and radii[-1]==2.6
    np.testing.assert_allclose(h[0],.02,rtol=1e-12)
    assert np.all(h>0) and np.all(np.diff(h)>0)
    np.testing.assert_allclose(h[1:]/h[:-1],np.full(9,h[1]/h[0]),rtol=2e-12)
    mesh=sphere_shell_mesh(1,2.6,10,1,first_height=.02)
    assert mesh.audit()['minimum_cell_volume']>0
    np.testing.assert_allclose(np.unique(np.round(np.linalg.norm(mesh.vertices,axis=1),12)),radii,atol=1e-12)
    for bad in (0,-1,.17,float('nan')):
        with pytest.raises(ValueError):shell_radial_coordinates(1,2.6,10,bad)
    with pytest.raises(ValueError):shell_radial_coordinates(1,2,1,.1)
    np.testing.assert_allclose(shell_radial_coordinates(1,2,1,1),[1,2])


def test_graded_shell_cartesian_overlap_and_unsafe_layout_rejection():
    background=CartesianGrid3D.from_bounds((-3,)*3,(3,)*3,(14,)*3)
    body=sphere_shell_mesh(1,2.6,10,1,first_height=.02)
    solid=SphereSolid((0,0,0),1)
    with pytest.raises(ValueError,match='orphan'):
        build_spherical_overset(background,body,solid,1.5,.3)
    connection=build_spherical_overset(background,body,solid,1.4,.3)
    fields=[1+m.cell_centers@np.array([.2,-.4,.7]) for m in connection.grids]
    for actual,expected in zip(connection.interpolate(fields),fields):
        np.testing.assert_allclose(actual,expected,atol=2e-13)
    for mapping,status in ((connection.background_from_body,connection.body_status),(connection.body_from_background,connection.background_status)):
        assert len(mapping.receiver_indices)>0
        assert np.all(status[mapping.donor_indices]==ACTIVE)
        assert np.all(status[mapping.containing_cells]==ACTIVE)


def test_circumscribed_wall_preserves_thin_fluid_layers():
    body=sphere_shell_mesh(1,2.6,10,1,first_height=.02,wall_geometry='circumscribed')
    # Every inner facet plane must exclude the analytic sphere. Centres alone
    # are insufficient to prove wall clearance.
    boundary=np.flatnonzero(body.neighbor<0)
    inner=boundary[np.linalg.norm(body.face_centers[boundary],axis=1)<1.5]
    distances=np.abs(np.einsum('ij,ij->i',body.face_centers[inner],body.area_vectors[inner]))/np.linalg.norm(body.area_vectors[inner],axis=1)
    assert np.all(distances>=1-2e-14)
    assert np.all(np.linalg.norm(body.cell_centers,axis=1)>1)
    radii=np.unique(np.round(np.linalg.norm(body.vertices,axis=1),12))
    np.testing.assert_allclose(radii[1]-radii[0],.02,atol=2e-12)
    background=CartesianGrid3D.from_bounds((-3,)*3,(3,)*3,(14,)*3)
    connection=build_spherical_overset(background,body,SphereSolid((0,0,0),1),1.4,.3)
    assert not np.any(connection.body_status==HOLE)
    assert len(connection.body_from_background.receiver_indices)>0
    with pytest.raises(ValueError,match='wall geometry'):
        sphere_shell_mesh(1,2.6,10,1,wall_geometry='invalid')

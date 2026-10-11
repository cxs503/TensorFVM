from types import SimpleNamespace
import numpy as np
import pytest
from tensorfvm.suboff_hybrid_mesh import adaptive_meridian_stations,structured_layers,wall_facet_geometry_audit
from tensorfvm.suboff_geometry import LENGTH_FT,RADIUS_OVER_LENGTH


def test_adaptive_tolerance_refinement_junctions_and_spacing():
    coarse,cs=adaptive_meridian_stations(.001)
    fine,fs=adaptive_meridian_stations(.0002)
    assert len(fine)>len(coarse)
    for x in (0,3.333333/LENGTH_FT,10.645833/LENGTH_FT,13.979167/LENGTH_FT,1):
        assert x in fine
    assert np.diff(fine).min()>=1e-5
    assert fs['maximum_dense_meridian_chord_error_L']<=.0002
    assert cs['maximum_dense_meridian_chord_error_L']<=.001


def test_adaptive_rejects_spacing_conflict_and_station_limit():
    with pytest.raises(ValueError,match='spacing'):adaptive_meridian_stations(1e-12)
    with pytest.raises(ValueError,match='limit'):adaptive_meridian_stations(.0002,max_stations=5)


def test_default_stations_remain_cosine_with_junction_replacements():
    points,_,_,_,_,s=structured_layers(axial=16,azimuthal=8,layers=1,first_height=.0001)
    x=.5*(1-np.cos(np.linspace(0,np.pi,17)))
    for junction in np.array([3.333333,10.645833,13.979167])/LENGTH_FT:
        x[1+np.argmin(np.abs(x[1:-1]-junction))]=junction
    assert np.array_equal(np.unique(points[:2+15*8,0]),np.sort(x))
    assert s['adaptive_sampling']=={}


def test_actual_facet_interior_has_sag_despite_exact_vertices():
    def mesh(angle):
        x=np.array([.4,.6,.5]);theta=np.array([-angle,-angle,angle])
        p=np.column_stack((x,RADIUS_OVER_LENGTH*np.cos(theta),RADIUS_OVER_LENGTH*np.sin(theta)))
        return SimpleNamespace(vertices=p,face_vertices=[(0,1,2)],patch_labels=np.array(['wall']))
    wide=wall_facet_geometry_audit(mesh(np.pi/8));narrow=wall_facet_geometry_audit(mesh(np.pi/16))
    assert wide['maximum_sampled_vertex_distance_L']<1e-9
    expected=RADIUS_OVER_LENGTH*(1-np.cos(np.pi/8))
    assert wide['maximum_sampled_interior_distance_L']==pytest.approx(expected,rel=1e-8)
    assert narrow['maximum_sampled_interior_distance_L']<wide['maximum_sampled_interior_distance_L']/3

import hashlib
from pathlib import Path
import numpy as np
import pytest
from tensorfvm.suboff_geometry import radius,suboff_radius_profile,RADIUS_OVER_LENGTH,SOURCE_SHA256


def test_mother_dimensions_and_segment_stations():
    assert RADIUS_OVER_LENGTH == pytest.approx(.8333333/14.291667)
    assert radius(0.)==0
    assert radius(1.)==0
    assert np.allclose(suboff_radius_profile([.3,.5,.7]),1)
    assert np.all(suboff_radius_profile([-1,2])==0)
    x=np.linspace(0,1,10001)
    r=radius(x)
    assert np.isfinite(r).all() and r.min()>=0 and r.max()==RADIUS_OVER_LENGTH
    assert np.all(np.diff(r[x<3.333333/14.291667])>=-1e-12)
    assert np.all(np.diff(r[x>10.645833/14.291667])<=1e-10)


def test_archived_source_sha_and_exact_numpy_port():
    source=Path(__file__).parents[1]/'docs/verification-suboff-mesh/tensorlbm-suboff-cad-source.py'
    assert hashlib.sha256(source.read_bytes()).hexdigest()==SOURCE_SHA256
    text=source.read_text();a=text.index('def suboff_radius_profile(');b=text.index('\n\n# ---------------------------------------------------------------------------',a)
    original=text[a:b].replace('    config: SuboffConfig | None = None,\n','').replace('    if config is None:\n        config = SuboffConfig()\n\n','')
    ns={'np':np};exec(original,ns)
    points=np.r_[np.linspace(-.1,1.1,10001),np.array([3.333333,10.645833,13.979167])/14.291667]
    assert np.array_equal(suboff_radius_profile(points),ns['suboff_radius_profile'](points))


def test_actual_generated_mesh_is_tetra_body_fitted(tmp_path):
    gmsh=pytest.importorskip('gmsh')
    from tensorfvm.suboff_mesh import generate_suboff_mesh
    result=generate_suboff_mesh(tmp_path/'suboff.msh',wall_size=.075,far_size=.6,stations=25)
    assert result['element_types']==[4]
    assert result['cells']>100 and result['nodes']>100
    assert result['max_wall_radial_error_L']<.001
    assert all(result['patch_surfaces'][p]>0 for p in ('wall','inlet','outlet','farfield'))
    assert result['boundary_layer'] is False and result['physics_qualified'] is False

"""Reject invalid multiple-body domains before interpolation or area integration."""
import numpy as np
import pytest
from tensorfvm.overset import annular_grid,cartesian_grid,circle_polygon,build_overset

@pytest.mark.parametrize('bodies', [(), (circle_polygon((0,0),.3),)*2, (circle_polygon((0,0),.3),circle_polygon((.2,0),.3))])
def test_multiple_bodies_require_nonempty_disjoint_solids(bodies):
    body=circle_polygon((0,0),.3)
    with pytest.raises(ValueError,match='physical body|Physical bodies'):
        build_overset(cartesian_grid((-2,2,-2,2),32,32),annular_grid((0,0),.3,1,64,8),body,circle_polygon((0,0),.65),body_polygons=bodies)

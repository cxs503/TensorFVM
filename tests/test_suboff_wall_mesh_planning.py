import importlib.util
from pathlib import Path
import pytest

script=Path(__file__).parents[1]/'scripts/report_suboff_wall_mesh.py'
spec=importlib.util.spec_from_file_location('wall_mesh_report',script)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_planning_uses_first_cell_center_not_full_layer_height():
    # Cf=.005 gives u_tau/U=.05. Re=1e6 and h/L=2e-5 have
    # first-center y/L=1e-5 and hence y+=.5, not the full-height value 1.
    assert module.planning_yplus(1e6,2e-5,cf=.005)==pytest.approx(.5)


def test_drag_re_planning_exceeds_one_without_hidden_height_adjustment():
    assert module.planning_yplus(12e6,4e-6)<1
    assert module.planning_yplus(13.2180568e6,4e-6)>1
    with pytest.raises(ValueError):module.planning_yplus(12e6,-4e-6)

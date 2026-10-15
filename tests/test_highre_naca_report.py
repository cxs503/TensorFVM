"""Reference orientation and error norms must not hide wrong surface pressures."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest
import sys

spec=importlib.util.spec_from_file_location('naca_report',Path(__file__).resolve().parents[1]/'scripts/report_highre_naca.py')
report=importlib.util.module_from_spec(spec);spec.loader.exec_module(report)


def test_full_curve_norm_is_exact_for_linear_error():
    reference=np.array([[0.,1.],[.2,1.],[1.,1.]])
    error,table=report.curve_error([1.,0.,.4],[2.,1.,1.4],reference)
    assert error['relative_l2']==pytest.approx(1/np.sqrt(3))
    assert error['coverage_chord']==1.
    assert np.allclose(table[:,1]-table[:,2],table[:,0])


def test_partial_surface_does_not_extrapolate_to_full_coverage():
    error,table=report.curve_error([.1,.9],[1.,1.],np.array([[0.,1.],[1.,1.]]))
    assert error['relative_l2']==0.
    assert error['coverage_chord']==pytest.approx(.8)
    assert table[0,0]==.1 and table[-1,0]==.9


def test_zero_reference_and_disjoint_support_rejected():
    with pytest.raises(ValueError):report.curve_error([0.,1.],[1.,1.],np.array([[0.,0.],[1.,0.]]))
    with pytest.raises(ValueError):report.curve_error([2.,3.],[1.,1.],np.array([[0.,1.],[1.,1.]]))


def test_tmr_cp_file_runs_lower_then_upper(tmp_path):
    p=tmp_path/'cp.dat'
    p.write_text('variables="x","cp"\nzone, t="alpha=10"\n1 .2\n.5 .5\n0 -3.8\n.5 -1\n1 .2\n')
    curves=report.reference_surfaces(p,10)
    np.testing.assert_array_equal(curves['lower'],[[0,-3.8],[.5,.5],[1,.2]])
    np.testing.assert_array_equal(curves['upper'],[[0,-3.8],[.5,-1],[1,.2]])


def test_grid_study_rejects_transients_and_repeated_meshes():
    sys.modules['report_highre_naca']=report
    spec=importlib.util.spec_from_file_location('naca_qualify',Path(__file__).resolve().parents[1]/'scripts/qualify_highre_naca.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    rows=[dict(cells=100*2**i,mesh_sha256=str(i),alpha=0.,reynolds=6e6,sources_sha256='same',reference_manifest={},
        coefficients={'drag':cd},reference_coefficients={'drag':.008},numerical_passed=False,
        wall_resolution_passed=True,source_integrity_verified=True,reference_passed=True)
        for i,cd in enumerate([.0081,.00802,.00801])]
    result=module.qualify(rows)
    assert result['drag_grid_passed']
    assert not result['numerical_and_wall_passed'] and not result['qualified']
    rows[1]['mesh_sha256']=rows[0]['mesh_sha256']
    with pytest.raises(ValueError,match='Distinct grids'):module.qualify(rows)

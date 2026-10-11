import numpy as np
import pytest
from tensorfvm.overset_flux import PlanarFluxTransfer


def rectangles(xs, ys):
    return np.array([[[a,c],[b,d]] for a,b in zip(xs[:-1],xs[1:]) for c,d in zip(ys[:-1],ys[1:])])


def test_nonmatching_3d_interface_preserves_each_flux_and_orientation():
    donor = rectangles([0,.3,1], [0,.2,.8,1])
    receiver = rectangles([0,.1,.7,1], [0,.4,1])
    operator = PlanarFluxTransfer.build(donor, receiver)
    flux = operator.donor_measure[:,None]*np.arange(1,19).reshape(6,3)
    mapped = operator.transfer(flux)
    assert np.max(abs(operator.balance(flux,mapped))) < 4e-15
    assert np.all(mapped < 0)
    # A constant physical flux density is reproduced face by face.
    assert np.allclose(operator.transfer(2*operator.donor_measure), -2*operator.receiver_measure)


def test_exact_common_subface_integrals_and_line_interface():
    operator = PlanarFluxTransfer.build([[[0],[.4]],[[.4],[1]]], [[[0],[.2]],[[.2],[.7]],[[.7],[1]]], normal_sign=1)
    assert np.allclose(operator.transfer([.8,1.8]), [.4,1.3,.9])
    assert abs(operator.balance([.8,1.8],[.4,1.3,.9])) < 1e-15


@pytest.mark.parametrize('receivers', [ [[[0],[.9]]], [[[0],[.6]],[[.5],[1]]], [[[0],[1]],[[0],[1]]] ])
def test_gaps_and_duplicate_coverage_fail(receivers):
    with pytest.raises(ValueError,match='coverage|Overlapping'):
        PlanarFluxTransfer.build([[[0],[1]]],receivers)


def test_invalid_geometry_and_flux_fail():
    with pytest.raises(ValueError):
        PlanarFluxTransfer.build([[[0],[0]]], [[[0],[1]]])
    op = PlanarFluxTransfer.build([[[0],[1]]], [[[0],[1]]])
    for q in ([np.nan], [1,2]):
        with pytest.raises(ValueError): op.transfer(q)
    with pytest.raises(ValueError): op.balance([1],[1,2])

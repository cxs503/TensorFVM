import numpy as np
import pytest
from tensorfvm.native_field_transfer3d import transfer_cell_values


@pytest.mark.parametrize('components',[1,4])
def test_transfer_reproduces_affine_samples_and_copies_coincident_centers(components):
    rng=np.random.default_rng(723)
    source=rng.uniform(-1,1,(80,3));target=np.vstack((source[:7],rng.uniform(-.8,.8,(30,3))))
    slope=rng.normal(size=(3,components));offset=rng.normal(size=components)
    data=source@slope+offset;expected=target@slope+offset
    if components==1:data=data[:,0];expected=expected[:,0]
    before=data.copy();answer,stats=transfer_cell_values(source,data,target)
    np.testing.assert_allclose(answer,expected,atol=3e-14,rtol=0)
    np.testing.assert_array_equal(answer[:7],data[:7]);np.testing.assert_array_equal(data,before)
    assert stats['exact_center_copies']==7 and stats['flow_solved'] is False


def test_transfer_rejects_singular_geometry_and_nonfinite_fields():
    x=np.arange(24,dtype=float).reshape(8,3)
    with pytest.raises(ValueError,match='rank-deficient'):
        transfer_cell_values(x,np.ones(8),np.array([[.2,.3,.4]]))
    with pytest.raises(ValueError,match='finite'):
        transfer_cell_values(x,np.full(8,np.nan),x[:1])

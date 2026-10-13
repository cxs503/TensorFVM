import numpy as np
import pytest
from tensorfvm.verification.channel_dns import import_profiles


def inputs(tmp_path):
    paths=[]
    for kind in ('mean_prof','mean_stdev'):
        path=tmp_path/(kind+'.dat')
        header=f'% Filename : LM_Channel_1000_{kind}.dat\n% nu = 0.00005\n% delta = 1\n% U_mean = 1\n% u_tau = 0.05\n% Re_tau = 1000\n'
        y=np.array([0.,.1,.5,.99]);a=np.column_stack((y,1000*y,np.ones((4,4))))
        with path.open('w') as f:f.write(header);np.savetxt(f,a)
        paths.append(path)
    return paths


def test_reference_import_preserves_values_and_normalization(tmp_path):
    mean,unc=inputs(tmp_path);out=tmp_path/'reference'
    m=import_profiles(mean,unc,out,case='1000')
    assert m['Cf']==pytest.approx(.005) and m['U_bulk_plus']==20
    np.testing.assert_array_equal(np.loadtxt(out/'mean.csv',delimiter=',',skiprows=1),np.loadtxt(mean,comments='%'))
    assert m['centerline_missing'] and m['source_values_changed'] is False
    with pytest.raises(ValueError,match='empty'):import_profiles(mean,unc,out,case='1000')


@pytest.mark.parametrize('change',[('Re_tau = 1000','Re_tau = 900'),('u_tau = 0.05','u_tau = -0.05'),('U_mean = 1','U_mean = 2'),('LM_Channel_1000','LM_Channel_0550')])
def test_reject_inconsistent_reference_headers(tmp_path,change):
    mean,unc=inputs(tmp_path);unc.write_text(unc.read_text().replace(*change))
    with pytest.raises(ValueError):import_profiles(mean,unc,tmp_path/'reference',case='1000')
    assert not (tmp_path/'reference').exists()

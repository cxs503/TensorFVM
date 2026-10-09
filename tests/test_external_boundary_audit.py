import unittest
import numpy as np
from tensorfvm.external_boundary_audit import scales, mask, field, moving

class ExternalBoundaryAuditTests(unittest.TestCase):
    def test_si_eos(self):
        a=scales(.01,.01,1000,.01,.001,.03,.08)
        self.assertAlmostEqual(a['tau'],.8)
        self.assertAlmostEqual(a['reynolds'],2.4)
        self.assertAlmostEqual(a['sound_speed_m_s'],1/3**.5)
        self.assertAlmostEqual(a['lattice_impulse_Ns'],.001)
        b=scales(.01,.005,1000,.01,.001,.03,.08)
        self.assertAlmostEqual(b['sound_speed_m_s'],2*a['sound_speed_m_s'])
        self.assertAlmostEqual(b['tau'],.65)
    def test_excluded_mask(self):
        solid=mask((8,8),(0,0),1,True)
        self.assertTrue(solid[0,7])
        f=np.ones((9,8,8));f[:,solid]=0
        m,p=field(f,solid)
        self.assertEqual(m,9*(64-solid.sum()))
        np.testing.assert_array_equal(p,[0,0])
        f[0,0,0]=1
        with self.assertRaisesRegex(ValueError,'inside solid'):field(f,solid)
    def test_invalid_units(self):
        with self.assertRaises(ValueError):scales(.01,0,1000,.01,.001,.03,.08)
    def test_independent_conversion_rejection(self):
        cfg=dict(nx=8,ny=8,center=[4,4],radius=1,velocity=[.03,0],fluid_velocity=[0,0],tau=.8)
        solid=mask((8,8),cfg['center'],1,True)
        f=np.zeros((9,8,8));f[0,~solid]=1
        h=dict(removed_momentum=[1,0],added_momentum=[0,0],conversion_impulse_on_solid=[0,0])
        raw=dict(state=dict(config=cfg,center=cfg['center'],f=f.tolist(),history=[h]),report=dict(si=dict(dx_m=.01,dt_s=.01,rho_kg_m3=1000,thickness_m=.01,nu_m2_s=.001,velocity_m_s=.03)))
        with self.assertRaisesRegex(ValueError,'conversion impulse'):moving(raw)

"""Independent invariants for the 3-D body-fitted shared-face kernel."""
import unittest
import torch
from tensorfvm.body_fitted_transient3d import BodyFittedTransient3D,TransientSettings
from tensorfvm.solver3d import Cylinder3DConfig

class Transient3DTests(unittest.TestCase):
    def solver(self):
        return BodyFittedTransient3D(Cylinder3DConfig(nx=32,ny=8,nz=4,mesh_type='body-fitted',time_step=.002),TransientSettings(initial_perturbation=0))
    def test_closed_cell_area_and_internal_flux_cancellation(self):
        s=self.solver()
        self.assertLess(float(s.integrate(s.area).abs().max()),1e-14)
        torch.manual_seed(14);f=torch.randn(len(s.owner),3,dtype=torch.float64)
        self.assertTrue(torch.allclose(s.integrate(f).sum(0),f[s.boundary].sum(0),atol=1e-12,rtol=0))
    def test_projection_closes_actual_nonorthogonal_mesh(self):
        s=self.solver();p,q,iterations,residual=s.project(s.flux)
        self.assertLess(float((s.integrate(q)/s.volumes).abs().max()),1e-8)
        self.assertLess(abs(float(q[s.boundary].sum())),1e-7)
        self.assertGreater(iterations,1)
    def test_pressure_matrix_matches_two_point_face_response(self):
        s=self.solver();torch.manual_seed(7);p=torch.randn(s.N,dtype=torch.float64)
        delta=p[s.safe]-p[s.owner];delta[s.boundary]=0;delta[s.outlet]=-p[s.owner[s.outlet]]
        direct=-s.integrate(s.coeff*delta)
        self.assertTrue(torch.allclose(direct,torch.from_numpy(s.pressure_matrix@p.numpy()),atol=1e-12,rtol=1e-12))
    def test_wale_zero_for_shear_and_positive_for_rotation(self):
        s=self.solver();g=torch.zeros((s.N,3,3),dtype=torch.float64);g[:,0,1]=1
        self.assertEqual(float(s.eddy_viscosity(g).max()),0)
        g[:,1,0]=-1
        self.assertTrue((s.eddy_viscosity(g)>0).all())
    def test_short_physical_steps_close_mass_and_momentum(self):
        s=self.solver()
        for _ in range(3):
            h=s.step();self.assertLess(h['continuity'],1e-8);self.assertLess(h['momentum_ledger'],1e-10);self.assertLess(h['cfl'],1)
        self.assertEqual(len(s.forces),3)
        self.assertTrue(torch.isfinite(s.velocity).all())

if __name__=='__main__':unittest.main()

class RestartTests(unittest.TestCase):
    def test_restart_is_identical_and_preserves_moments(self):
        import tempfile
        from pathlib import Path
        from tensorfvm.cylinder3900_checkpoint import FieldMoments,save_checkpoint,restore_checkpoint
        c=Cylinder3DConfig(nx=32,ny=8,nz=4,mesh_type='body-fitted',time_step=.002)
        a=BodyFittedTransient3D(c);ma=FieldMoments(a)
        for _ in range(2):a.step();ma.add(a)
        with tempfile.TemporaryDirectory()as t:
            path=Path(t)/'checkpoint.npz';save_checkpoint(path,a,ma,{'kernel':'test'})
            b=BodyFittedTransient3D(c);mb=FieldMoments(b);restore_checkpoint(path,b,mb,{'kernel':'test'})
            a.step();ma.add(a);b.step();mb.add(b)
            self.assertTrue(torch.equal(a.velocity,b.velocity));self.assertTrue(torch.equal(a.pressure,b.pressure));self.assertTrue(torch.equal(a.flux,b.flux))
            self.assertTrue(torch.equal(ma.velocity,mb.velocity));self.assertEqual(a.history,b.history)
            with self.assertRaises(ValueError):restore_checkpoint(path,b,mb,{'kernel':'changed'})

class StressAndStatisticsTests(unittest.TestCase):
    def test_affine_velocity_has_exact_interior_deviatoric_stress(self):
        s=BodyFittedTransient3D(Cylinder3DConfig(nx=32,ny=12,nz=4,mesh_type='body-fitted',time_step=.002),TransientSettings(sgs='none'))
        g=torch.tensor([[.2,.7,0],[-.3,-.1,0],[.4,.6,0]],dtype=torch.float64)
        u=s.centers@g.T
        _,visc,_=s.transport(u,torch.zeros_like(s.flux))
        j=s.owner//32%12;jn=s.neighbor.clamp_min(0)//32%12
        face=s.interior&(j>=2)&(j<10)&(jn>=2)&(jn<10)
        stress=s.config.density*s.config.kinematic_viscosity*(g+g.T-2/3*torch.trace(g)*torch.eye(3,dtype=torch.float64))
        expected=s.area[face]@stress.T
        self.assertTrue(torch.allclose(visc[face],expected,atol=2e-14,rtol=2e-12))
    def test_short_record_cannot_report_shedding_frequency(self):
        from tensorfvm.cylinder3900_statistics import statistics
        h=[dict(time=i*.1,Cd=1.,Cl=float(torch.sin(torch.tensor(i*.1))))for i in range(100)]
        st=statistics(h,discard_time=0)
        self.assertFalse(st['eligible']);self.assertIsNone(st['St'])
    def test_long_synthetic_record_recovers_known_spectrum_only_when_eligible(self):
        import numpy as np
        from tensorfvm.cylinder3900_statistics import statistics
        h=[dict(time=float(t),Cd=1.+.01*np.cos(2*np.pi*.2*t),Cl=float(np.sin(2*np.pi*.2*t)))for t in np.arange(0,601,.1)]
        st=statistics(h,discard_time=0)
        self.assertTrue(st['eligible']);self.assertAlmostEqual(st['St'],.2,delta=.003)

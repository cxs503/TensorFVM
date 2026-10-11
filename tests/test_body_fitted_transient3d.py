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

class TransientPressureCouplingTests(unittest.TestCase):
    def test_pressure_neumann_uses_actual_normal_at_skew_slip_faces(self):
        s=BodyFittedTransient3D(Cylinder3DConfig(nx=48,ny=16,nz=4,length=24,height=20,cylinder_x=8,cylinder_y=10,body_fitted_stretching=7,mesh_type='body-fitted',time_step=1e-5))
        # p=x has zero normal derivative on horizontal slip boundaries, even
        # when owner-to-face displacement includes a tangential x component.
        invalid=torch.zeros(s.N,dtype=torch.bool)
        invalid[s.owner[s.boundary&~s.slip]]=True
        selected=s.owner[s.slip];selected=selected[~invalid[selected]]
        self.assertGreater(len(selected),0)
        gradient=s.gradient(s.centers[:,0],pressure=True)[selected]
        exact=torch.tensor([1.,0.,0.],dtype=torch.float64)
        self.assertLess(float((gradient-exact).abs().max()),1e-10)

    def test_affine_pressure_force_on_stretched_skew_mesh(self):
        s=BodyFittedTransient3D(Cylinder3DConfig(nx=48,ny=16,nz=4,length=24,height=20,cylinder_x=8,cylinder_y=10,body_fitted_stretching=7,mesh_type='body-fitted',time_step=1e-5))
        # p is independent of periodic z. Exclude physical boundary stencils;
        # their homogeneous Neumann condition is not that of this manufacture.
        p=.7*s.centers[:,0]-.3*s.centers[:,1]+.2
        gradient=s.integrate(s.pressure_face_values(p)[:,None]*s.area)/s.volumes[:,None]
        radial=s.owner.new_tensor(range(s.N))//48%16
        mask=(radial>=2)&(radial<14)
        exact=torch.tensor([.7,-.3,0],dtype=torch.float64)
        self.assertLess(float((gradient[mask]-exact).abs().max()),2e-9)

    def test_no_momentum_increment_does_not_repeat_startup_pressure_impulse(self):
        # An isolated projection experiment: after the first incompressibility
        # impulse there is no acceleration. Pressure must not keep reapplying
        # that impulse because cell interpolation differs from native flux.
        s=BodyFittedTransient3D(Cylinder3DConfig(nx=32,ny=8,nz=4,mesh_type='body-fitted',time_step=.002),TransientSettings(initial_perturbation=0,sgs='none',pressure_tolerance=1e-11))
        zero=torch.zeros_like(s.area)
        s.transport=lambda u,q:(zero,zero,torch.full((s.N,),s.config.kinematic_viscosity,dtype=torch.float64))
        s.step();impulse=float(s.pressure.abs().max());previous=s.flux.clone()
        s.step()
        self.assertGreater(impulse,1.)
        self.assertLess(float(s.pressure.abs().max()),1e-5)
        self.assertLess(float((s.flux-previous).abs().max()),1e-8)
        self.assertLess(s.history[-1]['momentum_ledger'],1e-10)

    def test_fixed_inlet_is_not_added_again_to_flux_increment(self):
        s=BodyFittedTransient3D(Cylinder3DConfig(nx=32,ny=8,nz=4,mesh_type='body-fitted',time_step=.002))
        trial=s.velocity.clone();trial[:,0]+=.02
        flux=s.momentum_predictor_flux(trial)
        self.assertTrue(torch.equal(flux[s.prescribed],s.flux[s.prescribed]))
        self.assertTrue(torch.equal(flux[s.wall],s.flux[s.wall]))

    def test_diagnostic_detects_near_nyquist_not_only_alternating_mode(self):
        import numpy as np
        from tensorfvm.collocated_diagnostics import angular_pressure
        theta=np.arange(64)*2*np.pi/64
        smooth=angular_pressure(np.sin(theta),64,1,1)
        locked=angular_pressure(np.sin(29*theta),64,1,1)
        self.assertLess(smooth['high_angular_frequency_energy_fraction'],1e-20)
        self.assertGreater(locked['high_angular_frequency_energy_fraction'],.99)

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

class BoundaryAndFailureTests(unittest.TestCase):
    def test_slip_has_zero_normal_mass_and_zero_tangential_stress(self):
        s=BodyFittedTransient3D(Cylinder3DConfig(nx=32,ny=12,nz=4,mesh_type='body-fitted',time_step=.002))
        torch.manual_seed(27);u=torch.randn_like(s.velocity)
        values=s.face_values(u)
        self.assertLess(float((values[s.slip]*s.area[s.slip]).sum(1).abs().max()),1e-14)
        self.assertTrue(torch.equal(values[s.slip,0],u[s.owner[s.slip],0]))
        _,visc,_=s.transport(u,torch.zeros_like(s.flux));n=s.area[s.slip]/s.area_mag[s.slip,None]
        tangent=visc[s.slip]-(visc[s.slip]*n).sum(1)[:,None]*n
        self.assertLess(float(tangent.abs().max()),1e-14)
        h=s.step();self.assertLess(h['continuity'],1e-8);self.assertLess(h['momentum_ledger'],1e-10)
    def test_failed_diffusion_screen_preserves_complete_state(self):
        s=BodyFittedTransient3D(Cylinder3DConfig(nx=32,ny=12,nz=4,mesh_type='body-fitted',time_step=.002),TransientSettings(diffusion_cfl_limit=1e-20))
        before=[s.velocity.clone(),s.pressure.clone(),s.flux.clone()]
        with self.assertRaisesRegex(RuntimeError,'diffusion dt screen'):s.step()
        self.assertTrue(all(torch.equal(a,b)for a,b in zip(before,[s.velocity,s.pressure,s.flux])))
        self.assertEqual(s.time,0);self.assertFalse(s.history);self.assertFalse(s.forces)

class SpectralResolutionTests(unittest.TestCase):
    def test_dt002_long_record_resolves_actual_st0215_without_padding(self):
        import numpy as np
        from tensorfvm.cylinder3900_statistics import statistics
        h=[dict(time=float(t),Cd=1.015,Cl=float(np.sin(2*np.pi*.215*t)))for t in np.arange(0,601,.002)]
        st=statistics(h,discard_time=0)
        self.assertTrue(st['eligible']);self.assertLess(abs(st['St']-.215)/.215,.03)
        self.assertLessEqual(st['frequency_resolution'],.215*.03);self.assertGreater(st['segment_samples'],4096);self.assertFalse(st['zero_padding_used'])
    def test_short_segment_cannot_support_three_percent_st(self):
        import numpy as np
        from tensorfvm.cylinder3900_statistics import statistics
        h=[dict(time=float(t),Cd=1.,Cl=float(np.sin(2*np.pi*.215*t)))for t in np.arange(0,100,.002)]
        st=statistics(h,discard_time=0,minimum_duration=20.)
        self.assertTrue(st['duration_eligible']);self.assertFalse(st['frequency_resolution_gate_passed']);self.assertFalse(st['eligible']);self.assertIsNone(st['St'])
    def test_constant_lift_never_produces_a_shedding_frequency(self):
        from tensorfvm.cylinder3900_statistics import statistics
        h=[dict(time=i*.1,Cd=1.,Cl=.1)for i in range(6001)]
        st=statistics(h,discard_time=0)
        self.assertTrue(st['duration_eligible']);self.assertFalse(st['signal_present']);self.assertFalse(st['eligible']);self.assertIsNone(st['St']);self.assertIsNone(st['diagnostic_spectral_peak'])

class PressureBoundaryCompatibilityTests(unittest.TestCase):
    def solver(self,dt=1e-5,initialize=False):
        return BodyFittedTransient3D(Cylinder3DConfig(nx=48,ny=16,nz=4,length=24,height=20,cylinder_x=8,cylinder_y=10,body_fitted_stretching=7,mesh_type='body-fitted',time_step=dt),TransientSettings(initial_perturbation=0,sgs='none',pressure_tolerance=1e-10,nonorthogonal_iterations=150,initial_projection=initialize))

    def test_nonzero_neumann_affine_pressure_is_exact_at_all_physical_faces(self):
        s=self.solver();g=torch.tensor([.7,-.3,0.],dtype=torch.float64)
        pressure=s.centers@g+.2;face_pressure=s.fc@g+.2
        normal=s.area/s.area_mag[:,None]
        s.set_pressure_boundary_conditions(neumann=normal@g,outlet_values=face_pressure)
        self.assertLess(float((s.gradient(pressure,pressure=True)-g).abs().max()),1e-10)
        self.assertLess(float((s.pressure_face_values(pressure)-face_pressure).abs().max()),1e-10)
        self.assertLess(float((s.pressure_flux_gradient(pressure)-s.area@g).abs().max()),1e-10)
        gauss=s.integrate(s.pressure_face_values(pressure)[:,None]*s.area)/s.volumes[:,None]
        self.assertLess(float((gauss-g).abs().max()),1e-9)
        # Every shared-face pressure traction remains conservative.
        self.assertLess(float((s.integrate(face_pressure[:,None]*s.area).sum(0)-(face_pressure[:,None]*s.area)[s.boundary].sum(0)).abs().max()),1e-10)

    def test_manufactured_nonzero_neumann_projection_has_zero_velocity_impulse_defect(self):
        s=self.solver();g=torch.tensor([.7,-.3,0.],dtype=torch.float64)
        pressure=s.centers@g+.2;face_pressure=s.fc@g+.2
        s.set_pressure_boundary_conditions(neumann=(s.area/s.area_mag[:,None])@g,outlet_values=face_pressure)
        solved,flux,_,residual=s.project(s.area@g,correction_scale=1.)
        self.assertLess(residual,1e-10)
        self.assertLess(float((solved-pressure).abs().max()),1e-8)
        self.assertLess(float(flux.abs().max()),1e-8)
        corrected=g-s.integrate(s.pressure_face_values(solved)[:,None]*s.area)/s.volumes[:,None]
        self.assertLess(float(corrected.abs().max()),1e-8)

    def test_initial_hodge_projection_is_dt_independent_and_conservative(self):
        a=self.solver(initialize=True);b=self.solver(dt=5e-6,initialize=True)
        self.assertTrue(torch.equal(a.velocity,b.velocity));self.assertTrue(torch.equal(a.flux,b.flux))
        self.assertTrue(torch.equal(a.initial_velocity_potential,b.initial_velocity_potential))
        self.assertEqual(a.time,0);self.assertFalse(a.history);self.assertFalse(a.forces)
        self.assertEqual(float(a.pressure.abs().max()),0)
        self.assertEqual(float(a.pressure_neumann.abs().max()),0)
        self.assertLess(a.initialization['momentum_impulse_ledger'],1e-10)
        self.assertLess(a.initialization['native_continuity'],1e-10)
        self.assertLess(float(a.flux[a.wall].abs().max()),1e-12)
        # No nonconservative cell velocity reconstruction or filtering is used:
        # the sole impulse is the divergence of shared-face pressure traction.
        a.set_pressure_boundary_conditions(neumann=(a.area/a.area_mag[:,None])[:,0]*a.config.inlet_velocity*a.wall)
        phi_face=a.pressure_face_values(a.initial_velocity_potential)
        reconstructed=torch.zeros_like(a.velocity);reconstructed[:,0]=a.config.inlet_velocity
        reconstructed-=a.integrate(phi_face[:,None]*a.area)/a.volumes[:,None]
        self.assertLess(float((a.velocity-reconstructed).abs().max()),1e-12)
        a.set_pressure_boundary_conditions()

    def test_invalid_boundary_data_and_pressure_scale_are_rejected(self):
        s=self.solver()
        with self.assertRaisesRegex(ValueError,'one value per face'):
            s.set_pressure_boundary_conditions(neumann=torch.ones(2))
        with self.assertRaisesRegex(ValueError,'nonfinite'):
            s.set_pressure_boundary_conditions(neumann=float('nan'))
        s.set_pressure_boundary_conditions()
        with self.assertRaisesRegex(ValueError,'positive'):
            s.project(s.flux,correction_scale=0.)

class CompatibleInitializationRestartTests(unittest.TestCase):
    def test_hodge_checkpoint_roundtrip_recomputes_evidence_and_preserves_next_step(self):
        import tempfile
        from pathlib import Path
        from tensorfvm.cylinder3900_checkpoint import FieldMoments,save_checkpoint,restore_checkpoint
        c=Cylinder3DConfig(nx=32,ny=8,nz=4,mesh_type='body-fitted',time_step=.002)
        settings=TransientSettings(initial_projection=True)
        a=BodyFittedTransient3D(c,settings);ma=FieldMoments(a)
        for _ in range(2):a.step();ma.add(a)
        with tempfile.TemporaryDirectory()as t:
            path=Path(t)/'checkpoint.npz';producer={'kernel':'conservative-hodge-restart-test'}
            save_checkpoint(path,a,ma,producer)
            b=BodyFittedTransient3D(c,settings);mb=FieldMoments(b)
            # Init evidence is deterministically recomputed by the factory, not
            # stored by the current checkpoint schema. It is not evolving state.
            self.assertTrue(torch.equal(a.initial_velocity_potential,b.initial_velocity_potential))
            self.assertEqual(a.initialization,b.initialization)
            restore_checkpoint(path,b,mb,producer)
            self.assertEqual(float(b.pressure_neumann.abs().max()),0)
            self.assertEqual(float(b.pressure_outlet_values.abs().max()),0)
            a.step();ma.add(a);b.step();mb.add(b)
            for name in ('velocity','pressure','flux'):
                self.assertTrue(torch.equal(getattr(a,name),getattr(b,name)))
            self.assertEqual(a.time,b.time);self.assertEqual(a.history,b.history);self.assertEqual(a.forces,b.forces)
            self.assertEqual(ma.count,mb.count)
            for name in ('velocity','pressure','velocity_product'):
                self.assertTrue(torch.equal(getattr(ma,name),getattr(mb,name)))

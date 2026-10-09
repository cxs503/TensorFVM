import unittest
from tensorfvm.developed_boundary_audit import steady_gates

class DevelopedBoundaryQualificationTests(unittest.TestCase):
    def test_flat_force_still_accelerates(self):
        result=steady_gates(.005,95.5,100.)
        self.assertTrue(result['window_drift_passed'])
        self.assertFalse(result['net_acceleration_passed'])
        self.assertFalse(result['steady_qualified'])
    def test_force_balance_still_drifts(self):
        result=steady_gates(.02,100.,100.)
        self.assertFalse(result['steady_qualified'])
    def test_both_required(self):
        self.assertTrue(steady_gates(.005,99.5,100.)['steady_qualified'])
        self.assertFalse(steady_gates(.01,100.,100.)['steady_qualified'])
    def test_invalid_drive(self):
        with self.assertRaises(ValueError):steady_gates(.005,1.,0.)

class FalseSteadyBadgeTests(unittest.TestCase):
    def test_reconstruct_rejects_flat_window_false_steady(self):
        import numpy as np
        from tensorfvm.external_boundary_audit import mask
        from tensorfvm.developed_boundary_audit import reconstruct
        dx=.125;dt=.1;n=8
        solid=mask((n,n),(4,4),1)
        w=np.array([4/9,*([1/9]*4),*([1/36]*4)])
        initial=np.broadcast_to(w[:,None,None],(9,n,n)).copy();initial[:,solid]=0
        mass=1000*dx**2*.2*int((~solid).sum())
        t=(np.arange(10)+.5)*dt
        drives=mass*.02*np.sin(np.minimum(t/.25,1)*np.pi/2)**2*dt
        forces=.955*drives/dt
        impulse=float(drives.sum()-forces.sum()*dt)
        scale=1000*dx**3*.2/dt
        final=initial.copy();delta=impulse/scale/(2*(~solid).sum())
        final[1,~solid]+=delta;final[3,~solid]-=delta
        raw=dict(dt_s=dt,dx_m=dx,n=n,steady_window_s=.3,duration_s=1.,force_history_N=forces.tolist(),
                 center_m=[.5,.5],radius_m=.125,initial_population=initial.tolist(),final_population=final.tolist(),
                 rho_kg_m3=1000,thickness_m=.2,nu_m2_s=.02,acceleration_m_s2=.02,ramp_s=.25,
                 solid_impulse_Ns=[float(forces.sum()*dt),0],drive_impulse_Ns=[float(drives.sum()),0],
                 late_window_drift=0.,window_drift_passed=True,net_acceleration_passed=False,steady_qualified=True)
        with self.assertRaisesRegex(ValueError,'incorrect qualification steady_qualified'):reconstruct(raw)

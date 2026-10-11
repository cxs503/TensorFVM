"""Independent conservative reconstruction and current equation checks."""
import tempfile
import unittest
from pathlib import Path
import torch
from tests.test_three_element import _TINY_GMSH
from tensorfvm.solver import SolverConfig
from tensorfvm.multi_element_flow import MultiElementFlowSolver
from tensorfvm.hlpw30p30n import reference_zones,write_mesh_geo


def _two_row_mesh():
    nodes=[(x+.12*y*(2-y),y) for y in range(3) for x in range(4)]
    edges=[(4+i,i+1,i+2) for i in range(3)]
    edges += [(3,9+i,10+i) for i in range(3)]
    edges += [(1,1,5),(1,5,9),(2,4,8),(2,8,12)]
    elements=[f'{j+1} 1 2 {tag} 1 {a} {b}' for j,(tag,a,b) in enumerate(edges)]
    for y in range(2):
        for x in range(3):
            a=y*4+x+1
            elements.append(f'{len(elements)+1} 3 2 7 1 {a} {a+1} {a+5} {a+4}')
    header=_TINY_GMSH.split('$Nodes')[0]
    return header+'$Nodes\n12\n'+'\n'.join(f'{i+1} {x} {y} 0' for i,(x,y) in enumerate(nodes))+'\n$EndNodes\n$Elements\n16\n'+'\n'.join(elements)+'\n$EndElements\n'


class MultiElementFlowTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        mesh=Path(self.temp.name)/'mesh.msh';mesh.write_text(_TINY_GMSH)
        self.solver=MultiElementFlowSolver(SolverConfig(nx=1,ny=1,length=3,height=1,
            cylinder_radius=None,mesh_type='three-element',mesh_file=str(mesh),
            reynolds=100,turbulence_model='spalart-allmaras',max_iterations=1))

    def test_limited_flux_is_bounded_and_conservative(self):
        s=self.solver;u=torch.tensor([[.2,.1],[.8,.3],[.4,-.2]],dtype=torch.float64)
        gradient=torch.full((3,2,2),20.,dtype=u.dtype)
        flux=torch.ones_like(s.mass_flux)
        correction=s.linear_upwind_correction(u,flux,gradient)
        up,_=s._upwind_geometry(flux);face=u[up]+correction[s.f]
        self.assertTrue(bool((face>=torch.tensor([0.,-.2])-1e-14).all()))
        self.assertTrue(bool((face<=torch.tensor([1.,.3])+1e-14).all()))
        torch.testing.assert_close(s._sum(flux[:,None]*correction).sum(0),torch.zeros(2,dtype=u.dtype),atol=1e-14,rtol=0)
        self.assertTrue(bool((correction[s.mesh.boundary]==0).all()))

    def test_taylor_reconstruction_affine_exact_when_unlimited(self):
        s=self.solver;s.bounded=False
        gradient=torch.tensor([[2.,-1.],[3.,4.]],dtype=torch.float64).expand(3,2,2)
        centers=s.mesh.centers.reshape(-1,2);u=centers@gradient[0]
        flux=torch.ones_like(s.mass_flux);up,_=s._upwind_geometry(flux)
        face=u[up]+s.linear_upwind_correction(u,flux,gradient)[s.f]
        torch.testing.assert_close(face,s.mesh.face_centers[s.f]@gradient[0])

    def test_pressure_operator_and_sa_preconditioner_have_true_checks(self):
        s=self.solver;s.step()
        pressure=[r for r in s.linear_history if r['kind']=='pressure']
        self.assertTrue(pressure)
        self.assertLess(pressure[-1]['assembled_operator_relative_defect'],2e-12)
        self.assertTrue(any(r['kind']=='spalart-allmaras' for r in s.linear_history))
        self.assertIsNot(s._lu_cache['momentum'],s._sa_lu_cache['momentum'])
        self.assertTrue(all(r['true_residual']<=1.05*r['target'] for r in s.linear_history))

    def test_sa_uses_exactly_one_implicit_relaxation(self):
        from dataclasses import replace
        import numpy as np
        s=MultiElementFlowSolver(replace(self.solver.config,turbulence_relaxation=.4))
        s.sa_inner_iterations=1;old=s.nu_tilde.clone()
        d,a,b,rhs=s._sa_system(old);relaxed=d/.4
        expected=np.linalg.solve(s._matrix(relaxed,a,b).toarray(),(rhs+.6*relaxed*old).numpy())
        residual,_=s._turbulence_step()
        torch.testing.assert_close(s.nu_tilde,torch.from_numpy(expected).clamp_min(0),atol=2e-13,rtol=2e-12)
        d,a,b,rhs=s._sa_system(s.nu_tilde)
        actual=float((s._matvec(s.nu_tilde,d,a,b)-rhs).abs().max()/(s.config.density*s.config.inlet_velocity*s.config.reference_length*s.kinematic_viscosity))
        self.assertAlmostEqual(residual,actual,places=13)

    def test_stored_metrics_use_current_fields_and_local_flux_defect(self):
        s=self.solver;s.anderson_depth=3
        metrics=s.step()
        self.assertAlmostEqual(metrics['momentum'],max(metrics['u_residual'],metrics['v_residual']),places=13)
        d,_,_,_=s._momentum(s.velocity,s.p,s.mass_flux)
        physical,_=s._rhie_chow(s.velocity,s.p,s.volume/d)
        expected=float(((physical-s.mass_flux).abs()/(s.config.density*s.config.inlet_velocity*s.mesh.face_lengths)).max())
        self.assertAlmostEqual(metrics['rhie_chow_max_normal_velocity_defect'],expected,places=13)

    def test_coupled_frozen_reconstruction_operator(self):
        from tensorfvm.multi_element_coupled import CoupledMultiElementSolver
        s=CoupledMultiElementSolver(self.solver.config)
        matrix,defect=s.coupled_matrix()
        self.assertEqual(matrix.shape,(9,9))
        self.assertLess(defect,2e-11)

    def test_variable_viscosity_full_stress_response(self):
        s=self.solver
        s.turbulent_kinematic_viscosity=torch.tensor([.03,.07,.02],dtype=s.p.dtype)
        u=torch.tensor([[.2,.1],[.8,.3],[.4,-.2]],dtype=s.p.dtype)
        probe=torch.tensor([[.31,-.2],[-.17,.45],[.73,-.11]],dtype=s.p.dtype)
        blocks=s.stress_correction_matrices()
        expected=s.stress_correction(u+probe)-s.stress_correction(u)
        import numpy as np
        actual=np.stack([sum(blocks[j][k]@probe[:,k].numpy() for k in (0,1)) for j in (0,1)],axis=1)
        torch.testing.assert_close(torch.from_numpy(actual),expected,atol=1e-13,rtol=1e-12)

    def test_skew_pressure_face_force_and_assembled_gradient(self):
        from dataclasses import replace
        from tensorfvm.multi_element_skew_pressure import SkewPressureMultiElementSolver
        mesh=Path(self.temp.name)/'two-row.msh';mesh.write_text(_two_row_mesh())
        s=SkewPressureMultiElementSolver(replace(self.solver.config,mesh_file=str(mesh),height=2))
        field=torch.sin(torch.arange(s.count,dtype=s.p.dtype)*.37)
        H,I,G=s._geometry()
        gradient=s._gradient(field,pressure=True)
        import numpy as np
        actual=np.stack([g@field.numpy() for g in G],axis=1)
        torch.testing.assert_close(torch.from_numpy(actual),gradient,atol=2e-14,rtol=1e-13)
        face=s.reconstructed_pressure_faces(field)
        torch.testing.assert_close((gradient*s.volume[:,None]).sum(0),(face[s.mesh.boundary,None]*s.S[s.mesh.boundary]).sum(0),atol=2e-14,rtol=1e-13)
        s.step()
        pressure=[r for r in s.linear_history if r['kind']=='pressure']
        self.assertLess(pressure[-1]['assembled_operator_relative_defect'],2e-12)

    def test_full_momentum_response_uses_nonconstant_reconstruction(self):
        s=self.solver
        s.velocity.copy_(torch.tensor([[.2,.1],[.8,.3],[.4,-.2]],dtype=torch.float64))
        matrices,defect=s.momentum_correction_matrices(s.velocity,s.mass_flux)
        self.assertEqual(len(matrices),2);self.assertLess(defect,2e-11)
        s.step()
        momentum=[r for r in s.linear_history if r['kind']=='momentum']
        self.assertEqual(len(momentum),2)
        self.assertTrue(all(r['momentum_assembled_operator_relative_defect']<2e-11 for r in momentum))

    def test_exact_sa_newton_jacobian(self):
        from tensorfvm.sa_newton import NewtonSaMixin
        class Solver(NewtonSaMixin,MultiElementFlowSolver):pass
        s=Solver(self.solver.config)
        matrix,defect=s.sa_jacobian(s.nu_tilde)
        self.assertEqual(matrix.shape,(3,3));self.assertLess(defect,2e-10)
        for values in [[0.,0.,0.],[100.,20.,0.]]:
            _,boundary_defect=s.sa_jacobian(torch.tensor(values,dtype=torch.float64)*s.kinematic_viscosity)
            self.assertLess(boundary_defect,2e-10)
        s._turbulence_step()
        self.assertLessEqual(s.sa_newton_history[-1]['nonlinear_l2_after'],s.sa_newton_history[-1]['nonlinear_l2_before'])

    def test_primary_reference_requires_supported_conditions(self):
        zones=reference_zones(8.1);self.assertEqual(set(zones),{'slat','main','flap'})
        self.assertGreater(len(zones['main']),40)
        with self.assertRaises(ValueError):reference_zones(0.)
        geo=write_mesh_geo(Path(self.temp.name)/'official.geo').read_text()
        self.assertEqual(geo.count('Circle('),4)
        self.assertIn('500',geo)
        self.assertIn('Physical Curve("far-field")',geo)


if __name__=='__main__':unittest.main()

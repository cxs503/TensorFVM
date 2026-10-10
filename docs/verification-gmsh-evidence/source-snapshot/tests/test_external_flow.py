"""Check optional direct solves against the production Krylov equations."""
import unittest
import torch
from tensorfvm.external_linear import ExternalFlowSolver
from tensorfvm.external_cached import CachedExternalFlowSolver
from tensorfvm.external_anderson import AndersonExternalFlowSolver
from tensorfvm.external_coupled import CoupledExternalFlowSolver
from tensorfvm.sparse_simple import SparseBodyFittedSolver
from tensorfvm.verification.external_flow import configuration


class ExternalFlowTests(unittest.TestCase):
    def test_direct_and_krylov_advance_same_fields(self):
        torch.set_num_threads(1)
        for case in ('cylinder','naca'):
            with self.subTest(case=case):
                cfg=configuration(case,32,12,'cpu',5)
                direct=ExternalFlowSolver(cfg);krylov=SparseBodyFittedSolver(cfg);cached=CachedExternalFlowSolver(cfg)
                for _ in range(5):direct.step();krylov.step();cached.step()
                torch.testing.assert_close(direct.velocity,krylov.velocity,rtol=1e-7,atol=1e-8)
                torch.testing.assert_close(direct.p,krylov.p,rtol=1e-7,atol=1e-8)
                self.assertTrue(all(q['true_residual']<=q['target']*1.05 for q in direct.linear_history+cached.linear_history))
                torch.testing.assert_close(direct.velocity,cached.velocity,rtol=1e-7,atol=1e-8)
                torch.testing.assert_close(direct.p,cached.p,rtol=1e-7,atol=1e-8)

    def test_coupled_correction_preserves_steady_cylinder(self):
        torch.set_num_threads(1)
        cfg=configuration('cylinder',32,12,'cpu',1600)
        ordinary=ExternalFlowSolver(cfg);coupled=CoupledExternalFlowSolver(cfg)
        ordinary.solve();coupled.solve()
        self.assertTrue(ordinary.converged and coupled.converged)
        self.assertLess(coupled.history[-1]['rhie_chow_flux_defect'],1e-7)
        torch.testing.assert_close(coupled.velocity,ordinary.velocity,rtol=2e-4,atol=1e-6)
        torch.testing.assert_close(coupled.p,ordinary.p,rtol=2e-4,atol=1e-6)
        self.assertTrue(all(q['assembled_operator_relative_defect']<2e-11 for q in coupled.coupled_history))

    def test_airfoil_coupled_backend_is_rejected_until_verified(self):
        with self.assertRaises(ValueError):
            CoupledExternalFlowSolver(configuration('naca',32,12,'cpu',26))

    def test_anderson_final_state_matches_steady_airfoil_equations(self):
        torch.set_num_threads(1)
        cfg=configuration('naca',32,12,'cpu',1600)
        ordinary=CachedExternalFlowSolver(cfg);accelerated=AndersonExternalFlowSolver(cfg)
        ordinary.solve();accelerated.solve()
        self.assertTrue(ordinary.converged and accelerated.converged)
        self.assertLess(accelerated.history[-1]['rhie_chow_flux_defect'],1e-7)
        torch.testing.assert_close(accelerated.velocity,ordinary.velocity,rtol=2e-4,atol=1e-5)
        torch.testing.assert_close(accelerated.p,ordinary.p,rtol=2e-4,atol=1e-5)

if __name__=='__main__':unittest.main()

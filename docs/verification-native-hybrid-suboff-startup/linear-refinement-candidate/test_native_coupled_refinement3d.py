import numpy as np
from tensorfvm.native_hybrid_stokes3d import NativeHybridStokes3D
from tensorfvm.verification.native_hybrid_flow import channel_mesh


def test_small_global_residual_does_not_hide_continuity_defect(monkeypatch):
    import tensorfvm.native_hybrid_stokes3d as module
    m=channel_mesh(2,triangulate_faces=False);g=np.array([.7,-.3,.2])
    s=NativeHybridStokes3D(m,m.neighbor<0,body_force=g,gauge_value=.4+m.cell_centers[0]@g)
    face=np.flatnonzero(m.neighbor>=0)[0];axis=np.argmax(abs(m.area_vectors[face]))
    dof=3*s.N+3*face+axis;position=np.flatnonzero(s.viscous.free==dof)[0]
    real=module.splu
    def approximate_initial_factor(matrix):
        factor=real(matrix)
        class Factor:
            calls=0
            def solve(self,rhs):
                answer=factor.solve(rhs);self.calls+=1
                if self.calls==1:answer[position]+=1e-10
                return answer
        return Factor()
    monkeypatch.setattr(module,'splu',approximate_initial_factor)
    result=s.solve(algorithm='coupled',tolerance=1e-11)
    assert result['numerically_converged']
    record=s.coupled_linear_history[-1]['residual_refinement']
    assert record and record[0]['before']>1e-11 and record[-1]['after']<1e-11
    np.testing.assert_allclose(result['velocity'],0.,atol=1e-12)
    np.testing.assert_allclose(result['pressure'],.4+m.cell_centers@g,atol=1e-12)

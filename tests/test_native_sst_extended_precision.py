from decimal import Decimal, localcontext
import numpy as np
import pytest
from tensorfvm.native_sst3d import sst2003_closure,BETA2
from tensorfvm.native_hybrid_scalar3d import NativeHybridScalar3D
from tensorfvm.native_hybrid_sst3d import NativeHybridSSTTransport3D
from tensorfvm.native_hybrid_rans3d import NativeHybridRans3D
from tensorfvm.verification.native_hybrid_pressure import pressure_mesh
from tests.test_native_hybrid_rans3d import setup


pytestmark=pytest.mark.skipif(np.finfo(np.longdouble).nmant<=np.finfo(float).nmant,
                             reason='platform has no extra longdouble precision')


def test_large_omega_destruction_matches_independent_decimal_evaluation():
    omega=np.longdouble('10000000.0000000005')
    args=(0.,omega,.1,.1,np.zeros((3,3)),np.zeros(3),np.zeros(3))
    extended=sst2003_closure(*args,precision='longdouble')
    ordinary=sst2003_closure(*args)
    with localcontext() as ctx:
        ctx.prec=60
        expected=-Decimal.from_float(BETA2)*Decimal(str(omega))**2
        error_extended=abs(Decimal(str(extended['omega_source']))-expected)
        error_float=abs(Decimal.from_float(float(ordinary['omega_source']))-expected)
    assert error_extended<Decimal('0.000005')
    assert error_float>Decimal('0.0001')
    assert extended['omega_source'].dtype==np.dtype(np.longdouble)
    assert ordinary['omega_source'].dtype==np.dtype(float)
    assert extended['nu_t']==0 and extended['F1']==0


def test_extended_scalar_source_sink_storage_survive_float64_lu(monkeypatch):
    import tensorfvm.native_scalar_mmp_newton3d as newton
    mesh=pressure_mesh(1)
    op=NativeHybridScalar3D(mesh,correction='nonlinear_mmp',data_precision='longdouble')
    old=np.longdouble('1.00000000000000005')
    source=np.longdouble('0.30000000000000004')
    sink=np.longdouble('0.30000000000000003')
    seen=[];factor=newton.splu
    def checked(matrix,*args,**kwargs):
        seen.append(matrix.dtype)
        assert matrix.dtype==np.dtype(float)
        return factor(matrix,*args,**kwargs)
    monkeypatch.setattr(newton,'splu',checked)
    r=op.solve(source=source,sink=sink,old_values=old,dt=.2,boundary_values=1.)
    assert seen and op.ledger_acceptable(r)
    np.testing.assert_array_equal(r['production'],mesh.cell_volumes.astype(np.longdouble)*source)
    np.testing.assert_array_equal(r['destruction'],mesh.cell_volumes.astype(np.longdouble)*sink*r['cell_values'])
    np.testing.assert_array_equal(r['storage'],mesh.cell_volumes*5*(r['cell_values']-old))
    assert np.any(r['production']!=mesh.cell_volumes*float(source))
    audit=op.steady_residual(r['cell_values'],r['face_values'],source=source,sink=sink,boundary_values=1.)
    assert audit['cell_equation'].dtype==np.dtype(np.longdouble)


def test_extended_sst_step_and_restart_use_identical_closure_and_viscosity():
    baseline=setup(shear=.3);flow=baseline.flow
    s=NativeHybridRans3D(flow,.1,.1,k_boundary=.01,omega_boundary=2.,k_inflow=.01,omega_inflow=2.,
        scalar_correction='nonlinear_mmp',scalar_precision='longdouble',scalar_warm_start=True)
    s.initialize(.01,2.,face_initialization='cell_average')
    r=s.solve(max_iterations=1,dt=.1,tolerance=1e-8)
    assert s.closure['omega_source'].dtype==np.dtype(np.longdouble)
    assert r['current_rans_residual']['transport']['omega']['cell_equation'].dtype==np.dtype(np.longdouble)
    # Viscous matrices are explicitly rounded once to their supported dtype;
    # an unchanged extended closure must not invalidate flow factor caches.
    viscous=s.flow.viscous;s._refresh_viscosity();assert s.flow.viscous is viscous
    closure=s.closure['omega_source'].copy();k=s.k.copy()
    s.restore_accepted_fields(r)
    np.testing.assert_array_equal(s.closure['omega_source'],closure)
    np.testing.assert_array_equal(s.k,k)
    assert not s.converged


def test_extended_precision_configuration_cannot_silently_fall_back():
    mesh=pressure_mesh(1)
    for precision in ('bad','longdouble'):
        with pytest.raises(ValueError):NativeHybridScalar3D(mesh,data_precision=precision)
        with pytest.raises(ValueError):NativeHybridSSTTransport3D(mesh,scalar_precision=precision)


def test_strict_steady_ledger_gate_is_required_and_rechecked(monkeypatch):
    shear=.3;omega=np.sqrt(.44/.0828)*shear
    s=setup(shear=shear,k=0.,omega=omega)
    original=s._steady_ledger_acceptance
    def failed(current,flow):
        result=original(current,flow)
        assert result['passed']  # This actual Couette equilibrium passes.
        result['passed']=False
        return result
    monkeypatch.setattr(s,'_steady_ledger_acceptance',failed)
    r=s.solve(max_iterations=1,dt=.1,relaxation=1.,tolerance=1e-8,require_steady_ledgers=True)
    assert not r['numerically_converged']
    assert not r['rans_history'][-1]['steady_ledger_acceptance']['passed']
    monkeypatch.setattr(s,'_steady_ledger_acceptance',original)
    r=s.solve(max_iterations=2,dt=.1,relaxation=1.,tolerance=1e-8,require_steady_ledgers=True)
    assert r['numerically_converged'] and r['steady_ledger_acceptance']['passed']
    # Even a previously accepted state must be re-audited on result retrieval.
    monkeypatch.setattr(s,'_steady_ledger_acceptance',failed)
    assert not s.result()['numerically_converged']

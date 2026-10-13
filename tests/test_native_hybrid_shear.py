import numpy as np
import pytest
from tensorfvm.verification.native_hybrid_shear import shear_solver,shear_fields,shear_metrics


def test_continuous_variable_viscosity_solution_has_constant_shear():
    points=np.array([[.3,y,.5] for y in (.07,.21,.38,.62,.78,.93)])
    h=1e-6
    for kind in ('smooth','jump'):
        exact=shear_fields(points,kind)
        plus=shear_fields(points+[0,h,0],kind)['velocity'][:,0]
        minus=shear_fields(points-[0,h,0],kind)['velocity'][:,0]
        np.testing.assert_allclose(exact['viscosity']*(plus-minus)/(2*h),exact['shear_stress'],rtol=1e-9)


@pytest.mark.parametrize('algorithm',['coupled','simple','simplec'])
def test_viscosity_jump_conserves_exact_interface_and_wall_stress(algorithm):
    m,s,e=shear_solver(4,'jump')
    r=s.solve(algorithm=algorithm,tolerance=1e-10,max_iterations=6000)
    metric=shear_metrics(m,r,e)
    assert metric['numerically_converged']
    for key in ('velocity_relative_l2','lower_force_relative_error','upper_force_relative_error',
                'interface_traction_defect_max','global_force_balance_error'):
        assert metric[key]<1e-7,(key,metric)
    assert metric['pressure_maximum_absolute_error']<1e-7


def test_smooth_viscosity_refines_velocity_and_both_wall_forces():
    rows=[]
    for n in (4,8,16):
        m,s,e=shear_solver(n)
        r=s.solve(algorithm='coupled',tolerance=1e-10)
        rows.append(shear_metrics(m,r,e))
        assert rows[-1]['numerically_converged']
        assert rows[-1]['energy_balance_error']<1e-9
        assert rows[-1]['global_force_balance_error']<1e-9
    for key in ('velocity_relative_l2','lower_force_relative_error','upper_force_relative_error'):
        assert np.all(np.diff([r[key] for r in rows])<0),rows
        assert rows[-1][key]<.003,rows

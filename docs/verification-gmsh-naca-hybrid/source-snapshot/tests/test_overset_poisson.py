"""Actual simultaneous donor-constrained FV diffusion, not analytic field fill."""
import numpy as np
import pytest
from tensorfvm.overset import cartesian_grid,annular_grid,circle_polygon,build_overset,ACTIVE,FRINGE
from tensorfvm.overset_poisson import solve_overset_poisson


def make_connectivity(n):
    return build_overset(cartesian_grid((-2,2,-2,2),n,n),
        annular_grid((0,0),.3,1.,2*n,n//4),
        circle_polygon((0,0),.3),circle_polygon((0,0),.65))


def test_actual_coupled_diffusion_refinement_and_global_balance():
    errors=[];defects=[]
    for n in (32,64):
        result=solve_overset_poisson(make_connectivity(n));m=result.metrics
        errors.append(max(m['grid_0_relative_l2_error'],m['grid_1_relative_l2_error']))
        defects.append(m['physical_global_conservation_defect_relative'])
        assert m['relative_algebraic_residual']<1e-11
        assert m['maximum_donor_constraint_residual']<1e-11
        assert abs(m['summed_fv_balance_defect'])<1e-9
        assert not m['strict_local_conservation']
        assert not m['navier_stokes_verified']
        assert m['fringe_cells']>0
        # Defect uses the unique physical domain, not double-counted overlap.
        assert np.isclose(m['physical_global_conservation_defect'],
            m['physical_boundary_diffusive_flux']-m['physical_domain_integrated_source'])
        for values,exact,state in zip(result.fields,result.exact_fields,result.connectivity.states):
            assert np.max(abs(values[state==ACTIVE]-exact[state==ACTIVE]))>1e-5
    assert max(errors)<.03
    assert errors[1]<errors[0]/3
    assert defects[1]<defects[0]


def test_missing_receiver_constraints_rejected():
    from dataclasses import replace
    conn=make_connectivity(24)
    broken=replace(conn,stencils=())
    with pytest.raises(ValueError,match='missing donor'):
        solve_overset_poisson(broken)


def test_changed_manufactured_radius_rejected():
    with pytest.raises(ValueError,match='radius does not match'):
        solve_overset_poisson(make_connectivity(24),radius=.31)

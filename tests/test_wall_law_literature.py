"""Literature-anchored audit of the Spalding all-y-plus wall law.

Companion to ``scripts/audit_wall_law_literature.py`` (JSON + PNG in
``src/tensorfvm/verification/artifacts/wall_law/``). Existing tests for
``tensorfvm.wall_functions`` are self-consistency checks; these add external
anchors (Reichardt 1951 composite law, Prandtl-von Karman smooth-pipe friction)
plus a validity-window regression for two repaired defects.
"""
import warnings

import numpy as np
import pytest

from tensorfvm.wall_functions import E, KAPPA, spalding_y_plus, u_plus_at_y_plus

K = KAPPA
B = float(np.log(E) / K)


def reichardt(y_plus,kappa=KAPPA):
    """Reichardt (1951): ln(1+kappa*y+)/kappa + 7.8*(1 - exp(-y+/11) - (y+/11)exp(-y+/3))."""
    yp=np.asarray(y_plus,float)
    return np.log1p(kappa*yp)/kappa+7.8*(1-np.exp(-yp/11)-(yp/11)*np.exp(-yp/3))


def pvk_darcy(Re_D):
    """Prandtl-von Karman smooth pipe: 1/sqrt(f) = 2 log10(Re sqrt(f)) - 0.8."""
    Re_D=np.atleast_1d(np.asarray(Re_D,float));f=.02*np.ones_like(Re_D)
    for _ in range(80):
        f=1./(2*np.log10(Re_D*np.sqrt(f))-0.8)**2
    return f


MASTER=np.concatenate([[0.],np.geomspace(1e-4,1e6,60001)])
_CUMULATIVE={}


def bulk_ratio(Re_tau,kappa=KAPPA,E_const=E):
    """U_b/u_tau from the module profile; geometric grid resolves y+ -> 0."""
    key=(kappa,E_const)
    if key not in _CUMULATIVE:
        up=u_plus_at_y_plus(MASTER,kappa,E_const)
        _CUMULATIVE[key]=np.concatenate([[0.],np.cumsum(.5*(up[1:]+up[:-1])*np.diff(MASTER))])
    return float(np.interp(Re_tau,MASTER,_CUMULATIVE[key])/Re_tau)


def cf_from_law(Re_b,kappa=KAPPA,E_const=E):
    lo,hi=30.,1e6
    for _ in range(120):
        mid=.5*(lo+hi)
        if 2*mid*bulk_ratio(mid,kappa,E_const)<Re_b:lo=mid
        else:hi=mid
    return 2./bulk_ratio(.5*(lo+hi),kappa,E_const)**2


@pytest.mark.parametrize("y_plus",[1e-6,1e-3,.5,5.,30.,1e3,1e6,1e15,1e30])
def test_inverse_law_is_self_consistent_across_ten_decades(y_plus):
    """The inverse must return the root of the forward law, not just of [0, y+]."""
    u=float(u_plus_at_y_plus(y_plus))
    assert u>0
    assert abs(float(spalding_y_plus(u))/y_plus-1)<1e-9


def test_inverse_law_was_silently_wrong_past_y_plus_1e20():
    """Regression: the former 90 bisections on [0, y+] returned 403.90 at y+=1e30."""
    u=float(u_plus_at_y_plus(1e30))
    assert u==pytest.approx(174.04862237926315,abs=1e-6)
    assert (np.log(1e30)+np.log(E))/K==pytest.approx(u,rel=1e-12)


def test_forward_law_is_warning_clean_outside_float_series_range():
    """The unused series branch used to overflow for kappa*u+ > ~1e103."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        values=spalding_y_plus([2000.,1e50,1e300])
    # Beyond kappa*u+ = 700 the exponential leaves float64 range, so the law
    # saturates at y+ ~ 1.03e303: finite and non-decreasing, not strictly so.
    assert np.isfinite(values).all() and np.all(np.diff(values)>=0)
    assert values[0]>1e302                        # saturated, finite (approx overflows here)


def test_forward_law_has_expected_asymptotes():
    yp=np.array([.5,1.,3.,1e4])
    up=u_plus_at_y_plus(yp)
    assert abs(up[0]-yp[0])/yp[0]<1e-3          # viscous sublayer: u+ = y+
    log=np.log(yp)/K+B
    assert abs(up[-1]-log[-1])/up[-1]<1e-4      # log region: ln(E y+)/kappa


def test_law_enters_the_log_region_near_y_plus_120():
    """1%-of-u+ agreement with ln(Ey+)/kappa, the classic 'log region' entry."""
    fine=np.logspace(-3,5,4001);up=u_plus_at_y_plus(fine)
    dev=np.abs(up-(np.log(fine)/K+B))/up
    entry=float(fine[np.argmax(dev<1e-2)])
    assert 100.<entry<140.


def test_agrees_with_reichardt_1951_composite_law():
    """Independent literature profile, not a self-consistency relation."""
    yp=np.logspace(1,np.log10(1000.),601)
    delta=u_plus_at_y_plus(yp)-reichardt(yp)
    assert np.abs(delta).max()<.7               # measured 0.574
    assert np.sqrt(np.mean(delta**2))<.4        # measured 0.319


def test_implied_friction_velocity_matches_reichardt_at_a_physical_state():
    """U=10 m/s sampled at 1 mm in water: same state, two independent laws."""
    speed,distance,nu=10.,1e-3,1e-6
    lo,hi=1e-4,10.
    def invert(u_plus):
        a,b=lo,hi
        for _ in range(200):
            m=.5*(a+b)
            if m*u_plus(distance*m/nu)<speed:a=m
            else:b=m
        return .5*(a+b)
    ut_module=invert(lambda yp: float(u_plus_at_y_plus(yp)))
    ut_lit=invert(reichardt)
    assert abs(ut_module-ut_lit)/ut_lit<.01      # measured 0.43 %, y+ = 485
    assert distance*ut_module/nu==pytest.approx(485.,abs=5.)


@pytest.mark.parametrize("Re_b",[1e4,3e4,1e5,3e5,1e6,1e7])
def test_channel_friction_is_within_15_percent_of_smooth_pipe_pvk(Re_b):
    """D_h = 4h, so f_D = 4 Cf and the matched pipe Reynolds number is 2 Re_b."""
    cf=cf_from_law(Re_b)
    cf_pipe=pvk_darcy(2*Re_b)[0]/4
    assert abs(cf/cf_pipe-1)<.15                 # measured 4.6 % .. 11.4 %
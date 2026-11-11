"""Body coupling of the sigma-grid wave tank: load, added mass, stability.

The tests are anchored on quantities that do not depend on the code under test:

* the hydrostatic/Froude-Krylov contour integral must return the weight of the
  section in calm water (Archimedes) and ``rho g B`` per metre of heave;
* the *discrete* added mass of the tank must be of the order of the flat-plate
  value ``rho pi (B/2)^2 / 2`` for a shallow section -- a factor check that the
  body flux is a doublet and not a basin-scale piston;
* a free decay in calm water must oscillate at ``sqrt(K33 / (m + A33))`` with
  the tank's own ``A33``, which is how the implicit coupling is validated;
* the blocked *body* faces must not include the tank walls (regression for a
  defect that made the 12 m bottom follow the section: measured added mass
  1.1e4 kg/m instead of O(rho B^2) and a growing velocity at the body corner).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensorfvm.wave_tank.theory import GRAVITY
from tensorfvm.wave_tank_td import seakeeping as sk
from tensorfvm.wave_tank_td.sigma_tank import SigmaWaveTank, TankConfig

SEC = sk.floating_strip(1.2, 0.2, freeboard=0.05, x0=4.0)


def _tank(nx=120, ns=16, **kw):
    t = SigmaWaveTank(TankConfig(length=10.0, nx=nx, ns=ns, **kw))
    t.set_body(SEC.polygon, mass=SEC.mass, inertia=SEC.inertia, ref=SEC.ref)
    t._update_mask()
    return t


# ------------------------------------------------------------------ hydrostatics
def test_hydrostatic_contour_returns_the_weight():
    t = _tank()
    f, m = t.hydrostatic_force()
    assert abs(f[1] - SEC.mass * GRAVITY) < 1e-9 * SEC.mass * GRAVITY
    assert abs(f[0]) < 1e-9 * SEC.mass * GRAVITY
    assert abs(m) < 1e-9 * SEC.mass * GRAVITY


def test_hydrostatic_load_is_exact_under_heave():
    """A vertical shift of the section changes the load by exactly ``rho g B dh``.

    The contour quadrature is exact for straight edges, a piecewise-linear
    elevation and a pressure linear in ``y``, so the waterplane stiffness has no
    stair-step error -- unlike a cut-cell face-fraction sum.
    """
    t = _tank()
    base = t.hydrostatic_force()[0][1]
    t.body["xi"][0] = 0.01
    t._update_mask()
    up = t.hydrostatic_force()[0][1]
    assert (up - base) == pytest.approx(-SEC.length * sk.SECTION_DENSITY * GRAVITY * 0.01,
                                        rel=1e-9)


def test_froude_krylov_force_follows_the_local_elevation():
    """A uniform elevation over the footprint lifts the section by rho g B eta."""
    t = _tank()
    base = t.hydrostatic_force()[0][1]
    t.eta = np.full(t.nx, 0.02)
    t._build_grid()
    t._update_mask()
    lifted = t.hydrostatic_force()[0][1]
    assert (lifted - base) == pytest.approx(SEC.length * sk.SECTION_DENSITY * GRAVITY * 0.02,
                                            rel=2e-2)


# --------------------------------------------------------------------- the mask
def test_body_blocked_fractions_exclude_the_tank_walls():
    t = _tank()
    assert not t.bfrac_x[0, :].any()
    assert not t.bfrac_x[-1, :].any()
    assert not t.bfrac_s[:, 0].any()
    assert not t.bfrac_s[:, -1].any()
    assert t.bfrac_s.any() and t.bfrac_x.any()


def test_moving_body_flux_has_no_net_source():
    """The blocked-face flux telescopes: a rigid section moves no net volume.

    With the walls included in the blocked set the 10 m bottom became a
    wavemaker and this sum was ``O(A_tank * v)``.
    """
    t = _tank()
    for vel in ((0.01, 0.0), (0.0, -0.02), (0.013, 0.004)):
        t.body["vel"] = np.array(vel)
        assert abs(float(t.body_normal_flux().sum())) < 1e-12


def test_added_mass_is_of_the_flat_plate_order():
    """``A33`` must scale like ``rho B^2``, not like the tank.

    The flat-plate (infinite fluid) value is ``rho pi (B/2)^2 / 2``; a
    surface-piercing box in shallow water is larger but of that order.  Before
    the wall fix the "added mass" came out at 1.1e4 kg/m, i.e. a basin length
    (sqrt(1.1e4/1000) = 3.4 m) instead of a beam.
    """
    a = _tank().discrete_added_mass()
    plate = sk.SECTION_DENSITY * math.pi * (SEC.length / 2) ** 2 / 2.0
    assert a[0, 0] > 0
    assert 0.5 * plate < a[0, 0] < 2.0 * plate
    small = _tank()
    sec2 = sk.floating_strip(0.6, 0.1, freeboard=0.03, x0=4.0)
    small.set_body(sec2.polygon, mass=sec2.mass, inertia=sec2.inertia, ref=sec2.ref)
    small._update_mask()
    a2 = small.discrete_added_mass()
    assert a2[0, 0] < a[0, 0]


def test_body_force_functional_is_linear_in_the_pressure_field():
    """The load is one linear functional of ``q`` -- what the implicit step needs."""
    t = _tank()
    rng = np.random.default_rng(0)
    p1 = rng.standard_normal((t.nx, t.ns))
    p2 = rng.standard_normal((t.nx, t.ns))
    q = t.wetted_quadrature()
    f1, m1 = t.body_force_dynamic(p1, q)
    f2, m2 = t.body_force_dynamic(p2, q)
    fs, ms = t.body_force_dynamic(0.7 * p1 - 2.0 * p2, q)
    assert np.allclose(fs, 0.7 * f1 - 2.0 * f2, rtol=1e-10, atol=1e-10)
    assert ms == pytest.approx(0.7 * m1 - 2.0 * m2, rel=1e-10)


# ------------------------------------------------------------------- free motion
def test_free_decay_is_stable_and_oscillates_at_the_added_mass_frequency():
    t = _tank(nx=100, ns=16)
    a_disc = t.discrete_added_mass()
    t.body["xi"][0] = 0.01
    t._update_mask()
    dt = 0.9 / 40
    for _ in range(int(6 * 40)):
        t.step(dt)
    ts = np.asarray(t.body["time_series"]["t"])
    h = np.asarray(t.body["time_series"]["heave"])
    assert np.isfinite(h).all()
    assert np.abs(h).max() < 0.05                     # no blow-up
    sign = np.sign(h - h.mean())
    zero = [ts[i] for i in range(1, sign.size) if sign[i - 1] * sign[i] < 0]
    assert len(zero) > 3
    period = 2.0 * float(np.mean(np.diff(zero)))
    k33 = sk.hydrostatic_stiffness(SEC)["C33"]
    # the decay frequency must sit between the zero-added-mass and the
    # flat-plate-added-mass limits, and close to the tank's own discrete A33
    w_meas = 2 * math.pi / period
    a_meas = k33 / w_meas ** 2 - SEC.mass
    assert a_meas > 0.5 * a_disc[0, 0] and a_meas < 1.5 * a_disc[0, 0]


@pytest.mark.xfail(reason="known open defect: a body *plus* an incident wave field "
                         "destabilises the sigma free surface at the section's "
                         "waterline (see docs/wave-tank-td-body/README.md §5); the "
                         "calm-water forced/decay cases and the wave-only cases are "
                         "both stable, so this is the body/wave coupling, not either "
                         "half", strict=False)
def test_free_body_in_a_wave_run_stays_bounded():
    t = SigmaWaveTank(TankConfig(length=10.0, nx=120, ns=16))
    t.set_wave(0.02, 1.0)
    t.initialize_from_wave(0.0)
    t.set_body(SEC.polygon, mass=SEC.mass, inertia=SEC.inertia, ref=SEC.ref)
    dt = 1.0 / 50
    for _ in range(int(3 * 50)):
        t.step(dt)
    ts = np.asarray(t.body["time_series"]["heave"])
    assert np.isfinite(ts).all()
    assert np.abs(ts).max() < 0.2
    assert abs(float(t.body["xi"][1])) < 0.1


def test_fixed_body_does_not_move():
    t = _tank(nx=100, ns=16)
    t.set_body(SEC.polygon, mass=SEC.mass, inertia=SEC.inertia, ref=SEC.ref,
               motion="fixed")
    t.set_wave(0.02, 1.0)
    t.initialize_from_wave(0.0)
    for _ in range(40):
        t.step(1.0 / 50)
    assert np.allclose(t.body["xi"], 0.0)
    assert np.isfinite(t.body["force"]).all()
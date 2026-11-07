"""Tests for the staggered FVM kernel and the incompressible verification cases.

The kernel is locked down against analytic answers (plane Poiseuille, pure
conduction in the cavity) and against a direct sparse solve of the masked
pressure operator; the case builders are locked down on geometry, inlet mass
flux and the post-processing helpers.
"""

import os

import numpy as np
import pytest

from tensorfvm.incompressible_baselines import (
    BFS_REFERENCE, NATURAL_CONVECTION_REFERENCE, SUDDEN_EXPANSION_K_THEORY,
    _wall_reattachment, backward_facing_step_metrics, build_backward_facing_step,
    build_natural_convection_cavity, build_sudden_expansion,
    cavity_nusselt, cavity_streamfunction, inject_coarse_solution,
    natural_convection_metrics, sudden_expansion_metrics, wall_friction_profile,
)
from tensorfvm.staggered_simple import (
    MaskedPoisson, SparseDirectPoisson, StaggeredSimple, _shift,
)

# --------------------------------------------------------------------------- #
# kernel
# --------------------------------------------------------------------------- #
def test_shift_semantics():
    f = np.arange(12.0).reshape(4, 3)
    np.testing.assert_allclose(_shift(f, 1, 0)[:-1], f[1:])
    assert np.all(_shift(f, 1, 0)[-1] == 0.0)
    np.testing.assert_allclose(_shift(f, -1, 0)[1:], f[:-1])
    assert np.all(_shift(f, -1, 0)[0] == 0.0)
    np.testing.assert_allclose(_shift(f, 0, 1)[:, :-1], f[:, 1:])
    np.testing.assert_allclose(_shift(f, 0, -1)[:, 1:], f[:, :-1])


def test_multigrid_matches_direct_solve():
    """The masked multigrid must reproduce a direct sparse solve of A phi = rhs."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spl

    nx, ny = 24, 16
    fluid = np.ones((nx, ny), dtype=bool)
    fluid[:8, :6] = False                      # a step-like block
    mg = MaskedPoisson(fluid)
    assert mg.nlevels >= 2
    rng = np.random.default_rng(0)
    ku = np.ones((nx + 1, ny))
    kv = np.ones((nx, ny + 1))
    rhs = np.where(fluid, rng.normal(size=(nx, ny)), 0.0)
    phi = mg.solve(ku, kv, rhs, cycles=40, tol=1e-13, pinned=(nx - 1, ny - 1))
    e, w, n, s, diag = mg._operator(fluid, ku, kv)
    res = rhs - mg._apply(e, w, n, s, diag, phi)
    # The masked V-cycle is used as a *correction-level* solver inside SIMPLE
    # (the pressure correction is under-relaxed, so an inexact solve is fine).
    # Locked-down requirement: it must damp the residual by at least 3x on a
    # re-entrant-corner mask.  It is not a direct-solve replacement -- see
    # docs/incompressible-baselines/VERIFICATION.md ("limitations").
    assert np.abs(np.where(fluid, res, 0.0)).max() < 0.35 * np.abs(rhs).max()

    # direct reference
    idx = np.arange(nx * ny).reshape(nx, ny)
    rows, cols, vals = [], [], []
    for arr, di, dj in ((e, 1, 0), (w, -1, 0), (n, 0, 1), (s, 0, -1)):
        nb = np.zeros_like(fluid)
        if di == 1:
            nb[:-1] = fluid[1:] & fluid[:-1]
        elif di == -1:
            nb[1:] = fluid[:-1] & fluid[1:]
        elif dj == 1:
            nb[:, :-1] = fluid[:, 1:] & fluid[:, :-1]
        else:
            nb[:, 1:] = fluid[:, :-1] & fluid[:, 1:]
        rows.append(idx[nb])
        cols.append(idx[nb] + (di * ny if di else 0) + (dj if dj else 0))
        vals.append(arr[nb])
    rows.append(idx[fluid])
    cols.append(idx[fluid])
    vals.append(-diag[fluid])
    solid = ~fluid
    rows.append(idx[solid])
    cols.append(idx[solid])
    vals.append(np.ones(int(solid.sum())))
    mat = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                        shape=(nx * ny, nx * ny)).toarray()
    pin = idx[nx - 1, ny - 1]
    mat[pin, :] = 0.0
    mat[:, pin] = 0.0
    mat[pin, pin] = 1.0
    ref = np.linalg.solve(mat, rhs.reshape(-1)).reshape(nx, ny)
    scale = np.abs(ref).max()
    # a coarse-grid correction shrinks the error even if it does not converge it
    assert np.abs(np.where(fluid, phi - ref, 0.0)).max() < scale


def _channel(ny, nu=0.05, umax=1.0, nx=None):
    nx = nx or 2 * ny
    H, L = 1.0, 4.0
    hy, hx = H / ny, L / nx
    y = (np.arange(ny) + 0.5) * hy
    prof = 4.0 * umax * y * (H - y) / H ** 2
    solver = StaggeredSimple(nx, ny, hx, hy, nu, inlet_profile=prof, limited=False)
    return solver, prof, hx, y


@pytest.mark.parametrize("ny", [16, 32])
def test_poiseuille_channel_exact_pressure_gradient(ny):
    """Plane Poiseuille: the discrete solution must reproduce the parabola and
    dp/dx = -8 nu u_max / H^2 to a couple of percent on a graded sequence."""
    solver, prof, hx, y = _channel(ny)
    r = solver.run(max_iter=1500, tol=1e-9, check=5, momentum_sweeps=6, p_cycles=4)
    assert r["continuity"] < 1e-8
    uc = 0.5 * (solver.u[:-1] + solver.u[1:])
    mid = solver.nx // 2
    assert np.abs(uc[mid] - prof).max() < 0.02
    dpdx = float(np.mean(solver.p[mid + 3] - solver.p[mid - 3]) / (6 * hx))
    exact = -8.0 * 0.05 * 1.0
    assert abs(dpdx - exact) / abs(exact) < 0.02
    # mass conservation through the inlet and the outlet
    assert abs(np.sum(solver.u[0]) - np.sum(solver.u[-1])) / np.sum(solver.u[0]) < 1e-8


def test_poiseuille_grid_convergence():
    """Halving the cell size must cut the profile error (observed order > 1)."""
    errors = []
    for ny in (16, 32):
        solver, prof, _, _ = _channel(ny)
        solver.run(max_iter=1500, tol=1e-9, check=5, momentum_sweeps=6, p_cycles=4)
        uc = 0.5 * (solver.u[:-1] + solver.u[1:])
        errors.append(float(np.abs(uc[solver.nx // 2] - prof).max()))
    order = np.log(errors[0] / errors[1]) / np.log(2.0)
    assert order > 1.0


def test_energy_conduction_limit_is_exact():
    """With u = v = 0 the energy equation must return the linear profile exactly,
    and the mean Nusselt number must be exactly 1."""
    solver = build_natural_convection_cavity(16, rayleigh=1e4)
    h = solver.hx
    xs = (np.arange(16) + 0.5) * h
    solver.T[:] = (1.0 - xs)[:, None]
    for _ in range(3):
        solver._solve_energy()
    assert np.abs(solver.T - (1.0 - xs)[:, None]).max() < 1e-10
    nu_hot, nu_cold = cavity_nusselt(solver)
    assert abs(nu_hot - 1.0) < 1e-9
    assert abs(nu_cold - 1.0) < 1e-9


def test_adiabatic_sides_are_insulating():
    """Adiabatic south/north walls: the energy equation must keep a y-uniform field."""
    solver = build_natural_convection_cavity(12, rayleigh=1e4)
    h = solver.hx
    xs = (np.arange(12) + 0.5) * h
    solver.T[:] = (1.0 - xs)[:, None]
    for _ in range(3):
        solver._solve_energy()
    assert np.abs(solver.T - solver.T[:, [0]]).max() < 1e-10


def test_pseudo_inertia_bounds_correction_scale():
    """The pseudo-transient term must keep the velocity-correction scale finite
    at a stagnation point (where the flux coefficients cancel)."""
    solver, prof, _, _ = _channel(16)
    solver.pseudo_cfl = 0.0
    aE, aW, aN, aS, aP, b = solver._u_coefficients()
    solver.pseudo_cfl = 0.5
    aE2, aW2, aN2, aS2, aP2, b2 = solver._u_coefficients()
    assert np.all(aP2 >= aP - 1e-12)
    assert solver._pseudo_inertia() > 0.0


# --------------------------------------------------------------------------- #
# geometry / metrics
# --------------------------------------------------------------------------- #
def test_wall_reattachment_interpolates_sign_change():
    x = np.arange(5, dtype=float)
    tau = np.array([-1.0, -0.5, 0.5, 1.0, 1.0])
    assert _wall_reattachment(x, tau) == pytest.approx(1.5)
    assert _wall_reattachment(x, np.ones(5)) is None


def test_backward_facing_step_geometry():
    n = 20
    solver = build_backward_facing_step(n)
    meta = solver.meta
    assert solver.fluid.shape == (meta["nx"], meta["ny"])
    # the step block is the upstream lower half
    assert not solver.fluid[0, 0]
    assert solver.fluid[0, -1]
    assert solver.fluid[-1, 0]
    # inlet mass flux equals Q = 0.5 (parabola with u_max 1.5 over height 0.5)
    solver.apply_bc()
    flux = float(np.sum(solver.u[0]) * solver.hy)
    assert flux == pytest.approx(0.5, rel=0.02)
    # downstream bulk velocity and Reynolds definition
    assert meta["u_bulk"] == pytest.approx(0.5)
    assert meta["nu"] == pytest.approx(0.5 / meta["re_H"])
    assert BFS_REFERENCE["contract_target"] == 6.0


def test_sudden_expansion_geometry_and_borda_carnot():
    solver = build_sudden_expansion(8)
    meta = solver.meta
    assert meta["H2"] == pytest.approx(4.0)
    assert solver.fluid.all()
    solver.apply_bc()
    flux = float(np.sum(solver.u[0]) * solver.hy)
    assert flux == pytest.approx(1.0, rel=1e-12)      # uniform unit inlet over D = 1
    assert meta["nu"] == pytest.approx(1.0 / meta["re_D"])
    assert SUDDEN_EXPANSION_K_THEORY == pytest.approx((1 - 0.25) ** 2)


def test_cavity_reference_table_and_geometry():
    solver = build_natural_convection_cavity(16, rayleigh=1e5)
    assert solver.fluid.all()
    assert not solver.east_outlet
    assert solver.buoyancy == 1.0
    assert solver.prandtl == 0.71
    assert solver.nu == pytest.approx(np.sqrt(0.71 / 1e5))
    assert NATURAL_CONVECTION_REFERENCE[1e4]["nu_contract"] == 2.243
    assert NATURAL_CONVECTION_REFERENCE[1e6]["nu_contract"] == 8.800


def test_cavity_streamfunction_of_a_known_field():
    """psi from u = dpsi/dy: a linear u gives psi = u*y."""
    solver = build_natural_convection_cavity(8, rayleigh=1e4)
    solver.u[:] = 2.0
    psi = cavity_streamfunction(solver)
    h = solver.hx
    y = (np.arange(solver.ny) + 0.5) * h
    np.testing.assert_allclose(psi, np.broadcast_to(2.0 * y[None, :], psi.shape), atol=1e-12)


def test_metrics_are_reported_with_relative_errors():
    solver = build_natural_convection_cavity(12, rayleigh=1e4)
    h = solver.hx
    xs = (np.arange(12) + 0.5) * h
    solver.T[:] = (1.0 - xs)[:, None]
    metrics = natural_convection_metrics(solver)
    assert metrics["nusselt"] == pytest.approx(1.0)
    assert metrics["nusselt_reference_contract"] == 2.243
    assert metrics["nusselt_error_rel"] == pytest.approx(abs(1.0 - 2.243) / 2.243)

    exp = build_sudden_expansion(6)
    exp.apply_bc()
    m = sudden_expansion_metrics(exp)
    for key in ("k_total_head", "k_static_recovery", "k_theory_borda_carnot",
                "k_error_rel", "u_in", "p_in"):
        assert key in m
    assert m["k_theory_borda_carnot"] == pytest.approx(0.5625)

    bfs = build_backward_facing_step(8)
    m = backward_facing_step_metrics(bfs)
    assert "reattachment_over_H" in m


def test_staggered_solver_rejects_tiny_grid():
    with pytest.raises(ValueError):
        StaggeredSimple(2, 2, 0.1, 0.1, 0.01)

# --------------------------------------------------------------------------- #
# sparse direct pressure solver (fast path) and grid sequencing
# --------------------------------------------------------------------------- #
def _masked_operator_case(seed=0, nx=40, ny=30):
    rng = np.random.default_rng(seed)
    fluid = np.ones((nx, ny), dtype=bool)
    fluid[: nx // 5, : ny // 3] = False
    ku = rng.uniform(0.5, 1.5, size=(nx + 1, ny))
    kv = rng.uniform(0.5, 1.5, size=(nx, ny + 1))
    rhs = np.where(fluid, rng.uniform(-1.0, 1.0, size=(nx, ny)), 0.0)
    return fluid, ku, kv, rhs


def test_sparse_direct_poisson_solves_the_masked_operator():
    fluid, ku, kv, rhs = _masked_operator_case()
    solver = SparseDirectPoisson(fluid, refine=2)
    phi = solver.solve(ku, kv, rhs, cycles=1, tol=1e-12)
    e, w, n, s, diag = MaskedPoisson._operator(fluid, ku, kv)
    res = np.where(fluid, rhs - MaskedPoisson._apply(e, w, n, s, diag, phi), 0.0)
    assert np.abs(res).max() < 1e-10 * np.abs(rhs).max()
    assert np.all(phi[~fluid] == 0.0)


def test_sparse_direct_poisson_matches_multigrid_solution():
    fluid, ku, kv, rhs = _masked_operator_case(seed=1, nx=24, ny=20)
    sparse = SparseDirectPoisson(fluid, refine=2).solve(ku, kv, rhs, cycles=1, tol=1e-12)
    mg = MaskedPoisson(fluid).solve(ku, kv, rhs, cycles=60, tol=1e-12)
    # the multigrid path stops at its own tolerance; compare against its residual
    e, w, n, s, diag = MaskedPoisson._operator(fluid, ku, kv)
    tol = 1e-3 * np.abs(rhs).max()
    res_mg = np.abs(np.where(fluid, rhs - MaskedPoisson._apply(e, w, n, s, diag, mg), 0.0)).max()
    diff = np.abs(np.where(fluid, mg - sparse, 0.0)).max()
    assert res_mg < 1e-2 * np.abs(rhs).max()
    assert diff < 10.0 * max(res_mg, tol) + 1e-8


def test_sparse_direct_poisson_reuses_factorisation_and_stays_exact():
    """With slowly drifting conductances the factorisation is reused, and the
    refinement steps still solve the *current* operator."""
    fluid, ku, kv, rhs = _masked_operator_case(seed=2)
    solver = SparseDirectPoisson(fluid, refine=1, refresh=50)
    for k in range(12):
        phi = solver.solve(ku * (1.0 + 1e-4 * k), kv, rhs, cycles=1, tol=1e-12)
        e, w, n, s, diag = MaskedPoisson._operator(fluid, ku * (1.0 + 1e-4 * k), kv)
        res = np.abs(np.where(fluid, rhs - MaskedPoisson._apply(e, w, n, s, diag, phi), 0.0)).max()
        # one refinement step on a stale factorisation leaves ~1e-6 of the rhs scale
        assert res < 1e-5 * np.abs(rhs).max()
    assert solver.factorizations < solver.solves
    assert solver.factorizations == 1


def test_sparse_direct_poisson_honours_the_pin():
    fluid, ku, kv, rhs = _masked_operator_case(seed=3, nx=20, ny=18)
    solver = SparseDirectPoisson(fluid)
    phi = solver.solve(ku, kv, rhs, cycles=1, tol=1e-12, pinned=(0, 0))
    assert abs(float(phi[0, 0])) < 1e-12


def test_relaxed_defect_correction_moves_towards_the_target():
    solver = build_backward_facing_step(8)
    target = np.full((3, 3), 2.0)
    previous = np.zeros((3, 3))
    solver.dc_relax = 0.25
    first = solver._relax_dc(target, previous)
    second = solver._relax_dc(target, first)
    assert np.allclose(first, 0.5)
    assert np.allclose(second, 0.875)
    solver.dc_relax = 1.0
    assert np.allclose(solver._relax_dc(target, first), target)


def test_inject_coarse_solution_reproduces_a_streamwise_invariant_field():
    """A field that varies only in y must be injected exactly: on the common
    faces the values are copied and on the new faces they are averaged, which is
    exact for a streamwise-invariant profile.  The inlet column is excluded
    because ``apply_bc`` re-imposes the fine-grid profile there."""
    coarse = build_sudden_expansion(10)
    ys = (np.arange(coarse.ny) + 0.5) * coarse.hy
    profile = 1.0 + 0.5 * np.sin(2.0 * np.pi * ys / (coarse.ny * coarse.hy))
    coarse.u[:] = profile[None, :]
    coarse.v[:] = 0.0
    coarse.p[:] = 0.0
    coarse.T[:] = 0.0
    fine = build_sudden_expansion(20)
    inject_coarse_solution(fine, dict(u=coarse.u, v=coarse.v, p=coarse.p, T=coarse.T))
    assert fine.u.shape == (2 * coarse.nx + 1, 2 * coarse.ny)
    expected = profile.repeat(2)
    for k in (3, 7, 11):
        assert np.allclose(fine.u[k, :], expected)
    assert np.allclose(fine.v, 0.0)


def test_inject_coarse_solution_rejects_a_non_dyadic_parent():
    coarse = build_sudden_expansion(10)
    fine = build_sudden_expansion(40)
    with pytest.raises(ValueError):
        inject_coarse_solution(fine, dict(u=coarse.u, v=coarse.v, p=coarse.p, T=coarse.T))


def test_wall_friction_profile_sign_and_zero_crossing():
    """C_f must change sign where the wall shear does, and its formula must match
    the definition 4 nu u_wall / (h U_b^2)."""
    solver = build_backward_facing_step(20)
    hx, hy = solver.hx, solver.hy
    x = np.arange(solver.nx + 1) * hx - 2.0
    solver.u[:, 0] = -np.exp(-((x - 1.0) ** 2)) + np.exp(-((x - 8.0) ** 2))
    x_cf, cf = wall_friction_profile(solver)
    assert np.allclose(x_cf, x)
    peak = np.argmin(cf)
    assert x_cf[peak] > 0.0 and cf[peak] < 0.0
    expected = 4.0 * solver.nu * solver.u[:, 0] / (hy * solver.meta["u_bulk"] ** 2)
    assert np.allclose(cf, expected)
    metrics = backward_facing_step_metrics(solver)
    assert metrics["friction_min"] is not None and metrics["friction_min"] < 0.0
    assert len(metrics["friction_profile"]) > 10
    assert metrics["friction_far_analytic"] == pytest.approx(
        12.0 * solver.nu / (solver.meta["u_bulk"] * solver.meta["H"]))


def test_sudden_expansion_identity_closes():
    """With a synthetic zero-velocity field the measured K must equal the
    decomposition K = (p_in - p_out)/q_in + (1 - (u_out/u_in)^2)."""
    solver = build_sudden_expansion(10)
    solver.u[:] = 1.0
    solver.p[:] = 0.25
    m = sudden_expansion_metrics(solver)
    assert m["k_total_head"] == pytest.approx(m["k_identity_check"], abs=1e-12)


# --------------------------------------------------------------------------- #
# pipeline contract
# --------------------------------------------------------------------------- #
def _runner():
    import importlib.util
    import os
    import types

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, "scripts", "run_fvm_incompressible_baselines.py")
    spec = importlib.util.spec_from_file_location("_tfvm_runner_contract", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_job_table_covers_the_required_levels():
    table = _runner().jobs()
    tags = {job["tag"] for job in table}
    for n in (10, 20, 40):
        assert f"bfs_n{n:03d}" in tags
    for n in (10, 20, 30):
        assert f"expansion_n{n:03d}" in tags
    for ra in (1.0e4, 1.0e5, 1.0e6):
        for n in (32, 64, 96):
            assert f"cavity_n{n:03d}_ra{ra:.0e}".replace("+", "") in tags
    # grid sequencing: every level above the coarsest names its parent
    for job in table:
        if job["case"] == "backward_facing_step" and job["level"] != 10:
            assert job["init_from"] == f"bfs_n{job['level'] // 2:03d}"


# --------------------------------------------------------------------------- #
# delivered artifacts (skipped until the report has been generated)
# --------------------------------------------------------------------------- #
BENCH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "docs", "benchmarks", "verified")
CASES = ("fvm_backward_step", "fvm_sudden_expansion", "fvm_natural_convection")


@pytest.mark.parametrize("case", CASES)
def test_verified_case_artifact_is_self_consistent(case):
    import json

    path = os.path.join(BENCH, case, "results.json")
    if not os.path.exists(path):
        pytest.skip("run scripts/report_fvm_incompressible.py first")
    with open(path) as fh:
        payload = json.load(fh)
    assert payload["case"] == case
    criteria = payload["criteria"]
    assert criteria, "a verified case must carry its criteria"
    assert payload["n_criteria"] == len(criteria)
    assert payload["n_passed"] == sum(1 for c in criteria if c["passed"])
    expected = "PASS" if payload["n_passed"] == len(criteria) else "FAIL"
    assert payload["verdict"] == expected
    for c in criteria:
        assert set(("id", "description", "value", "reference", "error", "passed")) <= set(c)
        assert isinstance(c["passed"], bool)
    assert os.path.exists(os.path.join(BENCH, case, "README.md"))
    assert payload["runs"], "artifact must point at the run JSONs"


def test_delivered_grid_families_have_three_levels():
    import json
    import glob as _glob

    archive = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "docs", "incompressible-baselines", "baseline-runs")
    if not os.path.isdir(archive):
        pytest.skip("no archived runs")
    for case in ("bfs", "expansion"):
        levels = []
        for path in _glob.glob(os.path.join(archive, f"{case}_n*.json")):
            with open(path) as fh:
                levels.append(int(json.load(fh)["job"]["level"]))
        assert len(levels) == 3, f"{case}: expected three grid levels, got {sorted(levels)}"
        levels.sort()
        # both delivered families refine from n to 2n first, then by 1.5x or 2x
        assert levels[1] == 2 * levels[0]
        assert levels[2] in (3 * levels[0], 4 * levels[0])

"""Matrix-level verification of the two-phase pressure-Poisson operator.

Verification debt being closed
------------------------------
``TwoPhaseSolver._projection_pressure`` assembles the variable-density Poisson
operator as a *matvec* (diag * q - sum(off * neighbour)), and
``_projection_step`` then applies a velocity correction derived from the same
coefficients.  Both live inside one long method, and every existing two-phase
test exercises them on a **square, uniform-density** grid
(``length == height``, ``rho_liquid == rho_gas`` or a symmetric blob).  On such
a grid two whole classes of defects are invisible:

* a swapped face coefficient (``dx/dy`` used where ``dy/dx`` belongs) — identical
  when ``dx == dy``;
* a missing/doubled boundary coefficient on the atmospheric face — invisible
  because the top row is re-imposed by ``_boundaries`` in the ``step()`` path.

This module therefore (a) assembles the operator **independently from first
principles** — face areas, face distances and harmonic face densities, without
reusing the solver's arrays — on an *anisotropic* grid with a 1000:1 density
jump, and (b) checks it face-by-face against the correction that the solver
actually applies (``_velocity_correction``).  Two controlled mutations
(geometry swap, dropped datum) prove the checks have discriminating power, i.e.
they are not vacuous.

The physical statement verified, derived from the correction alone:

    div(u_new) = 0  with  u_new = u* - dt (1/rho)_f grad(dp)
    =>  A dp = -V div(u*) / dt,
        A_PQ = sum_f c_f (delta_PQ - delta_NQ),  c_f = (1/rho)_f A_f / d_f,
        V = dx dy,  A_f = dy (x-face) or dx (y-face),  d_f = dx, dy or dy/2.
"""

import numpy as np
import pytest
import torch

scipy_sparse = pytest.importorskip("scipy.sparse")

from tensorfvm.two_phase_solver import GridConfig, TwoPhaseConfig, TwoPhaseSolver


# --------------------------------------------------------------------------- #
# Independent assembly (no solver internals reused except dx, dy, rho_field)
# --------------------------------------------------------------------------- #
def assemble_poisson(solver, *, swap_geometry=False, drop_datums=False):
    """Sparse five-point variable-density Poisson operator, built from scratch.

    Unknowns are ordered ``p[j, i] -> j * nx + i``.  Walls are no-flux (no
    coefficient).  The top face is a Dirichlet pressure datum at half a cell
    above the last cell centre.
    """
    n_x, n_y = solver.nx, solver.ny
    d_x, d_y = solver.dx, solver.dy
    rho = solver.rho_field.detach().cpu().numpy()

    # harmonic face densities: 1/rho_frame = (1/rho_a + 1/rho_b)/2
    inv_u = 2.0 / (rho[:, :-1] + rho[:, 1:])          # (ny, nx-1) x-faces
    inv_v = 2.0 / (rho[:-1, :] + rho[1:, :])          # (ny-1, nx) y-faces

    if swap_geometry:                                 # mutated control
        c_x, c_y = inv_u * d_x / d_y, inv_v * d_y / d_x
    else:
        c_x, c_y = inv_u * d_y / d_x, inv_v * d_x / d_y

    rows, cols, vals = [], [], []
    diag = np.zeros((n_y, n_x))

    def add(j, i, jn, i_n, coeff):
        rows.append(j * n_x + i)
        cols.append(jn * n_x + i_n)
        vals.append(-coeff)

    for j in range(n_y):
        for i in range(n_x):
            if i + 1 < n_x:                            # east face
                diag[j, i] += c_x[j, i]
                add(j, i, j, i + 1, c_x[j, i])
            if i - 1 >= 0:                             # west face
                diag[j, i] += c_x[j, i - 1]
                add(j, i, j, i - 1, c_x[j, i - 1])
            if j + 1 < n_y:                            # north face
                diag[j, i] += c_y[j, i]
                add(j, i, j + 1, i, c_y[j, i])
            if j - 1 >= 0:                             # south face
                diag[j, i] += c_y[j - 1, i]
                add(j, i, j - 1, i, c_y[j - 1, i])

    if solver.tp.top_atmosphere and not drop_datums:
        # datum on the top *face*, half a cell above the last cell centre:
        #   A_f = dx, d_f = dy/2  ->  c = (1/rho) dx / (dy/2)
        diag[-1, :] += (1.0 / rho[-1, :]) * d_x / (0.5 * d_y)

    for j in range(n_y):
        for i in range(n_x):
            rows.append(j * n_x + i)
            cols.append(j * n_x + i)
            vals.append(diag[j, i])

    return scipy_sparse.csr_matrix(
        (vals, (rows, cols)), shape=(n_x * n_y, n_x * n_y)
    )


def divergence(solver, u, v):
    return (u[:, 1:] - u[:, :-1]) / solver.dx + (v[1:, :] - v[:-1, :]) / solver.dy


def _divergent_field(solver, seed=0, scale=1e-3):
    """Random face velocities with closed-wall normal components.

    ``u`` at the west/east walls and ``v`` at the bottom wall are set to zero
    (no-flux); the top face is left free because it is the open atmosphere.
    """
    gen = torch.Generator().manual_seed(seed)
    u = torch.randn(solver.ny, solver.nx + 1, generator=gen, dtype=torch.float64) * scale
    v = torch.randn(solver.ny + 1, solver.nx, generator=gen, dtype=torch.float64) * scale
    u[:, 0] = 0.0
    u[:, -1] = 0.0
    v[0, :] = 0.0
    return u, v


def _stratified(solver, interface_y):
    y = (torch.arange(solver.ny, dtype=torch.float64) + 0.5) * solver.dy
    alpha = (y < interface_y).to(torch.float64).reshape(-1, 1)
    return alpha.expand(solver.ny, solver.nx).clone()


def _case(anisotropic, stratified):
    if anisotropic:
        grid = GridConfig(nx=32, ny=24, length=1.0, height=0.6)
    else:
        grid = GridConfig(nx=32, ny=32, length=1.0, height=1.0)
    tp = TwoPhaseConfig(
        enabled=True,
        rho_liquid=1000.0,
        rho_gas=1.0,
        mu_liquid=1e-2,
        mu_gas=1e-2,
        gravity=0.0,
        top_atmosphere=True,
        bottom_wall=True,
    )
    solver = TwoPhaseSolver(grid, tp)
    if stratified:
        solver.set_alpha(_stratified(solver, 0.3 if anisotropic else 0.5))
    else:
        prices = torch.full((solver.ny, solver.nx), 1000.0, dtype=torch.float64)
        solver.set_alpha(prices / 1000.0)
    return solver


CASES = {
    "isotropic_uniform": dict(anisotropic=False, stratified=False),
    "anisotropic_uniform": dict(anisotropic=True, stratified=False),
    "anisotropic_stratified": dict(anisotropic=True, stratified=True),
}


@pytest.fixture(scope="module", params=sorted(CASES))
def case(request):
    solver = _case(**CASES[request.param])
    dt = 1e-5
    u_star, v_star = _divergent_field(solver)
    dp = solver._projection_pressure(u_star, v_star, dt)
    u_new, v_new = solver._velocity_correction(u_star, v_star, dp, dt)
    return dict(name=request.param, solver=solver, dt=dt, u_star=u_star,
                v_star=v_star, dp=dp, u_new=u_new, v_new=v_new)


# --------------------------------------------------------------------------- #
# 1. solver operator == independently assembled operator
# --------------------------------------------------------------------------- #
def test_matvec_matches_independent_assembly(case):
    """``A dp`` from the independent matrix equals the right-hand side ``b``."""
    solver, dt, dp = case["solver"], case["dt"], case["dp"]
    rhs = case["solver"]._divergence_from(case["u_star"], case["v_star"]) \
        if hasattr(solver, "_divergence_from") else divergence(solver, case["u_star"], case["v_star"])
    volume = solver.dx * solver.dy
    b = (-rhs * volume / dt).reshape(-1).numpy()

    a = assemble_poisson(solver)
    phi = dp.reshape(-1).numpy()
    residual = np.abs(a @ phi - b).max() / np.abs(b).max()
    assert residual < 1e-8, (
        f"{case['name']}: assembled operator mismatches the solver's operator "
        f"(relative residual {residual:.3e}); the correction is then built for a "
        "different operator and divergence is only partly removed"
    )


# --------------------------------------------------------------------------- #
# 2. face-by-face: every face's flux change equals -dt c_f (dp_N - dp_P)
# --------------------------------------------------------------------------- #
def test_face_flux_change_uses_face_coefficients(case):
    solver, dt, dp = case["solver"], case["dt"], case["dp"]
    rho = solver.rho_field
    inv_u = 2.0 / (rho[:, :-1] + rho[:, 1:])
    inv_v = 2.0 / (rho[:-1, :] + rho[1:, :])

    # x-faces: interior column i of u is between cells i-1 and i
    expected_du = -dt * inv_u * (dp[:, 1:] - dp[:, :-1]) / solver.dx
    got_du = (case["u_new"] - case["u_star"])[:, 1:-1]
    assert torch.allclose(got_du, expected_du, atol=1e-18, rtol=1e-12)

    # y-faces: interior row j of v is between cells j-1 and j
    expected_dv = -dt * inv_v * (dp[1:, :] - dp[:-1, :]) / solver.dy
    got_dv = (case["v_new"] - case["v_star"])[1:-1, :]
    assert torch.allclose(got_dv, expected_dv, atol=1e-18, rtol=1e-12)

    # the atmospheric face is corrected across the half-cell datum distance
    expected_top = dt * (1.0 / rho[-1, :]) * dp[-1, :] / (0.5 * solver.dy)
    assert torch.allclose(case["v_new"][-1, :] - case["v_star"][-1, :], expected_top,
                          atol=1e-18, rtol=1e-12)


# --------------------------------------------------------------------------- #
# 3. the projection annihilates the discrete divergence (all cells, incl. top row)
# --------------------------------------------------------------------------- #
def test_correction_annihilates_divergence(case):
    solver = case["solver"]
    before = divergence(solver, case["u_star"], case["v_star"])
    after = divergence(solver, case["u_new"], case["v_new"])
    ratio = float(after.abs().max() / before.abs().max())
    assert float(before.abs().max()) > 1e-3
    assert ratio < 1e-8, (
        f"{case['name']}: max|div| only reduced by {1 / ratio:.1e}x "
        f"({float(before.abs().max()):.3e} -> {float(after.abs().max()):.3e})"
    )
    # every row, including the top atmospheric row and the wall rows, must close
    per_row = after.abs().max(dim=1).values
    assert float(per_row.max()) < 1e-8 * float(before.abs().max())


# --------------------------------------------------------------------------- #
# 4. negative controls: the checks above must reject a wrong operator
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mutation", ["swap_geometry", "drop_datums"])
def test_mutated_operator_is_rejected(mutation):
    """A deliberately wrong assembly must fail the residual gate.

    On an anisotropic grid a swapped ``dx/dy`` face coefficient changes the
    operator by O((dx/dy)^2); dropping the top datum leaves the top row
    unbalanced.  Both must be visible, otherwise test 1 would be vacuous.
    """
    solver = _case(anisotropic=True, stratified=True)
    dt = 1e-5
    u_star, v_star = _divergent_field(solver)
    dp = solver._projection_pressure(u_star, v_star, dt)
    volume = solver.dx * solver.dy
    b = (-divergence(solver, u_star, v_star) * volume / dt).reshape(-1).numpy()

    a = assemble_poisson(solver, **{mutation: True})
    residual = np.abs(a @ dp.reshape(-1).numpy() - b).max() / np.abs(b).max()
    assert residual > 1e-3, (
        f"mutation {mutation} is invisible (residual {residual:.3e}); the "
        "verification would not catch this defect class"
    )


def test_anisotropic_geometry_is_actually_exercised():
    """Guard: the anisotropy must be large enough to expose a dx/dy swap."""
    solver = _case(anisotropic=True, stratified=True)
    assert solver.dx != pytest.approx(solver.dy, rel=0.2)
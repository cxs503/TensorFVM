"""CICSAM transport (Ubbink & Issa 1999) and the P167W2B-N03 verification gate.

The fast tests pin the *properties* the scheme must have (conservation,
boundedness, CBC limit, consistency of the normalised-variable helpers) and one
short rotating-disk run that guards the accuracy achieved at 64^2.  The heavy
sweeps (1/2/4/8 revolutions at 128^2 and 256^2, the Martin & Moyce dam break)
live in :mod:`tensorfvm.verification.vof_cicsam_gate` and are run as cluster
jobs; their artefacts are checked into ``docs/p167-n03``.
"""

import math

import pytest
import torch

from tensorfvm.vof_cicsam import (
    CICSAMVOF,
    blend_gamma,
    cbc_upper,
    cicsam_face_value,
    hyper_c,
    interface_thickness_cells,
    ultimate_quickest,
)
from tensorfvm.vof_verification.cases import (
    Transport,
    sample_faces,
    zalesak_case,
)
from tensorfvm.vof_verification.geometry import zalesak_alpha
from tensorfvm.vof_verification.metrics import field_metrics
from tensorfvm.verification.vof_cicsam_gate import (
    ACCURACY,
    eval_front,
    evaluate_criteria,
)


# --------------------------------------------------------------------------- #
# normalised-variable building blocks
# --------------------------------------------------------------------------- #
def test_hyper_c_is_the_cbc_bound():
    """Hyper-C must equal the CBC upper limit min(1, tilde_D / C)."""
    tilde = torch.tensor([0.0, 0.2, 0.5, 1.0], dtype=torch.float64)
    for cour in (0.25, 0.5, 1.0):
        c = torch.full_like(tilde, cour)
        expected = torch.clamp(tilde / cour, max=1.0)
        assert torch.allclose(hyper_c(tilde, c), expected)
        assert torch.allclose(hyper_c(tilde, c), cbc_upper(tilde, c))


def test_ultimate_quickest_limits():
    """UQ collapses onto the exact extremes at Courant 1 and onto QUICK at 0."""
    tilde = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
    one = torch.ones_like(tilde)
    zero = torch.zeros_like(tilde)
    at_one = ultimate_quickest(tilde, one)
    assert torch.allclose(at_one, tilde, atol=1e-12)
    quick = 0.375 + 0.75 * tilde
    assert torch.allclose(ultimate_quickest(tilde, zero), quick)
    # anti-diffusive at the interface mid value (downwind of the donor cell)
    assert float(ultimate_quickest(tilde, one * 0.5)[1]) > 0.5


def test_blend_gamma_orientation():
    """Compression when the normal is along the sweep, accuracy when across."""
    along = blend_gamma(torch.tensor([1.0]), torch.tensor([0.0]))
    across = blend_gamma(torch.tensor([0.0]), torch.tensor([1.0]))
    diag = blend_gamma(torch.tensor([1.0]), torch.tensor([1.0]))
    assert float(along) == pytest.approx(1.0)
    assert float(across) == pytest.approx(0.0)
    assert float(diag) == pytest.approx(0.5)
    # a uniform field carries no interface: the weight falls back to UQ
    assert float(blend_gamma(torch.tensor([0.0]), torch.tensor([0.0]))) == 0.0


def test_face_value_is_bounded_and_monotone():
    """The face value never leaves its stencil and grows with the blend weight."""
    ad = torch.full((5,), 0.5, dtype=torch.float64)
    aa = torch.full((5,), 1.0, dtype=torch.float64)
    au = torch.zeros(5, dtype=torch.float64)
    cour = torch.full((5,), 0.5, dtype=torch.float64)
    values = [cicsam_face_value(ad, aa, au, cour,
                                torch.full((5,), g, dtype=torch.float64))
              for g in (0.0, 0.5, 1.0)]
    for v in values:
        assert bool(((v >= 0.0) & (v <= 1.0)).all())
    assert float(values[0][0]) < float(values[1][0]) < float(values[2][0])
    # a reversed stencil (local extremum) falls back on the donor value
    rev = cicsam_face_value(torch.full((1,), 0.9, dtype=torch.float64),
                            torch.full((1,), 0.1, dtype=torch.float64),
                            torch.full((1,), 0.2, dtype=torch.float64),
                            torch.full((1,), 0.5, dtype=torch.float64),
                            torch.ones(1, dtype=torch.float64))
    assert float(rev) == pytest.approx(0.9)


def test_interface_thickness_metric():
    """1/max|d alpha| returns the transition length in cells."""
    n = 24
    step = torch.zeros(3, n, dtype=torch.float64)
    step[:, : n // 2] = 1.0                     # resolved step: no two-phase cell
    band = step.clone()
    band[:, n // 2 - 1] = 0.5                   # one half-cell -> 2-cell band
    wide = torch.zeros(3, n, dtype=torch.float64)
    x = torch.arange(n, dtype=torch.float64)
    wide[:] = 1.0 / (1.0 + torch.exp((x - 12.0) / 2.0))
    t_step = interface_thickness_cells(step)
    t_band = interface_thickness_cells(band)
    t_wide = interface_thickness_cells(wide)
    assert t_step["interface_cells"] == 0       # pure phases only
    assert t_band["median"] == pytest.approx(2.0, abs=0.05)
    assert 5.0 < t_wide["median"] < 12.0        # tanh over ~2-cell scale
    assert t_wide["median"] > t_band["median"]


# --------------------------------------------------------------------------- #
# transport properties
# --------------------------------------------------------------------------- #
def _translate(field: torch.Tensor, cells: int, axis: int,
               kg: float = 1.0) -> tuple[torch.Tensor, float]:
    """Translate ``field`` by ``cells`` cells with CICSAM (periodic, C = 0.5)."""
    ny, nx = field.shape
    vof = CICSAMVOF(nx, ny, 1.0 / nx, 1.0 / ny, boundary="periodic",
                    k_gamma=kg)
    u = torch.zeros((ny, nx + 1), dtype=torch.float64)
    v = torch.zeros((ny + 1, nx), dtype=torch.float64)
    if axis == 1:
        u[:, :] = 1.0
    else:
        v[:, :] = 1.0
    dt = 0.5 * (1.0 / nx)
    out = field.clone()
    for _ in range(2 * cells):
        out = vof.advect(out, u, v, dt)
    return out, 1.0 / nx


def test_cicsam_translates_a_step_without_smearing():
    """A colour-function step must translate without a diffuse precursor.

    The compressive branch is dispersive (a residual ripple of a few 1e-3 sits
    ahead of the front), so the assertion is on what matters physically: the
    step keeps its sharpness, the field stays bounded and the mass ledger
    closes to roundoff.
    """
    n = 64
    field = torch.zeros(3, n, dtype=torch.float64)
    field[:, : n // 4] = 1.0
    shifted, _ = _translate(field, 8, axis=1)
    v0 = float(field.sum().item())
    assert abs(float(shifted.sum().item()) - v0) / v0 < 1e-10
    assert float(shifted.min()) >= -1e-10
    assert float(shifted.max()) <= 1.0 + 1e-10
    l1 = float((shifted - torch.roll(field, 8, dims=1)).abs().sum().item())
    assert l1 / v0 < 0.02                       # < 2 % of the liquid volume
    profile = shifted[1]
    band = ((profile > 0.1) & (profile < 0.9)).sum().item()
    assert band <= 2                            # front still within two cells


def test_cicsam_conserves_mass_and_is_bounded():
    """Flux form + local repair: roundoff mass drift, hard bounds."""
    nx = 48
    x = (torch.arange(nx, dtype=torch.float64) + 0.5) / nx
    y = (torch.arange(nx, dtype=torch.float64) + 0.5) / nx
    dx2 = (x[:, None] - 0.5) ** 2 + (y[None, :] - 0.5) ** 2
    field = (dx2 < 0.15 ** 2).to(torch.float64)
    vof = CICSAMVOF(nx, nx, 1.0 / nx, 1.0 / nx, boundary="periodic")
    u = torch.full((nx, nx + 1), 1.0, dtype=torch.float64)
    v = torch.full((nx + 1, nx), 1.0, dtype=torch.float64)
    v0 = float(field.sum().item())
    out = field
    for k in range(60):
        out = vof.advect(out, u, v, 0.5 / nx, order="xy" if k % 2 == 0 else "yx")
    assert abs(float(out.sum().item()) - v0) / v0 < 1e-10
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


def test_cicsam_rejects_courant_gt_one():
    vof = CICSAMVOF(16, 16, 0.1, 0.1)
    alpha = torch.zeros((16, 16), dtype=torch.float64)
    u = torch.ones((16, 17), dtype=torch.float64)
    v = torch.zeros((17, 16), dtype=torch.float64)
    with pytest.raises(ValueError):
        vof.advect(alpha, u, v, 0.5)     # CFL = 5


def test_transport_dispatch_builds_cicsam():
    transport = Transport(scheme="cicsam", cfl=0.5)
    state = transport.build(16, 16, 0.0625, 0.0625)
    assert isinstance(state, CICSAMVOF)
    alpha = torch.zeros((16, 16), dtype=torch.float64)
    alpha[:8, :] = 1.0
    u = torch.zeros((16, 17), dtype=torch.float64)
    v = torch.zeros((17, 16), dtype=torch.float64)
    out = transport.step(state, alpha, u, v, 1e-3, 0.0625, 0.0625)
    assert out.shape == alpha.shape


# --------------------------------------------------------------------------- #
# rotating disk: accuracy guard
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def zalesak_one_revolution():
    """One revolution of the slotted disk at 64^2 (CICSAM and FCT)."""
    nx = 64
    dx = 1.0 / nx
    case = zalesak_case()
    exact = zalesak_alpha(nx, nx, 1.0, 1.0, (0.5, 0.75), 0.15, 0.025)
    u0, _ = sample_faces(case, nx, nx, 0.0)
    umax = float(u0.abs().max().item())
    dt = 0.5 * dx / umax
    steps = int(math.ceil(1.0 / dt))
    dt = 1.0 / steps
    out = {}
    for scheme in ("cicsam", "fct", "plic"):
        transport = Transport(scheme=scheme, cfl=0.5)
        state = transport.build(nx, nx, dx, dx)
        alpha = exact.clone()
        v0 = float(alpha.sum().item())
        for k in range(steps):
            u, v = sample_faces(case, nx, nx, k * dt, (k + 1) * dt)
            alpha = transport.step(state, alpha, u, v, dt, dx, dx)
        metrics = field_metrics(alpha, exact, dx * dx)
        metrics.update({"mass_drift": abs(float(alpha.sum().item()) - v0) / v0,
                        "thickness": interface_thickness_cells(alpha),
                        "peak": float(alpha.max().item())})
        out[scheme] = metrics
    return out


def test_zalesak_conservation_is_roundoff(zalesak_one_revolution):
    """Every scheme must conserve the disk area to roundoff over one turn."""
    for scheme, metrics in zalesak_one_revolution.items():
        assert metrics["mass_drift"] < 1e-10, scheme
        assert metrics["bounded"][0] >= 0.0 and metrics["bounded"][1] <= 1.0


def test_cicsam_is_sharper_and_more_accurate_than_fct(zalesak_one_revolution):
    """Regression guard: CICSAM beats FCT on L1 error and keeps the plateau."""
    cicsam = zalesak_one_revolution["cicsam"]
    fct = zalesak_one_revolution["fct"]
    assert cicsam["relative_l1"] < fct["relative_l1"]
    assert cicsam["peak"] > fct["peak"]
    assert cicsam["relative_l1"] < 0.20


def test_cicsam_interface_stays_within_two_cells():
    """The gate's acceptance number, judged on the gate's grid (128^2, 1 turn)."""
    nx = 128
    dx = 1.0 / nx
    case = zalesak_case()
    exact = zalesak_alpha(nx, nx, 1.0, 1.0, (0.5, 0.75), 0.15, 0.025)
    u0, _ = sample_faces(case, nx, nx, 0.0)
    dt = 0.5 * dx / float(u0.abs().max().item())
    steps = int(math.ceil(1.0 / dt))
    dt = 1.0 / steps
    vof = CICSAMVOF(nx, nx, dx, dx, boundary="wall")
    alpha = exact.clone()
    for k in range(steps):
        u, v = sample_faces(case, nx, nx, k * dt, (k + 1) * dt)
        alpha = vof.advect(alpha, u, v, dt, order="xy" if k % 2 == 0 else "yx")
    thickness = interface_thickness_cells(alpha)
    assert thickness["interface_cells"] > 0
    assert thickness["median"] < 1.6
    assert thickness["p90"] < ACCURACY["interface_thickness_cells"]


def test_cicsam_compression_beats_upwind_blend():
    """The blend weight must do work: full CICSAM << UQ-only (k_gamma = 0)."""
    nx = 64
    dx = 1.0 / nx
    case = zalesak_case()
    exact = zalesak_alpha(nx, nx, 1.0, 1.0, (0.5, 0.75), 0.15, 0.025)
    u0, _ = sample_faces(case, nx, nx, 0.0)
    dt = 0.5 * dx / float(u0.abs().max().item())
    steps = int(math.ceil(1.0 / dt))
    dt = 1.0 / steps
    errors = {}
    for kg in (0.0, 1.0):
        vof = CICSAMVOF(nx, nx, dx, dx, boundary="wall", k_gamma=kg)
        alpha = exact.clone()
        for k in range(steps):
            u, v = sample_faces(case, nx, nx, k * dt, (k + 1) * dt)
            alpha = vof.advect(alpha, u, v, dt,
                               order="xy" if k % 2 == 0 else "yx")
        # one full revolution is a closed solution: the exact field is the
        # initial field, no rotation of the reference is needed
        errors[kg] = field_metrics(alpha, exact, dx * dx)["relative_l1"]
    assert errors[1.0] < 0.5 * errors[0.0]


# --------------------------------------------------------------------------- #
# gate plumbing (no simulation)
# --------------------------------------------------------------------------- #
def test_eval_front_matches_reference_curve():
    """A trajectory drawn on the digitised curve must report zero error."""
    t = [0.0, 1.0, 2.0, 3.0]
    x = [0.0, 1.40, 2.45, 3.30]
    out = eval_front(t, x, times=(1.0, 2.0, 3.0))
    assert out["points_reached"] == "3/3"
    assert out["worst_relative_error"] < 1e-3
    assert out["passed"] is True


def test_eval_front_flags_a_lagging_front():
    t = [0.0, 1.0, 2.0, 3.0]
    x = [0.0, 0.7, 1.2, 1.6]
    out = eval_front(t, x, times=(1.0, 2.0, 3.0))
    assert out["passed"] is False
    assert out["worst_relative_error"] > ACCURACY["dam_front_relative"]


def test_evaluate_criteria_thresholds():
    """The criterion evaluator must judge the node's four thresholds."""
    zalesak = {"cells": [128, 128], "schemes": {
        "cicsam": {"revolutions": {
            "1": {"mass_drift_relative": 1e-15, "relative_l1": 0.04,
                  "peak_retention": 1.0,
                  "thickness": {"p90": 1.6, "median": 1.2}},
            "8": {"mass_drift_relative": 5e-15, "relative_l1": 0.09,
                  "peak_retention": 1.0,
                  "thickness": {"p90": 1.7, "median": 1.3}}}},
        "plic": {"revolutions": {
            "1": {"mass_drift_relative": 1e-15, "relative_l1": 0.05,
                  "peak_retention": 1.0,
                  "thickness": {"p90": 1.1, "median": 1.0}}}}}}
    hydrostatic = [{"pressure_oscillation": 1e-4}]
    dam = [{"scheme": "cicsam", "grid": [100, 30],
            "evaluation": {"worst_relative_error": 0.05}}]
    criteria = evaluate_criteria(zalesak, hydrostatic, dam)
    assert criteria["G1_zalesak_mass"]["passed"] is True
    assert criteria["G2_cicsam_sharpness"]["passed"] is True
    assert criteria["G4_hydrostatic_pressure"]["passed"] is True
    assert criteria["G5_dam_front"]["passed"] is True

    bad = evaluate_criteria(
        {"cells": [128, 128], "schemes": {"cicsam": {"revolutions": {
            "1": {"mass_drift_relative": 3e-3, "relative_l1": 0.3,
                  "peak_retention": 0.8,
                  "thickness": {"p90": 2.5, "median": 2.1}}}}}},
        [{"pressure_oscillation": 0.02}],
        [{"scheme": "cicsam", "grid": [100, 30],
          "evaluation": {"worst_relative_error": 0.25}}])
    assert bad["G1_zalesak_mass"]["passed"] is False
    assert bad["G2_cicsam_sharpness"]["passed"] is False
    assert bad["G4_hydrostatic_pressure"]["passed"] is False
    assert bad["G5_dam_front"]["passed"] is False
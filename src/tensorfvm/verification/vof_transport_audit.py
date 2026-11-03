"""Verification audit of the algebraic VOF transport stencil (verification debt).

Motivation
----------
:mod:`tensorfvm.two_phase_vof` documents :class:`~tensorfvm.two_phase_vof.VOFTransport`
as a "donor-cell (CICSAM-like)" scheme that is "stable, conservative, and bounded
... for CFL <= 1".  That claim had never been checked against linear (von Neumann)
theory.  This module supplies the missing verification: the discrete-symbol /
modified-equation analysis of the stencil the library actually implements, plus
periodic benchmark flows with exact solutions.

The library face value is (see ``VOFTransport.advect``)::

    f_face = f_donor + C (f_acceptor - f_donor),   C = |F_face| dt / V  (= |u| dt/dx)

so the blend weight *is* the Courant number.  For a uniform velocity and a smooth
(0 < alpha < 1) field the discrete symbol is::

    G_blend(theta, C) = (1 - C + 2C^2) + (C - 2C^2) cos(theta) - i C sin(theta)

Expanding |G|^2 for small theta::

    |G_blend|^2 = 1 + (3C^2 - C) theta^2 + O(theta^4)

which implies an effective numerical diffusion coefficient

    D_blend = (u dx / 2) (1 - 3C)      (anti-diffusive for C > 1/3)

and therefore **von Neumann stability only for C <= 1/3**.  The classic
donor-cell (upwind) scheme, selectable with ``VOFTransport(..., scheme="upwind")``,
has::

    G_upwind(theta, C) = 1 - C (1 - cos theta) - i C sin theta
    D_upwind = (u dx / 2) (1 - C)

and is stable for all 0 <= C <= 1, and exact for C = 1.

The ``alpha in [0, 1]`` clipping and the "pure-donor pinning" inside ``advect``
mask the linear instability on sharp interfaces (they act as a hard limiter),
which is why the existing unit tests pass; they cannot mask it for smooth fields,
where clipping is inactive and mode amplitudes grow or smear exactly as the
symbol predicts.  This audit measures both regimes and quantifies the
mass-conservation error the limiter introduces.

Everything here is deterministic, CPU-only and runs in seconds.

Usage
-----
    python -m tensorfvm.verification.vof_transport_audit --output <dir>
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

__all__ = [
    "symbol_blend",
    "symbol_upwind",
    "symbol",
    "max_symbol_magnitude",
    "stability_limit",
    "advect_1d",
    "measure_symbol",
    "theory_diffusion",
    "diffusion_coefficient",
    "convergence_1d",
    "advect_2d",
    "zalesak_initial",
    "zalesak_rotation",
    "run_study",
    "SCHEMES",
]

SCHEMES = ("blend", "upwind")
THIRD = 1.0 / 3.0


# ---------------------------------------------------------------------------
# Closed-form symbols
# ---------------------------------------------------------------------------
def symbol_blend(theta: float, cour: float) -> complex:
    """Discrete symbol of the library's Courant-weighted donor/acceptor blend."""
    a = 1.0 - cour + 2.0 * cour * cour
    b = cour - 2.0 * cour * cour
    return complex(a + b * math.cos(theta), -cour * math.sin(theta))


def symbol_upwind(theta: float, cour: float) -> complex:
    """Discrete symbol of classic donor-cell (first-order upwind) transport."""
    return complex(1.0 - cour * (1.0 - math.cos(theta)), -cour * math.sin(theta))


def symbol(scheme: str, theta: float, cour: float) -> complex:
    if scheme == "blend":
        return symbol_blend(theta, cour)
    if scheme == "upwind":
        return symbol_upwind(theta, cour)
    raise ValueError(f"unknown scheme {scheme!r}")


def max_symbol_magnitude(scheme: str, cour: float, samples: int = 4001) -> float:
    """max over theta in [0, pi] of |G(theta, C)|."""
    theta = np.linspace(0.0, math.pi, samples)
    if scheme == "blend":
        a = 1.0 - cour + 2.0 * cour * cour
        b = cour - 2.0 * cour * cour
        mag2 = (a + b * np.cos(theta)) ** 2 + (cour * np.sin(theta)) ** 2
    elif scheme == "upwind":
        mag2 = (1.0 - cour * (1.0 - np.cos(theta))) ** 2 + (cour * np.sin(theta)) ** 2
    else:
        raise ValueError(f"unknown scheme {scheme!r}")
    return float(np.sqrt(mag2.max()))


def most_unstable_mode(scheme: str, cour: float, samples: int = 200001) -> Tuple[float, float]:
    """(theta, |G|) of the fastest-growing resolved mode; (0, 1) when stable."""
    theta = np.linspace(0.0, math.pi, samples)
    if scheme == "blend":
        a = 1.0 - cour + 2.0 * cour * cour
        b = cour - 2.0 * cour * cour
        mag2 = (a + b * np.cos(theta)) ** 2 + (cour * np.sin(theta)) ** 2
    elif scheme == "upwind":
        mag2 = (1.0 - cour * (1.0 - np.cos(theta))) ** 2 + (cour * np.sin(theta)) ** 2
    else:
        raise ValueError(f"unknown scheme {scheme!r}")
    i = int(np.argmax(mag2))
    return float(theta[i]), float(math.sqrt(mag2[i]))


def roundoff_blowup_steps(scheme: str, cour: float, gain: float = 1e14) -> float:
    """Steps for a 1e-16 round-off seed to reach O(1) under the fastest mode."""
    _, g = most_unstable_mode(scheme, cour)
    if g <= 1.0 + 1e-15:
        return float("inf")
    return math.log(gain) / math.log(g)


def stability_limit(scheme: str, samples: int = 4001) -> float:
    """Largest Courant number keeping max_theta |G| <= 1."""
    if max_symbol_magnitude(scheme, 1.0, samples) <= 1.0 + 1e-14:
        return 1.0
    lo, hi = 1e-6, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if max_symbol_magnitude(scheme, mid, samples) <= 1.0:
            lo = mid
        else:
            hi = mid
    return lo


# ---------------------------------------------------------------------------
# 1D periodic transport (stencil transcribed from VOFTransport.advect)
# ---------------------------------------------------------------------------
def advect_1d(alpha: np.ndarray, cour: float, steps: int, scheme: str = "blend") -> np.ndarray:
    """Advance a periodic 1D field with the library stencil (u > 0, dx = dt = 1)."""
    if scheme not in SCHEMES:
        raise ValueError(f"unknown scheme {scheme!r}")
    a = np.asarray(alpha, dtype=np.float64).copy()
    for _ in range(int(steps)):
        donor = np.roll(a, 1)      # west cell of face j (between cells j-1 and j)
        acceptor = a
        if scheme == "blend":
            af = donor + cour * (acceptor - donor)
        else:
            af = donor.copy()
        pure = (donor <= 0.0) | (donor >= 1.0)
        af = np.where(pure, donor, af)
        af = np.clip(af, 0.0, 1.0)
        a = a - cour * (np.roll(af, -1) - af)   # face j+1 minus face j
        a = np.clip(a, 0.0, 1.0)
    return a


def sine_field(n: int, modes: int = 1, mean: float = 0.5, amp: float = 0.45) -> np.ndarray:
    x = np.arange(n, dtype=np.float64) / n
    return mean + amp * np.sin(2.0 * math.pi * modes * x)


def measure_symbol(
    cour: float, steps: int, scheme: str = "blend", n: int = 256, modes: int = 4
) -> complex:
    """Measure per-step amplification |G| and phase arg G from a single Fourier mode.

    The probe stays strictly inside (0, 1) for every step so neither the
    pure-donor pinning nor the [0, 1] clip is active: the measurement isolates
    the linear stencil.
    """
    field0 = sine_field(n, modes=modes, amp=0.35)
    field = advect_1d(field0, cour, steps, scheme=scheme)
    ratio = np.fft.rfft(field)[modes] / np.fft.rfft(field0)[modes]
    mag = abs(ratio) ** (1.0 / steps)
    phase = np.angle(ratio) / steps
    return complex(mag * math.cos(phase), mag * math.sin(phase))


def theory_diffusion(cour: float, scheme: str = "blend") -> float:
    """D/(u dx) from the modified equation: upwind (1-C)/2, blend (1-3C)/2."""
    if scheme == "upwind":
        return (1.0 - cour) / 2.0
    if scheme == "blend":
        return (1.0 - 3.0 * cour) / 2.0
    raise ValueError(f"unknown scheme {scheme!r}")


def _steps_for(cour: float, theta: float, target_decay: float = 0.3) -> int:
    """Step count giving a measurable but modest mode decay."""
    decay_rate = abs(cour * theta ** 2 * 0.35)  # leading-order |G| decay per step
    steps = int(round(target_decay / max(decay_rate, 1e-12)))
    return max(1, min(steps, 200000))


def diffusion_coefficient(
    cour: float, scheme: str = "blend", n: int = 256, modes: Sequence[int] = (1, 2, 3, 4)
) -> float:
    """Effective numerical diffusion D/(u dx) measured from |G(theta)| as theta -> 0.

    Leading order: |G| = 1 - D k^2 dt + O(k^4), i.e.
    -ln|G| / (C theta^2) = D/(u dx) + O(theta^2); a linear fit in theta^2
    extrapolates the measurement to theta = 0 to remove that truncation bias.

    Valid only where the scheme is stable at *all* resolved wave numbers
    (C <= 1/3 for the blend scheme).  Above that limit intermediate modes grow by
    up to ~12% per step, saturate against the [0, 1] clip and contaminate the
    low-wave-number decay measurement — use :func:`most_unstable_mode` there.
    """
    theta = np.array([2.0 * math.pi * m / n for m in modes])
    decay = []
    for m, th in zip(modes, theta):
        g = measure_symbol(cour, _steps_for(cour, th), scheme=scheme, n=n, modes=m)
        decay.append(-math.log(abs(g)))
    ratios = np.array(decay) / (cour * theta ** 2)
    basis = np.vstack([np.ones_like(theta ** 2), theta ** 2]).T
    coef, *_ = np.linalg.lstsq(basis, ratios, rcond=None)
    return float(coef[0])


def convergence_1d(
    scheme: str = "blend",
    cour: float = 0.25,
    sizes: Iterable[int] = (32, 64, 128, 256),
    final_time: float = 1.0,
    modes: int = 1,
) -> List[Dict[str, float]]:
    """L1 error of one full period of smooth advection at fixed final time (u = 1)."""
    rows: List[Dict[str, float]] = []
    for n in sizes:
        dx = 1.0 / n
        steps = max(1, int(round(final_time / (cour * dx))))
        dt = final_time / steps
        field0 = sine_field(n, modes=modes)
        field = advect_1d(field0, dt / dx, steps, scheme=scheme)
        exact = field0.copy()  # periodic domain, integer number of periods
        diff = field - exact
        rows.append(
            dict(
                n=int(n),
                dx=dx,
                dt=dt,
                steps=steps,
                cour=dt / dx,
                l1=float(np.mean(np.abs(diff))),
                linf=float(np.max(np.abs(diff))),
                volume_drift=float(abs(field.sum() - field0.sum())),
            )
        )
    for prev, cur in zip(rows[:-1], rows[1:]):
        cur["order_l1"] = math.log(prev["l1"] / cur["l1"]) / math.log(prev["dx"] / cur["dx"])
    return rows


# ---------------------------------------------------------------------------
# 2D periodic transport reproducing VOFTransport.advect face by face
# ---------------------------------------------------------------------------
def advect_2d(
    alpha: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    dt: float,
    dx: float,
    dy: float,
    scheme: str = "blend",
) -> np.ndarray:
    """One periodic VOF step with the library stencil.

    ``u`` has shape (ny, nx+1) (east faces) and ``v`` (ny+1, nx) (north faces),
    exactly as :class:`~tensorfvm.two_phase_vof.VOFTransport`.  Boundary faces
    wrap periodically, so the update is conservative up to clipping.
    """
    if scheme not in SCHEMES:
        raise ValueError(f"unknown scheme {scheme!r}")
    ny, nx = alpha.shape
    cell_vol = dx * dy
    ue = np.asarray(u, dtype=np.float64)[:, :nx]   # faces 0..nx-1 (cell j-1 | cell j)
    vn = np.asarray(v, dtype=np.float64)[:ny, :]   # faces 0..ny-1 (cell i-1 | cell i)

    west, east = np.roll(alpha, 1, axis=1), alpha
    pos = ue >= 0.0
    donor_x = np.where(pos, west, east)
    acceptor_x = np.where(pos, east, west)
    cour_x = np.abs(ue) * dt / dx
    af_x = donor_x + cour_x * (acceptor_x - donor_x) if scheme == "blend" else donor_x.copy()
    af_x = np.where((donor_x <= 0.0) | (donor_x >= 1.0), donor_x, af_x)
    af_x = np.clip(af_x, 0.0, 1.0)
    flux_x = ue * af_x * dt / dx

    south, north = np.roll(alpha, 1, axis=0), alpha
    posy = vn >= 0.0
    donor_y = np.where(posy, south, north)
    acceptor_y = np.where(posy, north, south)
    cour_y = np.abs(vn) * dt / dy
    af_y = donor_y + cour_y * (acceptor_y - donor_y) if scheme == "blend" else donor_y.copy()
    af_y = np.where((donor_y <= 0.0) | (donor_y >= 1.0), donor_y, af_y)
    af_y = np.clip(af_y, 0.0, 1.0)
    flux_y = vn * af_y * dt / dy

    out = alpha - ((np.roll(flux_x, -1, axis=1) - flux_x) + (np.roll(flux_y, -1, axis=0) - flux_y))
    _ = cell_vol  # documented for readability; flux scaling above is per unit cell volume
    return np.clip(out, 0.0, 1.0)


def zalesak_initial(n: int) -> np.ndarray:
    """Zalesak (1979) slotted disc on the unit square, cell centred."""
    dx = 1.0 / n
    c = (np.arange(n) + 0.5) * dx
    X, Y = np.meshgrid(c, c)
    inside = (X - 0.5) ** 2 + (Y - 0.75) ** 2 <= 0.15 ** 2
    slot = (np.abs(X - 0.5) <= 0.025) & (Y >= 0.75) & (Y <= 0.85)
    return np.where(inside & ~slot, 1.0, 0.0)


def zalesak_rotation(
    n: int,
    scheme: str = "blend",
    cfl: float = 0.25,
    revolutions: float = 1.0,
) -> Tuple[Dict[str, float], np.ndarray]:
    """Rigid-body rotation of the slotted disc on a periodic unit square.

    u = -omega (y - 0.5), v = omega (x - 0.5) is an exact rigid rotation, so at
    t = T = 2 pi / omega the analytic solution equals the initial field and the
    error norms are pure scheme error.  Volume drift isolates the mass the
    [0, 1] clip adds or removes.
    """
    dx = 1.0 / n
    omega = 1.0  # one revolution per T = 2 pi
    t_end = revolutions * 2.0 * math.pi / omega
    umax = omega * 0.5
    steps = max(1, int(round(t_end / (cfl * dx / umax))))
    dt = t_end / steps

    c = (np.arange(n) + 0.5) * dx
    u = np.repeat((-omega * (c - 0.5))[:, None], n + 1, axis=1)
    v = np.repeat((omega * (c - 0.5))[None, :], n + 1, axis=0)

    alpha0 = zalesak_initial(n)
    alpha = alpha0.copy()
    for _ in range(steps):
        alpha = advect_2d(alpha, u, v, dt, dx, dx, scheme=scheme)

    vol0 = float(alpha0.sum() * dx * dx)
    vol = float(alpha.sum() * dx * dx)
    diff = alpha - alpha0
    metrics = dict(
        n=int(n),
        scheme=scheme,
        cfl=float(cfl),
        steps=int(steps),
        dt=float(dt),
        cour_max=float(umax * dt / dx),
        l1=float(np.abs(diff).sum() * dx * dx),
        linf=float(np.abs(diff).max()),
        volume_drift_rel=abs(vol - vol0) / vol0,
        volume_final=vol,
        alpha_min=float(alpha.min()),
        alpha_max=float(alpha.max()),
        smeared_cells=int(((alpha > 0.05) & (alpha < 0.95)).sum()),
        initial_smeared_cells=int(((alpha0 > 0.05) & (alpha0 < 0.95)).sum()),
    )
    return metrics, alpha


def volume_history(
    n: int = 128,
    scheme: str = "blend",
    cfl: float = 0.5,
    checkpoints: int = 24,
) -> Dict[str, List[float]]:
    """Relative liquid volume vs revolution fraction for the Zalesak rotation.

    Shows *when* the limiter starts manufacturing mass: the linear instability of
    the default scheme feeds modes that the [0, 1] clip then rectifies into net
    volume gain.
    """
    dx = 1.0 / n
    omega = 1.0
    t_end = 2.0 * math.pi / omega
    steps = max(1, int(round(t_end / (cfl * dx / (omega * 0.5)))))
    dt = t_end / steps
    c = (np.arange(n) + 0.5) * dx
    u = np.repeat((-omega * (c - 0.5))[:, None], n + 1, axis=1)
    v = np.repeat((omega * (c - 0.5))[None, :], n + 1, axis=0)
    alpha = zalesak_initial(n)
    vol0 = float(alpha.sum() * dx * dx)
    every = max(1, steps // checkpoints)
    times, drift = [0.0], [0.0]
    for step in range(1, steps + 1):
        alpha = advect_2d(alpha, u, v, dt, dx, dx, scheme=scheme)
        if step % every == 0 or step == steps:
            times.append(step / steps)
            drift.append((float(alpha.sum() * dx * dx) - vol0) / vol0)
    return dict(time=times, drift=drift, steps=[steps], n=[n])


# ---------------------------------------------------------------------------
# Study driver
# ---------------------------------------------------------------------------
def run_study(output, quick: bool = False) -> Dict[str, object]:
    """Execute every verification case; write results.json, report.md, figures."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    cour_values = (0.1, 0.2, 0.25, THIRD, 0.4, 0.5, 0.8, 1.0)
    symbol_rows: List[Dict[str, float]] = []
    for scheme in SCHEMES:
        for cour in cour_values:
            n, modes = 256, 4
            theta = 2.0 * math.pi * modes / n
            steps = max(20, min(400, int(round(1.5 / max(cour, 1e-3)))))
            measured = measure_symbol(cour, steps, scheme=scheme, n=n, modes=modes)
            ref = symbol(scheme, theta, cour)
            symbol_rows.append(
                dict(
                    scheme=scheme,
                    cour=float(cour),
                    theta=theta,
                    steps=int(steps),
                    measured_abs_G=abs(measured),
                    theory_abs_G=abs(ref),
                    measured_phase=math.atan2(measured.imag, measured.real),
                    theory_phase=math.atan2(ref.imag, ref.real),
                    rel_err_abs=abs(abs(measured) - abs(ref)) / abs(ref),
                    rel_err_complex=abs(measured - ref) / abs(ref),
                    max_abs_G=max_symbol_magnitude(scheme, cour),
                )
            )

    stability = {s: stability_limit(s) for s in SCHEMES}
    instability_rows = []
    for scheme in SCHEMES:
        for cour in (THIRD, 0.4, 0.5, 0.8, 1.0):
            theta_max, g_max = most_unstable_mode(scheme, cour)
            instability_rows.append(
                dict(
                    scheme=scheme,
                    cour=float(cour),
                    theta_max_over_pi=theta_max / math.pi,
                    max_abs_G=g_max,
                    growth_per_step=g_max - 1.0,
                    roundoff_blowup_steps=roundoff_blowup_steps(scheme, cour),
                )
            )

    diffusion_rows: List[Dict[str, float]] = []
    for scheme in SCHEMES:
        for cour in (0.05, 0.1, 0.2, 0.25, 0.3):
            measured = diffusion_coefficient(cour, scheme=scheme)
            theory = theory_diffusion(cour, scheme=scheme)
            diffusion_rows.append(
                dict(
                    scheme=scheme,
                    cour=float(cour),
                    measured_D_over_udx=measured,
                    theory_D_over_udx=theory,
                    rel_err=abs(measured - theory) / abs(theory),
                )
            )

    convergence = {s: convergence_1d(s, cour=0.25) for s in SCHEMES}
    probe = sine_field(64, modes=2)
    exact_shift = advect_1d(probe, 1.0, 64, scheme="upwind")
    exact_shift_err = float(np.max(np.abs(exact_shift - np.roll(probe, 64))))

    sizes = (48, 96) if quick else (64, 128, 256)
    zalesak: Dict[str, List[Dict[str, float]]] = {}
    fields: Dict[str, np.ndarray] = {}
    for scheme in SCHEMES:
        for cfl in (0.25, 0.5):
            rows = []
            field = None
            for n in sizes:
                metrics, field = zalesak_rotation(n, scheme=scheme, cfl=cfl)
                rows.append(metrics)
            for prev, cur in zip(rows[:-1], rows[1:]):
                cur["order_l1"] = math.log(prev["l1"] / cur["l1"]) / math.log(cur["n"] / prev["n"])
            key = f"{scheme}-cfl{cfl}"
            zalesak[key] = rows
            fields[key] = field

    history = {}
    for scheme in SCHEMES:
        history[scheme] = volume_history(n=96 if quick else 128, scheme=scheme, cfl=0.5)

    results: Dict[str, object] = dict(
        schema="tensorfvm.vof-transport-audit/1",
        module="tensorfvm.two_phase_vof.VOFTransport",
        stencil="f_face = f_donor + C*(f_acceptor - f_donor)",
        schemes=list(SCHEMES),
        stability_limit=stability,
        instability=instability_rows,
        solver_default_vof_cfl_limit=0.5,
        symbol=symbol_rows,
        diffusion=diffusion_rows,
        convergence_1d=convergence,
        upwind_exact_at_courant_one=exact_shift_err,
        zalesak=zalesak,
        volume_history_cfl05=history,
    )
    results["criteria"] = _criteria(results)
    try:
        _figures(results, fields, output)
        results["figures_written"] = True
    except Exception as error:  # pragma: no cover
        results["figures_written"] = False
        results["figures_error"] = f"{type(error).__name__}: {error}"
    (output / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (output / "report.md").write_text(_markdown(results), encoding="utf-8")
    return results


def _criteria(r: Dict[str, object]) -> Dict[str, object]:
    """Machine-checkable acceptance criteria and their verdicts."""
    worst_symbol = max(row["rel_err_complex"] for row in r["symbol"])
    upd = [row for row in r["diffusion"] if row["scheme"] == "upwind"]
    bld = [row for row in r["diffusion"] if row["scheme"] == "blend"]
    worst_up = max(row["rel_err"] for row in upd)
    worst_bld = max(row["rel_err"] for row in bld)
    conv = r["convergence_1d"]
    up_orders = [row.get("order_l1", 0.0) for row in conv["upwind"][1:]]
    bl_orders = [row.get("order_l1", 0.0) for row in conv["blend"][1:]]
    zup = {k: v for k, v in r["zalesak"].items() if k.startswith("upwind")}
    return dict(
        C1_stencil_identity=dict(
            criterion="measured per-step symbol vs closed form, |dG|/|G| < 1e-10",
            value=worst_symbol,
            passed=bool(worst_symbol < 1e-10),
        ),
        C2_blend_stability_limit=dict(
            criterion="von Neumann limit of the blend scheme equals 1/3 (tol 2e-3)",
            value=r["stability_limit"]["blend"],
            passed=bool(abs(r["stability_limit"]["blend"] - THIRD) < 2e-3),
        ),
        C3_upwind_stable_to_one=dict(
            criterion="upwind max_theta |G| <= 1 at C = 1",
            value=max_symbol_magnitude("upwind", 1.0),
            passed=bool(max_symbol_magnitude("upwind", 1.0) <= 1.0 + 1e-12),
        ),
        C4_upwind_diffusion=dict(
            criterion="measured D_upwind vs (u dx/2)(1-C) relative error < 3%",
            value=worst_up,
            passed=bool(worst_up < 0.03),
        ),
        C5_blend_diffusion=dict(
            criterion="measured D_blend vs (u dx/2)(1-3C) relative error < 3%",
            value=worst_bld,
            passed=bool(worst_bld < 0.03),
        ),
        C6_first_order=dict(
            criterion="smooth-field 1D L1 convergence order of both schemes in [0.8, 1.2]",
            value=dict(upwind=up_orders, blend=bl_orders),
            passed=bool(all(0.8 <= o <= 1.2 for o in up_orders + bl_orders)),
        ),
        C7_upwind_exact_shift=dict(
            criterion="upwind at C = 1 is exact translation (max error < 1e-12)",
            value=r["upwind_exact_at_courant_one"],
            passed=bool(r["upwind_exact_at_courant_one"] < 1e-12),
        ),
        C8_zalesak_upwind_volume=dict(
            criterion="upwind Zalesak relative volume drift < 1e-12",
            value={k: max(row["volume_drift_rel"] for row in v) for k, v in zup.items()},
            passed=bool(all(max(row["volume_drift_rel"] for row in v) < 1e-12 for v in zup.values())),
        ),
        C9_zalesak_error_monotone=dict(
            criterion="Zalesak L1 error decreases monotonically with n at CFL 0.25 "
                      "(discontinuous data, so order is not required to be 1)",
            value={k: [row["l1"] for row in v] for k, v in r["zalesak"].items()
                   if k.endswith("cfl0.25")},
            passed=bool(all(all(a > b for a, b in zip([row["l1"] for row in v],
                                                      [row["l1"] for row in v][1:]))
                            for k, v in r["zalesak"].items() if k.endswith("cfl0.25"))),
        ),
        C10_blend_cfl05_mass_defect=dict(
            criterion="DEFECT REPRODUCTION: default scheme at CFL 0.5 corrupts mass "
                      "(relative volume drift > 0.1) while upwind stays < 1e-12. "
                      "Invert this criterion if the default scheme changes.",
            value={k: max(row["volume_drift_rel"] for row in v)
                   for k, v in r["zalesak"].items() if k.endswith("cfl0.5")},
            passed=bool(max(row["volume_drift_rel"] for row in r["zalesak"]["blend-cfl0.5"]) > 0.1
                        and all(row["volume_drift_rel"] < 1e-12
                                for row in r["zalesak"]["upwind-cfl0.5"])),
        ),
    )


def _figures(results: Dict[str, object], fields: Dict[str, np.ndarray], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "savefig.dpi": 200})

    theta = np.linspace(0.0, math.pi, 401)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4))
    for ax, scheme, title in (
        (axes[0], "blend", "library blend (default scheme)"),
        (axes[1], "upwind", "donor-cell upwind (scheme='upwind')"),
    ):
        for cour in (0.1, 0.25, THIRD, 0.5, 0.8):
            ax.plot(theta / math.pi, [abs(symbol(scheme, t, cour)) for t in theta],
                    lw=1.2, label=f"C={cour:.3f}")
        ax.plot([r["theta"] / math.pi for r in results["symbol"] if r["scheme"] == scheme],
                [r["measured_abs_G"] for r in results["symbol"] if r["scheme"] == scheme],
                "k.", ms=6, label="measured")
        ax.axhline(1.0, color="k", lw=0.8, ls="--")
        ax.set_xlabel(r"$\theta/\pi$")
        ax.set_ylabel(r"$|G(\theta,C)|$")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6.5)
    axes[0].annotate("unstable for C > 1/3", xy=(0.55, 1.9), fontsize=8, color="crimson")
    fig.tight_layout()
    fig.savefig(output / "symbol-magnitude.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4))
    for ax, scheme, title in (
        (axes[0], "blend", "library blend (default scheme)"),
        (axes[1], "upwind", "donor-cell upwind (scheme='upwind')"),
    ):
        rows = [r for r in results["diffusion"] if r["scheme"] == scheme]
        c = [r["cour"] for r in rows]
        ax.plot([0, 1], [theory_diffusion(x, scheme) for x in (0, 1)], "k-", lw=1.0,
                label="modified-equation theory")
        ax.plot(c, [r["measured_D_over_udx"] for r in rows], "o", ms=4.5,
                label=r"measured from $|G|(\theta\to0)$")
        ax.axhline(0.0, color="crimson", lw=0.8, ls=":")
        ax.set_xlabel("Courant number C")
        ax.set_ylabel(r"$D/(u\,\Delta x)$")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "numerical-diffusion.png", bbox_inches="tight")
    plt.close(fig)

    n_max = max(row["n"] for rows in results["zalesak"].values() for row in rows)
    keys = ("blend-cfl0.25", "blend-cfl0.5", "upwind-cfl0.25")
    panels = [(zalesak_initial(n_max), "exact: t = 0 = T")] + [
        (fields[k], f"{k} (n={n_max})") for k in keys if k in fields
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.0 * len(panels), 3.1))
    for ax, (field, title) in zip(np.atleast_1d(axes), panels):
        im = ax.imshow(field, origin="lower", extent=(0, 1, 0, 1), vmin=0.0, vmax=1.0, cmap="viridis")
        ax.set_title(title, fontsize=7.5)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=list(np.atleast_1d(axes)), shrink=0.85, label=r"$\alpha$")
    fig.savefig(output / "zalesak-fields.png", bbox_inches="tight")
    plt.close(fig)

    hist = results["volume_history_cfl05"]
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for scheme, data in hist.items():
        ax.semilogy(data["time"], np.maximum(np.abs(data["drift"]), 1e-18), "o-", ms=3,
                    label=f"scheme={scheme}")
    ax.set_xlabel("revolution fraction  t/T")
    ax.set_ylabel("|relative liquid volume drift|")
    ax.set_title(f"Zalesak rotation at CFL 0.5 (n={hist['blend']['n'][0]})", fontsize=9)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "mass-drift-history.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4))
    for key, rows in results["zalesak"].items():
        axes[0].loglog([r["n"] for r in rows], [r["l1"] for r in rows], "o-", ms=4, label=key)
        axes[1].semilogy([r["n"] for r in rows],
                         [max(r["volume_drift_rel"], 1e-18) for r in rows], "o-", ms=4, label=key)
    axes[0].set_ylabel(r"$L_1$ error after one revolution")
    axes[1].set_ylabel("relative liquid-volume drift")
    for ax, title in zip(axes, ("Zalesak disc: shape error", "Zalesak disc: mass conservation")):
        ax.set_xlabel("cells per side n")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=6.5)
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output / "zalesak-convergence.png", bbox_inches="tight")
    plt.close(fig)


def _pct(x: float) -> str:
    return f"{100.0 * x:.3f}%"


def _markdown(r: Dict[str, object]) -> str:
    L: List[str] = []
    A = L.append
    A("# VOF transport stencil: verification audit (von Neumann + exact benchmarks)")
    A("")
    A("Verification-debt audit of `tensorfvm.two_phase_vof.VOFTransport`, generated by "
      "`python -m tensorfvm.verification.vof_transport_audit`.")
    A("")
    A("## 1. Case description")
    A("")
    A("`VOFTransport.advect` blends donor and acceptor face values with the Courant "
      "number as weight: `f = f_donor + C (f_acceptor - f_donor)`. Its docstring claims a "
      "\"donor-cell (CICSAM-like)\" scheme that is stable and bounded for CFL <= 1. "
      "This audit checks that claim at three levels: (i) the discrete symbol of the "
      "implemented stencil, (ii) the modified-equation diffusion coefficient against "
      "textbook theory, (iii) two periodic benchmarks with exact solutions "
      "(one full period of smooth advection; Zalesak slotted-disc rigid-body rotation "
      "on the unit square). All runs are float64, CPU, single rank.")
    A("")
    A("## 2. Method")
    A("")
    A("| item | setting |")
    A("|---|---|")
    A("| 1D probe grid | n = 256, Fourier mode m = 4 (theta = pi/32), amplitude 0.35 about 0.5 |")
    A("| symbol measurement | per-step amplification from FFT of a single mode, |G| = (|X^N|/|X^0|)^(1/N) |")
    A("| diffusion | D = -ln|G|/(C theta^2), least-squares extrapolation in theta^2 -> 0 |")
    A("| 1D convergence | 1 period of u = 1 advection, C = 0.25, n = 32/64/128/256 |")
    A("| Zalesak | unit square, r = 0.15 at (0.5, 0.75), slot 0.05 x 0.10, omega = 1, T = 2 pi |")
    A("| Zalesak grids | n = 64 / 128 / 256, CFL = 0.25 (and 0.5 for the default scheme) |")
    A("| periodic driver | transcribes `VOFTransport.advect` face by face (donor/acceptor, "
      "pure pinning, [0,1] clip); equivalence to the library class is a pytest gate |")
    A("")
    A("## 3. Results")
    A("")
    A("### 3.1 Closed-form and measured symbols")
    A("")
    A("`G_blend = (1-C+2C^2) + (C-2C^2) cos(theta) - i C sin(theta)`, "
      "`G_upwind = 1 - C(1-cos(theta)) - i C sin(theta)`.")
    A("")
    A("| scheme | C | measured \\|G\\| | theory \\|G\\| | rel err (complex) | max_theta \\|G\\| |")
    A("|---|---|---|---|---|---|")
    for row in r["symbol"]:
        A(f"| {row['scheme']} | {row['cour']:.4f} | {row['measured_abs_G']:.6e} | "
          f"{row['theory_abs_G']:.6e} | {row['rel_err_complex']:.2e} | {row['max_abs_G']:.6f} |")
    A("")
    stab = r["stability_limit"]
    A(f"Measured von Neumann stability limit: **blend = {stab['blend']:.5f}** "
      f"(= 1/3), **upwind = {stab['upwind']:.5f}**. "
      f"`TwoPhaseSolver.vof_cfl_limit` defaults to {r['solver_default_vof_cfl_limit']}, "
      "which lies inside the unstable band of the default scheme.")
    A("")
    A("### 3.2 Growth of the most unstable mode (severity of the instability)")
    A("")
    A("| scheme | C | theta_max/pi | max \\|G\\| | growth per step | steps for 1e-16 seed to reach O(1) |")
    A("|---|---|---|---|---|---|")
    for row in r["instability"]:
        steps = row["roundoff_blowup_steps"]
        steps_text = "inf" if not math.isfinite(steps) else f"{steps:.0f}"
        A(f"| {row['scheme']} | {row['cour']:.4f} | {row['theta_max_over_pi']:.4f} | "
          f"{row['max_abs_G']:.6f} | {row['growth_per_step']:+.3e} | {steps_text} |")
    A("")
    A("### 3.2b Effective numerical diffusion")
    A("")
    A("| scheme | C | measured D/(u dx) | theory D/(u dx) | rel err |")
    A("|---|---|---|---|---|")
    for row in r["diffusion"]:
        A(f"| {row['scheme']} | {row['cour']:.2f} | {row['measured_D_over_udx']:.6f} | "
          f"{row['theory_D_over_udx']:.6f} | {_pct(row['rel_err'])} |")
    A("")
    A("### 3.3 1D convergence (one full period, C = 0.25)")
    A("")
    A("| scheme | n | L1 | Linf | order |")
    A("|---|---|---|---|---|")
    for scheme, rows in r["convergence_1d"].items():
        for row in rows:
            order = f"{row['order_l1']:.3f}" if "order_l1" in row else "-"
            A(f"| {scheme} | {row['n']} | {row['l1']:.6e} | {row['linf']:.6e} | {order} |")
    A("")
    A(f"Upwind at C = 1 is an exact translation: max error "
      f"{r['upwind_exact_at_courant_one']:.3e}.")
    A("")
    A("### 3.4 Zalesak slotted disc, one revolution")
    A("")
    A("| case | n | L1 | Linf | order | volume drift | smeared cells (initial) | alpha range |")
    A("|---|---|---|---|---|---|---|---|")
    for key, rows in r["zalesak"].items():
        for row in rows:
            order = f"{row['order_l1']:.3f}" if "order_l1" in row else "-"
            A(f"| {key} | {row['n']} | {row['l1']:.5e} | {row['linf']:.5f} | {order} | "
              f"{row['volume_drift_rel']:.3e} | {row['smeared_cells']} ({row['initial_smeared_cells']}) | "
              f"[{row['alpha_min']:.3f}, {row['alpha_max']:.3f}] |")
    A("")
    A("> Note: pairs with `|l1|` still defaulting to zero for the coarsest grid have "
      "`order` computed only for n >= 96 rows.")
    A("")
    A("### 3.5 Mass history at CFL 0.5 (Zalesak rotation)")
    A("")
    A("| t/T | blend drift | upwind drift |")
    A("|---|---|---|")
    hb = r["volume_history_cfl05"]["blend"]
    hu = r["volume_history_cfl05"]["upwind"]
    for t, db, du in zip(hb["time"], hb["drift"], hu["drift"]):
        A(f"| {t:.3f} | {db:.3e} | {du:.3e} |")
    A("")
    A("## 4. Figures")
    A("")
    A("- `symbol-magnitude.png` - |G| vs theta for both schemes with measured points.")
    A("- `numerical-diffusion.png` - measured vs modified-equation D/(u dx).")
    A("- `zalesak-fields.png` - alpha field after one revolution, exact vs both schemes.")
    A("- `zalesak-convergence.png` - L1 error and volume drift vs grid.")
    A("- `mass-drift-history.png` - liquid-volume drift vs time at CFL 0.5.")
    A("")
    A("## 5. Acceptance criteria and error verdict")
    A("")
    A("| id | criterion | measured | < 3% / verdict |")
    A("|---|---|---|---|")
    for key, item in r["criteria"].items():
        value = item["value"]
        if isinstance(value, float):
            text = f"{value:.3e}"
        else:
            text = json.dumps(value, default=float)
        if len(text) > 120:
            text = text[:117] + "..."
        A(f"| {key} | {item['criterion']} | {text} | {'PASS' if item['passed'] else 'FAIL'} |")
    A("")
    n_fail = sum(1 for item in r["criteria"].values() if not item["passed"])
    A(f"**Verdict: {len(r['criteria']) - n_fail}/{len(r['criteria'])} criteria pass.**")
    A("")
    A("## 6. Notes, limitations, applicability")
    A("")
    A("1. The default (`blend`) scheme is *not* donor-cell: with C as the blend weight it is "
      "pure upwind at C = 0, central differencing at C = 0.5 and pure downwind at C = 1. "
      "Its stability limit is C = 1/3, so the documented \"bounded for CFL <= 1\" is wrong "
      "for smooth fields.")
    A("2. The [0, 1] clip plus pure-donor pinning act as a hard limiter: they keep the "
      "solution in range on sharp interfaces (which is why the existing interface tests "
      "pass) but they are exactly the mechanism that breaks geometric conservation when "
      "the linear growth would exceed the bounds.")
    A("3. `TwoPhaseSolver.vof_cfl_limit = 0.5` sits inside the unstable band; either use "
      "`scheme='upwind'` (added here, opt-in, default unchanged) or cap the VOF Courant "
      "number at <= 1/3.")
    A("4. This audit says nothing about 3D or about the two-phase coupling; it bounds the "
      "transport operator only. Interface reconstruction (PLIC) remains unimplemented.")
    A("")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quick", action="store_true", help="smaller Zalesak grids")
    args = parser.parse_args(argv)
    results = run_study(args.output, quick=args.quick)
    n_fail = sum(1 for item in results["criteria"].values() if not item["passed"])
    print(f"wrote {args.output}/results.json, report.md, figures")
    print(f"criteria: {len(results['criteria']) - n_fail}/{len(results['criteria'])} passed")
    for key, item in results["criteria"].items():
        print(f"  {key}: {'PASS' if item['passed'] else 'FAIL'} ({item['criterion']})")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
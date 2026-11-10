"""Exact-advection benchmark cases and transport wrappers for VOF verification.

Three analytic anchors are provided.  All of them are *exact-advection* tests: the
velocity field is prescribed, divergence-free and smoothly reversible, so the
analytic solution is known in closed form.

``zalesak`` -- solid-body rotation of a slotted disk (Zalesak 1979)
    ``u = -omega (y - c_y)``, ``v = omega (x - c_x)`` about the domain centre
    with ``omega = 2 pi`` (one revolution per unit time).  After one complete
    revolution the analytic solution *is* the initial field, so both the shape
    error and the volume drift are measured against a closed-form reference.
    This is the standard shape-preservation test: it exercises a rotating
    interface with a thin slot whose corners the scheme must not round off.

``single_vortex`` -- Rider & Kothe (1995) vortex in a shear deformation
    ``u = -sin^2(pi x) sin(2 pi y) g(t)``,
    ``v =  sin(2 pi x) sin^2(pi y) g(t)``, ``g(t) = cos(pi t / T)``, ``T = 8``,
    on ``[0,1]^2``.  The field stretches the disk into a thin filament and then
    reverses it; at ``t = T`` the velocity is zero again and the analytic
    solution equals the initial field.  The deformation is far stronger than
    rigid rotation, so the recovered shape measures numerical diffusion and
    the interface reconstruction quality, not just boundedness.

``double_vortex`` -- two counter-rotating cells on a 2:1 box
    Stream function ``psi = (1/pi) sin(pi x) sin(2 pi y)`` on ``[0,2] x [0,1]``,
    giving ``u = -2 sin(pi x) cos(2 pi y) g(t)``,
    ``v = cos(pi x) sin(2 pi y) g(t)``; vorticity ``-5 pi sin(pi x) sin(2 pi y)``
    changes sign across ``y = 1/2``, i.e. two counter-rotating cells.  The field
    is exactly divergence-free and exactly reversible, and its normal component
    vanishes on all four boundaries, so a wall boundary condition is *exact* for
    it (no periodicity assumption is needed).

Every field is sampled on the staggered MAC faces, and the discrete divergence
of the sampled field is reported as an audit number (it is exactly zero up to
round-off because each field is a sum of separable products).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from ..vof_phase0.advection import AlgebraicVOF
from ..vof_plic import advect_plic

__all__ = [
    "Case",
    "zalesak_case",
    "single_vortex_case",
    "double_vortex_case",
    "sample_faces",
    "face_divergence",
    "advance",
]


@dataclass
class Case:
    """A fully specified exact-advection benchmark."""

    name: str
    lx: float
    ly: float
    duration: float
    speed_scale: float
    velocity: object            # callable (x, y, t) -> (u, v), vectorised
    exact_initial: object       # callable (nx, ny) -> alpha tensor
    stream: object = None       # callable (x, y, t) -> psi, vertex-centred
    note: str = ""
    metadata: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# velocity fields
# --------------------------------------------------------------------------- #
def _rigid_rotation(lx: float, ly: float, omega: float = 2.0 * math.pi):
    # rotation about the *domain* centre (the classical Zalesak setup): the
    # slotted disk at (0.5, 0.75) orbits it at radius 0.25 and stays at least
    # 0.1 away from every wall, so the wall treatment never touches the interface
    cx, cy = 0.5 * lx, 0.5 * ly

    def velocity(x: torch.Tensor, y: torch.Tensor, t: float, t1=None):
        return -omega * (y - cy), omega * (x - cx)

    return velocity


def _modulation_mean(t0: float, t1: float, T: float) -> float:
    """Exact step-average of the reversal modulation ``g(t) = cos(pi t / T)``.

    Sampling ``g`` pointwise at ``t_k`` makes the discrete time integral
    ``sum_k g(t_k) dt`` differ from ``int_0^T g dt`` by an O(dt) amount, which in
    a *reversal* benchmark is not a small perturbation: the recovered shape then
    sits at the deformation state reached one step short, i.e. displaced by
    ``dt * |U| = CFL * h``.  That error dominates the measured shape error and
    masks the transport scheme's real accuracy.  Averaging the modulation
    exactly over the step removes it (documented deviation from the naive
    pointwise sampling, verified in ``test_vof_two_phase_verification``).
    """
    if t1 <= t0:
        return math.cos(math.pi * t0 / T)
    return (T / (math.pi * (t1 - t0))) * (math.sin(math.pi * t1 / T)
                                         - math.sin(math.pi * t0 / T))


def _rigid_rotation_stream(lx: float, ly: float,
                           omega: float = 2.0 * math.pi):
    cx, cy = 0.5 * lx, 0.5 * ly

    def stream(x: torch.Tensor, y: torch.Tensor, t: float, t1=None):
        return 0.5 * omega * ((x - cx) ** 2 + (y - cy) ** 2)

    return stream


def _single_vortex_stream(T: float = 8.0):
    def stream(x: torch.Tensor, y: torch.Tensor, t: float, t1=None):
        g = (_modulation_mean(t, t1, T) if t1 is not None
             else math.cos(math.pi * t / T))
        pi = math.pi
        return g * (-torch.sin(pi * x) ** 2 * torch.cos(2.0 * pi * y) / (2.0 * pi)
                    - torch.cos(2.0 * pi * x) / (4.0 * pi))

    return stream


def _double_vortex_stream(T: float = 8.0):
    def stream(x: torch.Tensor, y: torch.Tensor, t: float, t1=None):
        g = (_modulation_mean(t, t1, T) if t1 is not None
             else math.cos(math.pi * t / T))
        pi = math.pi
        return g * torch.sin(pi * x) * torch.sin(2.0 * pi * y) / pi

    return stream


def _single_vortex(T: float = 8.0):
    def velocity(x: torch.Tensor, y: torch.Tensor, t: float, t1=None):
        g = (_modulation_mean(t, t1, T) if t1 is not None
             else math.cos(math.pi * t / T))
        pi = math.pi
        u = -torch.sin(pi * x) ** 2 * torch.sin(2.0 * pi * y) * g
        v = torch.sin(2.0 * pi * x) * torch.sin(pi * y) ** 2 * g
        return u, v

    return velocity


def _double_vortex(T: float = 8.0):
    def velocity(x: torch.Tensor, y: torch.Tensor, t: float, t1=None):
        g = (_modulation_mean(t, t1, T) if t1 is not None
             else math.cos(math.pi * t / T))
        pi = math.pi
        u = -2.0 * torch.sin(pi * x) * torch.cos(2.0 * pi * y) * g
        v = torch.cos(pi * x) * torch.sin(2.0 * pi * y) * g
        return u, v

    return velocity


# --------------------------------------------------------------------------- #
# case constructors
# --------------------------------------------------------------------------- #
def zalesak_case(radius: float = 0.15, slot_half_width: float = 0.025) -> Case:
    from .geometry import zalesak_alpha

    def exact(nx: int, ny: int):
        return zalesak_alpha(nx, ny, 1.0, 1.0, (0.5, 0.75), radius,
                             slot_half_width)

    return Case(
        name="zalesak",
        lx=1.0, ly=1.0, duration=1.0, speed_scale=2.0 * math.pi * 0.5,
        velocity=_rigid_rotation(1.0, 1.0),
        exact_initial=exact,
        stream=_rigid_rotation_stream(1.0, 1.0),
        note="solid-body rotation, one revolution == analytic initial field",
        metadata={"radius": radius, "slot_half_width": slot_half_width,
                  "centre": [0.5, 0.75], "omega": 2.0 * math.pi},
    )


def single_vortex_case(T: float = 8.0, radius: float = 0.15) -> Case:
    from .geometry import circle_alpha

    def exact(nx: int, ny: int):
        return circle_alpha(nx, ny, 1.0, 1.0, (0.5, 0.75), radius)

    return Case(
        name="single_vortex",
        lx=1.0, ly=1.0, duration=T, speed_scale=1.0,
        velocity=_single_vortex(T),
        exact_initial=exact,
        stream=_single_vortex_stream(T),
        note="Rider-Kothe 1995 vortex; at t=T the analytic solution is the "
             "initial field",
        metadata={"T": T, "radius": radius, "centre": [0.5, 0.75]},
    )


def double_vortex_case(T: float = 8.0, radius: float = 0.15) -> Case:
    from .geometry import circle_alpha

    def exact(nx: int, ny: int):
        return circle_alpha(nx, ny, 2.0, 1.0, (0.5, 0.75), radius)

    return Case(
        name="double_vortex",
        lx=2.0, ly=1.0, duration=T, speed_scale=2.0,
        velocity=_double_vortex(T),
        exact_initial=exact,
        stream=_double_vortex_stream(T),
        note="two counter-rotating cells (psi = sin(pi x) sin(2 pi y)/pi) on a "
             "2:1 box; at t=T the analytic solution is the initial field",
        metadata={"T": T, "radius": radius, "centre": [0.5, 0.75],
                  "stream_function": "sin(pi x) sin(2 pi y)/pi"},
    )


# --------------------------------------------------------------------------- #
# sampling and transport
# --------------------------------------------------------------------------- #
def sample_faces(case: Case, nx: int, ny: int, t: float, t1: float | None = None):
    """Sample the case velocity on the MAC faces ``u(ny, nx+1)``/``v(ny+1, nx)``.

    ``t1`` (optional) requests the *step-averaged* velocity over ``[t, t1]``: the
    prescribed modulation factor is integrated exactly (see
    :func:`_modulation_mean`), the spatial field is sampled once.
    """
    dx, dy = case.lx / nx, case.ly / ny
    if case.stream is not None:
        # Vertex-centred stream function, differentiated by exact differences:
        # the discrete MAC divergence then vanishes to round-off by telescoping
        # (an analytically sampled field leaves an O(h^2) divergence, which the
        # dimensionally split VOF sweeps feel as a spurious volume source).
        xv = torch.arange(nx + 1, dtype=torch.float64) * dx
        yv = torch.arange(ny + 1, dtype=torch.float64) * dy
        psi = case.stream(xv[None, :].expand(ny + 1, nx + 1).contiguous(),
                          yv[:, None].expand(ny + 1, nx + 1).contiguous(), t, t1)
        u = -(psi[1:, :] - psi[:-1, :]) / dy
        v = (psi[:, 1:] - psi[:, :-1]) / dx
        return u.contiguous(), v.contiguous()
    xf = torch.arange(nx + 1, dtype=torch.float64) * dx          # u-face x
    yc = (torch.arange(ny, dtype=torch.float64) + 0.5) * dy      # u-face y
    u, _ = case.velocity(xf[None, :].expand(ny, nx + 1).contiguous(),
                         yc[:, None].expand(ny, nx + 1).contiguous(), t, t1)
    xc = (torch.arange(nx, dtype=torch.float64) + 0.5) * dx      # v-face x
    yf = torch.arange(ny + 1, dtype=torch.float64) * dy          # v-face y
    _, v = case.velocity(xc[None, :].expand(ny + 1, nx).contiguous(),
                         yf[:, None].expand(ny + 1, nx).contiguous(), t, t1)
    return u.contiguous(), v.contiguous()


def face_divergence(u: torch.Tensor, v: torch.Tensor, dx: float,
                    dy: float) -> float:
    """``max |div u|`` of a MAC face field (round-off level for these cases)."""
    div = (u[:, 1:] - u[:, :-1]) / dx + (v[1:, :] - v[:-1, :]) / dy
    return float(div.abs().max().item())


@dataclass
class Transport:
    """Dimensionally split transport wrapper used by the benchmark driver."""

    scheme: str = "plic"        # "plic" | "fct"
    cfl: float = 0.5
    sweep: str = "xy"

    def build(self, nx: int, ny: int, dx: float, dy: float):
        if self.scheme == "fct":
            return AlgebraicVOF(nx, ny, dx, dy, boundary="wall", bounds="global")
        return None

    def step(self, state, alpha: torch.Tensor, u: torch.Tensor, v: torch.Tensor,
             dt: float, dx: float, dy: float) -> torch.Tensor:
        order = self.sweep
        self.sweep = "yx" if self.sweep == "xy" else "xy"   # alternate
        if self.scheme == "plic":
            return advect_plic(alpha, u, v, dt, dx, dy, order=order, clip=True,
                               conservative_bounds=True)
        if self.scheme == "fct":
            return state.advect(alpha, u, v, dt, order=order)
        if self.scheme == "donor":
            return _donor_step(alpha, u, v, dt, dx, dy, order)
        raise ValueError(f"unknown scheme {self.scheme!r}")


def _donor_step(alpha: torch.Tensor, u: torch.Tensor, v: torch.Tensor, dt: float,
                dx: float, dy: float, order: str) -> torch.Tensor:
    """First-order donor-cell (upwind) transport -- the convergence baseline."""
    def sweep_x(a: torch.Tensor) -> torch.Tensor:
        ext = torch.cat([a[:, :1], a, a[:, -1:]], dim=1)     # wall: replicate
        left, right = ext[:, :-1], ext[:, 1:]
        flux = torch.where(u > 0.0, u * left, u * right)
        flux = flux.clone()
        flux[:, 0] = 0.0
        flux[:, -1] = 0.0
        return a - dt / dx * (flux[:, 1:] - flux[:, :-1])

    def sweep_y(a: torch.Tensor) -> torch.Tensor:
        ext = torch.cat([a[:1, :], a, a[-1:, :]], dim=0)
        low, high = ext[:-1, :], ext[1:, :]
        flux = torch.where(v > 0.0, v * low, v * high)
        flux = flux.clone()
        flux[0, :] = 0.0
        flux[-1, :] = 0.0
        return a - dt / dy * (flux[1:, :] - flux[:-1, :])

    out = sweep_x(alpha) if order == "xy" else sweep_y(alpha)
    out = sweep_y(out) if order == "xy" else sweep_x(out)
    return out.clamp(0.0, 1.0)


def advance(case: Case, nx: int, ny: int, transport: Transport,
            max_steps: int = 100000) -> dict:
    """Run one benchmark to ``case.duration`` and return the final field + audit.

    The time step is chosen from the initial (largest) speed and the directional
    CFL limit, and is held fixed so that the temporal splitting error scales
    with ``h`` exactly like the spatial error (dt = CFL * h / u_max).  A fixed
    ``dt`` also makes the two grids comparable: the errors are then driven by
    the grid, not by a changing Courant number.
    """
    from .metrics import field_metrics

    dx, dy = case.lx / nx, case.ly / ny
    alpha = case.exact_initial(nx, ny).clone()
    state = transport.build(nx, ny, dx, dy)
    u0, v0 = sample_faces(case, nx, ny, 0.0)
    umax = max(float(u0.abs().max().item()), float(v0.abs().max().item()))
    h = min(dx, dy)
    dt = transport.cfl * h / max(umax, 1e-12)
    steps = int(math.ceil(case.duration / dt))
    if steps > max_steps:
        raise ValueError(f"{case.name}: {steps} steps exceeds max_steps")
    dt = case.duration / steps
    v0_total = float(alpha.sum().item()) * dx * dy
    worst = 0.0
    for k in range(steps):
        t = k * dt
        u, v = sample_faces(case, nx, ny, t, t + dt)
        alpha = transport.step(state, alpha, u, v, dt, dx, dy)
        drift = abs(float(alpha.sum().item()) * dx * dy - v0_total) / v0_total
        worst = max(worst, drift)
    exact = case.exact_initial(nx, ny)
    m = field_metrics(alpha, exact, dx * dy)
    m.update({
        "cells": [nx, ny],
        "steps": steps,
        "dt": dt,
        "dx": dx,
        "cou": transport.cfl,
        "mass_initial": v0_total,
        "mass_final": float(alpha.sum().item()) * dx * dy,
        "mass_drift_relative": abs(float(alpha.sum().item()) * dx * dy
                                   - v0_total) / v0_total,
        "worst_step_drift_relative": worst,
        "alpha_min": float(alpha.min().item()),
        "alpha_max": float(alpha.max().item()),
        "divergence_max": face_divergence(u0, v0, dx, dy),
    })
    return {"metrics": m, "alpha": alpha, "exact": exact}
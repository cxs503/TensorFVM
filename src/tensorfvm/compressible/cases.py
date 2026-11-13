"""Initial conditions and analytic anchors for the multi-material cases.

Each builder returns ``(solver, meta)`` where ``meta`` holds the numeric setup
*and* the analytic anchor used by the verification report, so a figure and a
criterion can never drift apart from the numbers that produced them.

``sod``
    Classical Sod shock tube (Sod 1978): pins the solver to an exact solution.
``two_gas_interface``
    Air (post-shock Mach 1.5 state) against helium at rest.  The exact Riemann
    solution supplies the interface trajectory, so the material-interface
    *position* has an analytic target and a measurable convergence order.
``shu_osher``
    Mach-3 shock interacting with a sinusoidal entropy wave (Shu & Osher 1988,
    *J. Comput. Phys.* **77**, 439--471).  The post-shock state is computed from
    the Rankine-Hugoniot relations; the reference solution is a fine-grid
    (N=2000) run of the same solver, so the comparison is zero-hand-copy.
``richtmyer_meshkov``
    Mach 1.5 shock in air hitting an air/SF6 interface with a single-mode
    sinusoidal perturbation.  Richtmyer's impulsive model gives the post-shock
    amplitude ``a0+ = a0 (1 - u*/D_t)``, the post-shock Atwood number ``A+`` and
    the linear growth rate ``da/dt = k a0+ A+ u*`` that the simulation must
    reproduce.
``shock_bubble``
    Mach 1.22 shock in air over a helium cylinder (Haas & Sturtevant 1987
    configuration): the bubble is light, so the transmitted shock accelerates
    inside it and a *rarefaction* travels back into the air.  Both speeds come
    from the exact Riemann solution, which turns "the vortex pair forms when the
    shock leaves the bubble" into a timing criterion.
``vortex_pair``
    Two co-rotating Lamb-Oseen vortices in a compressible gas: the point-vortex
    pair rotates at ``Omega = Gamma/(pi b^2)`` (Lamb 1932).  Core merging at
    small separation is the planar analogue of vortex-ring merging.

Units are non-dimensional: ambient gas ``rho = 1``, ``p = 1``, so the sound
speed is ``sqrt(gamma)`` and the reference velocity is ``sqrt(p/rho) = 1``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import torch

from .eos import StiffenedGasMixture
from .riemann import ExactRiemann
from .solver import EulerSolver

__all__ = [
    "normal_shock_state",
    "sod",
    "two_gas_interface",
    "material_advection",
    "shu_osher",
    "ShuOsherConfig",
    "richtmyer_meshkov",
    "shock_bubble",
    "vortex_pair",
    "rm_analytic",
    "bubble_analytic",
    "vortex_analytic",
    "RMConfig",
    "BubbleConfig",
    "VortexConfig",
    "GAMMA_AIR",
    "GAMMA_HE",
    "GAMMA_SF6",
]

GAMMA_AIR = 1.4
GAMMA_HE = 5.0 / 3.0
GAMMA_SF6 = 1.09


def normal_shock_state(gamma: float, mach: float, rho: float = 1.0, p: float = 1.0) -> Dict[str, float]:
    """Post-shock state of a normal shock with Mach number ``mach``."""
    if mach <= 1.0:
        raise ValueError("a normal shock needs mach > 1")
    g = gamma
    ratio = ((g + 1.0) * mach * mach) / ((g - 1.0) * mach * mach + 2.0)
    p_post = p * (2.0 * g * mach * mach / (g + 1.0) - (g - 1.0) / (g + 1.0))
    c_pre = math.sqrt(g * p / rho)
    u_post = (2.0 / (g + 1.0)) * (mach - 1.0 / mach) * c_pre
    return {
        "rho": rho * ratio,
        "u": u_post,
        "p": p_post,
        "shock_speed": mach * c_pre,
        "c_pre": c_pre,
    }


def _grid_1d(nx: int) -> torch.Tensor:
    return (torch.arange(nx, dtype=torch.float64) + 0.5) / nx


# ----------------------------------------------------------------------
# 1-D verification cases
# ----------------------------------------------------------------------
def sod(nx: int = 400, *, flux_mode: str = "hllc", order: int = 2,
        limiter: str = "mc", t_end: float = 0.2, cfl: float = 0.4):
    """Sod shock tube; returns ``(solver, meta)`` with the exact solution."""
    eos = StiffenedGasMixture([GAMMA_AIR, GAMMA_AIR], names=["air", "air"])
    solver = EulerSolver(
        nx=nx, ny=1, lx=1.0, ly=1.0, eos=eos,
        bc=("outflow", "outflow", "periodic", "periodic"),
        flux_mode=flux_mode, order=order, limiter=limiter, cfl=cfl,
    )
    xc = _grid_1d(nx)
    left = xc < 0.5
    solver.set_primitive({
        "rho": torch.where(left, torch.ones_like(xc), torch.full_like(xc, 0.125)),
        "u": torch.zeros_like(xc),
        "v": torch.zeros_like(xc),
        "p": torch.where(left, torch.ones_like(xc), torch.full_like(xc, 0.1)),
        "Y": torch.zeros_like(xc),
    })
    exact = ExactRiemann(1.0, 0.0, 1.0, GAMMA_AIR, 0.125, 0.0, 0.1, GAMMA_AIR)
    exact.solve()
    meta = {"case": "sod", "nx": nx, "t_end": t_end, "xc": xc, "x0": 0.5, "exact": exact}
    return solver, meta


def two_gas_interface(nx: int = 400, *, flux_mode: str = "gfm", order: int = 2,
                      limiter: str = "mc", cfl: float = 0.4, t_end: float = 0.2):
    """Post-shock air against helium at rest: a genuine material contact."""
    eos = StiffenedGasMixture([GAMMA_AIR, GAMMA_HE], names=["air", "He"])
    post = normal_shock_state(GAMMA_AIR, 1.5)
    rho_he = 0.138
    solver = EulerSolver(
        nx=nx, ny=1, lx=1.0, ly=1.0, eos=eos,
        bc=("outflow", "outflow", "periodic", "periodic"),
        flux_mode=flux_mode, order=order, limiter=limiter, cfl=cfl,
    )
    xc = _grid_1d(nx)
    left = xc < 0.4
    solver.set_primitive({
        "rho": torch.where(left, torch.full_like(xc, post["rho"]), torch.full_like(xc, rho_he)),
        "u": torch.where(left, torch.full_like(xc, post["u"]), torch.zeros_like(xc)),
        "v": torch.zeros_like(xc),
        "p": torch.where(left, torch.full_like(xc, post["p"]), torch.ones_like(xc)),
        "Y": torch.where(left, torch.zeros_like(xc), torch.ones_like(xc)),
    })
    riem = ExactRiemann(post["rho"], post["u"], post["p"], GAMMA_AIR,
                        rho_he, 0.0, 1.0, GAMMA_HE)
    p_star, u_star = riem.solve()
    meta = {
        "case": "two_gas_interface",
        "nx": nx,
        "t_end": t_end,
        "xc": xc,
        "x0": 0.4,
        "exact": riem,
        "u_star": float(u_star),
        "p_star": float(p_star),
        "shock_speed_right": float(riem.shock_speed_r()),
        "reflected_wave_speed": float(riem.wave_speed_l()),
        "reflected_wave_kind": ("rarefaction" if float(p_star) < post["p"] else "shock"),
    }
    return solver, meta


def material_advection(nx: int = 400, *, u: float = 0.5, rho2: float = 5.0,
                       x0: float = 0.3, t_end: float = 0.4, cfl: float = 0.4,
                       flux_mode: str = "gfm", order: int = 2,
                       limiter: str = "mc"):
    """Uniform-flow transport of a material contact (no wave) with an exact answer.

    Pressure and velocity are uniform, so the interface must translate exactly at
    ``u`` and the pressure must stay exactly 1: any drift is the scheme's
    interface-transport error, isolated from gas dynamics.  This is the case that
    measures the *order* of the material-interface position.
    """
    eos = StiffenedGasMixture([GAMMA_AIR, GAMMA_SF6], names=["air", "SF6"])
    solver = EulerSolver(
        nx=nx, ny=1, lx=1.0, ly=1.0, eos=eos,
        bc=("outflow", "outflow", "periodic", "periodic"),
        flux_mode=flux_mode, order=order, limiter=limiter, cfl=cfl,
    )
    xc = _grid_1d(nx)
    left = xc < x0
    solver.set_primitive({
        "rho": torch.where(left, torch.ones_like(xc), torch.full_like(xc, rho2)),
        "u": torch.full_like(xc, u),
        "v": torch.zeros_like(xc),
        "p": torch.ones_like(xc),
        "Y": torch.where(left, torch.zeros_like(xc), torch.ones_like(xc)),
    })
    meta = {
        "case": "material_advection",
        "nx": nx,
        "t_end": t_end,
        "xc": xc,
        "x0": x0,
        "u": u,
        "x_exact": x0 + u * t_end,
        "initial_integrals": solver.integrals(),
    }
    return solver, meta



# ----------------------------------------------------------------------
# Shu-Osher: shock-entropy wave interaction
# ----------------------------------------------------------------------
@dataclass
class ShuOsherConfig:
    """Parameters for the Shu-Osher shock-entropy wave interaction case.

    The original problem (Shu & Osher 1988, *JCP* **77**, 439) uses a domain
    [0, 5] with a Mach-3 shock at x=4 interacting with a sinusoidal density
    perturbation.  The post-shock state is computed from the Rankine-Hugoniot
    relations for a Mach-3 normal shock in a perfect gas (gamma=1.4).
    """

    nx: int = 200
    lx: float = 5.0
    shock_pos: float = 4.0  # initial shock location
    mach: float = 3.0
    gamma: float = GAMMA_AIR
    rho_pre: float = 1.0
    p_pre: float = 1.0
    entropy_amp: float = 0.2  # amplitude of density perturbation
    entropy_wavenumber: float = math.pi  # k = pi for domain [0,5]
    t_end: float = 1.8
    cfl: float = 0.4
    order: int = 2
    limiter: str = "mc"
    flux_mode: str = "hllc"

    @property
    def dx(self) -> float:
        return self.lx / self.nx

    @property
    def post_shock(self) -> Dict[str, float]:
        """Post-shock state from Rankine-Hugoniot (zero hand-copying)."""
        return normal_shock_state(self.gamma, self.mach, self.rho_pre, self.p_pre)


def shu_osher(cfg: ShuOsherConfig | None = None):
    """Build the Shu-Osher shock-entropy wave interaction case.

    Returns ``(solver, meta)``.  The initial condition is:
    - Left of ``shock_pos``: post-shock state (from Rankine-Hugoniot)
    - Right of ``shock_pos``: pre-shock state with sinusoidal density perturbation
      ``rho = rho_pre + entropy_amp * sin(entropy_wavenumber * x)``

    The reference solution is obtained by running the same solver on a fine grid
    (N=2000) and is stored in ``meta["reference"]`` when ``with_reference=True``.
    """
    cfg = cfg or ShuOsherConfig()
    eos = StiffenedGasMixture([cfg.gamma, cfg.gamma], names=["air", "air"])
    solver = EulerSolver(
        nx=cfg.nx, ny=1, lx=cfg.lx, ly=1.0, eos=eos,
        bc=("outflow", "outflow", "periodic", "periodic"),
        flux_mode=cfg.flux_mode, order=cfg.order, limiter=cfg.limiter, cfl=cfg.cfl,
    )
    xc = (torch.arange(cfg.nx, dtype=torch.float64) + 0.5) * cfg.lx / cfg.nx
    left = xc < cfg.shock_pos
    post = cfg.post_shock
    # pre-shock: uniform p, u=0, sinusoidal density perturbation
    rho_pre = cfg.rho_pre + cfg.entropy_amp * torch.sin(cfg.entropy_wavenumber * xc)
    solver.set_primitive({
        "rho": torch.where(left, torch.full_like(xc, post["rho"]), rho_pre),
        "u": torch.where(left, torch.full_like(xc, post["u"]), torch.zeros_like(xc)),
        "v": torch.zeros_like(xc),
        "p": torch.where(left, torch.full_like(xc, post["p"]), torch.full_like(xc, cfg.p_pre)),
        "Y": torch.zeros_like(xc),
    })
    meta = {
        "case": "shu_osher",
        "config": cfg,
        "nx": cfg.nx,
        "lx": cfg.lx,
        "xc": xc,
        "shock_pos": cfg.shock_pos,
        "t_end": cfg.t_end,
        "post_shock": post,
        "initial_integrals": solver.integrals(),
    }
    return solver, meta


# ----------------------------------------------------------------------
# Richtmyer-Meshkov
# ----------------------------------------------------------------------
@dataclass
class RMConfig:
    """Geometry of the air/SF6 Richtmyer--Meshkov case (non-dimensional)."""

    ny: int = 96
    lx: float = 3.3
    ly: float = 0.5
    x_left: float = -1.8
    x0: float = -0.6
    shock_x: float = -1.2
    mach: float = 1.5
    a0: float = 0.04
    rho_sf6: float = 5.0
    t_end: float = 1.1
    cfl: float = 0.35
    order: int = 2
    limiter: str = "mc"
    flux_mode: str = "gfm"

    def grid(self) -> Tuple[int, float, float]:
        dy = self.ly / self.ny
        nx = int(round(self.lx / dy))
        return nx, self.lx / nx, dy


def rm_analytic(cfg: RMConfig) -> Dict[str, float]:
    """Richtmyer impulsive-model prediction of the post-shock state."""
    post = normal_shock_state(GAMMA_AIR, cfg.mach)
    riem = ExactRiemann(post["rho"], post["u"], post["p"], GAMMA_AIR,
                        cfg.rho_sf6, 0.0, 1.0, GAMMA_SF6)
    p_star, u_star = riem.solve()
    d_t = riem.shock_speed_r()
    rho_sf6_post = riem.density_star_r()
    atwood = (rho_sf6_post - post["rho"]) / (rho_sf6_post + post["rho"])
    compression = 1.0 - u_star / d_t
    a0_post = cfg.a0 * compression
    k = 2.0 * math.pi / cfg.ly
    return {
        "mach": cfg.mach,
        "post_rho": float(post["rho"]),
        "post_u": float(post["u"]),
        "post_p": float(post["p"]),
        "incident_shock_speed": float(post["shock_speed"]),
        "p_star": float(p_star),
        "u_star": float(u_star),
        "transmitted_shock_speed": float(d_t),
        "rho_sf6_post": float(rho_sf6_post),
        "atwood_post": float(atwood),
        "compression": float(compression),
        "a0_post": float(a0_post),
        "wavenumber": float(k),
        "k_a0_post": float(k * a0_post),
        "linear_growth_rate": float(k * a0_post * atwood * u_star),
        "shock_transit_time": float((cfg.x0 - cfg.shock_x) / post["shock_speed"]),
        "reflected_shock_speed": float(riem.wave_speed_l()),
    }


def richtmyer_meshkov(cfg: RMConfig | None = None):
    """Build the RM initial condition; returns ``(solver, meta)``."""
    cfg = cfg or RMConfig()
    nx, dx, dy = cfg.grid()
    eos = StiffenedGasMixture([GAMMA_AIR, GAMMA_SF6], names=["air", "SF6"])
    solver = EulerSolver(
        nx=nx, ny=cfg.ny, lx=cfg.lx, ly=cfg.ly, eos=eos,
        bc=("outflow", "outflow", "periodic", "periodic"),
        flux_mode=cfg.flux_mode, order=cfg.order, limiter=cfg.limiter,
        cfl=cfg.cfl, gfm_threshold=0.1,
    )
    post = normal_shock_state(GAMMA_AIR, cfg.mach)
    xc = cfg.x_left + (torch.arange(nx, dtype=torch.float64) + 0.5) * dx
    yc = (torch.arange(cfg.ny, dtype=torch.float64) + 0.5) * dy
    X = xc.reshape(-1, 1).expand(nx, cfg.ny)
    Y = yc.reshape(1, -1).expand(nx, cfg.ny)
    interface = cfg.x0 + cfg.a0 * torch.cos(2.0 * math.pi * Y / cfg.ly)
    in_sf6 = X > interface
    shocked = X < cfg.shock_x
    rho = torch.where(in_sf6, torch.full_like(X, cfg.rho_sf6), torch.ones_like(X))
    p = torch.ones_like(X)
    u = torch.zeros_like(X)
    rho = torch.where(shocked, torch.full_like(X, post["rho"]), rho)
    p = torch.where(shocked, torch.full_like(X, post["p"]), p)
    u = torch.where(shocked, torch.full_like(X, post["u"]), u)
    solver.set_primitive({
        "rho": rho, "u": u, "v": torch.zeros_like(X), "p": p,
        "Y": in_sf6.to(torch.float64),
    })
    meta = {
        "case": "richtmyer_meshkov",
        "config": cfg,
        "nx": nx,
        "ny": cfg.ny,
        "dx": dx,
        "dy": dy,
        "xc": xc,
        "yc": yc,
        "analytic": rm_analytic(cfg),
        "initial_integrals": solver.integrals(),
    }
    return solver, meta


# ----------------------------------------------------------------------
# shock-bubble interaction
# ----------------------------------------------------------------------
@dataclass
class BubbleConfig:
    """Geometry of the Mach-1.22 helium-cylinder in air case."""

    mesh: int = 160
    radius: float = 1.0
    x_min: float = -3.0
    x_max: float = 6.0
    y_half: float = 2.5
    mach: float = 1.22
    rho_he: float = 0.138
    shock_x: float = -1.5
    t_end: float = 2.2
    cfl: float = 0.3
    order: int = 2
    limiter: str = "mc"
    flux_mode: str = "gfm"

    def grid(self) -> Tuple[int, int, float]:
        d = self.radius / self.mesh
        nx = int(round((self.x_max - self.x_min) / d))
        ny = int(round(2.0 * self.y_half / d))
        return nx, ny, d


def bubble_analytic(cfg: BubbleConfig) -> Dict[str, float]:
    """Wave speeds of the He/air refraction problem and the crossing time."""
    post = normal_shock_state(GAMMA_AIR, cfg.mach)
    riem = ExactRiemann(post["rho"], post["u"], post["p"], GAMMA_AIR,
                        cfg.rho_he, 0.0, 1.0, GAMMA_HE)
    p_star, u_star = riem.solve()
    d_t = riem.shock_speed_r()
    return {
        "post_rho": float(post["rho"]),
        "post_u": float(post["u"]),
        "post_p": float(post["p"]),
        "incident_shock_speed": float(post["shock_speed"]),
        "transmitted_shock_speed": float(d_t),
        "reflected_wave_speed": float(riem.wave_speed_l()),
        "reflected_is_rarefaction": bool(float(p_star) < post["p"]),
        "interface_speed": float(u_star),
        "p_interface": float(p_star),
        "crossing_time": float(2.0 * cfg.radius / d_t),
        # the incident shock starts inside the domain, so the bubble is reached
        # after the remaining distance divided by the incident shock speed
        "arrival_time": float((-cfg.radius - cfg.shock_x) / post["shock_speed"]),
    }


def shock_bubble(cfg: BubbleConfig | None = None):
    """Build the helium-bubble initial condition; returns ``(solver, meta)``."""
    cfg = cfg or BubbleConfig()
    nx, ny, d = cfg.grid()
    eos = StiffenedGasMixture([GAMMA_AIR, GAMMA_HE], names=["air", "He"])
    solver = EulerSolver(
        nx=nx, ny=ny, lx=cfg.x_max - cfg.x_min, ly=2.0 * cfg.y_half, eos=eos,
        bc=("outflow", "outflow", "reflective", "reflective"),
        flux_mode=cfg.flux_mode, order=cfg.order, limiter=cfg.limiter,
        cfl=cfg.cfl, gfm_threshold=0.1,
    )
    xc = cfg.x_min + (torch.arange(nx, dtype=torch.float64) + 0.5) * d
    yc = -cfg.y_half + (torch.arange(ny, dtype=torch.float64) + 0.5) * d
    X = xc.reshape(-1, 1).expand(nx, ny)
    Y = yc.reshape(1, -1).expand(nx, ny)
    in_bubble = (X * X + Y * Y) < cfg.radius ** 2
    post = normal_shock_state(GAMMA_AIR, cfg.mach)
    shocked = X < cfg.shock_x
    rho = torch.where(in_bubble, torch.full_like(X, cfg.rho_he), torch.ones_like(X))
    p = torch.ones_like(X)
    u = torch.zeros_like(X)
    rho = torch.where(shocked, torch.full_like(X, post["rho"]), rho)
    p = torch.where(shocked, torch.full_like(X, post["p"]), p)
    u = torch.where(shocked, torch.full_like(X, post["u"]), u)
    solver.set_primitive({
        "rho": rho, "u": u, "v": torch.zeros_like(X), "p": p,
        "Y": in_bubble.to(torch.float64),
    })
    meta = {
        "case": "shock_bubble",
        "config": cfg,
        "nx": nx,
        "ny": ny,
        "d": d,
        "xc": xc,
        "yc": yc,
        "analytic": bubble_analytic(cfg),
        "initial_integrals": solver.integrals(),
    }
    return solver, meta


# ----------------------------------------------------------------------
# co-rotating vortex pair (planar vortex merging)
# ----------------------------------------------------------------------
@dataclass
class VortexConfig:
    """Two equal co-rotating Lamb-Oseen vortices in a compressible gas."""

    n: int = 256
    size: float = 2.0 * math.pi
    circulation: float = 1.0
    core: float = 0.3
    separation: float = 1.2
    mach_vortex: float | None = None
    t_end: float = 8.0
    cfl: float = 0.35
    order: int = 2
    limiter: str = "mc"

    def velocity_scale(self) -> float:
        return self.circulation / (2.0 * math.pi * self.core)


def vortex_analytic(cfg: VortexConfig) -> Dict[str, float]:
    """Point-vortex rotation rate and the Lamb-Oseen peak swirl speed."""
    omega = cfg.circulation / (math.pi * cfg.separation ** 2)
    peak = cfg.velocity_scale() * (1.0 - math.exp(-1.0))
    return {
        "rotation_rate": float(omega),
        "rotation_period": float(2.0 * math.pi / omega),
        "peak_swirl": float(peak),
        "circulation": float(cfg.circulation),
        "separation": float(cfg.separation),
        "core_radius": float(cfg.core),
        "separation_over_core": float(cfg.separation / cfg.core),
    }


def vortex_pair(cfg: VortexConfig | None = None):
    """Build the co-rotating vortex pair; returns ``(solver, meta)``.

    The pressure field is the corresponding steady balance of the Lamb-Oseen
    swirl (``dp/dr = rho u_theta^2 / r``) so the initial condition carries no
    spurious acoustic transient.
    """
    from .vortex import lamb_oseen_pair

    cfg = cfg or VortexConfig()
    eos = StiffenedGasMixture([GAMMA_AIR, GAMMA_AIR], names=["air", "air"])
    solver = EulerSolver(
        nx=cfg.n, ny=cfg.n, lx=cfg.size, ly=cfg.size, eos=eos,
        bc=("periodic", "periodic", "periodic", "periodic"),
        flux_mode="hllc", order=cfg.order, limiter=cfg.limiter, cfl=cfg.cfl,
    )
    prim = lamb_oseen_pair(
        n=cfg.n, size=cfg.size, circulation=cfg.circulation, core=cfg.core,
        separation=cfg.separation, mach_vortex=cfg.mach_vortex, gamma=GAMMA_AIR,
    )
    solver.set_primitive(prim)
    meta = {
        "case": "vortex_pair",
        "config": cfg,
        "nx": cfg.n,
        "ny": cfg.n,
        "d": cfg.size / cfg.n,
        "analytic": vortex_analytic(cfg),
        "initial_integrals": solver.integrals(),
    }
    return solver, meta

# ----------------------------------------------------------------------
# Woodward-Colella two interacting blast waves
# ----------------------------------------------------------------------
#: Initial data of the "two interacting blast waves" problem of Woodward &
#: Colella (1984, J. Comput. Phys. 54, 115-173).  The 1984 article is not open
#: access, so the pressure plateaus below are taken from two independent
#: open-access restatements of that initial datum, which agree digit for digit:
#:
#:   * arXiv:1902.07773 (Dumbser et al., J. Comput. Phys. 2019), Sec. 5.1.4,
#:     (rho0, u0, p0) = [1, 0, 1e3] on [0, 0.1], [1, 0, 1e-2] on (0.1, 0.9),
#:     [1, 0, 1e2] on [0.9, 1]; "the expected solution has been computed via
#:     the pure RPJ on a mesh of 3200 cells"; T = 0.038.
#:   * arXiv:2109.14340 (Fu et al.), Eq. (37): the same plateaus, reflective
#:     walls at x = 0 and x = 1, T = 0.038.
#:
#: Both restatements cite W&C 1984.  ``docs/woodward-colella-double-blast/
#: provenance.md`` records the fetched documents and their sha256 so the digit
#: strings can be re-checked instead of trusted.
WC_BLAST_X1 = 0.1
WC_BLAST_X2 = 0.9
WC_BLAST_P_LEFT = 1000.0
WC_BLAST_P_MIDDLE = 0.01
WC_BLAST_P_RIGHT = 100.0
WC_BLAST_RHO = 1.0
WC_BLAST_T_END = 0.038


@dataclass
class DoubleBlastConfig:
    """Resolution and scheme knobs for the two interacting blast waves."""

    nx: int = 800
    ny: int = 1
    order: int = 2
    limiter: str = "mc"
    cfl: float = 0.4
    t_end: float = WC_BLAST_T_END
    flux_mode: str = "hllc"


def blast_pressure(x: torch.Tensor) -> torch.Tensor:
    """Three-plateau initial pressure of the Woodward-Colella blast problem."""
    return torch.where(
        x < WC_BLAST_X1,
        torch.full_like(x, WC_BLAST_P_LEFT),
        torch.where(x < WC_BLAST_X2,
                    torch.full_like(x, WC_BLAST_P_MIDDLE),
                    torch.full_like(x, WC_BLAST_P_RIGHT)),
    )


def double_blast(cfg: DoubleBlastConfig | None = None):
    """Two interacting blast waves in a closed tube: returns ``(solver, meta)``.

    ``ny == 1`` is the one-dimensional problem as posed in the paper.  ``ny > 1``
    extrudes the same profile in ``y`` and closes *all* walls with reflective
    boundaries; the exact solution of that configuration is y-independent, so the
    extrusion is a round-off-level test of the two-dimensional path (a slab has
    no y flux), not a new physics case.
    """
    cfg = cfg or DoubleBlastConfig()
    eos = StiffenedGasMixture([GAMMA_AIR, GAMMA_AIR], names=["air", "air"])
    bc = (("reflective",) * 4 if cfg.ny > 1
          else ("reflective", "reflective", "periodic", "periodic"))
    solver = EulerSolver(
        nx=cfg.nx, ny=cfg.ny, lx=1.0, ly=1.0, eos=eos, bc=bc,
        flux_mode=cfg.flux_mode, order=cfg.order, limiter=cfg.limiter, cfl=cfg.cfl,
    )
    x = (torch.arange(cfg.nx, dtype=torch.float64) + 0.5) / cfg.nx
    p0 = blast_pressure(x)
    zeros = torch.zeros(cfg.nx, dtype=torch.float64)
    if cfg.ny > 1:  # extrude the 1-D datum over the y cells
        x, p0, zeros = x.reshape(cfg.nx, 1), p0.reshape(cfg.nx, 1), zeros.reshape(cfg.nx, 1)
    solver.set_primitive({
        "rho": torch.full_like(x, WC_BLAST_RHO),
        "u": zeros,
        "v": zeros,
        "p": p0,
        "Y": torch.zeros_like(x),
    })
    meta = {
        "case": "double_blast",
        "config": cfg,
        "nx": cfg.nx,
        "ny": cfg.ny,
        "x": x,
        "p0": p0,
        "t_end": cfg.t_end,
        "initial_integrals": solver.integrals(),
        # exact reference states for the two initial discontinuities, used by
        # the early-time criterion (before either wave reaches a wall)
        "raster": 0.0,
    }
    meta.update(blast_early_anchors())
    return solver, meta


def blast_early_anchors() -> Dict[str, Dict[str, float]]:
    """Exact Riemann states of the two initial discontinuities.

    Up to the time the first rarefaction head reaches a wall
    (``t < 0.1/37.416 = 2.67e-3``) each initial jump is an isolated Riemann
    problem, so its shock speed and post-shock plateau are exact.  ``left`` is
    the jump at ``x = 0.1`` (high pressure on the left of the jump), ``right``
    the jump at ``x = 0.9`` (high pressure on the right of the jump), both with
    the plateau pair read off the same initial datum.
    """
    out: Dict[str, Dict[str, float]] = {}
    left = ExactRiemann(WC_BLAST_RHO, 0.0, WC_BLAST_P_LEFT, GAMMA_AIR,
                        WC_BLAST_RHO, 0.0, WC_BLAST_P_MIDDLE, GAMMA_AIR)
    left.solve()
    out["left"] = {
        "x_jump": WC_BLAST_X1,
        "p_star": float(left.pstar),
        "u_star": float(left.ustar),
        "shock_speed": float(left.shock_speed_r()),      # moves +x
        "rarefaction_head": float(left.wave_speed_l()),  # moves -x
        "rho_post_shock": float(left.density_star_r()),
    }
    right = ExactRiemann(WC_BLAST_RHO, 0.0, WC_BLAST_P_MIDDLE, GAMMA_AIR,
                         WC_BLAST_RHO, 0.0, WC_BLAST_P_RIGHT, GAMMA_AIR)
    right.solve()
    out["right"] = {
        "x_jump": WC_BLAST_X2,
        "p_star": float(right.pstar),
        "u_star": float(right.ustar),
        "shock_speed": float(right.shock_speed_l()),     # moves -x
        "rarefaction_head": float(right.wave_speed_r()),  # moves +x
        "rho_post_shock": float(right.density_star_l()),
    }
    return out

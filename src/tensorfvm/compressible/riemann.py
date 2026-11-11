"""Riemann solvers for the compressible multi-material surface.

Two solvers live here and they serve different purposes:

``hllc_flux``
    The HLLC approximate Riemann solver used as the production face flux.  It
    is written for *two different* materials on the two sides of a face, so a
    material contact inside a face costs nothing special: the contact wave is
    resolved exactly by HLLC, which is what keeps the material interface from
    diffusing faster than the flow requires.  Wave-speed estimates follow the
    Davis/Einfeldt bracket ``S_L = min(u_L - c_L, u_R - c_R)``,
    ``S_R = max(u_L + c_L, u_R + c_R)``.

``ExactRiemann``
    Toro's exact iterative solver for two stiffened gases (Toro 2009,
    *Riemann Solvers and Numerical Methods for Fluid Dynamics*, 3rd ed.,
    ch. 4).  It is used (a) as the reference in the shock-tube verification,
    (b) to obtain the analytic post-shock interface state
    ``(p_*, u_*, D_transmitted)`` that anchors the Richtmyer--Meshkov and
    shock--bubble timing predictions, and (c) as the interface state of the
    ghost-fluid treatment (``EulerSolver.flux_mode = "gfm"``).

Convention: a "state" carries ``(rho, un, ut, p, E, rhoY, c, pinf)`` where
``un`` is the face-normal velocity, ``ut`` the tangential one, ``E`` the total
energy density and ``rhoY`` the partial density of material 2.  All functions
are broadcast safe over arbitrary leading shapes (flattened faces, whole
fields, scalars).
"""

from __future__ import annotations

from typing import Tuple

import torch

__all__ = ["hllc_flux", "ExactRiemann", "riemann_star", "physical_flux"]

_TINY = 1.0e-30


def physical_flux(
    rho: torch.Tensor,
    un: torch.Tensor,
    ut: torch.Tensor,
    p: torch.Tensor,
    E: torch.Tensor,
    rhoY: torch.Tensor,
) -> Tuple[torch.Tensor, ...]:
    """Euler flux in the normal direction, ``F(U)``, component by component."""
    return (
        rho * un,
        rho * un * un + p,
        rho * un * ut,
        un * (E + p),
        rhoY * un,
    )


def _star_state(rho, un, ut, p, E, rhoY, pinf, S, SM):
    """HLLC star state ``U*_K`` of one side of a face."""
    denom = S - un
    safe = torch.where(torch.abs(denom) < 1e-12, torch.full_like(denom, 1e-12), denom)
    fac = rho * denom / (S - SM)
    return (
        fac,
        fac * SM,
        fac * ut,
        fac * (E / rho + (SM - un) * (SM + (p + pinf) / (rho * safe))),
        rhoY * denom / (S - SM),
    )


def hllc_flux(left: dict, right: dict) -> Tuple[torch.Tensor, ...]:
    """HLLC flux through a face whose two sides may be different materials.

    ``left``/``right`` must carry ``rho, un, ut, p, E, rhoY, c, pinf``; ``c`` is
    the *mixture* sound speed and ``pinf`` the mixture stiffened-gas offset.
    ``pinf == 0`` on both sides is the validated ideal-gas path; see
    ``tests/test_compressible_eos_riemann.py`` for the exact-solution
    comparison.
    """
    rho_l, un_l, ut_l, p_l = left["rho"], left["un"], left["ut"], left["p"]
    E_l, rhoY_l, c_l, pinf_l = left["E"], left["rhoY"], left["c"], left["pinf"]
    rho_r, un_r, ut_r, p_r = right["rho"], right["un"], right["ut"], right["p"]
    E_r, rhoY_r, c_r, pinf_r = right["E"], right["rhoY"], right["c"], right["pinf"]

    S_l = torch.minimum(un_l - c_l, un_r - c_r)
    S_r = torch.maximum(un_l + c_l, un_r + c_r)
    num = p_r - p_l + rho_l * un_l * (S_l - un_l) - rho_r * un_r * (S_r - un_r)
    den = rho_l * (S_l - un_l) - rho_r * (S_r - un_r)
    den = torch.where(torch.abs(den) < _TINY, torch.full_like(den, _TINY), den)
    SM = num / den
    # a contact leaving the fan would be non-physical; keep SM inside [S_l, S_r]
    SM = torch.clamp(SM, min=torch.minimum(S_l, S_r), max=torch.maximum(S_l, S_r))

    F_l = physical_flux(rho_l, un_l, ut_l, p_l, E_l, rhoY_l)
    F_r = physical_flux(rho_r, un_r, ut_r, p_r, E_r, rhoY_r)

    st_l = _star_state(rho_l, un_l, ut_l, p_l, E_l, rhoY_l, pinf_l, S_l, SM)
    st_r = _star_state(rho_r, un_r, ut_r, p_r, E_r, rhoY_r, pinf_r, S_r, SM)

    u_l = (rho_l, rho_l * un_l, rho_l * ut_l, E_l, rhoY_l)
    u_r = (rho_r, rho_r * un_r, rho_r * ut_r, E_r, rhoY_r)
    F_ls = tuple(f + S_l * (s - k) for f, s, k in zip(F_l, st_l, u_l))
    F_rs = tuple(f + S_r * (s - k) for f, s, k in zip(F_r, st_r, u_r))

    out = []
    for k in range(5):
        out.append(
            torch.where(
                S_l >= 0.0,
                F_l[k],
                torch.where(SM >= 0.0, F_ls[k], torch.where(S_r > 0.0, F_rs[k], F_r[k])),
            )
        )
    return tuple(out)


def pressure_function(p, rho_k, p_k, gamma_k, pinf_k):
    """Toro's ``f_K(p)`` and ``f'_K(p)`` for a stiffened gas (eq. 4.6/4.7)."""
    c_k = torch.sqrt(torch.clamp(gamma_k * (p_k + pinf_k) / rho_k, min=_TINY))
    A = 2.0 / ((gamma_k + 1.0) * rho_k)
    # Toro eq. (4.7): B_K = p_K (gamma-1)/(gamma+1); the stiffened-gas form is the
    # same expression written in the shifted pressure p~ = p + p_inf.
    B = (p_k + pinf_k) * (gamma_k - 1.0) / (gamma_k + 1.0) - pinf_k
    q_shock = torch.sqrt(A / (p + B))
    f_shock = (p - p_k) * q_shock
    fd_shock = q_shock * (1.0 - (p - p_k) / (2.0 * (B + p)))

    ratio = (p + pinf_k) / (p_k + pinf_k)
    e1 = (gamma_k - 1.0) / (2.0 * gamma_k)
    e2 = -(gamma_k + 1.0) / (2.0 * gamma_k)
    f_rare = 2.0 * c_k / (gamma_k - 1.0) * (torch.pow(ratio, e1) - 1.0)
    fd_rare = torch.pow(ratio, e2) / (rho_k * c_k)

    shock = p > p_k
    return torch.where(shock, f_shock, f_rare), torch.where(shock, fd_shock, fd_rare)


class ExactRiemann:
    """Exact Riemann solution for two stiffened gases (Toro ch. 4).

    All constructor arguments accept tensors of any (broadcastable) shape, so a
    single instance solves thousands of face problems at once.  ``solve``
    returns ``(p_star, u_star)``; ``sample`` returns the full
    ``(rho, u, p)`` state at similarity coordinates ``xi = x/t``.
    """

    def __init__(self, rho_l, u_l, p_l, gamma_l, rho_r, u_r, p_r, gamma_r,
                 pinf_l=None, pinf_r=None) -> None:
        u_l = torch.as_tensor(u_l, dtype=torch.float64)
        u_r = torch.as_tensor(u_r, dtype=torch.float64)
        u_l, u_r = torch.broadcast_tensors(u_l, u_r)
        as_t = lambda v: torch.as_tensor(v, dtype=torch.float64) + torch.zeros_like(u_l)  # noqa: E731
        self.u_l, self.u_r = u_l, u_r
        self.rho_l, self.p_l = as_t(rho_l), as_t(p_l)
        self.gamma_l = as_t(gamma_l)
        self.pinf_l = as_t(0.0 if pinf_l is None else pinf_l)
        self.rho_r, self.p_r = as_t(rho_r), as_t(p_r)
        self.gamma_r = as_t(gamma_r)
        self.pinf_r = as_t(0.0 if pinf_r is None else pinf_r)
        self.c_l = torch.sqrt(torch.clamp(
            self.gamma_l * (self.p_l + self.pinf_l) / self.rho_l, min=_TINY))
        self.c_r = torch.sqrt(torch.clamp(
            self.gamma_r * (self.p_r + self.pinf_r) / self.rho_r, min=_TINY))
        self.pstar = None
        self.ustar = None

    # -- pressure iteration ----------------------------------------------
    def _residual(self, p):
        f_l, fd_l = pressure_function(p, self.rho_l, self.p_l, self.gamma_l, self.pinf_l)
        f_r, fd_r = pressure_function(p, self.rho_r, self.p_r, self.gamma_r, self.pinf_r)
        return f_l, f_r, (f_l + f_r + self.u_r - self.u_l), (fd_l + fd_r)

    def solve(self, iterations: int = 16, tol: float = 1e-13
              ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Newton iteration on ``f(p*) = 0`` with a PVRS initial guess.

        Six iterations are enough from the PVRS guess; ``iterations`` is the
        safety cap and the loop exits early once the scaled residual is below
        ``tol``.  Deep iteration counts are expensive because every iteration is
        a handful of kernel launches, which matters when thousands of face
        problems are solved per time step.
        """
        c_bar = 0.5 * (self.c_l + self.c_r)
        rho_bar = 0.5 * (self.rho_l + self.rho_r)
        p = 0.5 * (self.p_l + self.p_r) - 0.5 * (self.u_r - self.u_l) * rho_bar * c_bar
        p_min = 1e-8 * torch.maximum(self.p_l, self.p_r)
        p = torch.clamp(p, min=p_min)
        scale = rho_bar * c_bar * (torch.abs(self.u_r - self.u_l) + c_bar)
        for _ in range(iterations):
            _, _, f, fd = self._residual(p)
            if bool(torch.all(torch.abs(f) <= tol * scale)):
                break
            fd = torch.where(torch.abs(fd) < _TINY, torch.full_like(fd, _TINY), fd)
            p = torch.clamp(p - f / fd, min=p_min)
        f_l, f_r, _, _ = self._residual(p)
        u = 0.5 * (self.u_l + self.u_r) + 0.5 * (f_r - f_l)
        self.pstar, self.ustar = p, u
        return p, u

    def residual_norm(self) -> torch.Tensor:
        """Max ``|f(p*)| / (rho c |u|)``-scaled residual, for reporting."""
        if self.pstar is None:
            raise RuntimeError("solve() must be called first")
        _, _, f, _ = self._residual(self.pstar)
        scale = torch.maximum(self.rho_l * self.c_l, self.rho_r * self.c_r) * (
            torch.abs(self.u_l) + torch.abs(self.u_r) + self.c_l + self.c_r
        )
        return torch.max(torch.abs(f) / scale)

    def shock_speed_l(self) -> torch.Tensor:
        """Speed of the left-going shock, ``S_L`` (lab frame)."""
        ratio = (self.pstar + self.pinf_l) / (self.p_l + self.pinf_l)
        g = self.gamma_l
        return self.u_l - self.c_l * torch.sqrt(
            (g + 1.0) / (2.0 * g) * ratio + (g - 1.0) / (2.0 * g)
        )

    def shock_speed_r(self) -> torch.Tensor:
        """Speed of the right-going shock, ``S_R`` (lab frame)."""
        ratio = (self.pstar + self.pinf_r) / (self.p_r + self.pinf_r)
        g = self.gamma_r
        return self.u_r + self.c_r * torch.sqrt(
            (g + 1.0) / (2.0 * g) * ratio + (g - 1.0) / (2.0 * g)
        )

    def density_star_r(self) -> torch.Tensor:
        """Star-region density on the right side.

        Hugoniot behind a right-going shock (``p* > p_R``), isentrope inside a
        right-going rarefaction; ``_sample_side`` applies the same branch, so the
        accessor cannot disagree with the sampled exact solution.
        """
        ratio = (self.pstar + self.pinf_r) / (self.p_r + self.pinf_r)
        g = self.gamma_r
        hugoniot = self.rho_r * (
            (ratio + (g - 1.0) / (g + 1.0)) / ((g - 1.0) / (g + 1.0) * ratio + 1.0)
        )
        isentrope = self.rho_r * torch.pow(torch.clamp(ratio, min=_TINY), 1.0 / g)
        return torch.where(self.pstar > self.p_r, hugoniot, isentrope)

    def density_star_l(self) -> torch.Tensor:
        """Star-region density on the left side (shock: Hugoniot, else isentrope)."""
        ratio = (self.pstar + self.pinf_l) / (self.p_l + self.pinf_l)
        g = self.gamma_l
        hugoniot = self.rho_l * (
            (ratio + (g - 1.0) / (g + 1.0)) / ((g - 1.0) / (g + 1.0) * ratio + 1.0)
        )
        isentrope = self.rho_l * torch.pow(torch.clamp(ratio, min=_TINY), 1.0 / g)
        return torch.where(self.pstar > self.p_l, hugoniot, isentrope)

    def wave_speed_l(self) -> torch.Tensor:
        """Leading-edge speed of the left-going wave, whichever family it is.

        Shock (``p* > p_L``): ``S_L``; rarefaction: the fan head ``u_L - c_L``.
        Use this instead of :meth:`shock_speed_l` when the family is not known in
        advance -- ``shock_speed_l`` evaluates the shock branch unconditionally
        and therefore returns a speed *inside* the fan when the left wave is a
        rarefaction.
        """
        head = self.u_l - self.c_l
        return torch.where(self.pstar > self.p_l, self.shock_speed_l(), head)

    def wave_speed_r(self) -> torch.Tensor:
        """Leading-edge speed of the right-going wave (see :meth:`wave_speed_l`)."""
        head = self.u_r + self.c_r
        return torch.where(self.pstar > self.p_r, self.shock_speed_r(), head)

    # -- exact sampling ---------------------------------------------------
    def sample(self, xi: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Exact ``(rho, u, p)`` at similarity coordinate ``xi = x/t``."""
        if self.pstar is None:
            self.solve()
        rho_l, u_l, p_l = self._sample_side(
            xi, self.rho_l, self.u_l, self.p_l, self.gamma_l, self.pinf_l, self.c_l,
            right=False)
        rho_r, u_r, p_r = self._sample_side(
            xi, self.rho_r, self.u_r, self.p_r, self.gamma_r, self.pinf_r, self.c_r,
            right=True)
        left = xi <= self.ustar
        return (
            torch.where(left, rho_l, rho_r),
            torch.where(left, u_l, u_r),
            torch.where(left, p_l, p_r),
        )

    def _sample_side(self, xi, rho_k, u_k, p_k, gamma_k, pinf_k, c_k, right: bool):
        """Exact state of one side; ``right`` selects the outgoing wave family."""
        p_star, u_star = self.pstar, self.ustar
        pk_pinf = p_k + pinf_k
        ratio = (p_star + pinf_k) / pk_pinf
        g = gamma_k
        gm1 = g - 1.0
        rho_star_shock = rho_k * ((ratio + gm1 / (g + 1.0)) / (gm1 / (g + 1.0) * ratio + 1.0))
        rho_star_rare = rho_k * torch.pow(torch.clamp(ratio, min=_TINY), 1.0 / g)
        is_shock = p_star > p_k
        rho_star = torch.where(is_shock, rho_star_shock, rho_star_rare)
        c_star = torch.sqrt(torch.clamp(g * (p_star + pinf_k) / rho_star, min=_TINY))
        wave = torch.sqrt((g + 1.0) / (2.0 * g) * ratio + gm1 / (2.0 * g))

        if right:
            shock_speed = u_k + c_k * wave
            head = u_k + c_k
            tail = u_star + c_star
            # right-going rarefaction: u - 2c/(g-1) is the constant invariant
            # and xi = u + c, hence c = (xi - inv)(g-1)/(g+1)
            inv = u_k - 2.0 * c_k / gm1
            c_fan = torch.clamp((g - 1.0) / (g + 1.0) * (xi - inv), min=_TINY)
            u_fan = xi - c_fan
            out_p = torch.where(
                xi > head, p_k, torch.where(xi > tail, fan_pressure(c_fan, c_k, g, pk_pinf, pinf_k), p_star))
            out_u = torch.where(xi > head, u_k, torch.where(xi > tail, u_fan, u_star))
            out_rho = torch.where(
                xi > head, rho_k,
                torch.where(xi > tail, fan_density(c_fan, c_k, g, rho_k), rho_star))
            # shock branch
            sh_rho = torch.where(xi > shock_speed, rho_k, rho_star)
            sh_u = torch.where(xi > shock_speed, u_k, u_star)
            sh_p = torch.where(xi > shock_speed, p_k, p_star)
        else:
            shock_speed = u_k - c_k * wave
            head = u_k - c_k
            tail = u_star - c_star
            # left-going rarefaction: u + 2c/(g-1) is the constant invariant
            # and xi = u - c, hence c = (inv - xi)(g-1)/(g+1)
            inv = u_k + 2.0 * c_k / gm1
            c_fan = torch.clamp((g - 1.0) / (g + 1.0) * (inv - xi), min=_TINY)
            u_fan = xi + c_fan
            out_p = torch.where(
                xi < head, p_k, torch.where(xi < tail, fan_pressure(c_fan, c_k, g, pk_pinf, pinf_k), p_star))
            out_u = torch.where(xi < head, u_k, torch.where(xi < tail, u_fan, u_star))
            out_rho = torch.where(
                xi < head, rho_k,
                torch.where(xi < tail, fan_density(c_fan, c_k, g, rho_k), rho_star))
            sh_rho = torch.where(xi < shock_speed, rho_k, rho_star)
            sh_u = torch.where(xi < shock_speed, u_k, u_star)
            sh_p = torch.where(xi < shock_speed, p_k, p_star)
        return (
            torch.where(is_shock, sh_rho, out_rho),
            torch.where(is_shock, sh_u, out_u),
            torch.where(is_shock, sh_p, out_p),
        )


def fan_pressure(c_fan, c_k, g, pk_pinf, pinf_k):
    """Isentropic pressure inside a centred rarefaction fan."""
    return pk_pinf * torch.pow(c_fan / c_k, 2.0 * g / (g - 1.0)) - pinf_k


def fan_density(c_fan, c_k, g, rho_k):
    """Isentropic density inside a centred rarefaction fan."""
    return rho_k * torch.pow(c_fan / c_k, 2.0 / (g - 1.0))


def riemann_star(rho_l, un_l, p_l, gamma_l, rho_r, un_r, p_r, gamma_r,
                 pinf_l=None, pinf_r=None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convenience wrapper: ``(p_*, u_*)`` for a normal two-material face."""
    return ExactRiemann(rho_l, un_l, p_l, gamma_l, rho_r, un_r, p_r, gamma_r,
                        pinf_l, pinf_r).solve()
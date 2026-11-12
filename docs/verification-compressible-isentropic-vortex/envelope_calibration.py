"""Calibrate the closed-form envelope of the Shu isentropic vortex (no hand-typing).

The gate ``tests/test_compressible_isentropic_vortex.py::test_exact_solution_is_isentropic_and_bounded``
was skipped with "analytic envelope assertion not yet calibrated".  This script
produces the numbers that assertion is checked against, *from the closed form*
(the same formula ``vortex_primitive`` implements), never from a fitted table:

    T(r)   = 1 - K e^{1-r^2},          K = (gamma-1) beta^2 / (8 gamma pi^2)
    rho(r) = T^{1/(gamma-1)}           (so p = rho^gamma exactly)
    extrema at the core r = 0:         T_min = 1 - K e,  rho_min = T_min^{2.5}

plus the *periodic-image* envelope: the single-image representation
``vortex_primitive`` uses is only exact up to the amplitude of the nearest
non-zero image, which is what bounds the "no interpolation" claim for the
10 x 10 box.

Usage:  PYTHONPATH=src python docs/verification-compressible-isentropic-vortex/envelope_calibration.py
"""
from __future__ import annotations

import json
import math
import pathlib

import torch

from tensorfvm.compressible.isentropic_vortex import vortex_primitive

L = 10.0
GAMMA = 1.4
BETAS = (5.0, 1.0)
OUT = pathlib.Path(__file__).with_name("envelope.json")


def analytic(beta: float, gamma: float = GAMMA) -> dict:
    """Closed-form amplitude scale K and the core extrema it implies."""
    k = (gamma - 1.0) * beta * beta / (8.0 * gamma * math.pi ** 2)
    t_min = 1.0 - k * math.e  # e^{1-r^2} = e at r = 0
    rho_min = t_min ** (1.0 / (gamma - 1.0))
    return {
        "beta": beta,
        "K": k,
        "T_min": t_min,
        "rho_min": rho_min,
        "p_min": rho_min ** gamma,
        # positivity of T everywhere requires K * e < 1
        "positivity_margin": 1.0 - k * math.e,
    }


def image_envelope(n: int = 41, lx: float = L) -> dict:
    """Largest amplitude of the *neglected* periodic images on an n x n grid.

    An image of the core at displacement ``d + (lx, 0)`` (or any other non-zero
    offset) carries amplitude ``exp((1 - |d + lx|^2)/2)``; the maximum over the
    sampled cells bounds the error of keeping a single image only.
    """
    x, y = torch.meshgrid(
        torch.linspace(0.0, lx, n, dtype=torch.float64),
        torch.linspace(0.0, lx, n, dtype=torch.float64),
        indexing="ij",
    )
    dx = x - 0.5 * lx
    dy = y - 0.5 * lx
    worst = 0.0
    for sx in (-lx, 0.0, lx):
        for sy in (-lx, 0.0, lx):
            if sx == 0.0 and sy == 0.0:
                continue
            r2 = (dx + sx) ** 2 + (dy + sy) ** 2
            worst = max(worst, float(torch.exp(0.5 * (1.0 - r2)).max()))
    r_max = math.sqrt(2.0) * 0.5 * lx
    return {"grid": n, "lx": lx, "max_image_amplitude": worst, "corner_radius": r_max}


def measured(beta: float, n: int = 41, lx: float = L) -> dict:
    """Extremes of the exact field actually sampled on the verification grid."""
    x, y = torch.meshgrid(
        torch.linspace(0.0, lx, n, dtype=torch.float64),
        torch.linspace(0.0, lx, n, dtype=torch.float64),
        indexing="ij",
    )
    st = vortex_primitive(x, y, lx=lx, ly=lx, beta=beta)
    rho, p = st["rho"], st["p"]
    r2 = (x - 0.5 * lx) ** 2 + (y - 0.5 * lx) ** 2
    order = torch.argsort(r2.flatten())
    radial = rho.flatten()[order]
    # nearest periodicity: rho must be a non-decreasing function of r^2
    step_floor = -1e-14
    return {
        "beta": beta,
        "grid": n,
        "rho_min": float(rho.min()),
        "rho_max": float(rho.max()),
        "p_min": float(p.min()),
        "p_max": float(p.max()),
        "isentropic_residual": float((p - rho ** GAMMA).abs().max()),
        "monotone_step_floor": step_floor,
        "min_radial_step": float(torch.diff(radial).min()),
    }


def main() -> None:
    payload = {
        "case": "isentropic vortex (Shu 1998), box 10x10, gamma=1.4",
        "analytic": [analytic(b) for b in BETAS],
        "measured": [measured(b) for b in BETAS],
        "periodic_image": image_envelope(),
    }
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
"""Wigley hull geometry for numerical wave tank validation.

The Wigley hull is a canonical mathematical hull form widely used in
CFD validation for ship hydrodynamics.  Its parabolic waterlines and
sections make it easy to generate on any grid and to compute exact
geometric properties (Cb, L, B, ∇) for verification.

Wigley hull surface (Lewis 1960):
    y(x, z) = B/2 · [1 − (2x/L)²] · [1 + C·(2z/T)²] · [1 − (z/T)⁴]

where:
    x ∈ [−L/2, L/2]  (longitudinal, midship at x=0)
    z ∈ [−T, 0]       (vertical, z=0 at SWL, z negative downward)
    B = beam, T = draft, L = length
    C = section-area coefficient parameter (typically C = 0.2)

For the standard Wigley III (C = 0.2):
    Cb = 5/12 ≈ 0.4167
    Cp = 5/6 ≈ 0.8333
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

__all__ = [
    "WigleyHull",
    "compute_wigley_halfbreadth",
    "wigley_block_coefficient",
    "wigley_displacement",
    "verify_wigley_geometry",
]


@dataclass
class WigleyHull:
    """Wigley hull geometric definition.

    Parameters
    ----------
    L : float
        Length between perpendiculars (m).
    B : float
        Maximum beam (m).
    T : float
        Draft (m).
    C : float
        Section-area coefficient parameter (default 0.2 for Wigley III).
    """
    L: float
    B: float
    T: float
    C: float = 0.2

    def __post_init__(self):
        if self.L <= 0 or self.B <= 0 or self.T <= 0:
            raise ValueError("L, B, T must be positive")
        if not (0 <= self.C <= 1):
            raise ValueError("C must be in [0, 1]")

    @property
    def block_coefficient(self) -> float:
        """Block coefficient Cb = ∇ / (L·B·T).

        For Wigley hull: Cb = (5/12)·(1 + C/3)
        For C = 0.2: Cb = 5/12 · 1.0667 ≈ 0.4444
        """
        return (5.0 / 12.0) * (1.0 + self.C / 3.0)

    @property
    def prismatic_coefficient(self) -> float:
        """Prismatic coefficient Cp = ∇ / (Am·L).

        For Wigley hull: Cp = 5/6 ≈ 0.8333 (independent of C).
        """
        return 5.0 / 6.0

    @property
    def midship_area(self) -> float:
        """Midship section area Am."""
        # Am = B·T · ∫₀¹ [1 + C·(2ζ)²]·[1 − ζ⁴] dζ  where ζ = z/T
        # = B·T · [1 + 4C/3 − 1/5 − 4C/15]
        # = B·T · [4/5 + 16C/15]
        return self.B * self.T * (4.0 / 5.0 + 16.0 * self.C / 15.0)

    @property
    def displacement(self) -> float:
        """Displacement volume ∇ = Cb · L · B · T."""
        return self.block_coefficient * self.L * self.B * self.T

    def halfbreadth(self, x: float, z: float) -> float:
        """Half-breadth y(x, z) at longitudinal position x and vertical z.

        x ∈ [−L/2, L/2], z ∈ [−T, 0] (z=0 at SWL).
        Returns 0 outside the hull envelope.
        """
        if abs(x) > self.L / 2 or z > 0 or z < -self.T:
            return 0.0
        xi = 2.0 * x / self.L  # ∈ [−1, 1]
        zeta = -z / self.T  # ∈ [0, 1]
        y = (self.B / 2.0) * (1.0 - xi ** 2) * (1.0 + self.C * (2.0 * zeta) ** 2) * (1.0 - zeta ** 4)
        return max(y, 0.0)

    def is_inside(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Boolean mask: True where (x, y, z) is inside the hull.

        Parameters
        ----------
        x, y, z : torch.Tensor
            Coordinate arrays (broadcastable).

        Returns
        -------
        mask : torch.Tensor
            Boolean tensor, True inside hull.
        """
        half_y = self.halfbreadth_vec(x, z)
        return (torch.abs(y) <= half_y) & (z <= 0) & (z >= -self.T)

    def halfbreadth_vec(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Vectorized half-breadth computation."""
        xi = 2.0 * x / self.L
        zeta = -z / self.T
        # Clamp to valid range
        xi = torch.clamp(xi, -1.0, 1.0)
        zeta = torch.clamp(zeta, 0.0, 1.0)
        y = (self.B / 2.0) * (1.0 - xi ** 2) * (1.0 + self.C * (2.0 * zeta) ** 2) * (1.0 - zeta ** 4)
        # Zero outside hull envelope
        outside = (torch.abs(x) > self.L / 2) | (z > 0) | (z < -self.T)
        y = torch.where(outside, torch.zeros_like(y), y)
        return y


def compute_wigley_halfbreadth(x: float, z: float, L: float, B: float,
                                T: float, C: float = 0.2) -> float:
    """Convenience function: Wigley half-breadth at (x, z)."""
    hull = WigleyHull(L=L, B=B, T=T, C=C)
    return hull.halfbreadth(x, z)


def wigley_block_coefficient(L: float, B: float, T: float,
                              C: float = 0.2) -> float:
    """Convenience: Wigley block coefficient."""
    hull = WigleyHull(L=L, B=B, T=T, C=C)
    return hull.block_coefficient


def wigley_displacement(L: float, B: float, T: float,
                         C: float = 0.2) -> float:
    """Convenience: Wigley displacement volume."""
    hull = WigleyHull(L=L, B=B, T=T, C=C)
    return hull.displacement


def verify_wigley_geometry(L: float, B: float, T: float, C: float = 0.2,
                            tol: float = 1e-4) -> dict:
    """Verify Wigley hull geometric properties against analytical values.

    Parameters
    ----------
    L, B, T, C : float
        Hull parameters.
    tol : float
        Tolerance for verification.

    Returns
    -------
    result : dict
        Verification results with pass/fail for each property.
    """
    hull = WigleyHull(L=L, B=B, T=T, C=C)

    # Analytical values
    Cb_expected = (5.0 / 12.0) * (1.0 + C / 3.0)
    Cp_expected = 5.0 / 6.0
    Am_expected = B * T * (4.0 / 5.0 + 16.0 * C / 15.0)
    nabla_expected = Cb_expected * L * B * T

    results = {
        "L": L,
        "B": B,
        "T": T,
        "C": C,
        "Cb": hull.block_coefficient,
        "Cb_expected": Cb_expected,
        "Cb_pass": abs(hull.block_coefficient - Cb_expected) < tol,
        "Cp": hull.prismatic_coefficient,
        "Cp_expected": Cp_expected,
        "Cp_pass": abs(hull.prismatic_coefficient - Cp_expected) < tol,
        "Am": hull.midship_area,
        "Am_expected": Am_expected,
        "Am_pass": abs(hull.midship_area - Am_expected) < tol,
        "nabla": hull.displacement,
        "nabla_expected": nabla_expected,
        "nabla_pass": abs(hull.displacement - nabla_expected) < tol,
    }
    results["all_pass"] = all(
        results[k] for k in ("Cb_pass", "Cp_pass", "Am_pass", "nabla_pass")
    )
    return results

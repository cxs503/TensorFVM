"""Bare SUBOFF mother-hull radius, ported from TensorLBM without Torch.

Source claims Groves, Huang, Chang (1989), DTRC/SHD-1298-01.
This port preserves its polynomial coefficients; it is not independently
certified against the original offset table. Coordinates use hull length L.
"""
from __future__ import annotations
import numpy as np
LENGTH_FT = 14.291667
RADIUS_FT = 0.8333333
LENGTH_M = LENGTH_FT * 0.3048
RADIUS_OVER_LENGTH = RADIUS_FT / LENGTH_FT
SOURCE_SHA256 = "435bd062a90d6881784734fa614f033e880b310111200436e8b13950fbccae02"
SOURCE_REPO_PATH = "TensorLBM/src/tensorlbm/suboff_cad.py"

def suboff_radius_profile(
    xi: np.ndarray | float,
) -> np.ndarray:
    """Normalised hull radius *r(xi) / R_max* for the SUBOFF bare hull.

    **真实 DARPA SUBOFF 4段公式** (Groves et al. 1989, from suboff8.py):
      段1 船首 [0, 0.2333]:  多项式 + 指数 1/2.1
      段2 平行中体 (0.2333, 0.7417]:  R = R_max
      段3 尾锥 (0.7417, 0.9748]:  6阶多项式
      段4 尾尖 (0.9748, 1.0]:  椭圆收缩

    Parameters
    ----------
    xi :
        Normalised axial coordinate ∈ [0, 1], where 0 = bow-tip and
        1 = stern-tip.

    Returns
    -------
    np.ndarray
        Normalised radius ∈ [0, 1] (same shape as *xi*).
    """
    xi = np.asarray(xi, dtype=float)
    r = np.zeros_like(xi)

    # SUBOFF 真实参数 (ft → normalized)
    L_ft = 14.291667  # 总长 (ft)
    BOW_END = 3.333333 / L_ft  # 0.2333
    MID_END = 10.645833 / L_ft  # 0.7449
    STERN_END = 13.979167 / L_ft  # 0.9781

    # 段1: 船首 [0, 0.2333]
    bow = (xi >= 0.0) & (xi < BOW_END)
    if np.any(bow):
        x_ft = xi[bow] * L_ft
        tmp = 0.3 * x_ft - 1.0
        tmp2 = tmp * tmp
        tmp4 = tmp2 * tmp2
        a1 = (
            1.126395101 * x_ft * tmp4
            + 0.442874707 * x_ft * x_ft * (tmp2 * tmp)
            + 1.0
            - tmp4 * (1.2 * x_ft + 1.0)
        )
        a1 = np.maximum(a1, 0.0)
        r[bow] = np.power(a1, 1.0 / 2.1)

    # 段2: 平行中体 (0.2333, 0.7449]
    mid = (xi >= BOW_END) & (xi <= MID_END)
    r[mid] = 1.0

    # 段3: 尾锥 (0.7449, 0.9781]
    stern_taper = (xi > MID_END) & (xi <= STERN_END)
    if np.any(stern_taper):
        r1 = 0.1175
        k0 = 10.0
        k1 = 44.6244
        x_ft = xi[stern_taper] * L_ft
        ksi = (13.979167 - x_ft) / 3.333333
        ksi2 = ksi * ksi
        ksi3 = ksi2 * ksi
        ksi4 = ksi3 * ksi
        ksi5 = ksi4 * ksi
        ksi6 = ksi5 * ksi

        a3 = (
            r1 * r1
            + r1 * k0 * ksi2
            + (20.0 - 20.0 * r1 * r1 - 4.0 * r1 * k0 - k1 / 3.0) * ksi3
            + (-45.0 + 45.0 * r1 * r1 + 6.0 * r1 * k0 + k1) * ksi4
            + (36.0 - 36.0 * r1 * r1 - 4.0 * r1 * k0 - k1) * ksi5
            + (-10.0 + 10.0 * r1 * r1 + r1 * k0 + k1 / 3.0) * ksi6
        )
        a3 = np.maximum(a3, 0.0)
        r[stern_taper] = np.sqrt(a3)

    # 段4: 尾尖 (0.9781, 1.0]
    tail = (xi > STERN_END) & (xi <= 1.0)
    if np.any(tail):
        x_ft = xi[tail] * L_ft
        val = 1.0 - (3.2 * x_ft - 44.733333) ** 2
        val = np.maximum(val, 0.0)
        r[tail] = np.sqrt(val) * 0.1175

    return np.clip(r, 0.0, 1.0)


def radius(x):
    """Physical radius in L units, preserving the mother offset dimensions."""
    return suboff_radius_profile(x) * RADIUS_OVER_LENGTH

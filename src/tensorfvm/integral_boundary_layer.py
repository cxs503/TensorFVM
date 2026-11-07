"""Integral boundary layer: Thwaites (laminar) + Head (turbulent).

The twin-hull resistance analysis needs a wall-shear and displacement-thickness
field on the hull surface, marched from bow to stern.  This module implements the
classical two-equation integral method used for ship form-factor work:

* laminar: Thwaites' method, ``theta^2 = 0.45 nu / Ue^6 * int Ue^5 ds``, with the
  Thwaites shape-factor and skin-friction correlations;
* turbulent: Head's entrainment method,
  ``dtheta/ds = cf/2 - (H+2) theta/Ue dUe/ds`` and
  ``d(Ue theta H1)/ds = Ue * 0.0306 (H1-3)^(-0.6169)``,
  closed by the Ludwieg-Tillmann skin-friction law
  ``cf = 0.246 10^(-0.678 H) Re_theta^(-0.268)`` and the Cebeci-Bradshaw
  ``H1(H)`` relation;
* transition: ``Re_theta = 320`` (Thwaites/forced-transition flag available).

The turbulence closure is therefore *integral*; no k/omega transport equation is
solved here (this is not an SST k-omega field solution).

Grid convention: all array arguments are indexed as ``(station, girth)`` with
index 0 at the **bow**, i.e. in streamwise marching order.  ``ds[i]`` is the
streamwise length of the surface strip ending at station ``i`` and ``areas[i, j]``
the wetted area of that strip on girth line ``j``.

Verification: a flat plate marched with the turbulent branch reproduces the
ITTC-1957 line to a few percent (see ``tests/test_twin_hull_form_factor.py``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def ittc57_cf(Re: float) -> float:
    """ITTC-1957 (Schoenherr) flat-plate friction line."""
    return 0.075 / (math.log10(Re) - 2.0) ** 2


def h1_of_h(H):
    """Cebeci-Bradshaw H1(H) relation."""
    H = np.clip(np.asarray(H, dtype=float), 1.0001, 12.0)
    return np.where(
        H <= 1.6,
        3.3 + 0.8234 * np.power(np.maximum(H - 1.1, 1e-4), -1.287),
        3.3 + 1.5501 * np.power(np.maximum(H - 0.6778, 1e-4), -3.064),
    )


_H1_SPLIT = float(np.atleast_1d(h1_of_h(np.array([1.6])))[0])


def h_of_h1(H1):
    """Inverse of :func:`h1_of_h`.

    ``h1_of_h`` is the Cebeci-Bradshaw relation, which is *not* monotonic across
    its branch switch: values below ``H1(1.6) = 5.309`` belong to the ``H > 1.6``
    branch and values above it to the ``H <= 1.6`` branch.  Selecting the branch
    the other way round made the inverse return ``H ~ 10`` for ``H = 4`` and
    tripped the ``H > H_max`` separation guard as soon as ``H`` exceeded 1.6.
    """
    H1 = np.clip(np.asarray(H1, dtype=float), 3.31, 60.0)
    lo = 1.1 + np.power(0.8234 / np.maximum(H1 - 3.3, 1e-6), 1.0 / 1.287)
    hi = 0.6778 + np.power(1.5501 / np.maximum(H1 - 3.3, 1e-6), 1.0 / 3.064)
    return np.where(H1 <= _H1_SPLIT, hi, lo)


def ludwieg_tillmann_cf(H: float, Re_theta: float) -> float:
    """Local skin friction, ``0.246 10^(-0.678 H) Re_theta^(-0.268)``."""
    return float(0.246 * 10.0 ** (-0.678 * H) * max(Re_theta, 1.0) ** (-0.268))


@dataclass
class BoundaryLayerResult:
    """Integral BL fields on the (station, girth) grid plus integrated friction."""

    cf: np.ndarray
    theta: np.ndarray
    H: np.ndarray
    delta_star: np.ndarray
    separated: np.ndarray
    transition_station: np.ndarray
    R_F: float
    S_wet: float
    C_F: float
    C_F_ittc57: float
    Re: float

    def as_dict(self, mask: np.ndarray | None = None) -> dict:
        """Statistics of the BL fields.

        ``mask`` restricts the statistics to the strips that carry wetted area;
        the extrapolated (vanishing) fore/aft tips are excluded because the strip
        boundary layer is not defined there.
        """
        sel = np.ones(self.cf.shape, dtype=bool) if mask is None else np.asarray(mask, bool)
        sel = sel & ~self.separated  # separated strips are post-model, excluded from statistics
        finite = np.isfinite(self.H) & sel
        cf_s = self.cf[sel]
        return dict(
            R_F=self.R_F, S_wet=self.S_wet, C_F=self.C_F, C_F_ittc57=self.C_F_ittc57,
            Re=self.Re,
            cf_min=float(np.nanmin(cf_s)), cf_max=float(np.nanmax(cf_s)),
            H_min=float(np.nanmin(self.H[finite])), H_max=float(np.nanmax(self.H[finite])),
            theta_max=float(np.nanmax(self.theta[sel])),
            delta_star_max=float(np.nanmax(self.delta_star[sel])),
            separated_fraction=float(np.mean(self.separated[sel])),
        )


def march_boundary_layer(Ve: np.ndarray, ds: np.ndarray, areas: np.ndarray, *,
                         nu: float, U: float, Re: float, rho: float = 998.0,
                         transition_Re_theta: float = 320.0,
                         force_turbulent: bool = False,
                         n_sub: int = 6, H_max: float = 4.0,
                         valid: np.ndarray | None = None) -> BoundaryLayerResult:
    """March the integral BL along axis 0 (bow -> stern) on every girth line.

    ``valid`` (optional, same shape as ``Ve``) marks strips that carry wetted
    area.  The march is confined to the contiguous valid station range and the
    end values are extrapolated, because the parametrically collapsing fore/aft
    tips give an edge velocity that is meaningless for a strip BL (the section
    vanishes).  Excluded strips carry ~0 area, so the friction integral is not
    affected; only the displacement field at the very tips is extrapolated.
    """
    Ve = np.asarray(Ve, dtype=float)
    ds = np.asarray(ds, dtype=float)
    areas = np.asarray(areas, dtype=float)
    nstat, ngirth = Ve.shape
    cf = np.zeros_like(Ve)
    theta = np.zeros_like(Ve)
    Hf = np.full_like(Ve, np.nan)
    dstar = np.zeros_like(Ve)
    sep = np.zeros(Ve.shape, dtype=bool)
    trans = np.full(ngirth, -1, dtype=int)
    floor = 0.05 * U

    for j in range(ngirth):
        if not np.any(areas[:, j] > 0):
            continue
        if valid is None:
            i_lo, i_hi = 0, nstat - 1
        else:
            idx = np.where(valid[:, j])[0]
            if idx.size < 4:
                continue
            i_lo, i_hi = int(idx[0]), int(idx[-1])
        ve = np.maximum(Ve[:, j], floor)
        s = np.maximum(ds[:, j], 1e-12)
        # the first strip starts at the stagnation/leading edge: its whole
        # length contributes to int(Ue^5 ds)
        int5 = ve[i_lo] ** 5 * s[i_lo]
        turb_from = nstat
        for i in range(i_lo, i_hi + 1):
            if i > 0:
                int5 += 0.5 * (ve[i - 1] ** 5 + ve[i] ** 5) * s[i]
            th_l = math.sqrt(max(0.45 * nu * int5, 0.0) / ve[i] ** 6)
            lam = 0.0
            if i > 0:
                lam = th_l * th_l / nu * (ve[i] - ve[i - 1]) / s[i]
            H_l = float(np.clip(2.61 - 3.75 * lam + 5.24 * lam * lam, 1.4, 4.0))
            f_lam = float(np.clip(0.22 + 1.57 * lam - 1.8 * lam * lam, 0.0, 1.0))
            theta[i, j] = th_l
            Hf[i, j] = H_l
            dstar[i, j] = H_l * th_l
            cf[i, j] = 2.0 * f_lam * nu / max(ve[i] * th_l, 1e-30)
            if not force_turbulent and th_l * ve[i] / nu > transition_Re_theta:
                turb_from = i
                trans[j] = i
                break
        if force_turbulent:
            turb_from = i_lo
            trans[j] = i_lo
        if turb_from >= nstat:
            continue
        # ---- turbulent branch (Head's entrainment method)
        i0 = turb_from
        th_c = theta[i0, j] if theta[i0, j] > 0 else math.sqrt(0.45 * nu * s[i0] / ve[i0])
        H1_c = float(np.atleast_1d(h1_of_h(np.array([1.45])))[0])
        frozen = False
        for i in range(i0, nstat):
            if frozen:
                # an integral method is invalid once the layer has separated; the
                # march is frozen instead of letting theta run away
                theta[i, j] = theta[i - 1, j]
                Hf[i, j] = Hf[i - 1, j]
                dstar[i, j] = dstar[i - 1, j]
                cf[i, j] = cf[i - 1, j]
                sep[i, j] = True
                continue
            for k in range(n_sub):
                frac0, frac1 = k / n_sub, (k + 1) / n_sub
                if i == i_lo:
                    v_now = ve[i]
                    dvds = 0.0
                else:
                    v_now = ve[i - 1] + (ve[i] - ve[i - 1]) * frac0
                    dvds = (ve[i] - ve[i - 1]) / s[i]
                v_next = v_now
                if i > i_lo:
                    v_next = ve[i - 1] + (ve[i] - ve[i - 1]) * frac1
                ds_sub = s[i] / n_sub
                H_c = float(np.atleast_1d(h_of_h1(np.array([H1_c])))[0])
                Re_th = max(v_now * th_c / nu, 1.0)  # noqa: F841 (kept for clarity)
                cf_c = ludwieg_tillmann_cf(H_c, Re_th)
                dth = cf_c / 2.0 - (H_c + 2.0) * th_c / v_now * dvds
                ent = 0.0306 * max(H1_c - 3.0, 0.05) ** (-0.6169)
                dH1 = (v_now * ent - H1_c * (v_now * dth + th_c * dvds)) / max(v_now * th_c, 1e-30)
                th_c = max(th_c + dth * ds_sub, 1e-12)
                H1_c = float(np.clip(H1_c + dH1 * ds_sub, 3.35, 40.0))
                if float(np.atleast_1d(h_of_h1(np.array([H1_c])))[0]) > H_max:
                    sep[i, j] = True
                    H1_c = float(np.atleast_1d(h1_of_h(np.array([H_max])))[0])
                v_now = v_next
            H_c = float(np.atleast_1d(h_of_h1(np.array([H1_c])))[0])
            Re_th = max(ve[i] * th_c / nu, 1.0)
            theta[i, j] = th_c
            Hf[i, j] = H_c
            dstar[i, j] = H_c * th_c
            cf[i, j] = ludwieg_tillmann_cf(H_c, Re_th)
            frozen = bool(sep[i, j])
        # extrapolate the end values across the excluded (vanishing) tip strips
        for arr in (cf, theta, Hf, dstar):
            arr[:i_lo, j] = arr[i_lo, j]
            arr[i_hi + 1:, j] = arr[i_hi, j]
        sep[:i_lo, j] = sep[i_lo, j]
        sep[i_hi + 1:, j] = sep[i_hi, j]

    tau = 0.5 * rho * cf * np.where(np.isfinite(Ve), Ve, 0.0) ** 2
    R_F = float(np.sum(tau * areas))
    S_wet = float(np.sum(areas))
    q = 0.5 * rho * U * U
    return BoundaryLayerResult(cf, theta, Hf, dstar, sep, trans, R_F, S_wet,
                               R_F / (q * S_wet) if S_wet > 0 else float("nan"),
                               ittc57_cf(Re), Re)
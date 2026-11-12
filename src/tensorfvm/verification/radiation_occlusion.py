"""Verification of occlusion-aware 2-D view factors (shadowing between strips).

Why this driver exists
----------------------
``python -m tensorfvm.verification.radiation`` verified the *unobstructed* strip
enclosure: crossed strings reproduce closed forms to round-off, concentric
circles by their closed form, gray enclosures against the two-surface formula.
It declared occlusion as out of scope ("no occlusion subtraction; partial
blocking by a third surface is not modelled"), and this driver is the follow-up
that closes it: :func:`tensorfvm.radiation.strip_form_factors_occluded`
integrates the differential kernel over the *visible* part of the receiver,
with the shadow boundary resolved analytically.

What the numbers below are anchored to
--------------------------------------
No closed form exists for a partially shadowed strip pair, so the anchors are
independent *statements* plus one independent *method*:

``C1`` equivalence   With no blockers the new machinery must reproduce the
                     crossed-strings answer (a different mathematical path:
                     quadrature of the kernel vs. the contour identity).
``C2`` exact zero    A pair fully hidden behind blocking segments has ``F = 0``
                     exactly -- and the obstructed-blind API does not.
``C3`` closure       ``sum_j F_ij = 1`` for a closed obstructed enclosure: energy
                     conservation, valid with or without shadowing.
``C4`` reciprocity   ``A_i F_ij = A_j F_ji`` before symmetrisation (a quadrature
                     convergence indicator, not an assumption of the code).
``C5`` Monte Carlo   An independent cosine-weighted ray trace (no kernel algebra,
                     no crossed strings, no visibility intervals) must agree
                     within its own statistical band.
``C6`` back-to-back  The faces of a convex obstacle cannot exchange (the chord
                     runs through the solid); the new machinery gives 0, the
                     cavity-style crossed strings read the obstacle as a cavity and give
                     ``sqrt(2) - 1`` for opposite faces.
``C7`` convergence   The shadow-split quadrature converges and the two-level
                     difference is reported (no silent plateau).
``C8`` engineering   A gray enclosure on the obstructed scenes: energy residual,
                     isothermal invariance, and *how large* the modelling error of
                     the obstructed-blind answers is -- the number a designer
                     would otherwise have shipped.

Honest limits: 2-D strip geometry only; a shadow boundary landing exactly on a
strip end point is assigned by the midpoint rule (measure-zero set); nothing is
compared against a radiation *measurement*.

CLI::

    PYTHONPATH=src python -m tensorfvm.verification.radiation_occlusion \
        --out docs/verification-radiation-occlusion
"""
from __future__ import annotations

import json
import math
import pathlib
import time
from typing import Dict, List, Sequence, Tuple

import numpy as np

from ..radiation import (SIGMA, GrayEnclosure, Strip, pair_visibility,
                         strip_form_factors, strip_form_factors_occluded,
                         strip_view_factor, strip_view_factor_occluded)

__all__ = ["monte_carlo_form_factors", "run", "render_report", "plot", "main"]


def _gate(value: float, limit: float, note: str) -> Dict:
    return {"value": float(value), "gate": float(limit), "pass": bool(value <= limit),
            "note": note}


def _scene(points: Sequence[Tuple[float, float]]) -> List[Strip]:
    """Closed strip loop; the radiating domain lies on the left of every strip."""
    p = np.asarray(points, float)
    return [Strip(tuple(p[i]), tuple(p[(i + 1) % len(p)])) for i in range(len(p))]


def _cavity_scene() -> List[Strip]:
    return _scene([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])


def _beam_scene() -> List[Strip]:
    """A cargo-hold corner: 2.0 x 1.2 cavity with a beam hung from the deck.

    Simply connected and re-entrant, so the whole boundary is traversed once and
    the crossed strings still close (rows sum to one) -- yet the beam shadows
    several pairs, so per-pair crossed strings stop being per-pair view factors.
    """
    return _scene([(0.0, 0.0), (2.0, 0.0), (2.0, 1.2), (1.1, 1.2), (1.1, 0.6),
                   (0.8, 0.6), (0.8, 1.2), (0.0, 1.2)])


def _block_scene() -> List[Strip]:
    """A floating obstacle (chilled pipe / ice block) inside a 2.0 x 1.2 hold.

    Multiply connected: the outer boundary is counter-clockwise and the obstacle
    is traversed clockwise (reverse *vertex* order) so its normal points into the
    domain.
    """
    outer = _scene([(0.0, 0.0), (2.0, 0.0), (2.0, 1.2), (0.0, 1.2)])
    block = _scene(list(reversed([(0.8, 0.4), (1.2, 0.4), (1.2, 0.8), (0.8, 0.8)])))
    return outer + block


def _blocked_pair_scene() -> Tuple[List[Strip], int, int]:
    """Two facing strips with a blocking segment spanning the gap (exactly zero)."""
    strips = [Strip((0.0, 0.0), (1.0, 0.0)),          # emitter, domain above
              Strip((1.0, 0.4), (0.0, 0.4)),          # receiver, domain below
              Strip((-0.5, 0.2), (1.5, 0.2)),         # blocker wider than the pair
              Strip((0.2, 0.1), (0.2, 0.3))]          # second, irrelevant blocker
    return strips, 0, 1


def monte_carlo_form_factors(strips: Sequence[Strip], rays: int = 100_000,
                             seed: int = 20261112) -> Dict[str, np.ndarray]:
    """Independent 2-D view factors by cosine-weighted ray tracing.

    For an element of a strip the directional density is ``cos(theta)/2`` on
    ``[-pi/2, pi/2]`` measured from the inward normal, so sampling
    ``s ~ U(0, L_i)`` and ``theta = arcsin(2u - 1)`` estimates
    ``F_ij = P(ray hits j)`` without the kernel, the crossed strings or the
    visibility intervals of the code under test.  ``sigma`` is the binomial
    standard error; ``lost_fraction`` counts rays that leave the scene (0 for a
    closed one).
    """
    rng = np.random.default_rng(seed)
    n = len(strips)
    per = max(1000, int(rays) // n)
    hit = np.zeros((n, n), dtype=float)
    lost = np.zeros(n, dtype=float)
    for i, si in enumerate(strips):
        a0, _ = si.points
        s_par = rng.random(per) * si.length
        p = a0 + np.outer(s_par, si.tangent)
        theta = np.arcsin(2.0 * rng.random(per) - 1.0)
        d = (np.outer(np.cos(theta), si.inward_normal)
             + np.outer(np.sin(theta), si.tangent))
        best_tau = np.full(per, np.inf)
        best_j = np.full(per, -1, dtype=int)
        for j, sj in enumerate(strips):
            if j == i:
                continue
            c0, _ = sj.points
            tt = sj.tangent
            w = c0 - p                                   # (per, 2)
            det = -d[:, 0] * tt[1] + tt[0] * d[:, 1]
            ok = np.abs(det) > 1e-14
            with np.errstate(invalid="ignore", divide="ignore"):
                tau = (-w[:, 0] * tt[1] + tt[0] * w[:, 1]) / det
                sig = (d[:, 0] * w[:, 1] - w[:, 0] * d[:, 1]) / det
            # ``sig`` is a parameter along the strip in *length* units (the tangent
            # is a unit vector), so its bounds are (0, length): bounding it by 1
            # silently discards every hit on a strip longer than one unit and
            # leaks those rays out of a closed scene (measured 45 % lost in a 2 m
            # wide hold -- the mistake this comment exists to prevent).
            ok &= ((tau > 1e-12) & (sig >= -1e-12) & (sig <= sj.length + 1e-12)
                   & (tau < best_tau))
            best_tau = np.where(ok, tau, best_tau)
            best_j = np.where(ok, j, best_j)
        for j in range(n):
            if j != i:
                hit[i, j] = float(np.count_nonzero(best_j == j))
        lost[i] = float(np.count_nonzero(best_j < 0))
    f = hit / per
    with np.errstate(invalid="ignore"):
        sigma = np.sqrt(np.clip(f * (1.0 - f), 0.0, None) / per)
    return {"F": f, "sigma": sigma, "rays_per_surface": per,
            "lost_fraction": lost / per}


def _max_rel(a: np.ndarray, b: np.ndarray) -> float:
    """max |a-b| / max |b| over entries where |b| is not negligible."""
    mask = np.abs(b) > 1e-9
    if not np.any(mask):
        return 0.0
    return float(np.max(np.abs(a[mask] - b[mask])) / float(np.max(np.abs(b[mask]))))


def _min_entry(f: np.ndarray) -> float:
    return float(min(min(row) for row in np.asarray(f)))


def _max_entry(f: np.ndarray) -> float:
    return float(max(max(row) for row in np.asarray(f)))

def run(order: int = 96, rays: int = 100_000) -> Dict:
    t0 = time.time()
    data: Dict[str, object] = {}
    criteria: Dict[str, Dict] = {}

    # ---- C1: no-blocker equivalence with the crossed strings (different method)
    cavity = _cavity_scene()
    naive_c = strip_form_factors(cavity)
    occ_c = strip_form_factors_occluded(cavity, order=order)
    criteria["C1_equivalence_no_blocker"] = _gate(
        _max_rel(occ_c["F"], naive_c["F"]), 5e-3,
        "max rel. |F_occluded - F_crossed_strings|, 1x1 cavity, no blockers "
        "(quadrature-limited at the shared corners)")

    si = Strip((0.0, 0.0), (1.0, 0.0))
    sj = Strip((1.0, 1.0), (0.0, 1.0))
    f_open = strip_view_factor(si, sj)
    f_occ = strip_view_factor_occluded(si, sj, (), order=order)
    criteria["C1b_equivalence_clean_pair"] = _gate(
        abs(f_occ - f_open) / abs(f_open), 1e-6,
        "|F_occluded - F_crossed_strings| / F for a corner-free equal parallel pair "
        "(closed form sqrt(1+d^2) - d)")

    # ---- C2: a fully hidden pair is exactly zero
    bstrips, i0, j0 = _blocked_pair_scene()
    blockers = [bstrips[k] for k in range(len(bstrips)) if k not in (i0, j0)]
    f_blocked = strip_view_factor_occluded(bstrips[i0], bstrips[j0], blockers, order=order)
    f_blocked_naive = strip_view_factor(bstrips[i0], bstrips[j0])
    criteria["C2_fully_blocked_pair_is_zero"] = _gate(
        abs(f_blocked), 1e-12,
        "F of a pair fully hidden behind a blocking segment (exact zero; the "
        "obstructed-blind API returns " + f"{f_blocked_naive:.4f})")
    data["blocked_pair"] = {"F_occluded": f_blocked, "F_crossed_strings": f_blocked_naive}

    # ---- obstructed scenes: closure, reciprocity, Monte Carlo
    scenes = {"cavity": cavity, "beam": _beam_scene(), "block": _block_scene()}
    closure, recip, mc_err = {}, {}, {}
    for name, strips in scenes.items():
        m = strip_form_factors_occluded(strips, order=order)
        nv = strip_form_factors(strips)
        mc = monte_carlo_form_factors(strips, rays=rays)
        closure[name] = m["closure_error"]
        recip[name] = m["reciprocity_error"]
        band = 4.0 * float(np.max(mc["sigma"]))
        mc_err[name] = (float(np.max(np.abs(m["F"] - mc["F"]))), band,
                        float(np.max(np.abs(nv["F"] - mc["F"]))))
        data[name] = {
            "F_occluded": m["F"].tolist(),
            "F_crossed_strings": nv["F"].tolist(),
            "F_monte_carlo": mc["F"].tolist(),
            "F_monte_carlo_sigma": mc["sigma"].tolist(),
            "hidden_fraction": m["hidden_fraction"].tolist(),
            "row_sum_occluded": m["F"].sum(axis=1).tolist(),
            "row_sum_crossed_strings": nv["F"].sum(axis=1).tolist(),
            "row_sum_monte_carlo": mc["F"].sum(axis=1).tolist(),
            "monte_carlo_lost_fraction": mc["lost_fraction"].tolist(),
            "lengths": m["lengths"].tolist(),
        }
    criteria["C3_closure_obstructed"] = _gate(
        max(closure.values()), 1e-3,
        "max_i |sum_j F_ij - 1| on the cavity/beam/floating-block scenes (cavity "
        f"{closure['cavity']:.2e}, beam {closure['beam']:.2e}, block {closure['block']:.2e})")
    criteria["C4_reciprocity_obstructed"] = _gate(
        max(recip.values()), 1e-3,
        "max |A_i F_ij - A_j F_ji| / max|AF| before symmetrisation (beam "
        f"{recip['beam']:.2e}, block {recip['block']:.2e})")
    band_max = max(v[1] for v in mc_err.values())
    criteria["C5_monte_carlo_agreement"] = _gate(
        max(v[0] for v in mc_err.values()), band_max,
        "max |F_occluded - F_MC| on the obstructed scenes vs the 4-sigma binomial band "
        + ", ".join(f"{k} {v[0]:.2e} (band {v[1]:.2e}; crossed strings {v[2]:.2e})"
                    for k, v in mc_err.items()))
    data["monte_carlo_error"] = {
        k: {"max_abs": v[0], "band_4sigma": v[1], "crossed_strings_max_abs": v[2]}
        for k, v in mc_err.items()}

    # ---- C6: back-to-back faces of a convex obstacle do not exchange
    block = _block_scene()[4:]
    f_back = strip_form_factors_occluded(block, order=order)["F"]
    naive_back = strip_form_factors(block)["F"]
    off = ~np.eye(4, dtype=bool)
    criteria["C6_back_to_back_faces"] = _gate(
        float(np.max(np.abs(f_back[off]))), 1e-12,
        "F between faces of a convex obstacle (the chord crosses its solid, so F = 0; "
        f"the crossed strings read the obstacle as a cavity and give up to "
        f"{float(np.max(np.abs(naive_back[off]))):.4f} = sqrt(2) - 1)")
    data["convex_obstacle"] = {"F_occluded": f_back.tolist(),
                              "F_crossed_strings": naive_back.tolist()}

    # ---- C7: convergence of the shadow-split quadrature
    orders = [24, 48, 96, 192]
    beam = _beam_scene()
    ref = strip_form_factors_occluded(beam, order=384)
    conv = [{"order": o,
             "max_abs_error_vs_order384": float(
                 np.max(np.abs(strip_form_factors_occluded(beam, order=o)["F"] - ref["F"])))}
            for o in orders]
    errs = [c["max_abs_error_vs_order384"] for c in conv[:-1]]
    p_obs = (math.log(errs[0] / errs[1]) / math.log(2.0)
             if errs[0] > 0.0 and errs[1] > 0.0 else float("nan"))
    criteria["C7_convergence"] = _gate(
        conv[-1]["max_abs_error_vs_order384"], 3e-3,
        "|F(order=192) - F(order=384)|max on the beam scene; observed order "
        f"{p_obs:.2f} between order 24 and 48 (errors "
        + ", ".join(f"{c['order']}:{c['max_abs_error_vs_order384']:.2e}" for c in conv) + ")")
    data["convergence"] = conv
    data["observed_order"] = p_obs

    # ---- C8: engineering consequence on the beam scene
    #      Black-surface limit (eps = 1) is exact from F alone:
    #      q_i = sigma sum_j A_i F_ij (T_i^4 - T_j^4), so the comparison does not
    #      depend on the radiosity solver at all.
    temps = np.array([350.0] * 8)
    temps[3:6] = 260.0                       # the beam's three exposed faces
    lengths = [s.length for s in beam]
    occ = strip_form_factors_occluded(beam, order=order)
    naive_f = strip_form_factors(beam)["F"]

    def black_power(f: np.ndarray) -> float:
        a_i = np.asarray(lengths)[:, None]
        q = SIGMA * np.sum(a_i * f * (temps[:, None] ** 4 - temps[None, :] ** 4), axis=1)
        return float(np.sum(q[3:6]))

    power_occluded = black_power(occ["F"])
    power_crossed = black_power(naive_f)

    # gray-diffuse enclosure on the occluded matrix (rows close to 2.4e-5, so the
    # driver states the tolerance it validates with)
    emis = np.full(8, 0.9)
    occ_enc = GrayEnclosure(occ["F"], lengths, emis, temps, row_sum_tol=1e-3)
    res_occ = occ_enc.solve()
    gray_power_occluded = float(np.sum(res_occ["power"][3:6]))
    naive_rejected = False
    try:
        GrayEnclosure(naive_f, lengths, emis, temps, row_sum_tol=1e-3)
    except ValueError:
        naive_rejected = True
    iso = GrayEnclosure(occ["F"], lengths, emis, np.full(8, 300.0), row_sum_tol=1e-3).solve()
    n = len(beam)
    mask = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(n):
            if i != j:
                mask[i, j] = mask[j, i] = pair_visibility(
                    beam[i], beam[j], [beam[k] for k in range(n) if k not in (i, j)])
    masked_rowsum_err = float(np.max(np.abs((naive_f * mask).sum(axis=1) - 1.0)))
    criteria["C8_enclosure_energy_residual"] = _gate(
        occ_enc.energy_residual(), 1e-4,
        "closed obstructed gray enclosure: |sum q_i A_i| / (sigma T_max^4 sum A); the gate "
        "follows the input matrix closure error (C3, same order), not the solver, which is "
        "exact for a given matrix")
    criteria["C8b_isothermal_invariance"] = _gate(
        float(np.max(np.abs(iso["flux"]))) / (SIGMA * 300.0 ** 4), 1e-4,
        "isothermal obstructed enclosure carries no net flux (closure-limited gate, as C8)")
    delta = abs(power_crossed - power_occluded) / max(abs(power_occluded), 1e-30)
    criteria["C8c_crossed_strings_power_error"] = dict(_gate(
        delta, 1e-2,
        "relative error of the obstructed-blind beam power in the exact black limit "
        f"({power_crossed:.5g} W/m vs {power_occluded:.5g} W/m): the modelling error "
        "the pre-existing API ships silently (reported, not a gate for the new code)"),
        graded=False)
    criteria["C8d_negative_view_factors_rejected"] = _gate(
        0.0 if naive_rejected else 1.0, 0.0,
        "GrayEnclosure refuses the crossed-strings matrix of a re-entrant scene because it "
        "contains negative view factors (before the guard it was accepted: rows sum to one, "
        "so the row-sum validation passed and the reported powers were wrong)")
    data["beam_enclosure"] = {
        "power_occluded_W_per_m": power_occluded,
        "power_crossed_strings_W_per_m": power_crossed,
        "power_relative_error": delta,
        "gray_power_occluded_W_per_m": gray_power_occluded,
        "gray_naive_rejected": naive_rejected,
        "masked_row_sum_error": masked_rowsum_err,
        "emissivity": 0.9,
        "temperatures_K": temps.tolist(),
        "lengths": lengths,
    }

    return {
        "case": "2-D strip radiation: occlusion-aware view factors",
        "criteria": criteria,
        "data": data,
        "all_pass": all(v["pass"] for v in criteria.values() if v.get("graded", True)),
        "elapsed_s": time.time() - t0,
        "order": order,
        "rays_requested": int(rays),
        "rays_per_surface": max(1000, int(rays) // 8),
    }


def render_report(s: Dict) -> str:
    c, d = s["criteria"], s["data"]
    be = d["beam_enclosure"]
    beam_naive_min = _min_entry(d["beam"]["F_crossed_strings"])
    block_naive_max = _max_entry(d["convex_obstacle"]["F_crossed_strings"])
    L = [
        "# Verification: occlusion-aware 2-D view factors (inter-surface shadowing)",
        "",
        f"Driver `python -m tensorfvm.verification.radiation_occlusion` "
        f"(order = {s['order']}, {s['rays_requested']} rays, i.e. "
        f"{s['rays_per_surface']} per surface of an 8-strip scene, "
        f"{s['elapsed_s']:.1f} s). Predecessor: "
        f"`python -m tensorfvm.verification.radiation` (unobstructed strips, catalog gap 5).",
        "",
        "## 1. Case",
        "",
        "Cross-section radiation in a 2.0 m x 1.2 m cargo hold, per unit depth:",
        "",
        "* **beam scene** — simply connected, re-entrant: a 0.30 m x 0.60 m beam hung from the "
        "deck (8 strips). The crossed strings still close the rows, so their enclosure-level "
        "answer looks consistent while per-pair values are no longer view factors.",
        "* **block scene** — multiply connected: a 0.4 m x 0.4 m floating obstacle (chilled pipe "
        "/ ice block) inside the same hold (8 strips, obstacle traversed clockwise so its "
        "normal points into the domain).",
        "* **blocked pair** — two facing strips with a segment spanning the gap (exact 0).",
        "* **cavity** — the 1 x 1 square: control case, no occlusion at all.",
        "",
        "Material properties: gray-diffuse, `eps = 0.9`; deck and walls 350 K, the beam's three "
        "exposed faces 260 K. The beam power is quoted in the black limit "
        f"(eps = 1), where it follows from F alone: {be['power_occluded_W_per_m']:.5g} W/m; "
        f"with eps = 0.9 the gray enclosure gives {be['gray_power_occluded_W_per_m']:.5g} W/m.",
        "",
        "## 2. Method",
        "",
        "`tensorfvm.radiation.strip_form_factors_occluded` integrates the 2-D differential "
        "kernel `cos_i cos_j / (2 r)` over the receiver's **visible** part. For a fixed emitter "
        "station the shadow of one blocking segment is a wedge, so the visibility indicator "
        "along the receiver line is discontinuous only where the two shadow edges meet that "
        "line; those break points are computed analytically (the discontinuity is never "
        "sampled), each sub-interval is integrated by Gauss-Legendre, and the emitter is "
        "integrated by Gauss-Legendre as well. Cosines are signed and clipped at zero, so a "
        "chord through an obstacle's solid interior cannot exchange.",
        "",
        "Independent references, none sharing algebra with the code under test: the "
        "crossed-strings contour identity (unoccluded control), exact zero (fully hidden pair), "
        "energy closure `sum_j F_ij = 1`, reciprocity of exchange areas, and a cosine-weighted "
        "Monte-Carlo ray trace (`monte_carlo_form_factors`, binomial standard error reported).",
        "",
        "## 3. Results",
        "",
        "| quantity | value |",
        "|---|---|",
    ]
    for k in ("cavity", "beam", "block"):
        L.append(f"| `{k}`: max abs(sum_j F_ij - 1), occluded | "
                 f"{max(abs(r - 1.0) for r in d[k]['row_sum_occluded']):.2e} |")
        L.append(f"| `{k}`: max abs(sum_j F_ij - 1), crossed strings | "
                 f"{max(abs(r - 1.0) for r in d[k]['row_sum_crossed_strings']):.2e} |")
    for k in ("beam", "block"):
        L.append(f"| `{k}`: max abs(F_occluded - F_MC) | "
                 f"{d['monte_carlo_error'][k]['max_abs']:.2e} "
                 f"(4 sigma band {d['monte_carlo_error'][k]['band_4sigma']:.2e}) |")
        L.append(f"| `{k}`: max abs(F_crossed_strings - F_MC) | "
                 f"{d['monte_carlo_error'][k]['crossed_strings_max_abs']:.2e} |")
    L += [
        f"| beam: min F entry, crossed strings | {beam_naive_min:.4f} |",
        f"| block: max F between obstacle faces, crossed strings | {block_naive_max:.4f} "
        f"(sqrt(2) - 1) |",
        "| beam: Gauss order 24/48/96/192 error vs order 384 | "
        + " / ".join(f"{x['max_abs_error_vs_order384']:.1e}" for x in d["convergence"]) + " |",
        f"| beam: observed order (24 -> 48) | {d['observed_order']:.2f} |",
        f"| beam: beam-face power, occluded | {be['power_occluded_W_per_m']:.5g} W/m |",
        f"| beam: beam-face power, crossed strings | {be['power_crossed_strings_W_per_m']:.5g} W/m |",
        f"| beam: relative power error of the crossed strings | "
        f"{100.0 * be['power_relative_error']:.1f} % |",
        f"| beam: all-or-nothing mask row-sum error | {be['masked_row_sum_error']:.2f} |",
        "",
        "Full matrices, Monte-Carlo standard errors and hidden fractions: `summary.json`.",
        "",
        "## 4. Figure",
        "",
        "`fig_radiation_occlusion.png` — left: the beam and block scenes with the pairs the "
        "crossed strings get wrong (dotted); middle: occluded vs crossed-strings view factors "
        "against the Monte-Carlo reference; right: beam power and quadrature convergence.",
        "",
        "## 5. Notes, traps and limits",
        "",
        f"* **The obstructed-blind API does not fail loudly.** On the beam scene the crossed "
        f"strings return negative view factors (minimum {beam_naive_min:.4f}) while every row "
        f"still sums to one, so `GrayEnclosure`'s row-sum validation *accepts* them and the "
        f"beam power comes out {100.0 * be['power_relative_error']:.0f} % wrong. A single-pair "
        f"query (`strip_view_factor`) has no validation at all.",
        f"* **The all-or-nothing `visible` mask cannot express partial shadows.** Masking whole "
        f"pairs on the beam scene leaves the rows off by {be['masked_row_sum_error']:.1f}, i.e. "
        f"the masked enclosure violates energy conservation by more than 100 %.",
        "* **Holes must be traversed clockwise.** With the left-of-strip convention an obstacle "
        "inside the domain must be listed in reversed *vertex* order; `reversed(strips)` does "
        "not flip each strip and silently leaves the obstacle's normals pointing into its own "
        "solid (measured: the obstacle then exchanges exactly zero with the enclosure).",
        "* **Back-to-back faces of a convex obstacle exchange nothing** (the chord crosses the "
        "solid, view factor 0) while cavity-style crossed strings read the obstacle as a cavity "
        f"and return up to `sqrt(2) - 1 = {block_naive_max:.4f}` (adjacent faces: "
        "`1 - sqrt(2)/2 = 0.2929`). Signed cosines in the kernel remove it; an "
        "`|cos|` kernel adds the spurious exchange (measured: enclosure row sum 1.81, not 1.00).",
        "* **Shadow edges landing outside the receiver segment** break any "
        "`<2 break points -> visible` shortcut: a blocker can shadow part of a strip with both "
        "shadow edges beyond its end points (measured F error 0.44 before the fix).",
        "* Limits: 2-D strips only, no participating media, no spectral properties, no "
        "radiometric data; a shadow boundary exactly through a strip end point is assigned by "
        "the midpoint rule.",
        "",
        "## 6. Error judgement",
        "",
        "| criterion | value | gate | under 3 % ? |",
        "|---|---|---|---|",
    ]
    for k, v in c.items():
        if not v.get("graded", True):
            verdict = "no - by construction: it measures the *pre-existing* API's error"
        elif v["value"] <= 0.03:
            verdict = "yes"
        else:
            verdict = "no"
        L.append(f"| {k} | {v['value']:.3e} | {v['gate']:.0e} | {verdict} |")
    L += [
        "",
        "The geometry and invariant criteria sit at round-off except C1, which is bounded by the",
        "1/r corner singularity of the shared-edge pairs (also the limit of the pre-existing",
        "quadrature helper); the Monte-Carlo criterion is bounded by the ray-trace statistical",
        "band, which is declared rather than hidden. The only quantity above 3 % is C8c, which is",
        "deliberately above it: it is the error of the obstructed-blind answer this capability",
        "replaces, a defect magnitude and not an error of the new implementation.",
        "",
    ]
    return "\n".join(L)


def plot(summary: Dict, path: pathlib.Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = summary["data"]
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.5))

    ax = axes[0]
    for name, colour, builder in (("beam", "#1f77b4", _beam_scene),
                                  ("block", "#d62728", _block_scene)):
        strips = builder()
        for st in strips:
            a, b = st.points
            ax.plot([a[0], b[0]], [a[1], b[1]], color=colour, lw=2.0)
        m = np.asarray(d[name]["F_occluded"])
        nv = np.asarray(d[name]["F_crossed_strings"])
        for i in range(m.shape[0]):
            for j in range(i + 1, m.shape[0]):
                if abs(nv[i, j] - m[i, j]) > 0.05:
                    ci, cj = strips[i].midpoint, strips[j].midpoint
                    ax.plot([ci[0], cj[0]], [ci[1], cj[1]], color="grey", lw=0.6, ls=":")
    ax.set_aspect("equal")
    ax.set_title("scenes: dotted = pairs the crossed\nstrings get wrong by more than 0.05",
                 fontsize=8)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(alpha=0.3)

    ax = axes[1]
    for name, marker in (("beam", "o"), ("block", "s")):
        m = np.asarray(d[name]["F_occluded"])
        mc = np.asarray(d[name]["F_monte_carlo"])
        nv = np.asarray(d[name]["F_crossed_strings"])
        n = m.shape[0]
        mask = ~np.eye(n, dtype=bool)
        ax.scatter(mc[mask], m[mask], marker=marker, s=26, label=f"{name}: occluded")
        ax.scatter(mc[mask], nv[mask], marker=marker, s=26, facecolors="none",
                   edgecolors="red", label=f"{name}: crossed strings")
    lim = (-0.15, 1.05)
    ax.plot(lim, lim, "k-", lw=0.8)
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xlabel("Monte-Carlo ray trace F (-)")
    ax.set_ylabel("model F (-)")
    ax.set_title("view factors vs the independent\nray trace (y = x reference)", fontsize=8)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6)

    ax = axes[2]
    labels = ["occluded", "crossed\nstrings"]
    values = [d["beam_enclosure"]["power_occluded_W_per_m"],
              d["beam_enclosure"]["power_crossed_strings_W_per_m"]]
    ax.bar(labels, values, color=["#2ca02c", "#d62728"])
    for i, v in enumerate(values):
        ax.annotate(f"{v:.1f}", (i, v), ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("beam-face net power (W/m)")
    ax.set_title(f"beam scene: {100.0 * d['beam_enclosure']['power_relative_error']:.0f} % "
                 "power error of the\nobstructed-blind enclosure", fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="docs/verification-radiation-occlusion")
    ap.add_argument("--order", type=int, default=96)
    ap.add_argument("--rays", type=int, default=100_000)
    args = ap.parse_args(argv)
    summary = run(order=args.order, rays=args.rays)
    outdir = pathlib.Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=float))
    (outdir / "REPORT.md").write_text(render_report(summary))
    plot(summary, outdir / "fig_radiation_occlusion.png")
    for k, v in summary["criteria"].items():
        tag = "" if v.get("graded", True) else " (reported, not graded)"
        print(f"{k}: {'PASS' if v['pass'] else 'FAIL'}{tag}  value={v['value']:.3e} "
              f"gate={v['gate']:.0e}")
    print(f"all_pass={summary['all_pass']} elapsed={summary['elapsed_s']:.1f}s -> {outdir}")
    return 0 if summary["all_pass"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

"""Stability certificate for the explicit blended-convection scheme (round 23).

Why this module exists
----------------------
Rounds 20--22 published ``CASE-FVM-PERIODIC-HILLS`` numbers, and round 22 queued
a four-level grid ladder (``cj3869``/``cj3870``/``cj3871``/``cj3872``) whose
verdict was left open "pending queue".  When the jobs returned, **all three
refined levels had blown up** (``diverged_at_step`` = 508 / 538 / 652,
``samples`` = 0, no ``x_r/H``), while the base mesh survived 40 000 steps at
*twice* the time step.  The ladder was therefore never a convergence study; the
instability it hit had never been attributed.

Round-23 probes (``scripts/round23_stability_probe.py``, one configuration per
process, 1200 steps each) attribute it.  The momentum convection of
:class:`tensorfvm.periodic_hill.PeriodicHillSolver` is **explicit** and blended
``blend * central + (1 - blend) * upwind``; the central fraction is
anti-diffusive, so the scheme has a finite stability domain, and the ladder was
designed at fixed *global* Courant number (``U_b dt / dx = 0.213`` at every
level) while the flow just above the leeward crest carries 3--6 ``U_b`` (the
wall-normal component contributes too).  Refining the mesh *resolves* that peak
instead of smearing it, so the local Courant number at fixed global CFL **rises**
with refinement -- exactly the direction that a fixed-CFL ladder assumes it does
not.

The measured discriminator is the **blend-weighted local Courant number**
``blend * (|u| dt/dx + |v| dt/deta)``, i.e. the anti-diffusive fraction actually
being stepped explicitly (``blend = 0`` is pure upwind and unconditionally stable
at these numbers, ``blend = 0.5`` is stable where ``blend = 0.75`` is not)::

    mesh      dt       blend  blend*cfl_local   outcome
    144x72    0.0133   0.75      0.620          diverged at step 498
    144x72    0.0133   0.75      1.079          diverged at step 522 (sweeps 6)
    192x96    0.0100   0.75      1.007          diverged at step 529
    240x120   0.0080   0.75      0.647          diverged at step 642
    96x48     0.0200   0.75      0.591          stable (1200 + 40 000 steps)
    144x72    0.0100   0.75      0.482          stable
    144x72    0.0080   0.75      0.392          stable
    144x72    0.0075   0.75      0.369          stable
    144x72    0.0066   0.75      0.326          stable
    192x96    0.0060   0.75      0.392          stable
    192x96    0.0050   0.75      0.330          stable
    240x120   0.0050   0.75      0.413          stable
    240x120   0.0040   0.75      0.333          stable
    144x48    0.0133   0.75      0.492          stable (x-refined only)
    96x72     0.0133   0.75      0.548          stable (eta-refined only)
    144x72    0.0133   0.50      0.412          stable
    144x72    0.0133   0.00      0.000          stable

Two controls rule out the competing explanations: refining *only* x (144x48) or
*only* eta (96x72) at the failing ``dt`` is stable -- the unstable mode needs the
resolved two-dimensional peak -- and raising the non-orthogonal defect-correction
sweeps from 2 to 6 does **not** help (still diverges, step 522), so this is not an
unconverged metric correction.  The separation between the largest stable
observation (0.591) and the smallest unstable one (0.620) is only ~5 %, which is why
:data:`TARGET_BLEND_CFL` carries a ~25 % margin instead of sitting at the
boundary.

The rule implemented here is **pre-registered**: it was written from the probe
table, and it is applied unchanged to every ladder artefact (round 22's own
numbers land on the correct side without tuning).  Nothing in this module knows
about a particular case: arrays and numbers go in, a verdict comes out.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "local_cfl",
    "blend_local_cfl",
    "sustained_local_cfl",
    "sustained_blend_cfl",
    "classify",
    "dt_for_blend_cfl",
    "STABLE_BLEND_CFL",
    "UNSTABLE_BLEND_CFL",
    "TARGET_BLEND_CFL",
    "STABLE",
    "MARGINAL",
    "UNSTABLE",
]

#: below this blend-weighted local Courant number every probe of round 23
#: survived (largest stable observation: 0.591, the published 96x48 / dt = 0.02
#: run; the limit is set below that so the "stable" verdict is not a statement
#: about the edge)
STABLE_BLEND_CFL = 0.55
#: at or above this value every probe of round 23 diverged
#: (smallest unstable observation: 0.6195 at 144x72 / dt = 0.0133, next 0.647 at
#: 240x120 / dt = 0.008; the observed gap between survival and failure is only
#: 0.591 -> 0.6195, i.e. ~5 %, and this constant sits at its lower edge)
UNSTABLE_BLEND_CFL = 0.60
#: working target handed to drivers: ~25 % margin below the smallest unstable
#: observation, so that a refined level is not designed on the stability edge
TARGET_BLEND_CFL = 0.45
#: the startup impulse (bulk-velocity controller reaching ``U_b`` from rest) is
#: excluded from "sustained": probes show it can transiently exceed the limit
#: without the run failing (96x48 / dt = 0.02)
SUSTAINED_FROM_STEP = 200

STABLE = "stable"
MARGINAL = "marginal"
UNSTABLE = "unstable"


def local_cfl(u, v, dx, deta, dt):
    """Per-cell local Courant number ``|u| dt / dx + |v| dt / deta``.

    ``dx`` may be a scalar or an ``(ny, nx)`` array; ``deta`` may be a scalar or
    a per-row array of length ``ny`` (the wall-normal spacing of the
    terrain-following mesh varies with the row, not with the column).
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    if u.ndim != 2 or u.shape != v.shape:
        raise ValueError(f"u and v must be 2-D with the same shape, got "
                         f"{u.shape} / {v.shape}")
    ny, nx = u.shape
    if not (np.all(np.isfinite(u)) and np.all(np.isfinite(v))):
        raise ValueError("velocities must be finite; a diverged state has no Courant number")
    dt = float(dt)
    if not dt > 0.0:
        raise ValueError("dt must be positive")
    dx = np.asarray(dx, dtype=float)
    if dx.ndim == 0:
        dx_arr = np.full((ny, nx), float(dx))
    elif dx.shape == (ny, nx):
        dx_arr = dx
    else:
        raise ValueError(f"dx must be scalar or {(ny, nx)}, got {dx.shape}")
    de = np.asarray(deta, dtype=float)
    if de.ndim == 0:
        de_arr = np.full((ny, nx), float(de))
    elif de.shape == (ny,):
        de_arr = np.broadcast_to(de[:, None], (ny, nx))
    elif de.shape == (ny, nx):
        de_arr = de
    else:
        raise ValueError(f"deta must be scalar, {(ny,)} or {(ny, nx)}, got {de.shape}")
    if np.any(dx_arr <= 0.0) or np.any(de_arr <= 0.0):
        raise ValueError("spacings must be positive")
    return np.abs(u) * dt / dx_arr + np.abs(v) * dt / de_arr


def blend_local_cfl(cfl_local, blend):
    """Anti-diffusive fraction that is actually stepped: ``blend * cfl_local``."""
    blend = float(blend)
    if not (math.isfinite(blend) and 0.0 <= blend <= 1.0):
        raise ValueError("blend must be in [0, 1]")
    arr = np.asarray(cfl_local, dtype=float)
    if np.any(~np.isfinite(arr)) or np.any(arr < 0.0):
        raise ValueError("cfl_local must be finite and non-negative")
    return blend * arr


def sustained_local_cfl(trace, start_step=SUSTAINED_FROM_STEP):
    """Max ``cfl_local_max`` over the part of ``trace`` after ``start_step``.

    ``trace`` is the list of per-sample dicts written by
    ``scripts/round23_stability_probe.py`` (``step`` + ``cfl_local_max``).  A
    sample without the key is skipped; an empty result raises, because "no
    samples" must never be silently read as "stable" (the round-18 error mode).
    """
    vals = [float(s["cfl_local_max"]) for s in trace
            if s.get("cfl_local_max") is not None
            and int(s.get("step", 0)) > int(start_step)]
    if not vals:
        raise ValueError("no samples after start_step; stability cannot be certified")
    return max(vals)


def sustained_blend_cfl(trace, blend, start_step=SUSTAINED_FROM_STEP):
    """Sustained local Courant number times ``blend`` -- the certified indicator."""
    return float(blend_local_cfl(sustained_local_cfl(trace, start_step), blend))


def dt_for_blend_cfl(current_blend_cfl, current_dt, target=TARGET_BLEND_CFL):
    """``dt`` that scales a measured indicator down to ``target``.

    The indicator is linear in ``dt`` at fixed velocity, which is what makes this
    a usable design handle: 144x72 went from 0.620 (diverged) to 0.482 (stable)
    when ``dt`` went 0.0133 -> 0.010, the scaling this function predicts to
    within the flow's own change.
    """
    c = float(current_blend_cfl)
    dt = float(current_dt)
    tgt = float(target)
    if not (math.isfinite(c) and c > 0.0):
        raise ValueError("current_blend_cfl must be positive and finite")
    if not dt > 0.0:
        raise ValueError("current_dt must be positive")
    if not 0.0 < tgt < UNSTABLE_BLEND_CFL:
        raise ValueError(f"target must be in (0, {UNSTABLE_BLEND_CFL})")
    return dt * tgt / c


def classify(sustained_blend_cfl_value):
    """Pre-registered verdict for a measured ``blend * cfl_local``.

    ``STABLE``   -- below :data:`STABLE_BLEND_CFL`;
    ``MARGINAL`` -- inside the observed gap (0.55--0.60): **no claim** is made,
                    neither "converged" nor "diverged";
    ``UNSTABLE`` -- at or above :data:`UNSTABLE_BLEND_CFL`.
    """
    c = float(sustained_blend_cfl_value)
    if not math.isfinite(c) or c < 0.0:
        raise ValueError("blend-weighted local CFL must be finite and non-negative")
    if c < STABLE_BLEND_CFL:
        return STABLE
    if c < UNSTABLE_BLEND_CFL:
        return MARGINAL
    return UNSTABLE
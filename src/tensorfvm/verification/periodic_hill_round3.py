"""Round-3 verification of the periodic-hill finite-volume solver.

Mission 16672 (selfstart, TensorFVM).  Round 2 (mission 16503,
``docs/case-periodic-hills/ROUND2.md``) left the case at an honest FAIL: the
plane-channel gate was closed to ``2.6e-4`` but the hill at ``Re = 2800``
diverged, and the cell-level divergence stayed ``O(1)``.  Round 2 blamed the
non-orthogonal treatment of the pressure correction and listed it as round 3's
first item.  This driver *measures* that item instead of assuming it.

Two defects are established here, both in ``PeriodicHillSolver.compact_gradient``:

* **(vi) south-face index shift.**  The eta-face term was assembled as
  ``c_n[j] n[j] (phi[j+1] - phi[j])`` for *every* row, i.e. each cell received
  its north jump as if it were the south one: row 0 was handed a wall face
  driven by an interior jump while row ``ny-1`` lost its south face entirely.
  The compact gradient was therefore not even consistent on a smooth field
  (measured: :math:`\\approx 19\\,\\%` relative L2 error on a periodic wave, and
  *independent of mesh size*, which is the signature of an inconsistency
  rather than a discretisation error).
* **(vii) no non-orthogonal correction on either face family.**  The mesh is
  terrain following, so a face normal is nowhere parallel to the
  centre-to-centre vector ``d = alpha n + beta t``: ``|beta/alpha|`` reaches
  0.79 on the eta faces and the east/west faces are skewed as well.  The implied
  one-sided relation ``jump/alpha`` differs from ``(grad phi).n`` by
  ``(beta/alpha)(grad phi).t``, which is *first order and O(1) relative* on the
  hill columns.  The deferred correction

      g_n = jump/alpha - (beta/alpha) (grad phi).t

  is applied explicitly with the *wide* least-squares gradient as the
  tangential estimate (iterating from the compact estimate itself does not
  converge once ``|beta/alpha| ~ 1``).

Criteria (pre-declared, all machine-evaluated here):

C1  the corrected compact gradient converges at first order on a periodic wave:
    error(32x16) > error(64x32) > error(128x64), ratios in [1.5, 5], and the
    finest-mesh error < 1 %.
C2  the *uncorrected* operator is order-zero inconsistent: its error varies by
    less than 10 % between 32x16 and 128x64 (no convergence).
C3  the correction is worth at least 10x on the finest mesh (sw=1 vs sw=0).
C4  the plane-channel gate survives the new operator: the Poiseuille
    ``u_max/u_b`` measured by the solver is within 3 % of the analytic value
    computed from the solver's *own* bulk functional (zero hand-copying).
C5  the reported non-orthogonality is real, not a rounding artefact:
    ``max |beta/alpha|`` on the eta faces > 0.5 on every mesh.
C6  honest status at ``Re = 2800``: the 48x24 reach run is finite at 500 steps
    and blows up before 1500, i.e. the reach item of round 2's plan is **not**
    delivered by this fix and ``x_r/H`` remains withheld.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..periodic_hill import PeriodicHillConfig, PeriodicHillSolver

__all__ = ["run", "wave_fields", "gradient_error", "poiseuille_band_ratio"]

LENGTH = 9.0
KX = 2.0 * np.pi / LENGTH


def wave_fields():
    """A smooth field, periodic in x, with an analytic gradient."""

    def field(x, y):
        return np.sin(KX * x) * np.cos(np.pi * y / 3.0)

    def grad(x, y):
        return (KX * np.cos(KX * x) * np.cos(np.pi * y / 3.0),
                -np.pi / 3.0 * np.sin(KX * x) * np.sin(np.pi * y / 3.0))

    return field, grad


def gradient_error(nx, ny, sweeps):
    """Relative L2 error of ``compact_gradient`` on the periodic wave."""
    solver = PeriodicHillSolver(PeriodicHillConfig(nx=nx, ny=ny, reynolds=100.0))
    mesh = solver.mesh
    field, grad = wave_fields()
    phi = field(mesh.xc, mesh.yc)
    ex, ey = grad(mesh.xc, mesh.yc)
    gx, gy = solver.compact_gradient(phi, nonorthogonal_sweeps=sweeps)
    scale = max(np.abs(ex).max(), np.abs(ey).max())
    return float(np.sqrt(np.mean((gx - ex) ** 2 + (gy - ey) ** 2)) / scale)


def poiseuille_band_ratio(solver):
    """Analytic ``u_max / u_b`` of plane Poiseuille, in the solver's own norm."""
    mesh = solver.mesh
    height = float(solver.config.channel_top_m)
    profile = mesh.yc * (height - mesh.yc)
    return float(profile.max() / solver.bulk_velocity(profile))


def run(out_dir=None):
    grids = ((32, 16), (64, 32), (128, 64))
    c1 = {f"{nx}x{ny}": gradient_error(nx, ny, 1) for nx, ny in grids}
    c2 = {f"{nx}x{ny}": gradient_error(nx, ny, 0) for nx, ny in grids}
    c3_gain = c2["128x64"] / c1["128x64"]

    solver = PeriodicHillSolver(PeriodicHillConfig(nx=8, ny=32, reynolds=100.0,
                                                  hill_amplitude=0.0,
                                                  channel_top_m=3.036,
                                                  time_step_s=0.05))
    solver.advance(2000)
    u_max_ratio = float(solver.u.max() / solver.bulk_velocity())
    analytic_ratio = poiseuille_band_ratio(solver)
    c4 = abs(u_max_ratio - analytic_ratio) / analytic_ratio
    c4_div = float(np.abs(solver.divergence()).max())

    mesh = solver.mesh
    c5_eta = 0.0
    for nx, ny in grids:
        m = PeriodicHillSolver(PeriodicHillConfig(nx=nx, ny=ny, reynolds=100.0)).mesh
        alpha = m.dn_n[1:-1]
        c5_eta = max(c5_eta, float(np.max(np.abs(m.beta_n[1:-1]) / alpha)))
        c5_e = float(np.max(np.abs(m.beta_e) / m.dn_e))

    reach = PeriodicHillSolver(PeriodicHillConfig(nx=48, ny=24, reynolds=2800.0,
                                                  time_step_s=0.01,
                                                  body_force_relaxation=0.2))
    trace = []
    for _ in range(3):
        reach.advance(500)
        trace.append(float(max(np.abs(reach.u).max(), np.abs(reach.v).max())))
    finite_500 = bool(np.isfinite(trace[0]) and trace[0] < 1e3)
    blown_up = bool(not (np.all(np.isfinite(trace)) and max(trace) < 1e3))

    criteria = {
        "C1_first_order_convergence": {
            "error_sweeps1": c1,
            "ratio_32_64": c1["32x16"] / c1["64x32"],
            "ratio_64_128": c1["64x32"] / c1["128x64"],
            "pass": bool(c1["32x16"] > c1["64x32"] > c1["128x64"]
                         and 1.5 <= c1["32x16"] / c1["64x32"] <= 5.0
                         and 1.5 <= c1["64x32"] / c1["128x64"] <= 5.0
                         and c1["128x64"] < 1.0e-2),
        },
        "C2_uncorrected_is_order_zero": {
            "error_sweeps0": c2,
            "ratio_32_128": c2["32x16"] / c2["128x64"],
            "pass": bool(abs(c2["32x16"] / c2["128x64"] - 1.0) < 0.10),
        },
        "C3_correction_gain": {"gain_finest": c3_gain, "pass": bool(c3_gain > 10.0)},
        "C4_channel_gate": {
            "u_max_over_ub": u_max_ratio,
            "analytic": analytic_ratio,
            "rel_error": c4,
            "max_cell_divergence": c4_div,
            "pass": bool(c4 < 0.03 and c4_div < 1e-10),
        },
        "C5_nonorthogonality": {"max_beta_over_alpha_eta": c5_eta,
                                "max_beta_over_dn_east": c5_e,
                                "pass": bool(c5_eta > 0.5)},
        "C6_reach_honest_fail": {
            "max_abs_velocity_trace": trace,
            "finite_at_500": finite_500,
            "blows_up_before_1500": blown_up,
            "x_reattachment": "withheld",
            "pass": bool(finite_500 and blown_up),
        },
    }
    summary = {
        "node": "CASE-FVM-PERIODIC-HILLS",
        "round": 3,
        "criteria": criteria,
        "all_pass": all(item["pass"] for item in criteria.values()),
        "note": ("C1-C5 are the round-3 deliverable; C6 records the honest state "
                 "of the Re=2800 reach metric (not delivered)."),
    }
    if out_dir is not None:
        path = Path(out_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "round3_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":  # pragma: no cover
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else "docs/case-periodic-hills"
    result = run(target)
    for name, item in result["criteria"].items():
        print(f"{name:32s} {'PASS' if item['pass'] else 'FAIL'}")
    print("all_pass:", result["all_pass"])

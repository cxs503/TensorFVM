"""Verification driver: rotational added inertia (``M_44..M_66``) of a spheroid.

Produces ``docs/verification-added-inertia-panel/``:

* ``rotational_added_inertia.json`` -- every number in the report, computed here
  (no hand-copied digits);
* ``figures/fig1_convergence.png``   -- panel convergence + Richardson vs the
  semi-analytic reference;
* ``figures/fig2_field.png``         -- surface potential of the rotation mode
  (the solved field, not a redraw).

Run:  python scripts/verify_rotational_added_inertia.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from tensorfvm.panel_added_inertia import (  # noqa: E402
    lamb_rotational_inertia,
    rotational_added_inertia,
)
from tensorfvm.panel_added_mass import spheroid_panels  # noqa: E402
from tensorfvm.panel_potential import solve_sources  # noqa: E402
from verify_rotational_added_inertia_reference import (  # noqa: E402
    _surface_potential,
    q_legendre_1,
    spheroid_rotation_reference,
    spheroid_surface_nodes,
)

OUT = ROOT / "docs" / "verification-added-inertia-panel"
MESHES = [(24, 18), (40, 28), (60, 42), (80, 56)]


def richardson(values, panels):
    """Power-law fit ``v(N) = v_inf + K N^{-q}`` -> observed order ``q`` and ``v_inf``.

    The order is *not* assumed: it is the ``q`` minimising the residual of the
    linear least-squares fit for ``(v_inf, K)`` over a scan of ``q``.  Only the
    panel values enter, so the extrapolated limit is independent of the exact
    reference it is later compared with.
    """
    v = np.asarray(values, dtype=float)
    N = np.asarray(panels, dtype=float)
    best = (np.inf, np.nan, np.nan)
    for q in np.linspace(0.1, 5.0, 490):
        y = N ** (-q)
        A = np.stack([np.ones_like(y), y], axis=1)
        coef, res, *_ = np.linalg.lstsq(A, v, rcond=None)
        r = float(np.sum((A @ coef - v) ** 2))
        if r < best[0]:
            best = (r, q, float(coef[0]))
    return float(best[1]), float(best[2])


def sphere_probe(radius=1.0, meshes=((16, 12), (24, 18), (32, 24), (40, 30))):
    """Sphere: the exact rotational added inertia is 0 for every axis."""
    rows = []
    for nt, nph in meshes:
        p = spheroid_panels(radius, radius, n_theta=nt, n_phi=nph)
        I = rotational_added_inertia(p, order=3)
        rows.append({"n_theta": nt, "n_phi": nph, "n_panels": int(p.n_panels),
                     "I_diag": np.diag(I).tolist(),
                     "offdiag_max": float(np.abs(I - np.diag(np.diag(I))).max()),
                     "I_max_abs_scaled": float(np.abs(I).max() / (radius**5))})
    return rows


def spheroid_ladder(a=2.0, b=1.0, meshes=MESHES):
    ref = spheroid_rotation_reference(a, b, n_terms=12, n_mu_fit=48, n_phi_fit=8)
    exact = float(lamb_rotational_inertia(a, b)[1, 1])
    vol = 4.0 / 3.0 * np.pi * a * b * b
    rows = []
    for nt, nph in meshes:
        p = spheroid_panels(a, b, n_theta=nt, n_phi=nph)
        I = rotational_added_inertia(p, order=3)
        rows.append({
            "n_theta": nt, "n_phi": nph, "n_panels": int(p.n_panels),
            "I_transverse": float(I[1, 1]),
            "I_symmetry_axis": float(I[0, 0]),
            "offdiag_max": float(np.abs(I - np.diag(np.diag(I))).max()),
            "relerr_vs_lamb": float(abs(I[1, 1]) / exact - 1.0),
        })
    p_obs, extrap = richardson([r["I_transverse"] for r in rows], [r["n_panels"] for r in rows])
    return {
        "a": a, "b": b, "volume": vol, "meshes": rows,
        "lamb_closed_form": exact,
        "semi_analytic_reference": ref["I_perp"],
        "reference_neumann_residual": ref["neumann_residual"],
        "reference_far_field_ratio": ref["far_field_ratio"],
        "reference_vs_closed_form_rel": float(abs(ref["I_perp"] / exact - 1.0)),
        "observed_order": p_obs,
        "richardson_extrapolated": extrap,
        "richardson_rel": float(abs(extrap / exact - 1.0)),
    }


def aspect_sweep(aspects=(1.05, 1.5, 2.0, 3.0, 6.0), n_terms=12):
    rows = []
    for ar in aspects:
        a, b = float(ar), 1.0
        ref = spheroid_rotation_reference(a, b, n_terms=n_terms, n_mu_fit=4 * n_terms,
                                          n_phi_fit=8)
        exact = float(lamb_rotational_inertia(a, b)[1, 1])
        rows.append({"aspect": ar, "reference": ref["I_perp"], "closed_form": exact,
                     "rel_diff": float(abs(ref["I_perp"] / exact - 1.0)) if exact else 0.0,
                     "neumann_residual": ref["neumann_residual"]})
    return rows


def surface_field(a=2.0, b=1.0, n_terms=12, n_mu=60, n_phi=120):
    """Surface potential of the rotation mode (reference series) for the figure."""
    ref = spheroid_rotation_reference(a, b, n_terms=n_terms, n_mu_fit=48, n_phi_fit=8)
    A, orders, zeta0 = ref["coefficients"], ref["orders"], ref["zeta0"]
    q0 = np.array([q_legendre_1(int(n), zeta0) for n in orders])
    mu1 = np.linspace(-0.999, 0.999, n_mu)
    ph1 = np.linspace(0.0, 2.0 * np.pi, n_phi)
    MU, PH = np.meshgrid(mu1, ph1, indexing="ij")
    phi = _surface_potential(orders, q0, A, MU.ravel(), PH.ravel()).reshape(MU.shape)
    return MU, PH, phi


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)
    payload = {
        "case": "rotational added inertia M_44..M_66 of a prolate spheroid in an ideal fluid",
        "sphere_exact_zero": sphere_probe(),
        "spheroid_a2b1": spheroid_ladder(),
        "aspect_sweep_reference_vs_closed_form": aspect_sweep(),
    }
    (OUT / "rotational_added_inertia.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sp = payload["spheroid_a2b1"]
    exact = sp["lamb_closed_form"]
    npan = [m["n_panels"] for m in sp["meshes"]]
    err = [abs(m["I_transverse"] / exact - 1.0) * 100.0 for m in sp["meshes"]]

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].semilogx(npan, [m["I_transverse"] for m in sp["meshes"]], "o-",
                   label="source-panel BEM")
    ax[0].axhline(exact, color="k", ls="--", label="semi-analytic / Lamb (exact)")
    ax[0].axhline(sp["richardson_extrapolated"], color="tab:red", ls=":",
                  label=f"Richardson (p={sp['observed_order']:.2f})")
    ax[0].set_xlabel("panels"); ax[0].set_ylabel(r"$I_\perp/\Omega$  (rho=1)")
    ax[0].set_title("transverse added inertia, spheroid a/b=2"); ax[0].legend(fontsize=8)
    ax[1].loglog(npan, err, "s-", label="panel error")
    ax[1].axhline(3.0, color="tab:orange", ls="--", label="3% gate")
    ax[1].axhline(1.0, color="tab:green", ls=":", label="1% gate")
    ax[1].set_xlabel("panels"); ax[1].set_ylabel("relative error to exact [%]")
    ax[1].set_title("discretisation error vs exact"); ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "fig1_convergence.png", dpi=140)
    plt.close(fig)

    MU, PH, phi = surface_field()
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    cs = ax.contourf(MU, PH, phi, levels=24, cmap="RdBu_r")
    fig.colorbar(cs, ax=ax, label=r"$\phi$ on the surface (units $a^2U$)")
    ax.set_xlabel(r"$\mu=\cos\theta$ (along the symmetry axis)")
    ax.set_ylabel(r"azimuth $\varphi$"); ax.set_title("rotation-mode surface potential (a/b=2)")
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "fig2_field.png", dpi=140)
    plt.close(fig)

    sp_rows = sp["meshes"]
    print(f"exact I_perp = {exact:.6f}  (V={sp['volume']:.4f}, I/V={exact/sp['volume']:.6f})")
    for m in sp_rows:
        print(f"  panels {m['n_panels']:5d}  I={m['I_transverse']:9.6f}  "
              f"err={m['relerr_vs_lamb']*100:+.2f}%  sym-axis={m['I_symmetry_axis']:.2e}")
    print(f"  Richardson p={sp['observed_order']:.2f} -> {sp['richardson_extrapolated']:.6f} "
          f"({sp['richardson_rel']*100:+.3f}%)")
    print(f"  semi-analytic ref {sp['semi_analytic_reference']:.6f} "
          f"(Neumann residual {sp['reference_neumann_residual']:.1e}, "
          f"vs closed form {sp['reference_vs_closed_form_rel']*100:.2e}%)")
    print("sphere (exact 0):",
          ["%.2e" % r["I_max_abs_scaled"] for r in payload["sphere_exact_zero"]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
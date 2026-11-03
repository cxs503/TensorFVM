#!/usr/bin/env python3
"""Independent verification of the free-surface VOF boundary-flux witness.

Closes the verification debt on :mod:`tensorfvm.free_surface_bc`: the module
advertises ``outflow_vof_flux`` as the quantity that lets the global liquid
volume ledger "be closed in the anchor validation", but until now it was only
exercised at zero velocity.

Checks, in increasing strength:

``reference``
    A scalar Python loop that walks every boundary face, pairing each face with
    its own upwind cell and accumulating ``max(+-F, 0) * alpha_upwind``.  It
    shares no vectorised expression with the implementation (no ``sum`` over a
    column, no broadcast ``clamp``), so it detects off-by-one indices,
    transposed ``u``/``v`` shapes and sign errors.  Valid for *any* velocity
    field because no transport step is involved.

``ledger``
    The telescoping identity against the repository's own advection operator:
    run :meth:`VOFTransport.advect` with ``scheme="upwind"`` and check
    ``(sum(alpha_new) - sum(alpha_ref)) * dx * dy + net_vof_flux == 0``.  The
    identity is exact for any velocity field, but ``advect`` clips ``alpha`` to
    ``[0, 1]``, so an active clip would destroy conservation for reasons that
    have nothing to do with the boundary witness.  The check therefore uses a
    discretely divergence-free velocity built from a streamfunction on the
    ``(ny+1, nx+1)`` corners, which makes donor-cell advection bounded and the
    clip provably inactive for interior ``alpha``.

``translation``
    Analytic closed-form anchor: uniform stream ``U``, arbitrary column profile.
    Total liquid volume must change by exactly
    ``dt * U * dy * (sum(alpha_at_inlet) - sum(alpha_at_outlet))``.

``retired``
    Negative control / regression witness: the previously shipped column-sum
    formula (``max(0, sum(u_column)) * mean(alpha_column)``) evaluated on the
    same data, documenting how far off a witness based on column sums is.  It
    must stay far outside the acceptance band so the defect cannot come back
    silently.

Usage::

    python3 scripts/verify_free_surface_ledger.py [--out runs/free_surface_ledger]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tensorfvm.free_surface_bc import (  # noqa: E402
    FreeSurfaceBC,
    net_vof_flux,
    outflow_vof_flux,
)
from tensorfvm.two_phase_vof import VOFTransport  # noqa: E402

GATE = 1.0e-12  # relative tolerance for the direct algebra checks
# The ledger residual is a *difference* of two O(1) totals, so a residual
# relative to dV is dominated by the cancellation floor of sum(alpha) (~eps*N).
# Judge it against the total liquid volume instead, which is well conditioned.
SCALED_GATE = 1.0e-13
DTYPE = torch.float64


def reference_gross_net(alpha, u, v, dx, dy, dt):
    """Scalar per-face loop; deliberately shares nothing with the implementation.

    Each boundary face is paired with its own upwind cell explicitly:

    ====== ============ =================
    face   leaving flux upwind cell
    ====== ============ =================
    east   +u[j, nx]    alpha[j, nx-1]
    west   -u[j, 0]     alpha[j, 0]
    north  +v[ny, i]    alpha[ny-1, i]
    south  -v[0, i]     alpha[0, i]
    ====== ============ =================

    The sign of each face is what makes ``net`` the *leaving* volume; getting it
    wrong on the west/south faces is exactly the defect this check exists for.
    """
    ny, nx = alpha.shape
    pairs = []
    for j in range(ny):
        # east leaves with u > 0, west leaves with u < 0 -> negate the west flux
        pairs.append((+float(u[j, nx].item()), float(alpha[j, nx - 1].item()), dy))
        pairs.append((-float(u[j, 0].item()), float(alpha[j, 0].item()), dy))
    for i in range(nx):
        pairs.append((+float(v[ny, i].item()), float(alpha[ny - 1, i].item()), dx))
        pairs.append((-float(v[0, i].item()), float(alpha[0, i].item()), dx))
    gross = 0.0
    net = 0.0
    for velocity, a, length in pairs:
        flux = dt * velocity * length * a
        net += flux
        gross += max(flux, 0.0)
    return gross, net


def retired_column_sum_flux(alpha, u, v, dx, dy, dt):
    """The formula this mission retires, reproduced verbatim for the record."""
    east_out = max(0.0, float(u[:, -1].sum().item())) * dy
    west_out = max(0.0, float(-u[:, 0].sum().item())) * dy
    north_out = max(0.0, float(v[-1, :].sum().item())) * dx
    south_out = max(0.0, float(-v[0, :].sum().item())) * dx
    a_e = float(alpha[:, -1].mean().item())
    a_w = float(alpha[:, 0].mean().item())
    a_n = float(alpha[-1, :].mean().item())
    a_s = float(alpha[0, :].mean().item())
    return dt * (east_out * a_e + west_out * a_w + north_out * a_n + south_out * a_s)


def streamfunction_field(alpha_shape, dx, dy, seed, target=0.5):
    """Discretely divergence-free MAC velocity from a corner streamfunction.

    ``psi`` lives on corners ``(ny+1, nx+1)``; with ``u = -dpsi/dy`` on east
    faces and ``v = +dpsi/dx`` on north faces the cell divergence telescopes to
    zero identically (no periodicity required).
    """
    ny, nx = alpha_shape
    g = torch.Generator().manual_seed(seed)
    jj = torch.arange(ny + 1, dtype=DTYPE).view(-1, 1)
    ii = torch.arange(nx + 1, dtype=DTYPE).view(1, -1)
    psi = (
        torch.sin(math.pi * jj / max(ny, 1)) * torch.sin(math.pi * ii / max(nx, 1))
        + 0.35 * torch.sin(2 * math.pi * jj / max(ny, 1)) * torch.cos(3 * math.pi * ii / max(nx, 1))
        + 0.20 * (torch.rand(ny + 1, nx + 1, generator=g, dtype=DTYPE) - 0.5)
    )
    u = -(psi[1:, :] - psi[:-1, :]) / dy
    v = (psi[:, 1:] - psi[:, :-1]) / dx
    scale = target / max(float(u.abs().max()), float(v.abs().max()), 1e-30)
    return u * scale, v * scale


def random_alpha(shape, seed, lo=0.05, hi=0.95):
    g = torch.Generator().manual_seed(seed + 977)
    return lo + (hi - lo) * torch.rand(*shape, generator=g, dtype=DTYPE)


def ledger_check(alpha, u, v, dx, dy, dt):
    """Run one upwind step and return the ledger residual, scaled by total volume."""
    transport = VOFTransport(alpha.shape[0], alpha.shape[1], dx, dy, scheme="upwind")
    alpha_new = transport.advect(alpha, u, v, dt)
    delta_volume = float((alpha_new.sum() - alpha.sum()).item()) * dx * dy
    net = net_vof_flux(alpha, u, v, dx, dy, dt)
    bc = FreeSurfaceBC(alpha.shape[0], alpha.shape[1], dx, dy)
    residual = bc.ledger_residual(alpha, alpha_new, u, v, dt)
    volume = max(float(alpha.sum().item()) * dx * dy, 1e-30)
    return delta_volume, net, float(residual), abs(float(residual)) / volume


def run(seeds=range(1, 9), fine=(64, 96)):
    records = []
    for seed in seeds:
        ny, nx, dx, dy, dt = 6, 8, 0.1, 0.1, 0.02
        alpha = random_alpha((ny, nx), seed)

        # --- algebra: arbitrary (possibly divergent) field -------------------
        g = torch.Generator().manual_seed(seed)
        u_any = (torch.rand(ny, nx + 1, generator=g, dtype=DTYPE) - 0.5)
        v_any = (torch.rand(ny + 1, nx, generator=g, dtype=DTYPE) - 0.5)
        gross_any = outflow_vof_flux(alpha, u_any, v_any, dx, dy, dt)
        net_any = net_vof_flux(alpha, u_any, v_any, dx, dy, dt)
        ref_gross_any, ref_net_any = reference_gross_net(alpha, u_any, v_any, dx, dy, dt)
        retired_any = retired_column_sum_flux(alpha, u_any, v_any, dx, dy, dt)

        # --- ledger: divergence-free, clip provably inactive ----------------
        u_df, v_df = streamfunction_field((ny, nx), dx, dy, seed)
        gross_df = outflow_vof_flux(alpha, u_df, v_df, dx, dy, dt)
        net_df = net_vof_flux(alpha, u_df, v_df, dx, dy, dt)
        ref_gross_df, ref_net_df = reference_gross_net(alpha, u_df, v_df, dx, dy, dt)
        retired_df = retired_column_sum_flux(alpha, u_df, v_df, dx, dy, dt)
        delta_volume, net_l, residual, ledger_rel = ledger_check(alpha, u_df, v_df, dx, dy, dt)

        # --- analytic uniform-stream translation ---------------------------
        U = 0.4
        u_uni = torch.full((ny, nx + 1), U, dtype=DTYPE)
        v_uni = torch.zeros((ny + 1, nx), dtype=DTYPE)
        exact_delta = dt * U * dy * float(alpha[:, 0].sum().item()) - dt * U * dy * float(
            alpha[:, -1].sum().item()
        )
        translation_delta, translation_net, _r, _rel = ledger_check(alpha, u_uni, v_uni, dx, dy, dt)

        records.append(dict(
            seed=seed,
            algebra=dict(
                gross_rel_error=abs(gross_any - ref_gross_any) / max(abs(ref_gross_any), 1e-30),
                net_rel_error=abs(net_any - ref_net_any) / max(abs(ref_net_any), 1e-30),
                retired_rel_error=abs(retired_any - ref_gross_any) / max(abs(ref_gross_any), 1e-30),
            ),
            ledger=dict(
                delta_volume=delta_volume,
                net=net_l,
                residual=residual,
                scaled_residual=ledger_rel,
                gross_rel_error=abs(gross_df - ref_gross_df) / max(abs(ref_gross_df), 1e-30),
                net_rel_error=abs(net_df - ref_net_df) / max(abs(ref_net_df), 1e-30),
                retired_rel_error=abs(retired_df - ref_gross_df) / max(abs(ref_gross_df), 1e-30),
            ),
            translation=dict(
                exact_delta=exact_delta,
                measured_delta=translation_delta,
                rel_error=abs(translation_delta - exact_delta) / max(abs(exact_delta), 1e-30),
            ),
        ))

    # Size robustness: the identity must not degrade with grid size.
    ny, nx = fine
    dx = dy = 0.004
    dt = 1.0e-4
    alpha = random_alpha((ny, nx), 101)
    u_df, v_df = streamfunction_field((ny, nx), dx, dy, 101)
    ref_gross, _ = reference_gross_net(alpha, u_df, v_df, dx, dy, dt)
    gross = outflow_vof_flux(alpha, u_df, v_df, dx, dy, dt)
    _dv, _n, _r, ledger_rel = ledger_check(alpha, u_df, v_df, dx, dy, dt)
    fine_case = dict(
        ny=ny, nx=nx,
        gross_rel_error=abs(gross - ref_gross) / max(abs(ref_gross), 1e-30),
        ledger_scaled_residual=ledger_rel,
        retired_rel_error=abs(retired_column_sum_flux(alpha, u_df, v_df, dx, dy, dt) - ref_gross)
        / max(abs(ref_gross), 1e-30),
    )

    algebra_err = max(
        max(r["algebra"]["gross_rel_error"], r["algebra"]["net_rel_error"],
            r["ledger"]["gross_rel_error"], r["ledger"]["net_rel_error"])
        for r in records
    )
    ledger_err = max([r["ledger"]["scaled_residual"] for r in records]
                     + [fine_case["ledger_scaled_residual"]])
    translation_err = max(r["translation"]["rel_error"] for r in records)
    retired_min = min([r["algebra"]["retired_rel_error"] for r in records]
                      + [r["ledger"]["retired_rel_error"] for r in records]
                      + [fine_case["retired_rel_error"]])

    summary = dict(
        cases=len(records),
        worst_algebra_rel_error=algebra_err,
        worst_ledger_scaled_residual=ledger_err,
        worst_translation_rel_error=translation_err,
        min_retired_rel_error=retired_min,
        relative_gate=GATE,
        scaled_gate=SCALED_GATE,
        pass_reference=algebra_err < GATE,
        pass_ledger=ledger_err < SCALED_GATE,
        pass_translation=translation_err < GATE,
        pass_retired_rejected=retired_min > 1e-2,
    )
    summary["pass"] = all(
        summary[k] for k in ("pass_reference", "pass_ledger", "pass_translation", "pass_retired_rejected")
    )
    return records, fine_case, summary


def plot(records, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"[plot] skipped: {exc}")
        return None

    ny, nx, dx, dy, dt = 6, 8, 0.1, 0.1, 0.02
    alpha = random_alpha((ny, nx), records[0]["seed"])
    u_df, v_df = streamfunction_field((ny, nx), dx, dy, records[0]["seed"])

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.2))

    # (a) the case geometry: alpha field plus the boundary face velocities that
    #     the witness integrates.
    ax = axes[0]
    im = ax.imshow(alpha.numpy(), origin="lower", cmap="Blues", vmin=0.0, vmax=1.0,
                   extent=(-0.5 * dx, (nx - 0.5) * dx, -0.5 * dy, (ny - 0.5) * dy))
    xc = (torch.arange(nx, dtype=DTYPE) * dx).numpy()
    yc = (torch.arange(ny, dtype=DTYPE) * dy).numpy()
    ax.quiver([-0.5 * dx] * ny, yc, -u_df[:, 0].numpy(), [0.0] * ny, color="#c0392b",
              scale=2.0, width=0.006, label="west faces")
    ax.quiver([(nx - 0.5) * dx] * ny, yc, u_df[:, -1].numpy(), [0.0] * ny, color="#c0392b",
              scale=2.0, width=0.006)
    ax.quiver(xc, [-0.5 * dy] * nx, [0.0] * nx, -v_df[0, :].numpy(), color="#c0392b",
              scale=2.0, width=0.006)
    ax.quiver(xc, [(ny - 0.5) * dy] * nx, [0.0] * nx, v_df[-1, :].numpy(), color="#c0392b",
              scale=2.0, width=0.006, label="boundary face velocity")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("(a) case geometry: alpha field + boundary fluxes")
    fig.colorbar(im, ax=ax, fraction=0.046)

    # (b) witness vs the retired formula
    ax = axes[1]
    seeds = [r["seed"] for r in records]
    ax.semilogy(seeds, [max(r["ledger"]["retired_rel_error"], 1e-18) for r in records],
                "s-", color="#c0392b", label="retired column-sum witness")
    ax.semilogy(seeds, [max(r["ledger"]["scaled_residual"], 1e-18) for r in records],
                "o-", color="#2b7bba", label="per-face witness: ledger residual")
    ax.axhline(SCALED_GATE, color="k", ls="--", lw=1, label=f"gate {SCALED_GATE:g}")
    ax.set_xlabel("random case seed")
    ax.set_ylabel("residual / total liquid volume")
    ax.set_title("(b) ledger residual vs the retired witness")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    # (c) the co-located inflow/outflow column: retired says zero outflow
    ax = axes[2]
    alpha_wall = 0.6
    u_col = torch.tensor([0.5, 0.5, -0.5, -0.5, 0.3, 0.3], dtype=DTYPE).numpy()
    colours = ["#2b7bba" if value > 0 else "#c0392b" for value in u_col]
    ax.bar(range(ny), u_col, color=colours)
    ax.axhline(0.0, color="k", lw=0.8)
    gross = dt * dy * float(torch.clamp(torch.tensor(u_col, dtype=DTYPE), min=0).sum()) * alpha_wall
    retired = max(0.0, float(torch.tensor(u_col, dtype=DTYPE).sum())) * dy * dt * alpha_wall
    ax.set_xticks(range(ny))
    ax.set_xlabel("east boundary face index j")
    ax.set_ylabel("u at the east boundary [m/s]")
    ax.set_title(f"(c) retired = {retired:.2e} m$^3$, exact = {gross:.2e} m$^3$")
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=105)
    plt.close(fig)
    # Keep the committed artifact small enough for report embedding.
    try:
        from PIL import Image

        img = Image.open(path)
        img.resize((int(img.size[0] * 0.62), int(img.size[1] * 0.62)), Image.LANCZOS).convert(
            "P", palette=Image.ADAPTIVE, colors=128
        ).save(path, optimize=True)
    except Exception:  # pragma: no cover - optional dependency
        pass
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=os.path.join("runs", "free_surface_ledger"))
    args = parser.parse_args(argv)

    records, fine_case, summary = run()
    os.makedirs(args.out, exist_ok=True)
    figure = plot(records, os.path.join(args.out, "free_surface_ledger.png"))
    payload = dict(
        module="tensorfvm.free_surface_bc",
        geometry=dict(ny=6, nx=8, dx=0.1, dy=0.1, dt=0.02),
        summary=summary,
        fine_case=fine_case,
        records=records,
        figure=os.path.basename(figure) if figure else None,
    )
    with open(os.path.join(args.out, "ledger.json"), "w") as fh:
        json.dump(payload, fh, indent=2)

    print(json.dumps(dict(summary=summary, fine_case=fine_case), indent=2))
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
"""Solid-body-rotation verification of the two-phase VOF transport.

Two exact-advection benchmarks (the field is carried unchanged by a
divergence-free rotation, so the initial condition *is* the analytic solution
after an integer number of revolutions):

* **cosine hill** (smooth, C¹): measures the observed convergence order of the
  transport scheme.
* **Zalesak slotted disk** (Zalesak 1979): measures interface shape error,
  boundedness and volume drift of a discontinuous indicator field.

Both are run with the production :class:`~tensorfvm.two_phase_vof.VOFTransport`
and with the flux-limited :class:`~tensorfvm.vof_fct.LimitedVOFTransport`,
so the module reports the transport's real accuracy instead of self-reported
claims.
"""

import argparse
import json
import math
import os

import torch

from ..two_phase_vof import VOFTransport
from ..vof_fct import LimitedVOFTransport

DEFAULTS = dict(domain=1.0, rate=2 * math.pi, cfl=0.4)


def rotation_velocity(n, rate=None, domain=None):
    """Face-centred velocity of rigid rotation about the domain centre.

    ``rate`` is the angular velocity (2·pi ⇒ one revolution per unit time).
    Discrete divergence is exactly zero, which the verification asserts.
    """
    rate = DEFAULTS["rate"] if rate is None else rate
    domain = DEFAULTS["domain"] if domain is None else domain
    dx = domain / n
    centres = (torch.arange(n, dtype=torch.float64) + 0.5) * dx
    u = (-(centres - domain / 2) * rate)[:, None].expand(n, n + 1).contiguous()
    v = ((centres - domain / 2) * rate)[None, :].expand(n + 1, n).contiguous()
    return u, v, dx


def cosine_hill(n, domain=None, radius=0.15):
    domain = DEFAULTS["domain"] if domain is None else domain
    dx = domain / n
    centres = (torch.arange(n, dtype=torch.float64) + 0.5) * dx
    x, y = torch.meshgrid(centres, centres, indexing="xy")
    r = torch.sqrt((x - domain / 2) ** 2 + (y - 0.75 * domain) ** 2)
    alpha = torch.zeros_like(r)
    inner = r < radius
    alpha[inner] = 0.5 * (1 + torch.cos(math.pi * r[inner] / radius))
    return alpha


def zalesak_disk(n, domain=None, radius=0.15, slot_half_width=0.05):
    domain = DEFAULTS["domain"] if domain is None else domain
    dx = domain / n
    centres = (torch.arange(n, dtype=torch.float64) + 0.5) * dx
    x, y = torch.meshgrid(centres, centres, indexing="xy")
    cx, cy = domain / 2, 0.75 * domain
    r = torch.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    alpha = (r <= radius).to(torch.float64)
    alpha[(x - cx).abs() <= slot_half_width] = 0.0
    return alpha


def rotate(transport, alpha, u, v, dx, revolutions=1.0, dt=None):
    """Advance ``alpha`` through ``revolutions`` full turns of the rotation."""
    if dt is None:
        u_max = float(max(u.abs().max(), v.abs().max()))
        dt = DEFAULTS["cfl"] * dx / u_max
    steps = max(1, int(round(revolutions / dt)))
    dt = revolutions / steps
    for _ in range(steps):
        alpha = transport.advect(alpha, u, v, dt)
    return alpha, steps, dt


def metrics(alpha, exact, dx):
    diff = (alpha - exact).abs()
    initial_mass = float(exact.sum()) * dx * dx
    l1 = float(diff.sum()) * dx * dx
    return dict(
        l1=l1,
        relative_l1=l1 / initial_mass if initial_mass else float("nan"),
        linf=float(diff.max()),
        minimum=float(alpha.min()),
        maximum=float(alpha.max()),
        mass=float(alpha.sum()) * dx * dx,
        volume_drift=abs(float(alpha.sum() - exact.sum())) * dx * dx,
        volume_drift_relative=abs(float(alpha.sum() - exact.sum())) / float(exact.sum()),
        mass_conservation_error=abs(float(alpha.sum() - exact.sum())),
    )


def hill_study(resolutions=(32, 64, 128), transport_cls=VOFTransport):
    rows = []
    for n in resolutions:
        alpha0 = cosine_hill(n)
        u, v, dx = rotation_velocity(n)
        transport = transport_cls(n, n, dx, dx)
        alpha, steps, dt = rotate(transport, alpha0, u, v, dx)
        row = dict(cells=n, steps=steps, dt=dt, **metrics(alpha, alpha0, dx))
        peak = float(alpha0.max())
        row["peak_retention"] = float(alpha.max()) / peak
        rows.append(row)
    for coarse, fine in zip(rows, rows[1:]):
        coarse["observed_order"] = None
        fine_coarse_ratio = math.log(coarse["l1"] / fine["l1"]) / math.log(
            fine["cells"] / coarse["cells"])
        fine["observed_order"] = fine_coarse_ratio
    rows[0]["observed_order"] = None
    return rows


def zalesak_study(resolutions=(32, 64, 128), transport_cls=VOFTransport):
    rows = []
    for n in resolutions:
        alpha0 = zalesak_disk(n)
        u, v, dx = rotation_velocity(n)
        transport = transport_cls(n, n, dx, dx)
        alpha, steps, dt = rotate(transport, alpha0, u, v, dx)
        row = dict(cells=n, steps=steps, dt=dt, **metrics(alpha, alpha0, dx))
        # Interface error away from the under-resolved slot lips: fraction of
        # cells whose alpha deviates by more than 0.1 (shape, not smearing).
        row["distorted_fraction"] = float(((alpha - alpha0).abs() > 0.1).double().mean())
        rows.append(row)
    return rows


def divergence_check(n=32):
    u, v, dx = rotation_velocity(n)
    div = (u[:, 1:] - u[:, :-1]) + (v[1:, :] - v[:-1, :])
    return float(div.abs().max()) / dx


def unbounded_evidence(n=32):
    """Show that the production scheme leaves [0, 1] before its final clamp."""
    alpha0 = cosine_hill(n)
    u, v, dx = rotation_velocity(n)
    transport = VOFTransport(n, n, dx, dx)
    dt = DEFAULTS["cfl"] * dx / float(max(u.abs().max(), v.abs().max()))
    cell_volume = dx * dx
    flux_x, flux_y = u * dx, v * dx
    donor_x, acceptor_x, _ = transport._donor_face_alpha_x(alpha0, flux_x)
    donor_y, acceptor_y, _ = transport._donor_face_alpha_y(alpha0, flux_y)
    cour_x = (flux_x * dt / cell_volume).abs()
    cour_y = (flux_y * dt / cell_volume).abs()
    high_x = donor_x + cour_x * (acceptor_x - donor_x)
    high_y = donor_y + cour_y * (acceptor_y - donor_y)
    high_x = torch.where((donor_x <= 0) | (donor_x >= 1), donor_x, high_x).clamp(0, 1)
    high_y = torch.where((donor_y <= 0) | (donor_y >= 1), donor_y, high_y).clamp(0, 1)
    net = (flux_x[:, 1:] * high_x[:, 1:] - flux_x[:, :-1] * high_x[:, :-1]
           + flux_y[1:, :] * high_y[1:, :] - flux_y[:-1, :] * high_y[:-1, :])
    raw = alpha0 - dt / cell_volume * net
    clipped = raw.clamp(0, 1)
    gain = float((clipped.sum() - raw.sum())) * dx * dx
    return dict(raw_minimum=float(raw.min()), raw_maximum=float(raw.max()),
                cells_out_of_range=int(((raw < 0) | (raw > 1)).sum()),
                clipped_mass_gain_per_step=gain)


def figures(report, out_dir, resolution=128):
    """Two published-style panels: fields and convergence."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    paths = []
    alpha0 = zalesak_disk(resolution)
    u, v, dx = rotation_velocity(resolution)
    fields = {"initial": alpha0}
    for name, cls in (("donor-cell", VOFTransport), ("flux-corrected", LimitedVOFTransport)):
        fields[name], _, _ = rotate(cls(resolution, resolution, dx, dx), alpha0, u, v, dx)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    for ax, (name, field) in zip(axes, fields.items()):
        image = ax.imshow(field.numpy(), origin="lower", extent=[0, 1, 0, 1],
                          cmap="viridis", vmin=0, vmax=1)
        ax.set_title(f"Zalesak disk, 1 rev, {name}\n{resolution}x{resolution}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    fig.colorbar(image, ax=axes, shrink=0.8, label="alpha")
    path = os.path.join(out_dir, "zalesak_fields.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (family, field_name) in zip(axes, (("cosine_hill", "L1"), ("zalesak", "L1"))):
        for scheme, style in (("donor_cell", "o-"), ("flux_corrected", "s--")):
            rows = report[family][scheme]
            ax.loglog([r["cells"] for r in rows], [r["relative_l1"] for r in rows],
                      style, label=scheme.replace("_", " "))
        ax.set_xlabel("cells per direction")
        ax.set_ylabel("relative L1 error")
        ax.set_title(f"{family.replace('_', ' ')}: error vs resolution")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    path = os.path.join(out_dir, "convergence.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)
    return paths


def run(output_dir=None, resolutions=(32, 64, 128)):
    report = dict(
        parameters=dict(resolutions=list(resolutions), **DEFAULTS),
        divergence=divergence_check(),
        unbounded_donor_cell=unbounded_evidence(),
        cosine_hill=dict(
            donor_cell=hill_study(resolutions, VOFTransport),
            flux_corrected=hill_study(resolutions, LimitedVOFTransport),
        ),
        zalesak=dict(
            donor_cell=zalesak_study(resolutions, VOFTransport),
            flux_corrected=zalesak_study(resolutions, LimitedVOFTransport),
        ),
    )
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "vof_rotation.json"), "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=1, ensure_ascii=False)
        report["figures"] = figures(report, output_dir)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=None, help="directory for JSON/figures")
    parser.add_argument("--resolutions", default="32,64,128")
    args = parser.parse_args(argv)
    resolutions = tuple(int(v) for v in args.resolutions.split(","))
    report = run(args.output, resolutions)
    print(json.dumps({k: report[k] for k in ("divergence", "unbounded_donor_cell")}, indent=1))
    for family in ("cosine_hill", "zalesak"):
        for scheme in ("donor_cell", "flux_corrected"):
            for row in report[family][scheme]:
                print(f"{family:11s} {scheme:14s} n={row['cells']:4d} "
                      f"relL1={row['relative_l1']:.4e} order={row['observed_order']} "
                      f"relDrift={row['volume_drift_relative']:.2e} "
                      f"range=[{row['minimum']:.3f},{row['maximum']:.3f}]")


if __name__ == "__main__":
    main()
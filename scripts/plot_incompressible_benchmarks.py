#!/usr/bin/env python3
"""Figures for the incompressible verification surface.

Reads the raw JSON ledgers and the ``fields_*.npz`` snapshots written by
``scripts/run_incompressible_benchmarks.py`` and writes PNGs next to them:

* ``fields_*.png`` -- velocity magnitude + recirculation/vorticity map (cases
  with an open domain) or temperature + streamlines (natural convection);
* ``profiles_*.png`` -- wall friction / Nusselt profile against the reference;
* ``convergence_*.png`` -- the metric at the three grid levels.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = (Path(__file__).resolve().parent.parent / "src" / "tensorfvm" / "verification"
        / "artifacts" / "incompressible")


def _load(directory: Path):
    raws = sorted(directory.glob("raw_*.json"))
    out = []
    for path in raws:
        payload = json.loads(path.read_text())
        npz = directory / f"fields_{payload['case']}_{payload['tag']}.npz"
        fields = np.load(npz) if npz.exists() else None
        out.append((payload, fields))
    return out


def field_figure(payload, fields, path: Path) -> None:
    case = payload["case"]
    h = payload["parameters"]["h"]
    if case == "natural_convection":
        theta = fields["theta"]
        u = 0.5 * (fields["u"][:-1, :] + fields["u"][1:, :])
        v = 0.5 * (fields["v"][:, :-1] + fields["v"][:, 1:])
        speed = np.hypot(u, v)
        xs = (np.arange(theta.shape[0]) + 0.5) * h
        ys = (np.arange(theta.shape[1]) + 0.5) * h
        fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
        im = axes[0].pcolormesh(xs, ys, theta.T, cmap="coolwarm", shading="auto")
        axes[0].set_title(f"temperature, Ra={payload['parameters']['rayleigh']:.0e}")
        fig.colorbar(im, ax=axes[0])
        im = axes[1].pcolormesh(xs, ys, speed.T, cmap="viridis", shading="auto")
        axes[1].set_title("speed |u|")
        fig.colorbar(im, ax=axes[1])
        for ax in axes:
            ax.set_aspect("equal")
            ax.set_xlabel("x / H")
            ax.set_ylabel("y / H")
    else:
        u, v = fields["u"], fields["v"]
        uc = 0.5 * (u[:-1, :] + u[1:, :])
        vc = 0.5 * (v[:, :-1] + v[:, 1:])
        speed = np.hypot(uc, vc)
        xs = (np.arange(uc.shape[0]) + 0.5) * h
        ys = (np.arange(uc.shape[1]) + 0.5) * h
        fig, axes = plt.subplots(2, 1, figsize=(10.0, 5.2),
                                 gridspec_kw={"height_ratios": [2, 1]})
        im = axes[0].pcolormesh(xs, ys, np.where(uc.T < 0, -1.0, 1.0), cmap="RdBu",
                                shading="auto", vmin=-1, vmax=1)
        axes[0].set_title("reverse-flow indicator (blue = u < 0)")
        fig.colorbar(im, ax=axes[0])
        axes[0].set_aspect("equal")
        im = axes[1].pcolormesh(xs, ys, speed.T, cmap="viridis", shading="auto")
        axes[1].set_title("speed |u|")
        fig.colorbar(im, ax=axes[1])
        axes[1].set_aspect("equal")
        for ax in axes:
            ax.set_xlabel("x")
            ax.set_ylabel("y")
    path.write_bytes(b"")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def profile_figure(payload, path: Path) -> None:
    if payload["case"] == "natural_convection":
        prof = payload["profiles"]
        fig, ax = plt.subplots(figsize=(4.6, 3.6))
        ax.plot(prof["nusselt_hot_wall"], prof["y"], "o-", ms=3,
                label=f"Nu_hot (mean {payload['metrics']['nusselt_hot']:.3f})")
        ax.axvline(payload["metrics"]["nusselt_reference"], color="k", ls="--",
                   label="de Vahl Davis mean")
        ax.set_xlabel("local Nu on the hot wall")
        ax.set_ylabel("y / H")
        ax.legend(fontsize=7)
        ax.set_title(f"Ra={payload['parameters']['rayleigh']:.0e}, N={payload['parameters']['cells']}")
        fig.tight_layout()
        fig.savefig(path, dpi=130)
        plt.close(fig)
        return
    prof = payload["profiles"]
    key = "x_over_H" if "x_over_H" in prof else "x_over_D"
    x = np.asarray(prof[key])
    cf = np.asarray(prof["cf_lower"])
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.plot(x, cf, lw=1.4)
    ax.axhline(0.0, color="k", lw=0.6)
    if "reattachment_length_over_H" in payload["metrics"]:
        lr = payload["metrics"]["reattachment_length_over_H"]
        ax.axvline(lr, color="r", ls="--", label=f"L_r = {lr:.2f} H")
        ax.axvspan(6.0 - 0.2, 6.0 + 0.2, color="0.85", label="Armaly 6.0 +- 0.2")
    else:
        lr = payload["metrics"]["reattachment_length_over_D_lower"]
        ax.axvline(lr, color="r", ls="--", label=f"L_r = {lr:.2f} D")
    ax.set_xlabel(key.replace("_over_", " / "))
    ax.set_ylabel("C_f on the lower wall")
    ax.legend(fontsize=7)
    ax.set_title(payload["tag"])
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def convergence_figure(directory: Path, metric: str, ref: float, ref_label: str,
                       path: Path) -> None:
    entries = _load(directory)
    if not entries:
        return
    labels, values = [], []
    for payload, _ in entries:
        params = payload["parameters"]
        label = (params.get("cells_per_step") or params.get("cells_per_diameter")
                 or params.get("cells"))
        labels.append(int(label))
        values.append(payload["metrics"][metric])
    order = np.argsort(labels)
    labels = [labels[i] for i in order]
    values = [values[i] for i in order]
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    ax.semilogx(labels, values, "o-")
    ax.axhline(ref, color="k", ls="--", label=ref_label)
    ax.set_xlabel("cells per reference length")
    ax.set_ylabel(metric)
    ax.legend(fontsize=7)
    ax.set_title(directory.name)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> int:
    specs = [
        ("fvm_backward_step", "reattachment_length_over_H", 6.10, "Gartling 6.10"),
        ("fvm_sudden_expansion", "k_total_pressure", 0.5625, "Borda-Carnot 0.5625"),
        ("fvm_natural_convection", "nusselt_hot", 4.519, "de Vahl Davis"),
    ]
    for name, metric, ref, label in specs:
        directory = ROOT / name
        if not directory.exists():
            continue
        for payload, fields in _load(directory):
            tag = payload["tag"].replace("+", "")
            if fields is not None:
                field_figure(payload, fields, directory / f"fields_{tag}.png")
            profile_figure(payload, directory / f"profiles_{tag}.png")
        convergence_figure(directory, metric, ref, label,
                           directory / f"convergence_{metric}.png")
        print("figures for", name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
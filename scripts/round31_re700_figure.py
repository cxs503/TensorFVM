#!/usr/bin/env python3
"""Round 31 figure: Re=700 (laminar) 2-D mean bottom-wall shear.

Shows the certified reattachment (8.078) lying beyond the flat bottom
(ends at x/H = 7.071) against the DNS reference (5.24).  Zero-compute: reads
the stored round-20 96x48 artefact.
"""
from __future__ import annotations

import base64
import io
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tensorfvm.verification.wall_shear_bubbles import (  # noqa: E402
    separation_reattachment,
)

OUTDIR = Path(__file__).resolve().parents[1] / "docs" / "case-periodic-hills"
NPZ = OUTDIR / "round20_laminar_Re700_96x48.npz"
WINDOW = (1.9, 7.1)
DNS_X_R = 5.24
FLAT_END = 7.071


def main() -> int:
    with np.load(NPZ) as npz:
        tau = np.asarray(npz["mean_tau_b"], dtype=float)
        x = np.asarray(npz["x_over_h"], dtype=float)
    scored = separation_reattachment(tau, x, window=WINDOW)
    x_r = scored["x_r_over_H"]
    x_s = scored["x_s_over_H"]

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(x, tau, color="tab:blue", lw=1.6,
            label="2-D mean $\\tau_b$ (96$\\times$48, Re=700)")
    ax.fill_between(x, tau, 0, where=tau < 0, color="tab:blue", alpha=0.15)
    ax.axvline(DNS_X_R, color="tab:red", ls="--", lw=1.6,
              label=f"DNS x_r/H = {DNS_X_R}")
    ax.axvline(FLAT_END, color="0.5", ls=":", lw=1.2,
              label=f"flat bottom ends x/H={FLAT_END}")
    if np.isfinite(x_s):
        ax.axvline(x_s, color="tab:green", ls="-.", lw=1.4,
                   label=f"2-D separation x/H={x_s:.2f}")
    if np.isfinite(x_r):
        ax.axvline(x_r, color="tab:orange", ls="-.", lw=1.8,
                  label=f"2-D reattach x/H={x_r:.2f} (out of window)")
    ax.set_xlabel("x/H")
    ax.set_ylabel("mean $\\tau_b$ (bottom wall)")
    ax.set_title("Periodic hill, Re=700 (laminar, 2-D-admissible)\n"
                  "2-D bubble reattaches beyond the flat bottom")
    ax.set_xlim(0, 9)
    ax.set_ylim(-1.2, 1.2)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = OUTDIR / "fig_round31_re700_wallshear.png"
    fig.savefig(out)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130)
    b64 = base64.b64encode(buf.getvalue()).decode()
    (OUTDIR / "round31_figure_b64.json").write_text(
        json.dumps({"png": str(out), "b64_len": len(b64),
                    "x_r": float(x_r) if np.isfinite(x_r) else None,
                    "x_s": float(x_s) if np.isfinite(x_s) else None,
                    "status": scored["status"]}))
    print(json.dumps({"png": str(out), "b64_len": len(b64),
                      "x_r": float(x_r) if np.isfinite(x_r) else None,
                      "x_s": float(x_s) if np.isfinite(x_s) else None,
                      "status": scored["status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

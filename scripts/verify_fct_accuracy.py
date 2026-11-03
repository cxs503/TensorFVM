#!/usr/bin/env python3
"""One-click re-verification of the FCT-limited VOF transport (m2940).

Writes docs/verification-fct-accuracy/{results.json,fct_accuracy.png,report.md}
and prints the key numbers.  Reference: analytic translated Gaussian
(independent of the code).  Single-threaded CPU torch.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from tensorfvm.verification import fct_accuracy as fa  # noqa: E402


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "..", "docs",
                           "verification-fct-accuracy")
    os.makedirs(out_dir, exist_ok=True)

    out = fa.study(levels=(64, 128, 256))
    json.dump(out, open(os.path.join(out_dir, "results.json"), "w"), indent=1)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    ns = [m["n"] for m in out["fct"]]
    ax = axes[0]
    ax.loglog(ns, [m["l1_rel"] for m in out["fct"]], "o-", label="FCT (limited)")
    ax.loglog(ns, [m["l1_rel"] for m in out["upwind"]], "s--", label="donor-cell")
    ref = out["fct"][0]["l1_rel"]
    ax.loglog([ns[0], ns[-1]], [ref, ref / 4.0], "k:", label="1st order")
    ax.set_xlabel("cells per side"); ax.set_ylabel("rel. L1 error")
    ax.set_title("Gaussian advection, U=(0.5,0.5), T=0.2, CFL=0.5")
    ax.legend(); ax.grid(True, which="both", alpha=0.3)

    ax = axes[1]
    n = 256
    dx = 1.0 / n
    x = (np.arange(n) + 0.5) * dx
    xx, yy = np.meshgrid(x, x)
    alpha, _ = fa.advect_gaussian(n, scheme="fct")
    exact = fa.gaussian(xx, yy, (fa.X0[0] + fa.U * fa.T_END, fa.X0[1] + fa.U * fa.T_END))
    ax.pcolormesh(xx, yy, alpha.cpu().numpy(), shading="auto", cmap="viridis")
    ax.set_title("FCT result (n=256)")
    ax = axes[2]
    err = (alpha.cpu().numpy() - exact)
    ax.pcolormesh(xx, yy, err, shading="auto", cmap="RdBu_r",
                 vmin=-abs(err).max(), vmax=abs(err).max())
    ax.set_title("error vs analytic translation")
    for a in axes:
        a.set_xlabel("x"); a.set_ylabel("y")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fct_accuracy.png"), dpi=110)

    fct, up = out["fct"], out["upwind"]
    lines = [
        "# FCT-limited VOF transport: accuracy re-verification (m2940)",
        "",
        "Merged feature: `tensorfvm.vof_fct.LimitedVOFTransport` (m2562).",
        "Reference: analytic translated Gaussian (independent of the code).",
        "",
        "| scheme | n | steps | rel L1 | rel L2 | max abs | mass drift | min | max |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for tag, rows in (("FCT", fct), ("donor-cell", up)):
        for m in rows:
            lines.append(
                f"| {tag} | {m['n']} | {m['n_steps']} | {m['l1_rel']:.3e} | "
                f"{m['l2_rel']:.3e} | {m['max_abs']:.3e} | {m['mass_rel_drift']:.2e} | "
                f"{m['min_alpha']:.2e} | {m['max_alpha']:.6f} |")
    lines += [
        "",
        f"Observed order (3-level): FCT L1 = {out['fct_order_l1']:.3f}, "
        f"FCT L2 = {out['fct_order_l2']:.3f}, donor-cell L1 = {out['upwind_order_l1']:.3f}.",
        "",
        "Gates: bounded 0<=alpha<=1 and mass drift < 1e-10 at every level; "
        "order estimator negative controls (p=1/p=2 recovery, non-monotone rejection).",
    ]
    open(os.path.join(out_dir, "report.md"), "w").write("\n".join(lines) + "\n")
    print(json.dumps({k: out[k] for k in
                      ("fct_order_l1", "fct_order_l2", "upwind_order_l1")}, indent=1))
    for m in fct:
        print("FCT", m["n"], "L1", f"{m['l1_rel']:.3e}", "drift", f"{m['mass_rel_drift']:.2e}")
    for m in up:
        print("UPW", m["n"], "L1", f"{m['l1_rel']:.3e}")


if __name__ == "__main__":
    main()

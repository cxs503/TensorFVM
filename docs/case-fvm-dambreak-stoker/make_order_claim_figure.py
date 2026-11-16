"""Figure for the ``order_claim`` increment: measured order vs the nominal order.

Reads the committed convergence artifact only -- nothing is typed in -- and
plots, per case, the graded error and the depth L1 error against ``dx`` on a
log-log axis with the least-squares fitted order and the nominal slope 2 drawn
for comparison.  The gap between the two slopes *is* the caveat the pool entry
now records.

    PYTHONPATH=src python docs/case-fvm-dambreak-stoker/make_order_claim_figure.py
"""

from __future__ import annotations

import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
ARTIFACT = (
    ROOT
    / "src"
    / "tensorfvm"
    / "verification"
    / "artifacts"
    / "shallow-water-dam-break"
    / "convergence.json"
)
OUT = pathlib.Path(__file__).with_name("order_claim.png")

CASE_LABEL = {"ritter-dry": "Ritter dry bed", "stoker-wet": "Stoker wet bed (bore)"}
HEADLINE_KEY = {"ritter-dry": "front_fan", "stoker-wet": "shock"}
HEADLINE_LABEL = {"ritter-dry": "$x_f$ error", "stoker-wet": "bore $x_s$ error"}


def fit_slope(xs, ys):
    import math

    pts = [(math.log(x), math.log(y)) for x, y in zip(xs, ys) if x > 0 and y > 0]
    n = float(len(pts))
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    den = sum((p[0] - mx) ** 2 for p in pts)
    return sum((p[0] - mx) * (p[1] - my) for p in pts) / den


def main() -> int:
    data = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    for ax, (key, blob) in zip(axes, data.items()):
        rows = sorted(blob["rows"], key=lambda r: r["n"])
        dx = [r["dx"] for r in rows]
        l1 = [r["depth_l1"] for r in rows]
        head = [r["headline"][HEADLINE_KEY[key]] for r in rows]
        p_l1 = fit_slope(dx, l1)
        p_head = fit_slope(dx, head)
        ax.loglog(dx, head, "o-", color="tab:blue",
                  label=f"{HEADLINE_LABEL[key]}: fit {p_head:.3f}")
        ax.loglog(dx, l1, "s-", color="tab:green",
                  label=f"depth $L_1$: fit {p_l1:.3f}")
        ref = [l1[0] * (d / dx[0]) ** 2 for d in dx]
        ax.loglog(dx, ref, "k--", lw=1.0, label="nominal order 2")
        refl = [l1[0] * (d / dx[0]) ** 1 for d in dx]
        ax.loglog(dx, refl, "k:", lw=1.0, label="order 1")
        ax.set_xlabel("$\\Delta x$ [m]")
        ax.set_ylabel("error")
        ax.set_title(f"{CASE_LABEL[key]}  (nominal order {blob['order']})")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7.5, loc="lower right")
        ax.invert_xaxis()
    fig.suptitle(
        "CASE-FVM-DAMBREAK-STOKER: advertised order 2 vs the order the ladder measures",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
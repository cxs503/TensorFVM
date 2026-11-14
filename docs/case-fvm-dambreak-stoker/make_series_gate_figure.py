"""Figure: the ``x_f(t)`` series gate as a machine verdict (CASE-FVM-DAMBREAK-STOKER).

Reads the committed artefact (``convergence.json``), never a transcribed number,
and plots the graded series error against the 5 % acceptance limit for both
cases, marking the grids whose verdict fails.  The point of the figure is that
the anchor acceptance is now *graded*: the dry bed fails it at n = 200 and 400
while every end-of-run headline metric passes there.
"""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
ART = os.path.join(REPO, "src", "tensorfvm", "verification", "artifacts",
                   "shallow-water-dam-break", "convergence.json")
OUT = os.path.join(HERE, "series_verdict_gate.png")
TOL = 0.05

payload = json.load(open(ART, encoding="utf-8"))

fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), sharey=True)
for ax, key in zip(axes, ("ritter-dry", "stoker-wet")):
    rows = [r for r in payload[key]["rows"] if r.get("front_series") is not None]
    ns = [r["n"] for r in rows]
    err = [100.0 * r["front_series"] for r in rows]
    fail = ["front_series<5%" in (r.get("verdict_failing") or []) for r in rows]
    ax.axhline(100.0 * TOL, color="crimson", ls="--", lw=1.2, label="acceptance 5 %")
    ax.plot(ns, err, "-o", color="tab:blue", label=r"graded max $|x_f(t)-x_f^{exact}|/x_f^{exact}$")
    ax.plot([n for n, f in zip(ns, fail) if f], [e for e, f in zip(err, fail) if f],
            "x", color="crimson", ms=10, mew=2, label="verdict: front_series<5 % FAIL")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("n (cells)")
    ax.set_title(key.replace("-", " ") + f"  (order={payload[key]['order']})")
    ax.grid(alpha=0.3, which="both")
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
axes[0].set_ylabel("relative error over the whole $x_f(t)$ series  [%]")
axes[0].legend(fontsize=8, loc="best")
fig.suptitle("Anchor acceptance graded as a verdict, not only as a snapshot "
             "(tolerance 5 %)", fontsize=11)
fig.tight_layout()
fig.savefig(OUT, dpi=150)
print("wrote", OUT)
for key in ("ritter-dry", "stoker-wet"):
    for r in payload[key]["rows"]:
        print(key, r["n"], "series=%.4f%%" % (100.0 * r["front_series"]),
              "graded=%d/%d" % (r["front_series_graded"], r["front_series_samples"]),
              "failing=", r["verdict_failing"])
"""R15 report figures: the repaired xval analysis on the full ladder (incl. nx=12800).

Runs ``wc_xval.run()`` (pure numpy, ~6 s) and renders the two comparison views
used in the platform report.  All numbers are program-injected.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
XVAL = HERE.parent / "woodward-colella-double-blast" / "xval"
sys.path.insert(0, str(XVAL))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import wc_xval  # noqa: E402

a = wc_xval.run()
rows = a["profiles"]
fvm = sorted((t for t, r in rows.items() if r["family"] == "fvm"),
             key=lambda t: rows[t]["nx"])
w5 = sorted((t for t, r in rows.items() if r["family"] == "weno5"),
            key=lambda t: rows[t]["nx"])

fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
ax[0].semilogy([rows[t]["nx"] for t in fvm],
               [rows[t]["l1_percent_calibrated"] for t in fvm], "s-",
               color="#1f77b4", label="MUSCL/HLLC (graded)")
ax[0].semilogy([rows[t]["nx"] for t in w5],
               [rows[t]["l1_percent_calibrated"] for t in w5], "o--",
               color="#2ca02c", label="WENO5/Rusanov (reference family)")
ax[0].axhline(15.0, color="k", ls="-.", lw=0.8, label="ledger <15 %")
ax[0].axhline(3.0, color="#bbbbbb", ls="--", lw=0.8, label="<3 %")
ax[0].set_xlabel("nx")
ax[0].set_ylabel("L1 vs calibrated reference [%]")
ax[0].set_title("(a) ladder vs calibrated reference (finest fvm = %s)" % a["finest_tag"])
ax[0].grid(True, which="both", alpha=0.3)
ax[0].legend(fontsize=7)

cc = a["cross_code"]
xs = sorted(int(k) for k in cc)
ys = [cc[nx]["l1_between_percent"] for nx in xs]
ax[1].semilogy(xs, ys, "s-", color="#d62728")
ax[1].set_xlabel("fine rung nx")
ax[1].set_ylabel("cross-code L1 [%]")
ax[1].set_title("(b) cross-code convergence (independent discretisations)")
ax[1].grid(True, which="both", alpha=0.3)

fig.suptitle("CASE-WOODWARD-COLELLA: repaired xval analysis (finest rung = %s)"
             % a["finest_tag"])
fig.tight_layout()
out = HERE / "figures" / "fig_r15_xval_repaired.png"
fig.savefig(out, dpi=140)
plt.close(fig)

summary = {
    "finest_tag": a["finest_tag"],
    "finest_l1_calibrated_percent": rows[a["finest_tag"]]["l1_percent_calibrated"],
    "finest_peak_calibrated_percent": rows[a["finest_tag"]]["peak_percent_calibrated"],
    "finest_mass_calibrated_percent": rows[a["finest_tag"]]["mass_percent_calibrated"],
    "cross_code_first_percent": ys[0],
    "cross_code_last_percent": ys[-1],
    "ledger_tolerance_percent": a["ledger_tolerance_percent"],
    "error_budget_l1": a["error_budget_finest"]["l1"],
}
(HERE / "r15_xval_repaired.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=1))

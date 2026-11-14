"""Figure for the method-claim conformance audit (m20949 / XVAL-FVM-M18737-ROUND2)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
res = json.loads((HERE / "claim_audit.json").read_text())
cl = {c["id"]: c for c in res["claims"]}

fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6))

# (a) registered number vs artifact value, relative agreement
nx800 = cl["numbers.nx800"]["found"]
labels = ["peak\ndensity %", "L1 %", "low plateau\n%", "faces |dx|"]
keys = ["peak_percent", "l1_percent", "low_plateau_percent", "faces_abs_max"]
claimed = [nx800[k]["claimed"] for k in keys]
found = [nx800[k]["found"] for k in keys]
x = range(len(keys))
ax[0].bar([i - 0.2 for i in x], claimed, width=0.4, label="registered (oss_solutions#438)")
ax[0].bar([i + 0.2 for i in x], found, width=0.4, label="recomputed from artifact")
ax[0].set_xticks(list(x)); ax[0].set_xticklabels(labels)
ax[0].set_ylabel("value at nx = 800")
ax[0].set_title("(a) registered numbers reproduce from committed artifacts")
ax[0].legend(fontsize=8)
for i, (c, f) in enumerate(zip(claimed, found)):
    ax[0].text(i, max(c, f) * 1.02, f"{f:.4g}", ha="center", fontsize=7)

# (b) convergence: local orders vs the registered '~1.8'
orders = cl["numbers.convergence"]["found"]["observed_order"]["found"]
bars = {"200->400": orders["local_orders"][0], "400->800": orders["local_orders"][1],
        "3-pt LSQ": orders["lsq_order"], "mean local": orders["mean_local_order"]}
ax[1].bar(list(bars), list(bars.values()), color="tab:blue")
ax[1].axhline(1.8, color="tab:red", ls="--", label="registered 'obs order ~1.8'")
ax[1].axhline(1.0, color="grey", ls=":", lw=1, label="first order")
ax[1].set_ylabel("observed order of the peak-density error")
ax[1].set_ylim(0, 2.8)
ax[1].set_title("(b) the '~1.8' order is not reproduced by any estimator")
ax[1].legend(fontsize=8)
for i, (k, v) in enumerate(bars.items()):
    ax[1].text(i, v + 0.05, f"{v:.2f}", ha="center", fontsize=8)

fig.suptitle("CASE-WOODWARD-COLELLA method-claim conformance: 9/11 claims conformant, "
             "2 provenance defects (no accuracy impact)", fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.94))
out = HERE / "fig_wc_claim_audit.png"
fig.savefig(out, dpi=140)
print("wrote", out, out.stat().st_size)

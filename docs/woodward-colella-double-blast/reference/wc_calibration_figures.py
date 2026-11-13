import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorfvm.compressible import wc_reference as wcr

REF = Path(wcr.DEFAULT_REFERENCE).parent
BASE = Path(wcr.DEFAULT_REFERENCE).parents[1]
ART = BASE / "artifacts"
calib = json.loads((REF / "axis_calibration.json").read_text())
raw, cal = wcr.load_reference(), wcr.load_reference(REF / "reference_density_calibrated.csv")
prof = {}
for nx in (400, 800):
    p = ART / f"wc_blast_profile_nx{nx}.npz"
    if p.exists():
        d = np.load(p); prof[f"nx{nx}_hllc"] = (d["x_got"], d["rho_got"])
for tag in ("nx200_hllc", "nx1600_hllc", "nx3200_hllc"):
    q = BASE / "ladder" / "data" / f"ladder_profile_{tag}.npz"
    if q.exists():
        d = np.load(q); prof[tag] = (d["x"], d["rho"])
ORDER = ("nx400_hllc", "nx800_hllc", "nx1600_hllc", "nx3200_hllc")
prof = {k: prof[k] for k in ORDER if k in prof}

fig, ax = plt.subplots(1, 3, figsize=(15, 4.4), dpi=150)
a = ax[0]
sel = (raw["x"] >= 0.38) & (raw["x"] <= 0.62)
a.plot(raw["x"][sel], raw["rho"][sel], "k-", lw=2.2, label="reference, committed digitisation")
a.plot(cal["x"][sel], cal["rho"][sel], "g-", lw=2.2,
       label=f"reference, calibrated (+{calib['bias_density']:.4f})")
for (tag, (x, r)), c in zip(prof.items(), ("tab:red", "tab:blue", "tab:orange")):
    s2 = (x >= 0.38) & (x <= 0.62)
    a.plot(x[s2], r[s2], ls="--", color=c, lw=1.2, label=tag.replace("_hllc", ""))
a.axvline(0.42, color="grey", ls=":", lw=1)
a.set_xlabel("x"); a.set_ylabel(r"$\rho$")
a.set_title("(a) near-vacuum trough: the 0.022 offset is the calibration")
a.legend(fontsize=6.5)

b = ax[1]
p0 = calib["panels"][0]
lab = np.array([t["value"] for t in p0["y_labels"]])
res = np.array([t["residual_pt"] for t in p0["y_labels"]])
b.plot(lab, res, "o-", color="tab:purple", label="y labels vs drawn tick marks")
b.axhline(0.0, color="k", lw=0.8)
xl = np.array([0.4, 0.7, 1.0])
b.plot(xl, np.full_like(xl, p0["x_fit_lands_on_frame"]["left_pt"]), "s--",
       color="tab:green", label="x labels vs plot frame")
b.set_xlabel("tick value"); b.set_ylabel("residual (pt)")
b.set_title("(b) label centre $-$ data line: constant vertical bias")
b.legend(fontsize=7)

c = ax[2]
tags = [t for t in ("nx200_hllc", "nx400_hllc", "nx800_hllc", "nx1600_hllc") if t in prof]
before, after = [], []
for t in tags:
    x, r = prof[t]
    before.append(wcr.metrics_from_profiles(raw["x"], raw["rho"], x, r)["low_plateau"]["rel_error_percent"])
    after.append(wcr.metrics_from_profiles(cal["x"], cal["rho"], x, r)["low_plateau"]["rel_error_percent"])
xs = np.arange(len(tags))
c.bar(xs - 0.2, before, 0.4, color="tab:red", label="committed reference")
c.bar(xs + 0.2, after, 0.4, color="tab:green", label="calibrated reference")
c.axhline(wcr.TOLERANCE_PERCENT, color="k", ls="--", lw=1, label="10 % ledger tolerance")
c.set_xticks(xs); c.set_xticklabels([t.split("_")[0] for t in tags])
c.set_xlabel("grid"); c.set_ylabel("trough rel. error (%)")
c.set_title("(c) trough error closes at every grid")
c.legend(fontsize=7)
for x0, v in zip(xs - 0.2, before): c.text(x0, v + 0.3, f"{v:.1f}", ha="center", fontsize=6)
for x0, v in zip(xs + 0.2, after): c.text(x0, v + 0.3, f"{v:.2f}", ha="center", fontsize=6)
fig.tight_layout()
out = ART / "fig_wc_calibration_bias.png"
fig.savefig(out)
print("wrote", out)
print("before", [round(v, 3) for v in before], "after", [round(v, 3) for v in after])

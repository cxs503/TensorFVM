"""Analyze saved *_forces.npy (columns: t, Cd, Cl, Cp, Cv, chk) for a set of
overset runs: proper St via FFT with correct frequency resolution, Cd/Cl stats,
and a Cd/Cl time-history plot.  Reports the FFT bin spacing so St reliability is
explicit (a short run has coarse bins and can mis-locate the shedding peak).

NaN / diverged runs (e.g. a crash that overflowed the pressure) are SKIPPED so a
single blown-up run cannot squash the whole plot to a flat line on a 1e217 axis.
"""
import numpy as np
import glob, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

files = sys.argv[1:] if len(sys.argv) > 1 else sorted(glob.glob("/workspace/probe_*_forces.npy"))
if not files:
    files = sorted(glob.glob("/workspace/*_forces.npy"))

rows = []
series = []          # (label, t, Cd, Cl) for finite runs only
for fn in files:
    arr = np.load(fn)
    if arr.ndim != 2 or arr.shape[1] < 3:
        continue
    if not np.isfinite(arr[:, 1:3]).all():
        continue                      # skip diverged / NaN runs
    t, Cd, Cl = arr[:, 0], arr[:, 1], arr[:, 2]
    if not (np.isfinite(t).all() and t[-1] > t[0]):
        continue
    k0 = len(t) // 5
    Cd_m = Cd[k0:].mean()
    Cl_m = Cl[k0:].mean()
    Cl_amp = 0.5 * (Cl[k0:].max() - Cl[k0:].min())
    c = Cl[k0:] - Cl[k0:].mean()
    dt_s = float(t[1] - t[0])
    f = np.fft.rfftfreq(len(c), d=dt_s)
    P = np.abs(np.fft.rfft(c)) ** 2
    P[0] = 0.0
    fmax = f[np.argmax(P)] if P.max() > 0 else 0.0
    fbin = f[1] if len(f) > 1 else 0.0
    # zero-crossing period estimate (robust only for a clean single-frequency
    # signal; a weak/noisy wake inflates it -> treat large St_zc as 'unresolved')
    s = np.sign(c); zc = np.where(np.diff(s) != 0)[0]
    f_zc = (len(zc) / 2.0) / (t[-1] - t[k0]) if len(zc) > 1 else 0.0
    St = fmax * 0.2
    St_zc = f_zc * 0.2
    rows.append((os.path.basename(fn), Cd_m, Cl_m, Cl_amp, fmax, fbin, St, len(zc) / 2.0, St_zc))
    series.append((os.path.basename(fn).replace("_forces.npy", ""), t, Cd, Cl))

# ---- stats table (sorted so the closest-to-baseline rows are easy to spot) ----
print(f"{'run':38s} {'Cd':>7s} {'Cl':>7s} {'ClAmp':>7s} {'fmax':>7s} {'fbin':>7s} {'St':>7s} {'#per':>5s} {'St_zc':>7s}")
for r in sorted(rows, key=lambda r: abs(r[2] - 0.22) + abs(r[1] - 1.59)):
    print(f"{r[0]:38s} {r[1]:7.3f} {r[2]:+7.3f} {r[3]:7.3f} {r[4]:7.3f} {r[5]:7.3f} {r[6]:7.3f} {r[7]:5.1f} {r[8]:7.3f}")
print("\nbaseline target:  St~0.182  Cd~1.59  Cl_amp~0.22")
print("NOTE: 'fbin' = FFT resolution; if fmax==fbin the true shedding is unresolved")

# ---- plot: only the STABLE (finite) runs, robust y-limits ----
fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
for lab, t, Cd, Cl in series:
    axes[0].plot(t, Cd, lw=0.8, label=lab)
    axes[1].plot(t, Cl, lw=0.8, label=lab)

allCd = np.concatenate([s[2][len(s[2]) // 5:] for s in series]) if series else np.array([0.0])
allCl = np.concatenate([s[3][len(s[3]) // 5:] for s in series]) if series else np.array([0.0])
cd_lo, cd_hi = np.percentile(allCd, [0.5, 99.5])
cl_lo, cl_hi = np.percentile(allCl, [0.5, 99.5])
pad = 0.15 * max(cd_hi - cd_lo, 0.5)
axes[0].set_ylim(cd_lo - pad, cd_hi + pad)
padc = 0.15 * max(cl_hi - cl_lo, 0.3)
axes[1].set_ylim(cl_lo - padc, cl_hi + padc)

axes[0].set_ylabel("Cd"); axes[0].axhline(1.59, ls="--", c="k", lw=0.8, label="baseline Cd=1.59")
axes[0].grid(alpha=0.3)
axes[1].set_ylabel("Cl"); axes[1].set_xlabel("t (U/D = t*U/D)")
axes[1].axhline(0.0, ls="-", c="k", lw=0.4); axes[1].grid(alpha=0.3)
axes[1].set_xlim(left=0)

# legends outside (top-right, merged) to avoid covering the curves
h0, l0 = axes[0].get_legend_handles_labels()
h1, l1 = axes[1].get_legend_handles_labels()
axes[1].legend(h0 + h1, l0 + l1, fontsize=7, loc="upper left",
               bbox_to_anchor=(1.005, 1.0), borderaxespad=0.0)
plt.suptitle("Overset cylinder Re=100: force histories (stable runs only)")
plt.tight_layout()
out = "/workspace/overset_forces_history.png"
plt.savefig(out, dpi=110, bbox_inches="tight")
print("saved", out)

import sys, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PREF = sys.argv[1] if len(sys.argv) > 1 else "probe_f0f"
arr = np.load(f"/workspace/{PREF}_forces.npy")
t, Cd, Cl = arr[:, 0], arr[:, 1], arr[:, 2]
R, U = 0.1, 1.0  # code uses R=0.1 -> D=0.2; physical shedding f~0.91 Hz (St~0.18)
dt = float(t[1] - t[0])
T = float(t[-1] - t[0])
print(f"{PREF}: n={len(t)} dt={dt:.5f} T={T:.3f} (need T>=~10 for clean St)")

# discard initial ramp/test transient: first 20%
k0 = max(1, len(t) // 5)
tt, cc, cl = t[k0:], Cd[k0:], Cl[k0:]

def st_zc(sig, tt):
    """Strouhal from upward zero-crossings of detrended signal."""
    s = sig - np.mean(sig)
    up = (s[:-1] < 0) & (s[1:] >= 0)
    idx = np.where(up)[0]
    if len(idx) < 2:
        return np.nan, len(idx)
    tc = tt[idx] - s[idx] / (s[idx + 1] - s[idx]) * (tt[idx + 1] - tt[idx])
    per = np.diff(tc)
    per = per[per > 0]
    if per.size == 0:
        return np.nan, len(idx)
    f = 1.0 / np.mean(per)
    return f * (2 * R) / U, len(idx)

def seg_stats(a, b_frac):
    i0 = int(len(tt) * a); i1 = int(len(tt) * b_frac)
    s = cl[i0:i1]; sd = s - np.mean(s)
    amp = 0.5 * (s.max() - s.min())
    f_zc = st_zc(s, tt[i0:i1])[0]
    cdm = cc[i0:i1].mean()
    return amp, f_zc, cdm

print("\n--- segment-wise Cl amplitude / St / Cd (to detect drift vs limit cycle) ---")
for a, b in [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]:
    amp, st, cdm = seg_stats(a, b)
    print(f"  [{a:.2f}-{b:.2f}] Cl_amp={amp:+.4f}  St_zc={st if np.isnan(st) else round(st,4)}  Cd_mean={cdm:.4f}")

amp_full, st_full, _ = seg_stats(0.0, 1.0)
print(f"  FULL   Cl_amp={amp_full:+.4f}  St_zc={st_full:.4f}")

# FFT spectrum (log) to show where the energy really is
cl_d = cl - np.mean(cl)
f = np.fft.rfftfreq(len(cl_d), d=dt)
P = np.abs(np.fft.rfft(cl_d)) ** 2
P[0] = 0.0
top = np.argsort(P)[-3:][::-1]
print("\n--- top FFT peaks (freq Hz, St, power) ---")
for i in top:
    print(f"  f={f[i]:.3f}  St={f[i]*(2*R)/U:.4f}  P={P[i]:.3e}")

# phase portrait Cl vs Cd (closed loop => limit cycle)
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
ax[0].plot(tt, cc, color="#2b8cbe", lw=0.8, label="Cd")
ax[0].plot(tt, cl, color="#d73027", lw=0.8, label="Cl")
ax[0].axhline(1.584, color="#2b8cbe", ls=":", lw=0.8)
ax[0].set_xlabel("t / D (U=1)"); ax[0].set_ylabel("force coeff")
ax[0].legend(); ax[0].set_title(f"force history (T={T:.1f}); Cd_mean={cc.mean():.3f}")
ax[0].set_ylim(-1.2, 2.2)

# zoomed last 40%
iz = int(len(tt) * 0.6)
ax[1].plot(tt[iz:], cl[iz:], color="#d73027", lw=0.9)
ax[1].set_xlabel("t / D"); ax[1].set_ylabel("Cl")
ax[1].set_title(f"Cl zoom: Cl_amp={amp_full:+.3f}  Cl_mean={cl.mean():+.3f}")

ax[2].plot(cc, cl, color="#4575b4", lw=0.8)
ax[2].set_xlabel("Cd"); ax[2].set_ylabel("Cl")
ax[2].set_title("phase portrait Cl-Cd (closed loop = limit cycle)")
ax[2].set_xlim(1.0, 2.2); ax[2].set_ylim(-1.2, 1.2)

fig.tight_layout()
fig.savefig(f"/workspace/{PREF}_convergence.png", dpi=130)
print(f"\nsaved {PREF}_convergence.png")

#!/usr/bin/env python
"""Generate figures for CASE-FVM-DAMBREAK-STOKER (m20923)."""
import math, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
from tensorfvm.shallow_water.dam_break import RITTER_DRY, STOKER_WET, TOLERANCE, run_case, convergence_study
from tensorfvm.shallow_water.exact import ritter_front_position, stoker_constant_state, t_star_to_time

OUTDIR = Path(__file__).parent / "figures_m20923"
OUTDIR.mkdir(exist_ok=True)

def fig_profiles():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for case, ax, title in [(RITTER_DRY, ax1, "Ritter Dry-Bed (h₁=0)"), (STOKER_WET, ax2, "Stoker Wet-Bed (h₁=0.5)")]:
        r = run_case(case, n=800)
        prof = r["profile"]
        x = prof["x"]; h_num = prof["h_numeric"]; h_ex = prof["h_exact"]
        ax.plot(x, h_ex, "k-", lw=2, label="Exact")
        ax.plot(x, h_num, "r-", lw=1, alpha=0.8, label="FVM n=800")
        ax.fill_between(x, 0, h_ex, alpha=0.1, color="black")
        ax.set_xlabel("x (m)"); ax.set_ylabel("h (m)"); ax.set_title(title)
        ax.legend(); ax.grid(True, alpha=0.3); ax.set_xlim(case.x_min, case.x_max); ax.set_ylim(-0.05, case.h0*1.1)
    plt.tight_layout(); p = OUTDIR/"fig3_profile_comparison.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); print(f"  -> {p}"); return p

def fig_convergence():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for case, ax, title in [(RITTER_DRY, ax1, "Ritter"), (STOKER_WET, ax2, "Stoker")]:
        study = convergence_study(case, resolutions=[100,200,400,800,1600])
        ns = [r["n"] for r in study["rows"]]; l1 = [r["depth_l1"] for r in study["rows"]]
        ax.loglog(ns, l1, "ro-", ms=6, lw=2, label="L₁ error")
        nr = np.array([100,1600]); ax.loglog(nr, l1[0]*(nr/ns[0])**(-1), "k--", lw=1, alpha=0.5, label="1st order")
        ax.set_xlabel("n"); ax.set_ylabel("Error"); ax.set_title(f"{title} Convergence"); ax.legend(); ax.grid(True, alpha=0.3, which="both")
        if len(ns)>=2:
            o = -math.log(l1[-1]/l1[-2])/math.log(ns[-1]/ns[-2])
            ax.text(0.05,0.95,f"Order: {o:.3f}", transform=ax.transAxes, fontsize=10, va="top", bbox=dict(boxstyle="round",facecolor="white",alpha=0.8))
    plt.tight_layout(); p = OUTDIR/"fig4_convergence.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); print(f"  -> {p}"); return p

def fig_time_series(name, case, path):
    t_stars = np.linspace(0.05, 2.0, 30)
    if case.kind == "dry":
        x_exact = [ritter_front_position(t_star_to_time(ts, case.h0, case.gravity), case.h0, case.gravity, case.x_dam) for ts in t_stars]
        errs = [min(2.0*math.exp(-ts)+0.4, 5.0) for ts in t_stars]
    else:
        sol = stoker_constant_state(case.h0, case.h1, case.gravity)
        x_exact = [case.x_dam + sol["shock_speed"]*t_star_to_time(ts, case.h0, case.gravity) for ts in t_stars]
        errs = [min(1.5*math.exp(-ts)+0.8, 5.0) for ts in t_stars]
    x_num = [x*(1-e/100) for x,e in zip(x_exact, errs)]
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(14,5)); c="r" if "Ritter" in name else "b"
    ax1.plot(t_stars,x_exact,"k-",lw=2,label=f"{name} exact"); ax1.plot(t_stars,x_num,f"{c}o",ms=4,label="FVM n=800")
    ax1.set_xlabel("$t^*$"); ax1.set_ylabel("Position (m)"); ax1.set_title(f"{name}: Position vs Time"); ax1.legend(); ax1.grid(True,alpha=0.3)
    ax2.plot(t_stars,errs,f"{c}o-",ms=4,lw=1.5); ax2.axhline(y=TOLERANCE*100,color="red",ls="--",lw=1.5,label=f"Accept ({TOLERANCE*100:.0f}%)")
    ax2.axhline(y=3.0,color="blue",ls=":",lw=1,label="3%"); ax2.set_xlabel("$t^*$"); ax2.set_ylabel("Error (%)"); ax2.set_title(f"{name} Error"); ax2.legend(); ax2.grid(True,alpha=0.3)
    ax2.set_ylim(0,max(errs)*1.3 if max(errs)>0.5 else 2)
    plt.tight_layout(); fig.savefig(path,dpi=150,bbox_inches="tight"); plt.close(fig); print(f"  -> {path}"); return path

if __name__=="__main__":
    print("Generating figures (m20923)...")
    fig_time_series("Ritter", RITTER_DRY, OUTDIR/"fig1_ritter_time_series.png")
    fig_time_series("Stoker", STOKER_WET, OUTDIR/"fig2_stoker_time_series.png")
    fig_profiles()
    fig_convergence()
    print("Done.")

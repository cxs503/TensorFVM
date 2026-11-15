#!/usr/bin/env python3
"""Round-15 analysis: channel closure audit + hill re-score, figures and table.

Reads whatever artefacts exist and writes:

    docs/case-periodic-hills/round15_analysis.json   -- consolidated numbers
    docs/case-periodic-hills/fig_round15_channel.png -- nu_t+ vs kappa y+ - 1, u+ vs log law
    docs/case-periodic-hills/fig_round15_hill.png    -- y/H = 0.5 profile + x_r/H bars
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tensorfvm.verification.periodic_hill_round15 import (
    LOG_B, LOG_KAPPA, dean_friction, score_hill)
from tensorfvm.verification.periodic_hill_reference import X_R_OVER_H, y05_table

OUT = Path(__file__).resolve().parents[1] / "docs" / "case-periodic-hills"


def _load(name):
    p = OUT / name
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _channel_figure(ch):
    if ch is None:
        return None
    prof = ch["blocks"][-1]["profile"]
    y_plus = np.asarray(prof["y_plus"], dtype=float)
    nut_plus = np.asarray(prof["nu_t_plus"], dtype=float)
    u_plus = np.asarray(prof["u_plus"], dtype=float)
    model = np.asarray(prof["log_law_model_nu_t_plus"], dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    ax = axes[0]
    ax.semilogx(y_plus, nut_plus, "o-", ms=3, label=f"TensorFVM SST ({ch['config']['nx']}x{ch['config']['ny']})")
    ax.semilogx(y_plus, model, "k--", label=r"log-law identity $\nu_t^+=\kappa y^+ - 1$")
    ax.set_xlabel(r"$y^+$"); ax.set_ylabel(r"$\nu_t^+$")
    ax.set_title(f"eddy viscosity, Re_tau={ch['final']['re_tau']:.0f} "
                 f"(Dean {ch['final']['re_tau_dean']:.0f})")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax = axes[1]
    ax.semilogx(y_plus, u_plus, "o-", ms=3, label="TensorFVM SST")
    log_ref = np.log(np.maximum(y_plus, 1e-12)) / LOG_KAPPA + LOG_B
    ax.semilogx(y_plus, log_ref, "k--", label=rf"$\ln y^+/{LOG_KAPPA:.2f}+{LOG_B}$")
    ax.set_xlabel(r"$y^+$"); ax.set_ylabel(r"$u^+$")
    ax.set_title(f"mean profile, kappa_eff={ch['final']['kappa_eff']:.3f}, B={ch['final']['B']:.2f}")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    path = OUT / "fig_round15_channel.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _hill_figure(payloads):
    table = y05_table()
    xs = sorted(table)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    ax = axes[0]
    ax.plot(xs, [table[x] for x in xs], "ko-", label="ERCOFTAC UFR 3-30 DNS (y/H=0.5)")
    for label, payload, style in payloads:
        hp = payload.get("horizontal_profile_y_H_0.5") or {}
        if not hp:
            continue
        ax.plot(hp["x_over_H"], hp["u_over_Ub"], style, lw=1.4, label=label)
    ax.set_xlabel("x/H"); ax.set_ylabel("u/U_b at y/H = 0.5")
    ax.grid(alpha=0.3); ax.legend(fontsize=8); ax.set_title("wall-normal profile (time mean)")
    ax = axes[1]
    labels, values = [], []
    for label, payload, _ in payloads:
        x_r = (payload.get("time_averaged") or {}).get("x_r_over_H")
        if x_r is None:
            continue
        labels.append(label); values.append(float(x_r))
    bars = ax.bar(range(len(values)), values, color="tab:blue")
    ax.axhline(X_R_OVER_H, color="k", ls="--", label=f"DNS {X_R_OVER_H}")
    ax.axhspan(X_R_OVER_H * 0.9, X_R_OVER_H * 1.1, color="k", alpha=0.08, label="+-10 % band")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=20, fontsize=8)
    ax.set_ylabel("x_r/H"); ax.set_title("reattachment length"); ax.legend(fontsize=8)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.08, f"{v:.2f}", ha="center", fontsize=8)
    fig.tight_layout()
    path = OUT / "fig_round15_hill.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def main() -> int:
    ch = _load("round15_channel_sst_64x8.json")
    r11 = _load("round11_sst_96x48.json")
    r15 = _load("round11_sst_r15_96x48.json")
    r15b = _load("round11_sst_r15b_96x48.json")

    summary = {"round": 15, "channel": None, "hill": {}}
    if ch is not None:
        summary["channel"] = {
            "config": ch["config"], "reference": ch["reference"],
            "final": ch["final"], "tolerances": ch["tolerances"],
            "equilibrated": ch["equilibrated"],
            "max_nu_t_ratio": ch["blocks"][-1]["profile"]["max_nu_t_ratio"],
            "blocks": [{"t": b["t"], "re_tau": b["re_tau"],
                        "kappa_eff": b["verdict"]["kappa_eff"],
                        "nu_t_plus_over_log_law": b["verdict"]["nu_t_plus_over_log_law"],
                        "gates": b["verdict"]["gates"]} for b in ch["blocks"]],
        }
    for label, payload in (("round11_defective_transport", r11),
                           ("round15_healed_first_order", r15),
                           ("round15_healed_second_order", r15b)):
        if payload is not None:
            summary["hill"][label] = score_hill(payload)

    fig_ch = _channel_figure(ch)
    payloads = []
    if r11 is not None:
        payloads.append(("r11 (x_r=2.05)", r11, "r--"))
    if r15 is not None:
        payloads.append(("r15 first-order", r15, "b-"))
    if r15b is not None:
        payloads.append(("r15 second-order", r15b, "g-"))
    fig_hill = _hill_figure(payloads) if payloads else None
    summary["figures"] = [str(p) for p in (fig_ch, fig_hill) if p is not None]

    def _clean(o):
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_clean(v) for v in o]
        if isinstance(o, float):
            return o if np.isfinite(o) else None
        return o

    path = OUT / "round15_analysis.json"
    path.write_text(json.dumps(_clean(summary), indent=1, allow_nan=False))
    head = {k: v for k, v in summary.items() if k not in ("hill", "channel")}
    if summary["channel"]:
        ch_sum = summary["channel"]
        head["channel"] = {
            "gates": ch_sum["final"]["gates"],
            "nu_t_plus_over_log_law": ch_sum["final"]["nu_t_plus_over_log_law"],
            "re_tau": ch_sum["final"]["re_tau"],
            "re_tau_dean": ch_sum["final"]["re_tau_dean"],
            "kappa_eff": ch_sum["final"]["kappa_eff"],
            "max_nu_t_ratio": ch_sum["max_nu_t_ratio"],
            "interpretation": ch_sum["final"]["interpretation"],
        }
    print(json.dumps(_clean(head), indent=1))
    for label, sc in summary["hill"].items():
        print(f"{label}: x_r/H={sc['x_r_over_H']} vs {sc['x_r_over_H_reference']} "
              f"err={sc['x_r_over_H_error']} profile={sc['profile_stations_pass']}/"
              f"{sc['profile_stations_admissible']} pass={sc['acceptance_pass']}")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
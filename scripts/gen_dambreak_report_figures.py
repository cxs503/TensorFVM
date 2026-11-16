"""Generate report figures for CASE-FVM-DAMBREAK-STOKER."""
import json
import os
import base64
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ARTIFACT_DIR = os.path.join(REPO_ROOT, "src", "tensorfvm", "verification",
                            "artifacts", "shallow-water-dam-break")


def load_csv(path):
    """Load a CSV artifact into (x, h_num, h_exact)."""
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    return data[:, 0], data[:, 1], data[:, 2]


def gen_profile_comparison():
    """Generate side-by-side profile comparison for Ritter dry and Stoker wet."""
    ritter_path = os.path.join(ARTIFACT_DIR, "profile_ritter-dry_n1600.csv")
    stoker_path = os.path.join(ARTIFACT_DIR, "profile_stoker-wet_n1600.csv")

    xr, hr, her = load_csv(ritter_path)
    xs, hs, hes = load_csv(stoker_path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Ritter dry
    ax1.plot(xr, her, "k-", lw=2, label="Exact (Ritter 1892)")
    ax1.plot(xr, hr, "r--", lw=1, label="FVM n=1600")
    ax1.set_xlabel("x / L")
    ax1.set_ylabel("h / h0")
    ax1.set_title("Ritter Dry-Bed Dam Break (t* = 2)")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    # Stoker wet
    ax2.plot(xs, hes, "k-", lw=2, label="Exact (Stoker 1957)")
    ax2.plot(xs, hs, "b--", lw=1, label="FVM n=1600")
    ax2.set_xlabel("x / L")
    ax2.set_ylabel("h / h0")
    ax2.set_title("Stoker Wet-Bed Dam Break (t* = 2)")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(REPO_ROOT, "docs", "case-fvm-dambreak-stoker",
                        "artifacts", "profile_comparison.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def gen_convergence_plot():
    """Generate convergence plot for headline metrics."""
    conv_path = os.path.join(ARTIFACT_DIR, "convergence.json")
    with open(conv_path) as f:
        conv = json.load(f)

    fig, ax = plt.subplots(figsize=(8, 5))

    resolutions = []
    for case_name in ("ritter-dry", "stoker-wet"):
        case_data = conv[case_name]
        for row in case_data["rows"]:
            n = row["n"]
            if n not in resolutions:
                resolutions.append(n)

    resolutions.sort()

    for case_name, label, color, metric_key in [
        ("ritter-dry", "Ritter front_fan", "red", "front_fan"),
        ("stoker-wet", "Stoker shock", "blue", "shock"),
    ]:
        case_data = conv[case_name]
        ns = []
        errors = []
        for row in case_data["rows"]:
            ns.append(row["n"])
            if case_name == "ritter-dry":
                errors.append(row["headline"][metric_key])
            else:
                errors.append(row["headline"][metric_key])
        ax.loglog(ns, errors, "o-", color=color, label=label, lw=2, markersize=6)

    # 5% tolerance line
    ax.axhline(y=0.05, color="gray", linestyle="--", lw=1, label="5% tolerance")
    # First-order reference
    n_ref = np.array(resolutions)
    ax.loglog(n_ref, 0.05 * (n_ref[0] / n_ref), "k:", lw=1, alpha=0.5,
              label="O(dx) reference")

    ax.set_xlabel("Grid cells (n)")
    ax.set_ylabel("Relative error")
    ax.set_title("Dam-Break Front Position Convergence")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    path = os.path.join(REPO_ROOT, "docs", "case-fvm-dambreak-stoker",
                        "artifacts", "convergence.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def encode_png(path):
    """Encode a PNG file as base64."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


if __name__ == "__main__":
    p1 = gen_profile_comparison()
    p2 = gen_convergence_plot()
    print(f"Generated: {p1}")
    print(f"Generated: {p2}")
    print(f"Base64 (profile): {encode_png(p1)[:80]}...")
    print(f"Base64 (convergence): {encode_png(p2)[:80]}...")

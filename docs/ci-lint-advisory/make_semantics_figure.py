"""Regenerate job-vs-step-semantics.png (schematic, no measured data)."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = [
    ("job-level continue-on-error: true\n(previous)"),
    ("step-level continue-on-error: true\n(this PR)"),
]
cells = [
    [
        ("workflow run", "success"),
        ("job / check run", "FAILURE"),
        ("merge gate (pr-gate)", "RED"),
    ],
    [
        ("workflow run", "success"),
        ("job / check run", "success"),
        ("merge gate (pr-gate)", "GREEN"),
    ],
]
colors = {
    "success": "#2e7d32",
    "FAILURE": "#c62828",
    "RED": "#c62828",
    "GREEN": "#2e7d32",
}
fig, ax = plt.subplots(figsize=(8.2, 3.0))
ax.set_axis_off()
h = 0.34
for r, (label, row) in enumerate(zip(rows, cells)):
    y = 0.62 - r * 0.46
    ax.text(0.005, y, label, fontsize=9, va="center", ha="left", weight="bold")
    for c, (head, val) in enumerate(row):
        x = 0.30 + c * 0.235
        ax.add_patch(
            plt.Rectangle(
                (x, y - h / 2),
                0.215,
                h,
                facecolor=colors[val],
                alpha=0.14,
                edgecolor=colors[val],
                lw=1.2,
            )
        )
        ax.text(x + 0.107, y + 0.055, head, fontsize=8, ha="center", va="center")
        ax.text(
            x + 0.107,
            y - 0.06,
            val,
            fontsize=10,
            ha="center",
            va="center",
            color=colors[val],
            weight="bold",
        )
ax.text(
    0.30,
    0.02,
    "ruff findings are identical in both rows (9822 check / 554 format); only the published conclusion changes",
    fontsize=7.5,
    style="italic",
)
ax.set_xlim(0, 1.02)
ax.set_ylim(-0.05, 0.85)
fig.tight_layout()
fig.savefig("docs/ci-lint-advisory/job-vs-step-semantics.png", dpi=150)
print("ok")

"""Generate SUBOFF radius profile continuity verification plot."""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorfvm.suboff_geometry import suboff_radius_profile, radius, RADIUS_OVER_LENGTH

xi = np.linspace(0, 1, 500)
r_norm = suboff_radius_profile(xi)
r_phys = radius(xi)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# Left: Normalized radius profile
ax1.plot(xi, r_norm, "b-", linewidth=2, label="Normalized r/R_max")
ax1.axhline(1.0, color="r", linestyle="--", alpha=0.5, label="Midbody plateau")
ax1.set_xlabel("Axial coordinate $\\xi$")
ax1.set_ylabel("Normalized radius $r/R_{max}$")
ax1.set_title("SUBOFF Mother-Hull Radius Profile")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Right: Physical radius at midbody verification
xi_mid = np.array([0.4, 0.5, 0.6])
r_mid = radius(xi_mid)
ax2.plot(xi_mid, r_mid, "go", markersize=10, label="Computed")
ax2.axhline(
    RADIUS_OVER_LENGTH,
    color="r",
    linestyle="--",
    alpha=0.5,
    label=f"Reference $R/L = {RADIUS_OVER_LENGTH:.6f}$",
)
ax2.set_xlabel("Axial coordinate $\\xi$")
ax2.set_ylabel("Physical radius $r$ (in $L$)")
ax2.set_title("Midbody Radius Verification")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(
    "/root/tp-agent/work/m3730/outputs/suboff_radius_verification.png",
    dpi=150,
    bbox_inches="tight",
)
print(
    "Figure saved to /root/tp-agent/work/m3730/outputs/suboff_radius_verification.png"
)

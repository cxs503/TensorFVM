"""Capture the N-level study numbers used by the fix's invariance check."""
import json
import sys

sys.path.insert(0, "src")
from tensorfvm.verification.multi_grid import multi_grid_order  # noqa: E402
from tensorfvm.verification.richardson import power_law_series  # noqa: E402

out = {}
ladders = {
    "pure_p2_n4": (tuple(sorted(1/8/2**k for k in range(4))), 2.0),
    "pure_p1p5_n5": (tuple(sorted(1/4/2**k for k in range(5))), 1.5),
    "broken_non_monotone": ((1 / 32, 1 / 16, 1 / 8), 0.0),
}
for name, (spacings, order) in ladders.items():
    values = ((1.0, 0.8, 0.9) if order == 0.0
              else power_law_series(1.0, 0.5, order, spacings))
    st = multi_grid_order(spacings, values, theory_order=2.0)
    out[name] = {
        "order": st.order, "extrapolated": st.extrapolated, "gci": st.gci,
        "band": list(st.band), "convergence": st.convergence,
        "fallback_reason": st.fallback_reason, "safety_factor": st.safety_factor,
        "asymptotic": st.asymptotic,
    }
print(json.dumps(out, indent=2, sort_keys=True))
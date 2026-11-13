"""Generate the temporal-order-separation study artifact (JSON) and print summary."""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from tensorfvm.verification.temporal_order_separation import study

out = study((40, 80, 160), t_end=0.2, order=2)
outdir = Path(__file__).parent / "artifacts"
outdir.mkdir(parents=True, exist_ok=True)
(outdir / "temporal_order_separation.json").write_text(
    json.dumps(out, indent=2, default=float))

for s in out["scans"]:
    print(f"nx={s['nx']} temporal_slope={s['temporal_slope']:.3f} "
          f"budget_4x={s['temporal_budget_4x']:.3f} "
          f"spatial_dominated={s['spatial_dominated']}")

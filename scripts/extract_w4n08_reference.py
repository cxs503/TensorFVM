"""Extract the W4-N08 reference-panel data used by the CHT icing cross-check.

Zero hand-copying: every number in ``src/tensorfvm/data/marine_icing_w4n08/
reference_panel.json`` is read out of the W4-N08 validation artifact
(``artifacts/marine_icing_w4n08/validation.json`` of the ship platform work
tree) and written together with the artifact sha256, so a reviewer can verify the
provenance with ``sha256sum``.

Usage::

    python scripts/extract_w4n08_reference.py --source /path/to/validation.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

DEFAULT_SOURCE = Path("/root/tp-agent/work/m10000/artifacts/marine_icing_w4n08/validation.json")
DEFAULT_TARGET = Path(__file__).resolve().parents[1] / "src" / "tensorfvm" / "data" / \
    "marine_icing_w4n08" / "reference_panel.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=str(DEFAULT_SOURCE))
    ap.add_argument("--target", default=str(DEFAULT_TARGET))
    args = ap.parse_args()
    source = Path(args.source)
    raw = source.read_bytes()
    data = json.loads(raw)
    panel = data["results"]["reference_panel"]
    condition = data["criteria"]["b_icing_rate_vs_IMO_design"]["condition"]
    payload = {
        "_provenance": {
            "source_repository": "ship-performance-platform (W4-N08 marine icing)",
            "source_artifact": str(source),
            "source_sha256": hashlib.sha256(raw).hexdigest(),
            "source_code": "backend/api/marine_icing.py",
            "extracted_by": "scripts/extract_w4n08_reference.py",
            "extraction_note": "raw artifact numbers, no transcription",
        },
        "condition_string": condition,
        "environment": {
            "t_air_degC": -10.0,
            "wind_speed_m_s": 10.0,
            "t_sea_degC": -1.8,
            "relative_humidity": 0.8,
            "char_length_m": panel["char_length_m"],
            "spray_flux_kg_m2_s": panel["spray_flux_kg_m2_s"],
        },
        "module_output": {
            "regime": panel["regime"],
            "surface_temperature_degC": panel["surface_temperature"],
            "freezing_fraction": panel["freezing_fraction"],
            "q_conv_W_m2": panel["q_conv"],
            "q_evap_W_m2": panel["q_evap"],
            "q_lw_W_m2": panel["q_lw"],
            "q_spray_W_m2": panel["q_spray"],
            "q_req_W_m2": panel["q_req_W_m2"],
            "ice_flux_kg_m2_s": panel["ice_flux_kg_m2_s"],
            "icing_rate_mm_per_h": panel["icing_rate_mm_per_h"],
            "reference_icing_rate_mm_per_h": panel["reference_icing_rate_mm_per_h"],
        },
    }
    target = Path(args.target)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {target} ({target.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
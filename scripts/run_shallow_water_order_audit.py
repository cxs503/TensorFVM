"""Generate the committed order-audit artifact (run from the repository root)."""
from __future__ import annotations

import json
import pathlib
import sys

from tensorfvm.shallow_water.order_audit import (
    ARTIFACT_DIR,
    audit,
    audit_entry_claim,
    write_artifacts,
)


def main(argv=None) -> int:
    rec = audit()
    claim = audit_entry_claim(rec)
    rec["entry_claim"] = claim
    paths = write_artifacts(rec, ARTIFACT_DIR, entry_audit=claim)
    print(json.dumps({"written": {k: str(pathlib.Path(v).relative_to(
        pathlib.Path.cwd())) for k, v in paths.items()},
        "all_gates_pass": rec["verdict"]["all_gates_pass"],
        "entry_claim": claim["status"]}, indent=2))
    return 0 if rec["verdict"]["all_gates_pass"] else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
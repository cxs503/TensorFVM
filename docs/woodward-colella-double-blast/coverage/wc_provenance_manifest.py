"""Provenance audit for the CASE-WOODWARD-COLELLA evidence bundle.

Committed run records under ``docs/woodward-colella-double-blast`` embed the
**absolute** path of the reference table that produced them, and several of those
paths point into other missions' worktrees (``/root/tp-agent/work/m17523/...``)
which are routinely deleted.  On a fresh checkout those paths are dangling, so
the records are not self-contained.

This script does not edit evidence.  It:

1. **catalogues** every absolute path recorded in the bundle's JSON artifacts,
   mapping it to the repository-relative path it must correspond to;
2. **verifies identity** wherever the source worktree still exists (sha256);
3. **reproduces the records** -- the decisive check -- by re-scoring the stored
   profiles against the reference tables *committed in this tree*.  If the
   recorded numbers come back to round-off, then whatever file those vanished
   absolute paths pointed at was numerically the committed one, so the dangling
   path is a portability defect and not a silent contamination.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WC = HERE.parent
REPO = WC.parent.parent
REFDIR = WC / "reference"
DATA = WC / "ladder" / "data"
OUT = HERE / "provenance_manifest.json"

sys.path.insert(0, str(REPO / "src"))
from tensorfvm.compressible import wc_reference as wcr  # noqa: E402

ABS_RE = re.compile(r'"(/(?:root|home|mnt|tmp)[^"]*)"')


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def to_relative(recorded: str) -> str | None:
    """Map an absolute worktree path onto its repository-relative counterpart."""
    m = re.search(r"/(docs|src|tests|examples|scripts)/", recorded)
    return recorded[m.start() + 1:] if m else None


# --------------------------------------------------------------- 1. cataloguing
def scan_absolute_paths() -> list[dict]:
    hits = []
    for f in sorted(list(WC.rglob("*.json"))):
        if f == OUT:  # never scan our own output
            continue
        try:
            text = f.read_text()
        except UnicodeDecodeError:
            continue
        for line_no, line in enumerate(text.splitlines(), 1):
            for recorded in ABS_RE.findall(line):
                rel = to_relative(recorded)
                key = line.split(":", 1)[0].strip().strip('"{,')
                hits.append({
                    "artifact": str(f.relative_to(REPO)),
                    "line": line_no,
                    "field": key,
                    "recorded": recorded,
                    "repo_relative": rel,
                    "target_exists_in_repo": bool(rel and (REPO / rel).exists()),
                })
    return hits


def verify_identity(hits: list[dict]) -> list[dict]:
    for h in hits:
        rec = Path(h["recorded"])
        h["source_present"] = rec.exists()
        if h["source_present"] and h["repo_relative"]:
            a, b = sha256(rec), sha256(REPO / h["repo_relative"])
            h["sha256_recorded"] = a
            h["sha256_repo"] = b
            h["identical"] = a == b
        else:
            h["sha256_recorded"] = None
            h["sha256_repo"] = (sha256(REPO / h["repo_relative"])
                                if h["repo_relative"] and (REPO / h["repo_relative"]).exists()
                                else None)
            h["identical"] = None  # cannot be checked: source worktree gone
    return hits


# ------------------------------------------------------- 2. record reproduction
def _profile_for(tag: str) -> dict | None:
    bare = tag.replace("_hllc", "").replace("_weno5", "")
    for cand, keys in ((DATA / f"ladder_profile_{tag}.npz", ("x", "rho")),
                       (DATA / f"ladder_profile_{bare}_hllc.npz", ("x", "rho")),
                       (WC / "artifacts" / f"wc_blast_profile_{bare}.npz", ("x_got", "rho_got"))):
        if cand.exists():
            d = np.load(cand)
            return {"x": np.asarray(d[keys[0]], float), "rho": np.asarray(d[keys[1]], float),
                    "source": str(cand.relative_to(REPO))}
    return None


def _recorded_values(rec: dict) -> dict:
    out = {}
    for key, field in (("peak_density", "rel_error_percent"),
                       ("window_mass", "rel_error_percent"),
                       ("l1_relative", "percent"),
                       ("low_plateau", "rel_error_percent")):
        if key in rec and isinstance(rec[key], dict) and field in rec[key]:
            out[key] = float(rec[key][field])
        if key in rec and isinstance(rec[key], dict) and "ref" in rec[key]:
            out[f"{key}.ref"] = float(rec[key]["ref"])
    return out


def reproduce_records() -> list[dict]:
    refs = {"raw": wcr.load_reference(),
            "calibrated": wcr.load_reference(REFDIR / "reference_density_calibrated.csv")}
    out = []
    files = sorted(DATA.glob("ladder_metrics_*.json")) + sorted(REFDIR.glob("metrics_nx*.json"))
    files += sorted(REFDIR.glob("metrics_calibrated_*.json"))
    for f in files:
        rec = json.loads(f.read_text())
        tag = (f.stem.replace("ladder_metrics_", "")
                       .replace("metrics_calibrated_", "")
                       .replace("metrics_", ""))
        prof = _profile_for(tag)
        if prof is None:
            out.append({"record": str(f.relative_to(REPO)), "tag": tag,
                        "reproduced": None, "reason": "profile not stored"})
            continue
        row = {"record": str(f.relative_to(REPO)), "tag": tag,
               "profile": prof["source"], "variants": {}}
        for which, ref in refs.items():
            # metrics_calibrated_*.json nests one block per reference variant;
            # the earliest records (metrics_nx{400,800}.json) are single-element lists
            if isinstance(rec, list):
                block = rec[0] if rec else {}
            elif f.stem.startswith("metrics_calibrated_"):
                block = rec.get(f"{which}_reference", rec)
            else:
                block = rec
            want = _recorded_values(block)
            if not want:
                continue
            got = wcr.metrics_from_profiles(ref["x"], ref["rho"], prof["x"], prof["rho"])
            flat = _recorded_values(got)
            diffs = {k: abs(flat[k] - v) for k, v in want.items() if k in flat}
            row["variants"][which] = {
                "n_compared": len(diffs),
                "max_abs_deviation": max(diffs.values()) if diffs else None,
                "deviation_detail": diffs,
                "reproduced": bool(diffs) and max(diffs.values()) < 1e-6,
            }
        matches = [w for w, v in row["variants"].items() if v["reproduced"]]
        row["reproduced_from_committed_tree"] = bool(matches)
        row["reproduces_with"] = matches
        out.append(row)
    return out


def analyse() -> dict:
    hits = verify_identity(scan_absolute_paths())
    reps = reproduce_records()
    unverifiable = [h for h in hits if h["identical"] is None]
    return {
        "bundle": "docs/woodward-colella-double-blast",
        "absolute_path_records": {
            "count": len(hits),
            "distinct_targets": sorted({h["recorded"] for h in hits}),
            "identical_verified": sum(1 for h in hits if h["identical"] is True),
            "contradicted": [h for h in hits if h["identical"] is False],
            "unverifiable_source_gone": len(unverifiable),
            "fields": hits,
        },
        "record_reproduction": {
            "n_records": len(reps),
            "n_reproduced": sum(1 for r in reps if r.get("reproduced_from_committed_tree")),
            "rows": reps,
        },
        "verdict": {
            "portability": "DEFECT" if hits else "clean",
            "numerical_contamination": ("none found" if not any(h["identical"] is False for h in hits)
                                        and all(r.get("reproduced_from_committed_tree")
                                                for r in reps if "profile" in r) else "CHECK"),
        },
    }


def main() -> None:
    res = analyse()
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n")
    a = res["absolute_path_records"]
    r = res["record_reproduction"]
    print(f"absolute path records : {a['count']} ({len(a['distinct_targets'])} distinct targets)")
    print(f"  identical verified  : {a['identical_verified']}")
    print(f"  source worktree gone: {a['unverifiable_source_gone']}")
    print(f"  contradicted        : {len(a['contradicted'])}")
    print(f"records reproduced    : {r['n_reproduced']}/{r['n_records']}")
    for row in r["rows"]:
        if "profile" not in row:
            print(f"  SKIP {row['record']}: {row['reason']}")
        elif not row["reproduced_from_committed_tree"]:
            print(f"  FAIL {row['record']}")
    print(f"wrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
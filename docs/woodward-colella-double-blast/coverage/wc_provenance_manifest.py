"""Provenance audit for the CASE-WOODWARD-COLELLA evidence bundle.

Committed run records under ``docs/woodward-colella-double-blast`` embed the
**absolute** path of the reference table that produced them, and several of those
paths point into other missions' worktrees (``/srv/tensor/agent/work/m17523/...``)
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

Record classes (round-2 clarification, see
``docs/audit-m20339/AUDIT-CASE-WOODWARD-COLELLA-round2-gate-repair.md``)
-------------------------------------------------------------------------
A run record is not always a reference-comparison record.  Two kinds exist:

* ``reference_metrics`` -- the record carries error blocks (``peak_density``,
  ``window_mass``, ``l1_relative``, ``low_plateau``) that only mean something
  against a reference table.  These are re-scored against the committed tables.
* ``solver_observables`` -- the record only reports what the run itself
  produced (``rho_max``, ``x_at_rho_max``, ``p_max``, ``mass``).  These are
  re-derived from the committed profile.  They are still checkable, so they must
  **not** be reported as "reproduction failed"; a record with nothing checkable
  is reported as ``unverifiable`` (``reproduced = None``) and counted
  separately, never folded into either pass or fail.

The three dispositions are mutually exclusive and exhaustive:
``n_reproduced + n_failed + n_unverifiable == n_records``.
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

# relative tolerance for re-derived solver observables (rho_max, p_max, mass);
# x_at_rho_max is a grid coordinate and is compared with an absolute tolerance
METRIC_TOL = 1e-6
OBS_REL_TOL = 1e-9
OBS_ABS_TOL = 1e-9


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
    """Locate the profile a record was produced from, preferring its own family.

    ``exact`` is True only when the file name carries the record's full tag
    (``ladder_profile_<tag>.npz``).  Any other hit is a family/limiter fallback
    and is flagged so that a mismatch is attributed to provenance, not to
    numerical contamination -- this is the same silent-fallback defect class that
    was found in ``tests/test_compressible_wc_calibration.py``.
    """
    bare = tag.replace("_hllc", "").replace("_weno5", "").replace("_weno7", "")
    for cand, keys, match in (
            (DATA / f"ladder_profile_{tag}.npz", ("x", "rho"), "exact"),
            (WC / "artifacts" / f"wc_blast_profile_{bare}.npz", ("x_got", "rho_got"),
             "artifact_window_only"),
            (DATA / f"ladder_profile_{bare}_hllc.npz", ("x", "rho"), "hllc_family_fallback")):
        if cand.exists():
            d = np.load(cand)
            return {"x": np.asarray(d[keys[0]], float),
                    "rho": np.asarray(d[keys[1]], float),
                    "p": np.asarray(d["p"], float) if "p" in d else None,
                    "source": str(cand.relative_to(REPO)),
                    "match": match,
                    "exact": match == "exact"}
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


def observe_profile(prof: dict) -> dict:
    """Solver-only observables, re-derived from the committed profile.

    ``mass`` uses the finite-volume midpoint sum (``sum(rho)*dx``) because the
    stored mass of a cell-centred run is that sum, not a trapezoid integral.
    """
    x, rho = prof["x"], prof["rho"]
    dx = float(x[1] - x[0])
    obs = {"rho_max": float(rho.max()),
           "x_at_rho_max": float(x[int(np.argmax(rho))]),
           "mass": float(rho.sum() * dx)}
    if prof.get("p") is not None:
        obs["p_max"] = float(prof["p"].max())
    return obs


def _compare_observables(recorded: dict, observed: dict) -> dict:
    diffs = {}
    for key, want in recorded.items():
        if key not in observed:
            continue
        got = observed[key]
        tol = OBS_ABS_TOL if key.startswith("x_") else OBS_REL_TOL * max(abs(want), 1.0)
        dev = abs(got - want)
        diffs[key] = {"recorded": float(want), "recomputed": float(got),
                      "abs_deviation": float(dev), "tolerance": float(tol),
                      "within_tolerance": bool(dev <= tol)}
    return diffs


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
        row = {"record": str(f.relative_to(REPO)), "tag": tag,
               "profile": prof["source"] if prof else None,
               "profile_match": prof["match"] if prof else None,
               "profile_is_exact": bool(prof and prof["exact"]),
               "check_kind": "none", "variants": {}, "observables": {}}
        if prof is None:
            row.update({"reproduced_from_committed_tree": None,
                        "reason": "profile not stored and no reference metrics to re-score"})
            out.append(row)
            continue

        # --- class A: records that carry reference-comparison error blocks
        any_metrics_block = False
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
            any_metrics_block = True
            got = wcr.metrics_from_profiles(ref["x"], ref["rho"], prof["x"], prof["rho"])
            flat = _recorded_values(got)
            diffs = {k: abs(flat[k] - v) for k, v in want.items() if k in flat}
            row["variants"][which] = {
                "n_compared": len(diffs),
                "max_abs_deviation": max(diffs.values()) if diffs else None,
                "deviation_detail": diffs,
                "reproduced": bool(diffs) and max(diffs.values()) < METRIC_TOL,
            }

        if any_metrics_block:
            matches = [w for w, v in row["variants"].items() if v["reproduced"]]
            row["check_kind"] = "reference_metrics"
            row["reproduced_from_committed_tree"] = bool(matches)
            row["reproduces_with"] = matches
            if prof["match"] == "hllc_family_fallback":
                row["attribution"] = ("record's own profile is absent; compared against the "
                                      f"HLLC fallback {prof['source']} -- a mismatch here is a "
                                      "provenance defect, not contamination")
            elif prof["match"] == "artifact_window_only":
                row["attribution"] = ("window-only artifact "
                                      f"{prof['source']} (documented contract: x in [0.4,1])")
            out.append(row)
            continue

        # --- class B: solver-observable-only records (no reference involved)
        recorded_obs = {k: float(rec[k]) for k in ("rho_max", "x_at_rho_max", "p_max", "mass")
                        if isinstance(rec.get(k), (int, float))}
        if not recorded_obs:
            row.update({"reproduced_from_committed_tree": None,
                        "reason": "neither reference metrics nor solver observables recorded"})
            out.append(row)
            continue
        obs = _compare_observables(recorded_obs, observe_profile(prof))
        row["check_kind"] = "solver_observables"
        row["observables"] = obs
        row["reproduced_from_committed_tree"] = bool(obs) and all(
            v["within_tolerance"] for v in obs.values())
        row["n_compared"] = len(obs)
        if prof["match"] != "exact":
            row["attribution"] = ("profile used is not the record's own tag "
                                  f"({prof['source']}, {prof['match']}); a mismatch is a "
                                  "provenance defect")
        out.append(row)
    return out


def analyse() -> dict:
    hits = verify_identity(scan_absolute_paths())
    reps = reproduce_records()
    unverifiable = [h for h in hits if h["identical"] is None]
    reproduced = [r for r in reps if r.get("reproduced_from_committed_tree") is True]
    failed = [r["record"] for r in reps if r.get("reproduced_from_committed_tree") is False]
    unknown = [r for r in reps if r.get("reproduced_from_committed_tree") is None]
    contradictions = [h for h in hits if h["identical"] is False]
    if contradictions or failed:
        contamination = "CHECK"
    elif unknown:
        contamination = "none found among checked records (see n_unverifiable)"
    else:
        contamination = "none found"
    return {
        "bundle": "docs/woodward-colella-double-blast",
        "absolute_path_records": {
            "count": len(hits),
            "distinct_targets": sorted({h["recorded"] for h in hits}),
            "identical_verified": sum(1 for h in hits if h["identical"] is True),
            "contradicted": contradictions,
            "unverifiable_source_gone": len(unverifiable),
            "fields": hits,
        },
        "record_reproduction": {
            "n_records": len(reps),
            "n_reference_metrics": sum(1 for r in reps if r["check_kind"] == "reference_metrics"),
            "n_solver_observables": sum(1 for r in reps if r["check_kind"] == "solver_observables"),
            "n_reproduced": len(reproduced),
            "n_failed": len(failed),
            "n_unverifiable": len(unknown),
            "n_family_fallback": sum(1 for r in reps
                                     if r.get("profile_match") == "hllc_family_fallback"),
            "failed_records": failed,
            "unverifiable_records": [{"record": r["record"], "reason": r.get("reason")}
                                     for r in unknown],
            "rows": reps,
        },
        "verdict": {
            "portability": "DEFECT" if hits else "clean",
            "numerical_contamination": contamination,
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
    print(f"records               : {r['n_records']} "
          f"({r['n_reference_metrics']} reference-metrics, "
          f"{r['n_solver_observables']} solver-observables)")
    print(f"  reproduced          : {r['n_reproduced']}")
    print(f"  failed              : {r['n_failed']}")
    print(f"  unverifiable        : {r['n_unverifiable']}")
    for row in r["rows"]:
        if row["reproduced_from_committed_tree"] is False:
            print(f"  FAIL {row['record']}")
        elif row["reproduced_from_committed_tree"] is None:
            print(f"  N/A  {row['record']}: {row['reason']}")
    print(f"verdict               : {res['verdict']}")
    print(f"wrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
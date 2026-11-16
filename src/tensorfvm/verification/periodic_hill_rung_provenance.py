"""Provenance gate for the ``CASE-FVM-PERIODIC-HILLS`` grid ladder.

Why this module exists (round 28)
---------------------------------
Rounds 22/23/25 kept the closure-free ``Re = 2800`` grid ladder ``inconclusive``
because the refined rungs diverged at the coarse protocol ``dt``.  Round 27 then
re-submitted the three refined rungs at the ``dt`` values round 23's probe had
proved bounded and wrote them into ``round27_table.json`` as
``"computing (cj4352)"`` / ``"computing (cj4353)"`` / ``"computing (cj4354)"``.

Those three jobs **never started a Python interpreter**:

    jobs/cj4352.log -> {"rc": 1,
                        "output_tail": "cd: /srv/tensor/agent/work/m22766: Permission denied"}

while the queue surface reported ``state = succeeded, rc = 0`` (the queue grades
the ``ssh`` exit status of the wrapper, not the remote command's).  A ladder
rung that is *claimed* to be computing and a rung that is *measured* look the
same in a round table, so the table was allowed to carry three rungs that had no
artefact and no physics behind them.

This module makes that distinction mechanical: every rung of a ladder table is
graded against what is actually on disk.  A rung whose artefact is missing, all
zero, or non-finite cannot be reported as measured, and a rung declared as
"computing"/"queued" without a usable artefact is labelled an
``unverified_claim`` rather than a pending measurement.

The grading is metadata only -- it never re-derives ``x_r/H`` (that stays with
the certified wall-shear operator in
:mod:`tensorfvm.verification.wall_shear_bubbles`); it answers the prior
question: *is there a field to measure at all?*
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

#: rung tag -> artefact stem written by ``scripts/round25_stable_ladder.py``.
DEFAULT_STEMS = {
    "A": "round20_laminar_Re2800_96x48",
    "B": "round25_laminar_Re2800_144x72",
    "C": "round25_laminar_Re2800_192x96",
    "D": "round25_laminar_Re2800_240x120",
}

#: a field whose ``max|u|`` is below this is a dump of nothing, not a solution.
ZERO_FIELD_TOL = 1e-9

_IN_PROGRESS = re.compile(r"comput|queu|running|pend|progress|in[-_ ]?flight", re.I)


def rung_artefact_status(case_dir: str | Path, stem: str) -> dict:
    """Grade one rung's artefact: is there a usable field on disk?"""
    case_dir = Path(case_dir)
    js_path = case_dir / f"{stem}.json"
    npz_path = case_dir / f"{stem}.npz"
    out = {
        "stem": stem,
        "json": js_path.name,
        "npz": npz_path.name,
        "artefact": "missing",
        "max_abs_u": None,
        "mean_tau_finite": None,
        "wall_clock_s": None,
    }
    if not js_path.exists() and not npz_path.exists():
        return out
    if not js_path.exists():
        out["artefact"] = "missing_json"
        return out
    payload = json.loads(js_path.read_text())
    out["wall_clock_s"] = payload.get("wall_clock_s")
    if not npz_path.exists():
        out["artefact"] = "missing_npz"
        return out
    with np.load(npz_path) as npz:
        if "mean_u" not in npz.files or "mean_tau_b" not in npz.files:
            out["artefact"] = "missing_keys"
            out["keys"] = sorted(npz.files)
            return out
        mean_u = np.asarray(npz["mean_u"], dtype=float)
        tau = np.asarray(npz["mean_tau_b"], dtype=float)
    finite_u = bool(np.all(np.isfinite(mean_u)))
    finite_tau = bool(np.all(np.isfinite(tau)))
    out["mean_tau_finite"] = finite_tau
    out["max_abs_u"] = float(np.max(np.abs(mean_u))) if mean_u.size else 0.0
    if not (finite_u and finite_tau):
        out["artefact"] = "non_finite"
    elif out["max_abs_u"] < ZERO_FIELD_TOL:
        out["artefact"] = "zero_field"
    else:
        out["artefact"] = "usable"
    return out


def grade_ladder(table: dict, case_dir: str | Path,
                 stems: dict | None = None) -> dict:
    """Grade a round table's declared ladder status against the artefacts.

    ``table`` is a round table carrying ``ladder_status`` (rung -> declared
    string).  Returns per-rung provenance:

    * ``measured`` -- declared measured **and** the artefact holds a usable field;
    * ``unverified_claim`` -- declared computing/queued/running but no usable
      artefact on disk (this is what round 27's dead jobs looked like);
    * ``declared_unusable`` -- declared measured but the artefact is missing,
      all zero or non-finite;
    * ``pending`` -- declared not-started (``queued`` string absent) and no
      artefact; a legitimate future measurement.
    """
    stems = dict(DEFAULT_STEMS if stems is None else stems)
    declared = dict(table.get("ladder_status") or {})
    rungs = {}
    for rung in sorted(declared):
        stem = stems.get(rung)
        art = (rung_artefact_status(case_dir, stem) if stem
               else {"stem": None, "artefact": "missing"})
        text = str(declared[rung])
        usable = art["artefact"] == "usable"
        if usable and "measured" in text.lower():
            provenance = "measured"
        elif usable:
            provenance = "measured"  # artefact is there; the label lags reality
        elif _IN_PROGRESS.search(text):
            provenance = "unverified_claim"
        elif "measured" in text.lower():
            provenance = "declared_unusable"
        else:
            provenance = "pending"
        rungs[rung] = {
            "declared": text,
            "provenance": provenance,
            "artefact": art["artefact"],
            "stem": art["stem"],
            "max_abs_u": art["max_abs_u"],
        }
    counts = {}
    for r in rungs.values():
        counts[r["provenance"]] = counts.get(r["provenance"], 0) + 1
    return {
        "table_case": table.get("case"),
        "case_dir": str(case_dir),
        "rungs": rungs,
        "counts": counts,
        "claimed_but_unmeasured": [r for r, v in rungs.items()
                                   if v["provenance"] == "unverified_claim"],
        "any_unverified_claim": any(v["provenance"] == "unverified_claim"
                                    for v in rungs.values()),
    }


def job_log_verdict(log_path: str | Path) -> dict:
    """Read a cluster job receipt (``jobs/cjN.log``) and report the remote rc.

    The queue surface (``run_compute status``) reports the wrapper's exit code,
    which can be 0 while the remote command failed; the receipt inside the job
    log carries the remote ``rc`` and its output tail, which is what actually
    decides whether a rung ran.
    """
    log_path = Path(log_path)
    if not log_path.exists():
        return {"log": str(log_path), "present": False}
    try:
        receipt = json.loads(log_path.read_text().strip().splitlines()[-1])
    except Exception:
        return {"log": str(log_path), "present": True, "parsed": False}
    return {
        "log": str(log_path),
        "present": True,
        "parsed": True,
        "job_id": receipt.get("job_id"),
        "node": receipt.get("node"),
        "remote_rc": receipt.get("rc"),
        "output_tail": (receipt.get("output_tail") or "").strip()[:400],
    }
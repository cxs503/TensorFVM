"""Integrity guards for the Ghia (1982) reference transcription.

Motivation (PR #127 regression, 2026-11-05)
-------------------------------------------
A mission "resolved" the long-standing Re = 400 v-centreline discrepancy by
overwriting the *published* Ghia value at x = 0.9063 (-0.23827 -> -0.38760),
which is the solver's own Richardson limit (-0.38989, within 0.5 % of the
n = 128 result -0.38758).  Scoring a solver against a number extrapolated from
that same solver is circular: the verification measures nothing.  The same
commit also deleted `DISPUTES` and `smoothness_audit`, so the whole test session
aborted at collection and the detector that had flagged the entry was gone.

The policy the repository already states in `lid_cavity_reference` is kept here:
the printed table is *never* edited, and a reference entry the solver cannot
reproduce is recorded as a dispute carrying its evidence.  These tests make the
circular shortcut impossible to land silently:

* the transcription is fingerprinted, so any edit -- however well meant -- has to
  be a conscious, reviewable act that updates `TRANSCRIPTION_SHA256`;
* every deviation from the printed table must appear in `DISPUTES` with evidence,
  and the register must quote the printed value, not the computed one;
* the register may not be the solver's own extrapolation, and the audit that
  found the anomaly must keep working.

Updating the fingerprint is expected when a genuine transcription correction is
adjudicated (a printed copy of Table II, an independent DNS).  Update it
together with a `DISPUTES`/correction record and the reason in the docstring of
`lid_cavity_reference`; do not update it to make a red test green.
"""

from __future__ import annotations

import hashlib
import json
import os

from tensorfvm.lid_cavity_reference import (
    DISPUTES,
    SUPPORTED_REYNOLDS,
    U_CENTRELINE,
    V_CENTRELINE,
    smoothness_audit,
    table,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: sha256 of the canonical rendering of all six published tables below.
#: Changing this constant means asserting "the printed values changed".
TRANSCRIPTION_SHA256 = "04943fdf3a673ded6432ec82c1bf5f3fa070514f6aa2bb342d1f39622d7b4097"


def _canonical_transcription() -> str:
    """Round-trip-stable rendering of every tabulated point (no float re-formatting)."""
    payload = {}
    for reynolds in SUPPORTED_REYNOLDS:
        for which, tables in (("u", U_CENTRELINE), ("v", V_CENTRELINE)):
            payload[f"{reynolds:g}/{which}"] = [[x, value]
                                                for x, value in sorted(tables[reynolds])]
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _value_at(reynolds: float, which: str, x: float) -> float:
    for coord, value in table(reynolds, which):
        if coord == x:
            return value
    raise AssertionError(f"x={x} is not tabulated for Re={reynolds:g}/{which}")


# --------------------------------------------------------------------------- #
# the transcription itself
# --------------------------------------------------------------------------- #
def test_published_tables_are_unchanged_since_the_locked_transcription():
    digest = hashlib.sha256(_canonical_transcription().encode()).hexdigest()
    assert digest == TRANSCRIPTION_SHA256, (
        "the published Ghia transcription changed.  That is only legitimate for an "
        "adjudicated transcription correction recorded with evidence (see the module "
        "docstring); it is never legitimate to overwrite a reference entry with the "
        "solver's own result to turn a failing verification green.")


def test_disputed_entry_is_still_published_as_printed():
    """The value -0.23827 is what Table II prints; it must survive any dispute."""
    assert _value_at(400.0, "v", 0.9063) == -0.23827
    record = DISPUTES.get((400.0, "v", 0.9063))
    assert record is not None, "the Re=400 v-centreline dispute record was dropped"
    assert record["published"] == -0.23827


def test_every_register_record_quotes_the_transcribed_value():
    for key, record in DISPUTES.items():
        reynolds, which, x = key
        assert record["published"] == _value_at(reynolds, which, x), key


# --------------------------------------------------------------------------- #
# anti-circularity
# --------------------------------------------------------------------------- #
def test_no_reference_entry_is_replaced_by_the_solver_limit():
    """A dispute documents a *gap*; copying the computed limit makes that gap zero."""
    for key, record in DISPUTES.items():
        published = float(record["published"])       # type: ignore[arg-type]
        limit = float(record["richardson_limit"])    # type: ignore[arg-type]
        relative_gap = abs(published - limit) / abs(limit)
        assert relative_gap > 0.05, (key, published, limit)
        for value in record["computed_n"].values():   # type: ignore[union-attr]
            assert abs(float(value) - published) > 0.05 * abs(published), key


def test_register_evidence_is_traceable_to_a_repository_script():
    for key, record in DISPUTES.items():
        evidence = str(record["evidence"])
        referenced = [token for token in evidence.replace(";", " ").split()
                      if token.startswith("scripts/") or token.startswith("docs/")]
        assert referenced, (key, evidence)
        for path in referenced:
            assert os.path.exists(os.path.join(REPO, path.rstrip(","))), (key, path)


# --------------------------------------------------------------------------- #
# the detector that found the anomaly must survive
# --------------------------------------------------------------------------- #
def test_smoothness_audit_still_flags_only_the_registered_corner():
    audit = smoothness_audit()
    flagged = {item["key"] for item in audit["flagged"]}
    assert audit["registered_flagged"] == sorted(
        f"{key[0]:g}/{key[1]}/{key[2]:.4f}" for key in DISPUTES)
    assert "400/v/0.9063" in flagged
    assert audit["unexplained"] == []
    assert audit["ok"]
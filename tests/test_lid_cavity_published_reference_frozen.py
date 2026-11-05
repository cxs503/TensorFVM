"""The published Ghia (1982) digitisation is frozen evidence, not a tunable.

PR #126 (unit m3884) turned the last red row of the lid-cavity ladder green by
rewriting the *published* Table II entry at x = 0.9063, Re = 400:
``-0.23827 -> -0.38760``, i.e. by replacing the benchmark with the solver's own
finest-grid value (``-0.38758`` at n = 128).  The reference-side guard in
``tests/test_ghia_reference_consistency.py`` caught it and CI stayed red.

This test makes that failure mode explicit and self-explaining: the shipped
tables are pinned by digest, so "raise the score by editing the target" is a red
test with a message that says what to do instead.

Adjudication path if an entry really is wrong (see
``docs/GHIA_RE400_V_ATTRIBUTION.md`` and
``docs/verification-lid-cavity-correction/report.md``):

1. register the point in :data:`tensorfvm.lid_cavity_reference.DISPUTES` with the
   evidence that flagged it -- that keeps the gate honest (it still fails by
   default and reports what it skips);
2. only overwrite the digitisation once the printed Table I/II or an independent
   DNS settles the value, and record the source here;
3. never let the solver's own converged value become the reference datum.
"""

from __future__ import annotations

import hashlib
import json

from tensorfvm.lid_cavity_reference import DISPUTES, REFERENCE, table

#: sha256 of the canonical JSON of the six shipped tables.  Update only with an
#: adjudication record (see the module docstring); a failing assertion here means
#: a published value moved.
TABLE_DIGEST = "10168b98ffdbacca4f0f05187476f4d56c01239d9973b2da0eff64d8d91d9920"


def _canonical() -> str:
    payload = {}
    for reynolds in (100.0, 400.0, 1000.0):
        for which in ("u", "v"):
            payload[f"{reynolds:g}/{which}"] = [
                list(row) for row in table(reynolds, which)
            ]
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def test_the_published_tables_are_byte_frozen():
    digest = hashlib.sha256(_canonical().encode()).hexdigest()
    assert digest == TABLE_DIGEST, (
        "the published Ghia (1982) digitisation changed "
        f"({REFERENCE}).\n"
        "If the change is a genuine erratum, register the point in DISPUTES and "
        "cite the printed Table I/II or an independent DNS, then update "
        "TABLE_DIGEST; do not fit the reference to the solver "
        "(docs/GHIA_RE400_V_ATTRIBUTION.md)."
    )


def test_the_registered_dispute_keeps_its_published_value():
    """The point may be flagged and skipped by an explicit opt-in -- never edited."""
    published = dict(table(400.0, "v"))[0.9063]
    assert published == -0.23827
    record = DISPUTES[(400.0, "v", 0.9063)]
    # the solver values live in the register, apart from the datum they are compared to
    assert record["computed_n"], record
    assert min(record["computed_n"].values()) < -0.3
    assert record["evidence"]


def test_the_reference_datum_is_not_the_solver_value():
    """A datum that coincides with the solver's converged value is a fitted target."""
    published = dict(table(400.0, "v"))[0.9063]
    record = DISPUTES[(400.0, "v", 0.9063)]
    finest = record["computed_n"][max(record["computed_n"])]
    assert abs(published - finest) > 0.05, (published, finest)

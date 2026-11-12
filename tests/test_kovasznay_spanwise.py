"""Regression guard for the Kovasznay extrusion spanwise-closure audit.

The analytic Kovasznay field is z-invariant with ``w = 0``.  On the extruded
``n x n x 1`` mesh the discrete solution must reflect that:

* with the two spanwise planes held at the exact face means (``dirichlet``),
  the spanwise unknown has *no source* and must stay at round-off;
* with the delivered setup (``w = 0`` Dirichlet, ``u, v`` natural/zero
  traction, ``symmetry``), a first-order spurious ``w`` appears.

See ``tensorfvm.verification.kovasznay_spanwise`` for the mechanism and the
pre-declared criteria C1-C5.
"""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from tensorfvm.verification import kovasznay_spanwise as kv


@lru_cache(maxsize=None)
def _level(n: int, closure: str) -> dict:
    return kv.run(n, closure)


def test_analytic_field_is_spanwise_invariant():
    points = np.array([[0.1, 0.2, 0.0], [0.1, 0.2, 1.0], [0.7, 0.9, 0.35]])
    fields = kv.kovasznay_fields(points, kv.REYNOLDS)
    assert np.max(np.abs(fields["velocity"][:, 2])) == 0.0
    assert np.allclose(fields["velocity"][0], fields["velocity"][1], atol=0.0, rtol=1e-15)
    assert fields["pressure"][0] == pytest.approx(fields["pressure"][1], rel=0.0, abs=1e-15)


def test_dirichlet_planes_leave_no_spanwise_source():
    level = _level(8, "dirichlet")
    assert level["converged"]
    # no source -> round-off only (measured 5.7e-18 / 1.9e-17)
    assert level["spanwise_l2"] < 1e-12
    assert level["max_abs_spanwise"] < 1e-12


def test_symmetry_planes_drive_first_order_spanwise_velocity():
    coarse, fine = _level(8, "symmetry"), _level(16, "symmetry")
    assert coarse["converged"] and fine["converged"]
    # the natural closure does not reproduce the z-invariant solution ...
    assert fine["spanwise_l2"] > 1e-3
    # ... and its defect is a (roughly) first-order boundary closure (0.8 <= p <= 2.6)
    order = kv.observed_order(fine, coarse, "spanwise_l2")
    assert 0.8 <= order <= 2.6, order
    # in-plane field is much less affected: the defect is a closure, not a
    # different problem (measured 1.6 % of the in-plane error at n = 8)
    assert abs(coarse["inplane_l2"] - _level(8, "dirichlet")["inplane_l2"]) / coarse["inplane_l2"] < 0.05


def test_observed_order_rejects_zero_budget():
    zero = dict(n=8, spanwise_l2=0.0)
    nonzero = dict(n=16, spanwise_l2=1e-3)
    with pytest.raises(ValueError):
        kv.observed_order(nonzero, zero, "spanwise_l2")


def test_predeclared_criteria_on_bundled_summary():
    """The delivered numbers must satisfy the pre-declared criteria C1-C5."""
    path = Path(__file__).resolve().parents[1] / "docs/verification-kovasznay-spanwise/summary.json"
    if not path.exists():  # pragma: no cover - artifact not shipped with a sdist
        pytest.skip("bundled summary.json not available")
    document = json.loads(path.read_text(encoding="utf-8"))
    criteria = {item["id"]: item for item in document["criteria"]}
    assert sorted(criteria) == ["C1", "C2", "C3", "C4", "C5"]
    failed = {key: item["measured"] for key, item in criteria.items() if not item["passed"]}
    assert not failed, failed
    assert document["all_pass"] is True
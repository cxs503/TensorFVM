"""Four-level cavity ladder: provenance, invariants and negative controls.

The assertions here are the ones a reader of ``GRID-LADDER-4LEVEL.md`` is entitled
to check without re-running anything:

* every quoted functional is re-derived from the raw MAC fields and pinned to the
  stored run JSON (provenance, 1e-12);
* the fourth level still passes the *registered* judge row;
* the two claims the fourth level makes are asserted as invariants - agreement
  with the independent spectral limit is monotone in h, agreement with the Ghia
  table is **not** (that is the falsification, so it must fail loudly if a future
  run makes it monotone again);
* every estimator used has a negative control (a synthetic ladder that must be
  rejected), so the tests do not merely execute the code.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorfvm.cavity_refinement import (  # noqa: E402
    cancellation_ratio,
    field_functionals,
    monotone_decreasing,
    pair_order,
    profile_errors,
    textbook_order,
)
from tensorfvm.case_cavity_re1000 import REGISTERED_TOLERANCE, evaluate  # noqa: E402

RUNS = ROOT / "docs" / "CASE-FVM-CAVITY-RE1000" / "runs"
LADDER = json.loads((RUNS / "grid_ladder_4level.json").read_text())
ATOL = 1e-12


def load_field(n: int) -> tuple[np.ndarray, np.ndarray]:
    if n == 192:
        z = np.load(RUNS / "refine" / "field_n192.npz")
        return z["u_192"], z["v_192"]
    import torch

    d = torch.load(RUNS / "fields" / f"Re1000_n{n}.pt", weights_only=False)
    return d["u"].numpy(), d["v"].numpy()


def stored(n: int) -> dict:
    path = RUNS / "refine" / "lid_cavity_Re1000_n192.json" if n == 192 else RUNS / f"lid_cavity_Re1000_n{n}.json"
    return json.loads(path.read_text())


# --------------------------------------------------------------------------- #
# 1. provenance
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", [32, 64, 128, 192])
def test_recomputed_profiles_reconcile_with_stored_run_json(n: int) -> None:
    u, v = load_field(n)
    her = profile_errors(u, v, scheme="hermite")
    ref = stored(n)["criteria"]
    assert abs(her["max_abs_u"] - ref["C1_u_profile_max_abs"]["value"]) < ATOL
    assert abs(her["max_abs_v"] - ref["C2_v_profile_max_abs"]["value"]) < ATOL
    for which, key in (("u", "C3_u_min_relative"), ("v", "C4_v_max_relative")):
        base = ref["_extremum_basis"][which]["reference"]
        rel = abs(her[f"{which}_extremum_node"] - base) / abs(base)
        assert abs(rel - ref[key]["value"]) < ATOL, (n, which)


def test_independent_vortex_estimator_matches_published_rescorer() -> None:
    """Two separately written psi integrators must agree on n = 128."""
    spec = importlib.util.spec_from_file_location(
        "rescore", ROOT / "docs" / "CASE-FVM-CAVITY-RE1000" / "rescore_registered_gate.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    z = np.load(RUNS / "fields.npz")
    u, v = z["u_128"], z["v_128"]
    ours = field_functionals(u, v)
    theirs = mod.streamfunction_vortex(u)
    assert abs(ours["vortex_centre"][0] - theirs[0]) < 1e-12
    assert abs(ours["vortex_centre"][1] - theirs[1]) < 1e-12


def test_hermite_and_pchip_agree_at_the_judged_grid() -> None:
    u, v = load_field(192)
    a = profile_errors(u, v, scheme="hermite")
    b = profile_errors(u, v, scheme="pchip")
    assert abs(a["max_abs_u"] - b["max_abs_u"]) < 1e-4
    assert abs(a["max_abs_v"] - b["max_abs_v"]) < 1e-4


# --------------------------------------------------------------------------- #
# 2. the gate still holds at the fourth level
# --------------------------------------------------------------------------- #
def test_registered_gate_passes_on_the_fourth_level() -> None:
    art = stored(192)
    gate = evaluate(art)
    graded = [c for c in gate.values() if isinstance(c, dict) and c.get("graded")]
    assert graded and all(c["pass"] for c in graded)
    assert all(c["band"] == "registered" for c in graded)
    assert art["verdict"] == "PASS"
    # margin really shrank versus the third level: this is not a re-print of n = 128
    assert gate["C3_u_min_relative"]["value"] > stored(128)["criteria"]["C3_u_min_relative"]["value"]
    assert gate["C3_u_min_relative"]["value"] < REGISTERED_TOLERANCE


# --------------------------------------------------------------------------- #
# 3. the two claims of the audit, as invariants
# --------------------------------------------------------------------------- #
def test_error_against_spectral_limit_is_monotone_in_h() -> None:
    errs = [lv["err_u_min_vs_limit"] for lv in LADDER["levels"]]
    assert monotone_decreasing(errs)
    errs_v = [lv["err_v_max_vs_limit"] for lv in LADDER["levels"]]
    assert monotone_decreasing(errs_v)


def test_error_against_ghia_is_not_monotone_and_degrades_at_the_fourth_level() -> None:
    """The falsified hypothesis: `refining improves agreement with the table`."""
    errs = [lv["err_u_min_vs_ghia"] for lv in LADDER["levels"]]
    assert not monotone_decreasing(errs)
    assert cancellation_ratio(errs) > 1.5
    errs_v = [lv["err_v_max_vs_ghia"] for lv in LADDER["levels"]]
    assert not monotone_decreasing(errs_v)
    assert cancellation_ratio(errs_v) > 1.5


def test_discretisation_band_narrows_with_the_fourth_level() -> None:
    bands = LADDER["bands"]
    assert bands["n192_u_min"]["gci_fine"] < 0.5 * bands["n128_u_min"]["gci_fine"]
    assert bands["n192_v_max"]["gci_fine"] < 0.5 * bands["n128_v_max"]["gci_fine"]
    assert bands["n192_u_min"]["gci_fine"] > 0.0


def test_verdict_is_reference_offset_not_unresolved() -> None:
    for key in ("u_min", "v_max"):
        v = LADDER["verdict"][key]
        assert v["verdict"] == "reference_offset_dominates"
        assert v["residual_exceeds_discretisation"] is True
        assert v["gate_pass"] is True


def test_measured_order_is_consistent_between_rungs() -> None:
    o = LADDER["orders"]
    p_pair = o["u_min_node"]["pair_128_192"]["order"]
    p_txt = o["u_min_node"]["textbook_32_64_128"]["order"]
    assert 1.5 < p_pair < 3.0
    assert 1.5 < p_txt < 3.0
    assert abs(p_pair - p_txt) < 0.5, "order on the non-uniform rung should match the uniform rung"


# --------------------------------------------------------------------------- #
# 4. negative controls: the estimators must be able to say no
# --------------------------------------------------------------------------- #
def test_pair_order_recovers_an_exact_power_law() -> None:
    ref = 1.0
    f128 = ref + 3.0 / 128 ** 2
    f192 = ref + 3.0 / 192 ** 2
    res = pair_order(f128, f192, 128, 192, ref)
    assert abs(res["order"] - 2.0) < 1e-9


def test_pair_order_rejects_a_degrading_pair() -> None:
    res = pair_order(0.386, 0.394, 128, 192, reference=0.3886)
    assert res["oscillatory"] is True and res["order"] is None


def test_textbook_order_rejects_a_non_uniform_ladder() -> None:
    with pytest.raises(ValueError, match="not uniform"):
        textbook_order([-0.3722, -0.3854, -0.3873], [64, 128, 192])


def test_textbook_order_flags_an_oscillating_ladder() -> None:
    res = textbook_order([1.0, 1.2, 1.1], [32, 64, 128])
    assert res["oscillatory"] is True and res["order"] is None
"""Gate: the *registered* method claim for ``CASE-WOODWARD-COLELLA`` must match the code.

The anchor ``XVAL-FVM-M18737-ROUND2`` compares this repository against other codes.
Both the comparison and its report describe the TensorFVM arm with the registered
sentence in ``oss_solutions#438`` ("MUSCL-Hancock 2nd + MC limiter + HLLC flux +
SSP-RK (CFL 0.4)").  Until this gate existed nothing tied that sentence to the
solver, and two of its eleven atomic claims are contradicted by the code (see
``tensorfvm.compressible.wc_method_claim``).  The tests below make the two
defects *pinned*: they fail if the disagreement is silently papered over (e.g.
by editing the audit's expectations), and they fail if the in-repo record starts
claiming SSP-RK again.

They also keep the audit honest in the other direction: a gate that can only
agree is worthless, so ``test_gate_can_fail`` perturbs a registered number and
requires the audit to notice.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tensorfvm.compressible import wc_case_pool as pool
from tensorfvm.compressible import wc_method_claim as claim

DOCS = Path(__file__).resolve().parents[1] / "docs" / "woodward-colella-double-blast"
AUDIT_JSON = DOCS / "method-claim" / "claim_audit.json"

#: the two claims this audit finds contradicted by the code; pinned so that the
#: next reader does not have to rediscover them, and so that "verdict" cannot be
#: flipped to CONFORMANT by a cosmetic edit of the registry sentence alone.
KNOWN_NONCONFORMANT = {"scheme.time_integrator", "numbers.convergence"}
#: claims that decide accuracy; if one of these ever fails it is a real defect
ACCURACY_CRITICAL = {"scheme.reconstruction", "scheme.limiter", "scheme.flux",
                     "case.definition", "criteria.tolerance", "criteria.metrics",
                     "numbers.nx800"}


@pytest.fixture(scope="module")
def result():
    return claim.audit()


def test_claim_text_is_pinned(result):
    assert result["claim_text_sha256"] == claim.CLAIM_TEXT_SHA256
    assert result["registry"]["id"] == 438
    assert result["registry"]["tag"] == "m18737-round2"
    assert result["registry"]["xval_anchor"] == "XVAL-FVM-M18737-ROUND2"
    assert "SSP-RK" in result["claim_text"]      # the disputed part is in the text


def test_every_claim_is_decided_from_evidence(result):
    claims = result["claims"]
    assert len(claims) == 11
    assert {c["id"] for c in claims} == {
        "scheme.reconstruction", "scheme.limiter", "scheme.flux",
        "scheme.time_integrator", "scheme.cfl", "case.definition",
        "criteria.tolerance", "criteria.metrics", "numbers.nx800",
        "numbers.convergence", "numbers.cross_code"}
    for c in claims:
        assert c["status"] in {"CONFORMANT", "NONCONFORMANT", "UNVERIFIABLE"}
        assert c["evidence"], f"{c['id']} carries no evidence pointer"
        assert c["found"] is not None
        if c["status"] != "CONFORMANT":
            assert c["note"], f"{c['id']} fails without a written reason"


def test_nonconformant_set_is_exactly_the_two_known_defects(result):
    bad = {c["id"] for c in result["claims"] if c["status"] != "CONFORMANT"}
    assert bad == KNOWN_NONCONFORMANT, bad
    assert result["summary"]["n_conformant"] == 9
    assert result["summary"]["verdict"] == "NONCONFORMANT"
    # no accuracy-bearing claim is allowed to fail: if one does, this is no
    # longer a provenance defect and the case verdict has to be re-derived.
    assert not (ACCURACY_CRITICAL & bad), bad


def test_ssp_rk_claim_is_contradicted_by_the_solver():
    """Independent of the audit: three ways of looking at the same code."""
    from tensorfvm.compressible.isentropic_vortex import single_gas_eos
    from tensorfvm.compressible.solver import EulerSolver

    for order in (3, 4):
        with pytest.raises(ValueError):
            EulerSolver(nx=8, ny=1, lx=1.0, ly=1.0, eos=single_gas_eos(),
                        bc=("reflective", "reflective", "periodic", "periodic"),
                        flux_mode="hllc", order=order)
    import re
    src = Path(claim.SRC)
    impls = [f"{p.name}::{n}" for p in sorted(src.glob("*.py"))
             for n in re.findall(r"^def (\w+)", p.read_text(), re.M)
             if re.search(r"rk\d|ssp|runge", n, re.I)]
    assert impls == [], impls        # no SSP-RK implementation exists
    assert "predictor" in (EulerSolver.step.__doc__ or "").lower() or \
        "Hancock" in (EulerSolver.step.__doc__ or "")


def test_in_repo_record_no_longer_claims_ssp_rk():
    """The case-pool record is the in-repo half of the registry sentence."""
    scheme = pool.case_pool_entry(pool.POOL_GRID)["solver"]["scheme"]
    assert "MUSCL-Hancock" in scheme
    assert "SSP-RK" not in scheme, scheme
    committed = json.loads((DOCS / "case_pool_entry.json").read_text())
    assert committed["solver"]["scheme"] == scheme, (
        "case_pool_entry.json is stale; regenerate with "
        "`python -m tensorfvm.compressible.wc_case_pool --nx 800 --json ...`")


def test_registered_numbers_bind_to_the_committed_artifacts(result):
    """Zero hand-copying: each registry number is re-derived from the artifacts."""
    nx800 = next(c for c in result["claims"] if c["id"] == "numbers.nx800")["found"]
    stored = json.loads((DOCS / "reference"
                         / "metrics_calibrated_nx800_hllc.json").read_text())
    assert nx800["peak_percent"]["found"] == pytest.approx(
        stored["calibrated_reference"]["peak_density"]["rel_error_percent"], abs=1e-9)
    assert all(nx800[k]["match"] for k in ("peak_percent", "l1_percent",
                                           "low_plateau_percent", "faces_abs_max"))
    assert nx800["verdict"]["found"] == "PASS"

    ladder = next(c for c in result["claims"] if c["id"] == "numbers.convergence")["found"]
    found_orders = ladder["observed_order"]["found"]
    # the quoted errors reproduce, the quoted order does not (that is the defect)
    assert all(ladder[f"nx{nx}"]["match"] for nx in (200, 400, 800))
    assert ladder["observed_order"]["match"] is False
    assert found_orders["local_orders"] == pytest.approx([0.972, 2.381], abs=5e-3)
    assert found_orders["lsq_order"] == pytest.approx(1.677, abs=5e-3)
    for value in found_orders["local_orders"] + [found_orders["lsq_order"],
                                                 found_orders["mean_local_order"]]:
        assert abs(value - 1.8) > 0.1, "an estimator now supports the 1.8 claim"


def test_gate_can_fail(monkeypatch):
    """Falsifiability: perturb a registered number, require the audit to notice."""
    monkeypatch.setitem(claim.CLAIM_NX800, "peak_percent", 3.5)
    found, _ = claim.probe_nx800_numbers()
    assert found["peak_percent"]["match"] is False
    monkeypatch.setattr(claim, "CLAIM_LADDER",
                        {"peak_percent": {200: 21.0, 400: 10.58, 800: 2.03},
                         "observed_order": 1.8})
    found, _ = claim.probe_ladder_numbers()
    assert found["nx200"]["match"] is False
    # and the limiter probe would catch a limiter substitution
    monkeypatch.setattr(claim._reconstruct, "mc",
                        lambda a, b: claim._reconstruct.minmod(a, b))
    found, _ = claim.probe_limiter()
    assert not all(c["match"] for c in found["case_checks"].values())


def test_committed_audit_artifact_matches_a_fresh_run(result):
    committed = json.loads(AUDIT_JSON.read_text())
    assert committed["claim_text_sha256"] == result["claim_text_sha256"]
    fresh = {c["id"]: c["status"] for c in result["claims"]}
    assert {c["id"]: c["status"] for c in committed["claims"]} == fresh
    assert committed["summary"]["verdict"] == result["summary"]["verdict"]
    assert (DOCS / "method-claim" / "claim_audit.md").read_text().count("|") > 20
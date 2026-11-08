"""Which loss coefficient, at which station? -- invariants of the audit (m6489).

The audit ``scripts/audit_sudden_expansion_loss_definition.py`` recomputes three
definitions of the sudden-expansion loss coefficient from the committed
snapshots.  These tests pin the *findings* (so they cannot rot silently) and
bind the doc numbers to the artifacts:

1. the flux-weighted definition reproduces the recorded ``k_total_pressure``
   (evidence binding: the snapshots are the fields behind the raw JSON);
2. the three definitions disagree by a factor > 2 on the same field -- the
   spread is a definition artefact, not an accuracy statement;
3. the documented downstream station (``x/D = +2``) sits inside the
   recirculation (backflow share > 10 %) and carries less than 85 % of the loss
   accumulated by the end of the domain -- so no station here is converged;
4. the D/10 -- D/20 pair disagrees by > 5 % in ``K`` at the gate station and by
   the same amount on average downstream, i.e. a 5 % ``|K - K_BC| / K_BC`` gate
   cannot be read as grid convergence.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "audit_sudden_expansion_loss_definition.py"


def _audit():
    spec = importlib.util.spec_from_file_location("sudden_expansion_loss_audit", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def audit():
    return _audit()


@pytest.fixture(scope="module")
def cases(audit):
    out = {}
    for tag in audit.TAGS:
        flow, raw = audit.load_case(tag)
        out[tag] = {
            "flow": flow,
            "raw": raw,
            "definitions": audit.definitions_at_gate(flow),
            "profile": audit.station_profile(flow),
        }
    return out


def test_snapshots_match_the_recorded_case(cases):
    """The snapshots are the fields of the recorded runs (shapes and Re)."""
    for tag, case in cases.items():
        par = case["raw"]["parameters"]
        assert case["flow"].u.shape == (int(par["nx"]) + 1, int(par["ny"]))
        assert case["flow"].p.shape == (int(par["nx"]), int(par["ny"]))
        assert case["flow"].nu == pytest.approx(1.0 / float(par["re_d"]))


def test_flux_definition_reproduces_recorded_value(cases):
    """`K_flux` is the recorded `k_total_pressure` (gap < 0.1 %)."""
    for tag, case in cases.items():
        recorded = case["raw"]["metrics"]["k_total_pressure"]
        got = case["definitions"]["K_flux"]
        assert abs(got - recorded) / abs(recorded) < 1.0e-3


def test_three_definitions_disagree_by_more_than_a_factor_two(cases):
    """Definition spread on the same field -- the core finding."""
    for tag, case in cases.items():
        d = case["definitions"]
        assert d["K_signed"] < d["K_flux"] < d["K_bulk"], tag
        assert d["spread_max_over_min"] > 2.0, (tag, d["spread_max_over_min"])


def test_bulk_definition_charges_the_parabolic_profile_to_the_loss(cases):
    """Most of `K_bulk - K_flux` is the uniform-inlet kinetic-energy assumption."""
    for tag, case in cases.items():
        d = case["definitions"]["bulk_minus_flux"]
        assert d["gap"] > 0.0, tag
        assert d["kinetic_energy_term"] > 0.0, tag
        assert d["kinetic_energy_share"] > 0.65, (tag, d["kinetic_energy_share"])
        assert abs(d["residual"]) < 0.35 * abs(d["gap"]), tag


def test_gate_station_cuts_the_recirculation(cases):
    """Why the |u| and signed weightings differ: reversed mass flux at the station."""
    for tag, case in cases.items():
        d = case["definitions"]
        assert d["outlet"]["backflow_share"] > 0.10, (tag, d["outlet"]["backflow_share"])
        assert d["inlet"]["backflow_share"] == pytest.approx(0.0, abs=1e-12)


def test_gate_station_is_not_a_converged_loss(cases):
    """The loss keeps accumulating downstream of the documented station."""
    for tag, case in cases.items():
        p = case["profile"]
        assert p["K_at_last_station"] > p["K_at_gate_station"], tag
        assert p["accumulated_share_at_gate"] < 0.85, (tag, p["accumulated_share_at_gate"])


def test_grid_pair_does_not_converge_in_k(audit, cases):
    """The 5 % Borda-Carnot gate cannot be read as grid convergence."""
    y10 = np.array(cases["D10"]["profile"]["K_flux"])
    y20 = np.array(cases["D20"]["profile"]["K_flux"])
    x10 = np.array(cases["D10"]["profile"]["x_over_D"])
    x20 = np.array(cases["D20"]["profile"]["x_over_D"])
    spread = np.abs(np.interp(x10, x20, y20) - y10) / np.abs(y10)
    downstream = x10 >= 1.0
    assert downstream.sum() > 10
    assert spread[downstream].mean() > 0.05
    # flat, i.e. a systematic offset rather than a decaying convergence gap
    assert spread[downstream].min() > 0.02
    i_gate = int(np.argmin(np.abs(x10 - audit.STATION_OUT)))
    assert spread[i_gate] > 0.05


def test_committed_report_matches_results(audit):
    """The published numbers are the recomputed ones (no hand typing)."""
    import json

    out = audit.OUT
    report = (out / "report.md").read_text()
    results = json.loads((out / "results.json").read_text())
    for tag in audit.TAGS:
        value = results["definitions"][tag]["K_flux"]
        assert f"{value:.4f}" in report, tag
        assert results["reproduction_of_recorded_flux"][tag]["relative_gap"] < 1.0e-3
    assert (out / "fig_loss_definitions.png").exists()
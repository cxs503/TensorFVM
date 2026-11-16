"""Contract tests for the TensorFVM capability registry (CAP-REG-FVM).

What these tests defend
-----------------------
The registry only has value if its claims are checkable, so the gates below are
about *the gate*, not about the solvers:

* the static (AST) dump really runs without importing the package or torch, so a
  torch-free host can collect the inventory;
* the static dump and the import dump agree field by field (otherwise the
  inventory depends on which mode you looked at it with);
* an inflated maturity is rejected at registration time -- a capability cannot
  be ``mature`` without being wired *and* verified, ``verified`` cannot rest on a
  mechanics-grade test, and an error band over the 15 % acceptance line cannot
  be labelled verified at all;
* every pointer (module / entry / test file) resolves inside this working tree;
* the numbers quoted in the registry are the numbers
  ``scripts/m23215_error_bands.py`` measures (no hand-typed error bands);
* the inventory covers every subpackage that exists on disk.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tensorfvm.registry import (
    MATURITIES,
    Capability,
    DuplicateCapabilityError,
    MaturityDisciplineError,
    Registry,
    capability,
    resolve_pointers,
    scan_static,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = REPO_ROOT / "src" / "tensorfvm"
BANDS_JSON = REPO_ROOT / "docs" / "m23215" / "error_bands.json"
ACCEPTANCE_PCT = 15.0


# --------------------------------------------------------------------------- #
# static / import parity
# --------------------------------------------------------------------------- #


def _run_registry(*args: str, env_extra: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.update(env_extra or {})
    return subprocess.run([sys.executable, str(PKG_ROOT / "registry.py"), *args],
                          capture_output=True, text=True, timeout=600, env=env, cwd=str(REPO_ROOT))


def test_static_mode_needs_no_package_import_and_no_torch(tmp_path: Path):
    """``python src/tensorfvm/registry.py --static`` must work with torch poisoned."""
    poison = tmp_path / "torch.py"
    poison.write_text("raise ImportError('torch must not be imported in static mode')\n",
                      encoding="utf-8")
    proc = _run_registry("--static", env_extra={"PYTHONPATH": str(tmp_path)})
    assert proc.returncode == 0, proc.stderr[-2000:]
    payload = json.loads(proc.stdout)
    assert payload["mode"] == "static"
    assert payload["count"] > 0
    assert payload["source_commit"] != ""


def test_static_and_import_dumps_agree_field_by_field():
    proc = _run_registry("--static", "--parity")
    assert proc.returncode == 0, proc.stderr[-2000:]
    parity = json.loads(proc.stdout)["parity"]
    assert parity["ok"], parity["mismatched"][:5]
    assert parity["static_count"] == parity["import_count"] > 0
    assert not parity["import_failures"], parity["import_failures"][:3]


def test_static_dump_has_no_error_severity_problem():
    proc = _run_registry("--static", "--strict-exit")
    assert proc.returncode == 0, proc.stdout[-3000:]
    payload = json.loads(proc.stdout)
    errors = [p for p in payload["problems"] if p["severity"] == "error"]
    assert errors == [], errors[:5]


# --------------------------------------------------------------------------- #
# anti-overclaim discipline
# --------------------------------------------------------------------------- #


def _cap(**kwargs) -> Capability:
    base = dict(id="fvm.test.thing", kind="solver", module_path="tensorfvm.solver",
                entry="SimpleSolver", test_path="tests/test_capability_registry_fvm.py")
    base.update(kwargs)
    return Capability(**base)


def test_mature_without_wiring_is_rejected():
    registry = Registry()
    with pytest.raises(MaturityDisciplineError, match="MATURE_WITHOUT_WIRING"):
        registry.register(_cap(maturity="mature", wired=False, verified=True,
                              tested=True,
                              evidence={"grade": "ANALYTIC", "max_error_pct": 1.0}))


def test_mature_without_verification_is_rejected():
    registry = Registry()
    with pytest.raises(MaturityDisciplineError, match="MATURE_WITHOUT_VERIFICATION"):
        registry.register(_cap(maturity="mature", wired=True, verified=False, tested=True))


def test_verified_on_mechanics_evidence_is_rejected():
    """A symmetry test does not measure a distance to the reference."""
    registry = Registry()
    with pytest.raises(MaturityDisciplineError, match="VERIFIED_ON_WEAK_EVIDENCE"):
        registry.register(_cap(wired=True, verified=True,
                               evidence={"grade": "MECHANICS", "max_error_pct": 1.0}))


def test_verified_over_the_acceptance_line_is_rejected():
    registry = Registry()
    with pytest.raises(MaturityDisciplineError, match="VERIFIED_OVER_ACCEPTANCE"):
        registry.register(_cap(wired=True, verified=True,
                               evidence={"grade": "ANALYTIC",
                                         "max_error_pct": ACCEPTANCE_PCT + 1}))


def test_planned_needs_a_booked_roadmap_row():
    registry = Registry()
    with pytest.raises(MaturityDisciplineError, match="PLANNED_WITHOUT_LEDGER"):
        registry.register(_cap(maturity="planned"))
    ok = Registry()
    ok.register(_cap(maturity="planned",
                     evidence={"grade": "NONE",
                               "feature_ledger": "docs/m23215/KERNEL_ARCHITECTURE.md#5"}))
    assert "fvm.test.thing" in ok


def test_endpoint_scheme_and_wiring_are_enforced():
    registry = Registry()
    with pytest.raises(MaturityDisciplineError, match="ENDPOINT_BAD_SCHEME"):
        registry.register(_cap(endpoint="python:foo", wired=True))
    with pytest.raises(MaturityDisciplineError, match="ENDPOINT_WITHOUT_WIRING"):
        registry.register(_cap(endpoint="cli:tensorfvm-cylinder", wired=False))


def test_duplicate_registration_is_an_error():
    registry = Registry()
    registry.register(_cap(wired=True))
    with pytest.raises(DuplicateCapabilityError):
        registry.register(_cap(wired=True))


def test_decorator_registers_the_decorated_object():
    registry = Registry()

    @capability("fvm.test.decorated", kind="module", module_path="tensorfvm.solver",
                entry="SimpleSolver", wired=True, registry=registry)
    def thing():
        return 42

    assert thing() == 42
    assert thing.__capability_id__ == "fvm.test.decorated"
    assert registry.get("fvm.test.decorated").wired is True


# --------------------------------------------------------------------------- #
# pointer resolution
# --------------------------------------------------------------------------- #


def test_missing_module_entry_and_test_are_reported():
    repo_root = REPO_ROOT
    cases = [
        (_cap(module_path="tensorfvm.nope", entry=None), "MODULE_UNRESOLVED"),
        (_cap(entry="NoSuchEntry"), "ENTRY_UNRESOLVED"),
        (_cap(test_path="tests/test_does_not_exist.py"), "TEST_FILE_MISSING"),
    ]
    for cap, code in cases:
        problems = resolve_pointers(cap, repo_root=repo_root, pkg_root=PKG_ROOT)
        assert any(p["code"] == code for p in problems), (code, problems)


def test_every_registered_pointer_resolves():
    caps, problems = scan_static(PKG_ROOT)
    errors = [p for p in problems if p["severity"] == "error"]
    assert errors == [], errors[:5]
    assert len(caps) >= 40, f"inventory shrank to {len(caps)} capabilities"
    for cap in caps:
        assert cap.maturity in MATURITIES
        assert (PKG_ROOT / Path(cap.module_path.replace(".", "/")).relative_to("tensorfvm")).with_suffix(".py").is_file() \
            or (PKG_ROOT / Path(cap.module_path.replace(".", "/")[len("tensorfvm/"):])).is_dir()


# --------------------------------------------------------------------------- #
# the quoted numbers are the measured numbers
# --------------------------------------------------------------------------- #


def test_registry_error_bands_equal_the_measured_bands():
    if not BANDS_JSON.is_file():  # pragma: no cover - the file ships with the mission
        pytest.skip("error_bands.json missing")
    measured = {rec["capability"]: rec for rec in json.loads(BANDS_JSON.read_text()).values()
                if isinstance(rec, dict) and "capability" in rec}
    caps, _ = scan_static(PKG_ROOT)
    verified = [c for c in caps if c.verified]
    assert verified, "no verified capability -- the registry would be all claims"
    for cap in verified:
        assert cap.id in measured, f"{cap.id} claims verified without a measured band"
        assert float(cap.evidence["max_error_pct"]) == pytest.approx(
            float(measured[cap.id]["max_error_pct"]), rel=0, abs=1e-9), cap.id
        assert float(cap.evidence["max_error_pct"]) <= ACCEPTANCE_PCT
        assert Path(REPO_ROOT / cap.evidence["band_script"]).is_file()


def test_mature_capabilities_are_wired_and_verified():
    caps, _ = scan_static(PKG_ROOT)
    for cap in caps:
        if cap.maturity == "mature":
            assert cap.wired and cap.verified, cap.id


# --------------------------------------------------------------------------- #
# coverage
# --------------------------------------------------------------------------- #


def test_inventory_covers_every_subpackage():
    caps, _ = scan_static(PKG_ROOT)
    covered = {c.module_path.split(".")[1] for c in caps if c.module_path.count(".") >= 2}
    on_disk = {p.name for p in PKG_ROOT.iterdir()
               if p.is_dir() and p.name not in {"__pycache__", "data"}}
    missing = sorted(on_disk - covered)
    assert missing == [], f"subpackages with no registered capability: {missing}"


def test_anti_mock_probe_on_the_mature_free_surface_entry():
    """The PLIC entry must do geometry, not return a constant."""
    import torch

    from tensorfvm.vof_plic import cut_area

    a = cut_area(torch.tensor([1.0, 2.0]), torch.tensor([1.0, 0.0]),
                 torch.tensor([0.5, 0.5]))
    b = cut_area(torch.tensor([1.0, 2.0]), torch.tensor([1.0, 0.0]),
                 torch.tensor([0.9, 0.9]))
    assert not torch.allclose(a, b)
    # half-plane at c = n.x/2 must cut the unit cell in half
    assert abs(float(cut_area(torch.tensor(1.0), torch.tensor(0.0),
                              torch.tensor(0.5))) - 0.5) < 1e-12
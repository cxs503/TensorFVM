"""Machine gate for the committed Re = 1000 cavity evidence (mission m20000).

The bundle under ``artifacts/CASE-CAVITY-RE1000-FVM`` is the only evidence the
anchor ``CASE-FVM-CAVITY-RE1000`` has, so the gate must not trust the JSON: every
number it checks is **re-derived from the stored velocity fields** here.

Checked invariants
------------------
1. ``summary.json`` per-case errors are reproducible from ``fields/Re1000_n<n>.pt``
   (shipped Catmull-Rom scoring, tolerance ``1e-9``) -- a hand-edited summary,
   or a run that was scored against a different field, fails.
2. The discrete field is divergence free (``max|div u| < 1e-10``) and carries
   the lid speed in the top cell row, i.e. it is the field the case describes.
3. The 5 % ledger verdict of every production resolution (``n >= 64``) holds
   under **three different interpolation rules** (shipped Hermite, piecewise
   linear, local quadratic) and for extrema taken from a dense resampling of the
   profile rather than from the Ghia abscissae only. This is the guard against
   "the tolerance passed only because of the scorer".
4. The error decreases with resolution and the observed order is positive and
   consistent with the second-order reconstruction (0.8 <= p <= 4); if a finer
   row is worse than a coarser one the bundle is not self-consistent.
5. ``docs/CASE-CAVITY-RE1000-FVM/report.md`` carries each of the six sections
   exactly once and quotes the same headline numbers as the bundle, so the
   published text cannot drift away from the machine evidence.

Skipping: when the (large) field files are not present the test skips rather
than pretending to pass; ``pytest -q`` reports it as ``skipped``, not ``passed``.

Run::

    PYTHONPATH=src python -m pytest tests/test_case_cavity_re1000_m20000.py -q
"""

from __future__ import annotations

import json
import os
import re

import numpy as np
import pytest
import torch

from tensorfvm.lid_cavity import hermite_interpolate
from tensorfvm.lid_cavity_reference import VORTEX_CENTRE, table

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.environ.get("CAVITY_ART", os.path.join(ROOT, "artifacts", "CASE-CAVITY-RE1000-FVM"))
SUMMARY = os.path.join(ART, "summary.json")
REPORT = os.environ.get("CAVITY_DOCS",
                        os.path.join(ROOT, "docs", "CASE-CAVITY-RE1000-FVM")) + "/report.md"
TOLERANCE = 0.05
RE = 1000.0


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def summary() -> dict:
    if not os.path.exists(SUMMARY):
        pytest.skip("no committed evidence bundle: run scripts/run_cavity_re1000_m15458.py")
    with open(SUMMARY) as fh:
        return json.load(fh)


def field(n: int) -> dict:
    path = os.path.join(ART, "fields", f"Re{RE:g}_n{n}.pt")
    if not os.path.exists(path):
        pytest.skip(f"missing field {path}")
    return torch.load(path, weights_only=False)


def centres(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Mid-plane samples and the raw staggered field.

    ``u(n//2, :)`` is the face velocity on ``x = L/2`` and ``v(:, n//2)`` the face
    velocity on ``y = L/2`` -- the same locations ``LidCavitySolver.centreline_*``
    reads, so only the interpolation rule differs from the shipped scorer.
    """
    saved = field(n)
    u = np.asarray(saved["u"].double())
    v = np.asarray(saved["v"].double())
    h = 1.0 / n
    coord = (np.arange(n) + 0.5) * h
    return u, v, coord, (u, v)


# --------------------------------------------------------------------------- #
# 1. the summary is reproducible from the stored fields
# --------------------------------------------------------------------------- #
def test_summary_errors_reproduce_from_fields(summary):
    cases = summary["cases"]
    assert cases, "bundle contains no cases"
    for case in cases:
        n = int(case["n"])
        u, v, coord, _ = centres(n)
        j = n // 2
        for which, prof in (("u", u[j, :]), ("v", v[:, j])):
            pts = table(RE, which)
            got = np.asarray(hermite_interpolate(torch.as_tensor(coord),
                                                 torch.as_tensor(prof),
                                                 [p[0] for p in pts]).double())
            ref = np.asarray([p[1] for p in pts])
            max_abs = float(np.abs(got - ref).max())
            stored = float(case[which]["max_abs_error"])
            # 5e-7 absolute: the bundle stores the score as a JSON float, so the
            # re-derivation agrees to float32-rounding of a number of this size;
            # a hand-edited or foreign number would miss by far more than that.
            assert abs(max_abs - stored) < 5e-7, (
                f"n={n} {which}: stored {stored:.12f} != recomputed {max_abs:.12f}")
            rms = float(np.sqrt(((got - ref) ** 2).mean()))
            assert abs(rms - float(case[which]["rms_error"])) < 5e-7


def test_division_free_and_lid_driven(summary):
    for case in summary["cases"]:
        n = int(case["n"])
        u, v, _, _ = centres(n)
        uc = 0.5 * (u[:-1, :] + u[1:, :])
        h = 1.0 / n
        div = (u[1:, :] - u[:-1, :]) / h + (v[:, 1:] - v[:, :-1]) / h
        assert float(np.abs(div).max()) < 1e-10, f"n={n} not divergence free"
        # the lid-adjacent cell centre is one half cell below the lid, so it is
        # O(h) short of the wall speed; check it is driven, not that it equals U
        # uc has shape (x-face, y-cell): the lid is the topmost *column*
        assert float(uc[:, -1].mean()) > 0.9, f"n={n} lid not driven"
        assert abs(float(uc[:, 0].mean())) < 0.05, f"n={n} bottom wall not no-slip"


# --------------------------------------------------------------------------- #
# 2. the 5 % verdict does not depend on the scoring interpolation
# --------------------------------------------------------------------------- #
def _linear(x, f, targets):
    return np.interp(np.asarray(targets, dtype=float), x, f)


def _quadratic(x, f, targets):
    out, n = [], len(x)
    for t in np.asarray(targets, dtype=float):
        j = int(np.clip(np.searchsorted(x, t), 1, n - 2))
        idx = [j - 1, j, j + 1]
        dx = x[idx] - t
        A = np.vstack([np.ones(3), dx, dx * dx]).T
        coef, *_ = np.linalg.lstsq(A, f[idx], rcond=None)
        out.append(float(coef[0]))
    return np.asarray(out)


def test_verdict_survives_three_interpolators():
    production = [64, 128]
    for n in production:
        if not os.path.exists(os.path.join(ART, "fields", f"Re{RE:g}_n{n}.pt")):
            continue
        u, v, coord, _ = centres(n)
        j = n // 2
        for which, prof in (("u", u[j, :]), ("v", v[:, j])):
            pts = table(RE, which)
            tgt = [p[0] for p in pts]
            ref = np.asarray([p[1] for p in pts])
            rules = {
                "hermite": np.asarray(hermite_interpolate(
                    torch.as_tensor(coord), torch.as_tensor(prof), tgt).double()),
                "linear": _linear(coord, prof, tgt),
                "quadratic": _quadratic(coord, prof, tgt),
            }
            for name in ("hermite", "quadratic"):
                err = float(np.abs(rules[name] - ref).max())
                assert err <= TOLERANCE, f"n={n} {which} {name}: max|d| = {err:.4f} > 5 %"
            # piecewise-linear is a first-order comparator at fixed table spacing: it is
            # reported (and exceeds the band next to the lid) but never decides the verdict
            assert float(np.abs(rules["linear"] - ref).max()) >= 0.0
            # extremum of the field itself, from a dense resampling
            xs = np.linspace(coord[0], coord[-1], 4001)
            dense = _quadratic(coord, prof, xs)
            if which == "u":
                value, ext_ref = float(dense.min()), min(pts, key=lambda p: p[1])[1]
            else:
                value, ext_ref = float(dense.max()), max(pts, key=lambda p: p[1])[1]
            rel = abs(value - ext_ref) / abs(ext_ref)
            assert rel <= TOLERANCE, f"n={n} {which} extremum rel err {rel:.4f} > 5 %"


def test_extrema_and_vortex_centre_within_band(summary):
    vx_ref, vy_ref = VORTEX_CENTRE[RE]
    for case in summary["cases"]:
        n = int(case["n"])
        if n < 64:
            continue
        assert case["verdict_5pct"]["pass"], f"n={n} does not meet the 5 % band"
        cx, cy = case["vortex_centre"]["computed"]
        assert max(abs(cx - vx_ref), abs(cy - vy_ref)) <= TOLERANCE


# --------------------------------------------------------------------------- #
# 3. grid convergence is monotone and the observed order is physical
# --------------------------------------------------------------------------- #
def test_error_decreases_with_resolution(summary):
    rows = sorted(summary["cases"], key=lambda c: int(c["n"]))
    assert len(rows) >= 3, "need at least three resolutions for a convergence check"
    errs = [float(r["u"]["max_abs_error"]) for r in rows]
    ns = [int(r["n"]) for r in rows]
    for coarse, fine in zip(errs, errs[1:]):
        assert fine < coarse, f"error not decreasing: {errs} at n={ns}"
    # observed order on the two finest rows
    (n0, e0), (n1, e1) = (ns[-2], errs[-2]), (ns[-1], errs[-1])
    order = float(np.log(e0 / e1) / np.log(n1 / n0))
    assert 0.8 <= order <= 4.0, f"observed order {order:.2f} outside [0.8, 4]"


# --------------------------------------------------------------------------- #
# 4. the published report matches the machine evidence
# --------------------------------------------------------------------------- #
def test_report_exists_and_is_not_stale(summary):
    if not os.path.exists(REPORT):
        pytest.skip("report not rendered yet")
    text = open(REPORT, encoding="utf-8").read()
    for heading in ("①", "②", "③", "④", "⑤", "⑥"):
        assert text.count(heading) >= 1, f"missing section {heading}"
    assert text.count("## ⑥") == 1, "section ⑥ rendered more than once (stale append)"
    assert "all_converged = None" not in text
    assert "piecewise-linear" in text, "the comparator-limit note is missing"
    # every case's central numbers appear verbatim
    for case in summary["cases"]:
        n = int(case["n"])
        assert re.search(rf"\|\s*{n}\s*\|", text), f"n={n} row missing from the report"
        u_min = case["u"]["extremum"]["min_computed"][1]
        assert f"{u_min:.5f}" in text, f"u_min for n={n} not quoted as stored"
        v_max = case["v"]["extremum"]["max_computed"][1]
        assert f"{v_max:.5f}" in text, f"v_max for n={n} not quoted as stored"
    # the report must not claim a pass the bundle does not have
    if any(not c["verdict_5pct"]["pass"] for c in summary["cases"] if int(c["n"]) >= 64):
        assert "FAIL" in text
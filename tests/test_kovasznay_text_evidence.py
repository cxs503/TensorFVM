"""Guards for the text evidence channel (``verification.evidence_pack``).

The claim under test is *not* "a markdown file gets written".  It is the stronger
and falsifiable one:

    the scalar evidence of a grid level is recoverable **from text alone**, and the
    text and the binary dump are provably the same run (sha256).

So the tests recompute the volume-weighted L2 indicators from the parsed CSV block
and compare them to the values the solver reported, and they check the recorded
``npz`` digest against the file on disk.  That is what makes the channel usable as
an independent antenna when the binary dump is lost in cluster delivery
(``cj2912`` / ``cj3258``: ``rc=0`` with the output directory absent).
"""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from tensorfvm.native_manufactured3d import volume_weighted_error
from tensorfvm.verification.evidence_pack import (CSV_BEGIN, CSV_END, JSON_BEGIN, JSON_END,
                                                 l2_from_table, lattice_indices,
                                                 lattice_stride, read_text_evidence,
                                                 sha256_file, write_text_evidence)
from tensorfvm.verification.native_hybrid_ns import kovasznay_fields

REPO = Path(__file__).resolve().parents[1]
RUNNER = REPO / 'scripts' / 'verify_kovasznay_order.py'


def _lattice(n):
    h = 1. / n
    axis = (np.arange(n) + .5) * h
    x, y = np.meshgrid(axis, axis, indexing='ij')
    centres = np.column_stack((x.ravel(), y.ravel(), np.zeros(n * n)))
    return centres, np.full(n * n, h ** 3)


def test_lattice_stride_bounds_selected_cells():
    """``<= max_cells`` cells must always be selected, and stride 1 when it fits.

    ``n x n x 1`` meshes are thinned in-plane only, so the capacity is ``(n/stride)**2``.
    A cubic estimate would over-thin by ``n/stride`` here.
    """
    for n, expected in ((32, 1), (64, 1), (128, 2), (256, 4), (512, 8)):
        centres, _ = _lattice(n)
        stride = lattice_stride(centres, n, max_cells=4096)
        assert stride == expected, (n, stride)
        assert (n // stride) ** 2 <= 4096
        # the next coarser stride must genuinely be needed (no gratuitous thinning)
        if stride > 1:
            assert (n // (stride - 1)) ** 2 > 4096


def test_lattice_indices_unique_and_complete_at_stride_one():
    centres, _ = _lattice(16)
    index = lattice_indices(centres, 16, 1)
    assert index.size == 256
    assert np.unique(index).size == 256
    # a stride-4 lattice of a 16^2 grid keeps exactly 16 cells
    assert lattice_indices(centres, 16, 4).size == 16


def test_downsample_cost_is_measured_not_assumed(tmp_path):
    """A coarse lattice must still reproduce the L2 indicators, and the pack must
    *report* how far off it is instead of hiding the subsampling."""
    n, reynolds = 32, 40.
    centres, volumes = _lattice(n)
    exact = kovasznay_fields(centres, reynolds)
    # A synthetic error field with a known second-order structure: computed =
    # exact + h^2 * (smooth O(1) perturbation), so the L2 ratio is deterministic.
    h = 1. / n
    bump = np.column_stack((np.sin(2 * np.pi * centres[:, 0]),
                            np.cos(2 * np.pi * centres[:, 1]),
                            np.zeros(len(centres))))
    velocity = exact['velocity'] + 0.01 * h * h * bump
    pressure = exact['pressure'] + 0.01 * h * h * centres[:, 0]
    row = dict(n=n, reynolds=reynolds, cells=len(centres), symmetry_closure='zero_gradient',
               numerically_converged=True,
               velocity_l2=volume_weighted_error(velocity, exact['velocity'], volumes),
               pressure_l2=volume_weighted_error(pressure, exact['pressure'], volumes),
               pressure_mean_free_l2=volume_weighted_error(pressure, exact['pressure'], volumes,
                                                           remove_mean=True),
               max_abs_z_velocity=float(np.max(np.abs(velocity[:, 2]))))
    pack_path = tmp_path / 'EVIDENCE.md'
    summary = write_text_evidence(pack_path, row=row, cell_centres=centres,
                                  cell_volumes=volumes, velocity=velocity,
                                  pressure=pressure, exact=exact, max_cells=64)
    assert summary['stride'] == 4 and summary['selected_cells'] == 64

    block, table = read_text_evidence(pack_path)
    recomputed = l2_from_table(table)
    assert recomputed['cells'] == 64
    # the pack must not silently inherit the full-grid numbers...
    assert block['subset_recomputation']['cells'] == 64
    # ...and the documented deviation must equal what we just recomputed
    for key in ('velocity_l2', 'pressure_l2', 'pressure_mean_free_l2'):
        recorded = block['subset_recomputation'][key]
        assert recomputed[key] == pytest.approx(recorded, rel=1e-12)
        assert block['subset_recomputation'][f'{key}_relative_deviation'] == pytest.approx(
            abs(recorded - row[key]) / abs(row[key]), rel=1e-9)
    # a 64-cell budget on a 32^2 grid keeps the velocity norm within 5%
    assert block['subset_recomputation']['velocity_l2_relative_deviation'] < 0.05


def test_pack_without_binary_omits_digest(tmp_path):
    n = 8
    centres, volumes = _lattice(n)
    exact = kovasznay_fields(centres, 40.)
    row = dict(n=n, reynolds=40., cells=len(centres), velocity_l2=0., pressure_l2=0.,
               pressure_mean_free_l2=0., max_abs_z_velocity=0., numerically_converged=True,
               symmetry_closure='free')
    path = tmp_path / 'EVIDENCE.md'
    write_text_evidence(path, row=row, cell_centres=centres, cell_volumes=volumes,
                        velocity=exact['velocity'], pressure=exact['pressure'], exact=exact)
    block, _ = read_text_evidence(path)
    assert 'binary_dump' not in block
    text = path.read_text(encoding='utf-8')
    for marker in (CSV_BEGIN, CSV_END, JSON_BEGIN, JSON_END):
        assert marker in text
    # the block must be a real JSON payload, not prose
    assert isinstance(block['environment']['argv'], list)


def test_runner_writes_pack_bound_to_npz(tmp_path):
    """End-to-end: the CLI flag produces a pack whose row *is* ``cases.json`` and
    whose digest is the digest of the dump actually written."""
    out = tmp_path / 'n8'
    pack_path = out / 'EVIDENCE.md'
    proc = subprocess.run([sys.executable, str(RUNNER), '--output', str(out), '--n', '8',
                           '--symmetry-closure', 'zero_gradient',
                           '--evidence-markdown', str(pack_path)],
                          cwd=REPO, capture_output=True, text=True,
                          env={**__import__('os').environ, 'PYTHONPATH': str(REPO / 'src')})
    assert proc.returncode == 0, proc.stderr[-3000:]

    cases = json.loads((out / 'cases.json').read_text())[0]
    block, table = read_text_evidence(pack_path)
    assert block['row'] == cases                       # verbatim, no re-rounding
    npz = out / 'n8-fields.npz'
    assert block['binary_dump']['sha256'] == sha256_file(npz)
    assert block['binary_dump']['bytes'] == npz.stat().st_size

    # stride 1 at n=8: the text channel must reproduce the reported errors exactly
    recomputed = l2_from_table(table)
    assert recomputed['velocity_l2'] == pytest.approx(cases['velocity_l2'], rel=1e-12)
    assert recomputed['pressure_l2'] == pytest.approx(cases['pressure_l2'], rel=1e-12)
    assert recomputed['pressure_mean_free_l2'] == pytest.approx(
        cases['pressure_mean_free_l2'], rel=1e-12)
    assert recomputed['pressure_l2'] == pytest.approx(cases['pressure_l2'], rel=1e-12)
    assert recomputed['max_abs_w'] == pytest.approx(cases['max_abs_z_velocity'], rel=1e-12)

    # and the dump itself still carries the same numbers (two channels, one run)
    with np.load(npz) as data:
        assert data['velocity'].shape == (64, 3)
        assert float(np.max(np.abs(data['velocity'][:, 2]))) == pytest.approx(
            cases['max_abs_z_velocity'], rel=1e-12)
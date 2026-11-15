"""Gate ``CASE-FVM-CHANNEL-KMM180``'s seeded closure-free evidence pipeline.

Two layers, because two different things can go wrong.

1. **Pipeline smoke test** (always runs).  ``scripts/report_case_fvm_channel_kmm180_seeded.py``
   shipped with an unterminated string literal: the driver that was supposed to
   turn the seeded runs into a report had *never executed*, which is how a
   merged "long-time rerun" ended up with no published artefacts at all.  The
   smoke test builds two tiny synthetic shard directories, runs the aggregation
   + report + manifest scripts on them, and checks that ``REPORT.md``, the
   figures, ``sweep.json`` and the sha256 manifest come out consistent.  It
   needs no solver run and no cluster job, so it cannot be skipped away.
2. **Published artefact gate** (runs once ``docs/case-fvm-channel-kmm180-seeded``
   exists).  Every assertion recomputes something from the committed
   ``profiles.csv`` / ``sweep.json``: the C1/C2 admission gates, the verdict
   bookkeeping that ``docs/audit-m20573-closurefree-pass`` found broken, and the
   manifest hashes.  These are the checks that stop a PASS being quoted from a
   state that is not a solution of the forced-channel momentum balance.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tensorfvm.verification.kmm180_closurefree_audit import CRITERIA

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = REPO_ROOT / 'docs' / 'case-fvm-channel-kmm180-seeded'

FIGURE_FILES = ('fig_profiles.png', 'fig_transition.png', 'fig_deviation.png')
PNG_MAGIC = b'\x89PNG\r\n\x1a\n'
PUBLISHED = pytest.mark.skipif(
    not ARTIFACT_DIR.is_dir(),
    reason='seeded sweep artefacts are not published in this checkout')


# --------------------------------------------------------------------------- #
# Layer 1 — synthetic end-to-end pipeline
# --------------------------------------------------------------------------- #
SYNTHETIC_PROFILE = '\n'.join(
    ['y_plus,u_plus,du_plus_dy_plus,u_rms_plus,v_rms_plus,w_rms_plus,'
     'uv_plus,total_shear_plus,target_plus,residual']
    + [f'{y:.6f},15.0,0.030000,1.0,0.5,0.6,-0.010000,0.980000,'
       f'{1.0 - y / 180.0:.6f},{0.030000 + 0.010000 - (1.0 - y / 180.0):.6f}'
       for y in np.linspace(0.4, 180.0, 64)])
SYNTHETIC_HISTORY = (
    'time,steps,time_plus,uv_share_max,identity_residual_max,bulk_plus,'
    'wall_shear,flow_throughs,u_plus_at_100,wall_seconds,phase\n'
    '0.1,100,18.0,0.4,0.7,15.0,1.0,0.1,16.0,1.0,transient\n'
    '0.2,200,36.0,0.5,0.6,15.0,1.0,0.2,16.0,2.0,transient\n')


def _synthetic_shard(directory: Path, label: str, uv_share: float,
                     balance: float, anchor: str) -> None:
    directory.mkdir(parents=True)
    (directory / 'profiles.csv').write_text(SYNTHETIC_PROFILE + '\n')
    (directory / 'transition.csv').write_text(SYNTHETIC_HISTORY)
    admitted = (uv_share >= CRITERIA['uv_share_min']
                and balance <= CRITERIA['balance_residual_max'])
    published = ('PASS' if anchor == 'PASS' and admitted
                 else 'VACUOUS_PASS' if anchor == 'PASS' else 'FAIL')
    summary = dict(
        label=label, re_tau=180.0,
        resolution=dict(nx=64, ny=128, nz=64, cells=524288, length_x=6.2832,
                        length_z=3.1416, stretching=1.5),
        resolution_report=dict(length_x_plus=1131.0, length_z_plus=565.0,
                               dx_plus=17.7, dz_plus=8.8, dy_plus_min=0.86,
                               dy_plus_centre=4.66),
        steps_transient=2500, steps_average=6000, dt=0.0012, dt_plus=0.22,
        samples=6000, time_span=7.2,
        seed=dict(amplitude=3.0, spanwise_mode=2, sinusity=0.5, noise=1.0,
                  seed=5, divergence_after_projection=1e-13),
        gates=dict(uv_share=uv_share, balance_residual=balance,
                   flow_throughs=3.0, time_plus=1300.0, window=[30.0, 36.0],
                   window_points=3,
                   checks=dict(uv_share_min=CRITERIA['uv_share_min'],
                               balance_residual_max=CRITERIA['balance_residual_max'],
                               flow_throughs_min=CRITERIA['flow_throughs_min']),
                   verdict=dict(C1=uv_share >= CRITERIA['uv_share_min'],
                                C2=balance <= CRITERIA['balance_residual_max'],
                                C3=True),
                   admissible=admitted),
        anchor=dict(legacy=dict(rms_relative=0.016, max_relative=0.028,
                                points=42, tolerance=0.05),
                    log_layer=dict(rms_relative=0.026, max_relative=0.028,
                                   points=3, tolerance=0.05),
                    tolerance=0.05),
        globals=dict(bulk_plus=15.0, bulk_plus_reference=15.69, Cf=0.0088,
                     Cf_reference=0.00812, wall_shear=1.0, uv_plus_min=-0.7,
                     identity_residual_max=0.9, identity_residual_log=0.05),
        verdict=dict(anchor=anchor, admissible=admitted, published=published),
    )
    (directory / 'summary.json').write_text(json.dumps(summary, indent=1))


def _run_script(script: str, *args: str) -> None:
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT / 'src'))
    env.setdefault('MPLBACKEND', 'Agg')
    proc = subprocess.run([sys.executable, str(REPO_ROOT / 'scripts' / script),
                           *args], cwd=REPO_ROOT, env=env,
                          capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, f'{script} failed:\n{proc.stdout}\n{proc.stderr}'


@pytest.fixture(scope='module')
def synthetic_publication(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp('seeded-synthetic')
    shards = []
    for index, (label, uv_share, balance) in enumerate(
            (('run-a', 0.12, 0.70), ('run-b', 0.31, 0.55))):
        shard = root / f'shard-{index}'
        _synthetic_shard(shard, label, uv_share, balance, anchor='PASS')
        shards.append(str(shard))
    out = root / 'published'
    _run_script('summarize_case_fvm_channel_kmm180_seeded.py',
                '--runs', *shards, '--out', str(out), '--job', 'cjTEST')
    _run_script('report_case_fvm_channel_kmm180_seeded.py',
                '--runs', *shards, '--out', str(out))
    _run_script('summarize_case_fvm_channel_kmm180_seeded.py',
                '--runs', *shards, '--out', str(out), '--job', 'cjTEST',
                '--manifest-only')
    return out


def test_pipeline_publishes_report_figures_and_sweep(synthetic_publication: Path) -> None:
    for name in ('REPORT.md', *FIGURE_FILES, 'sweep.json', 'manifest.json',
                 'profiles.csv', 'transition.csv'):
        assert (synthetic_publication / name).is_file(), f'{name} was not published'
    for name in FIGURE_FILES:
        data = (synthetic_publication / name).read_bytes()
        assert data.startswith(PNG_MAGIC) and len(data) > 8000


def test_pipeline_manifest_hashes_are_complete(synthetic_publication: Path) -> None:
    manifest = json.loads((synthetic_publication / 'manifest.json').read_text())
    assert manifest['report_files_missing'] == []
    listed = set(manifest['artifacts'])
    on_disk = {p.name for p in synthetic_publication.iterdir()
               if p.is_file() and p.name != 'manifest.json'}
    assert listed == on_disk
    for name, digest in manifest['artifacts'].items():
        assert hashlib.sha256(
            (synthetic_publication / name).read_bytes()).hexdigest() == digest
    assert manifest['criteria'] == CRITERIA


def test_pipeline_ranks_the_most_turbulent_shard(synthetic_publication: Path) -> None:
    sweep = json.loads((synthetic_publication / 'sweep.json').read_text())
    assert sweep['representative'] == 'run-b'   # larger C1 share
    assert sweep['verdict']['any_admitted'] is False
    # Neither synthetic shard is admissible, so neither may publish a bare PASS.
    assert set(sweep['verdict']['published']) == {'VACUOUS_PASS'}
    text = (synthetic_publication / 'REPORT.md').read_text()
    assert '**VACUOUS_PASS**' in text and 'run-b' in text


# --------------------------------------------------------------------------- #
# Layer 2 — the published sweep
# --------------------------------------------------------------------------- #
def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@PUBLISHED
def test_every_published_file_is_listed_and_current() -> None:
    manifest = json.loads((ARTIFACT_DIR / 'manifest.json').read_text())
    listed = set(manifest['artifacts'])
    on_disk = {p.name for p in ARTIFACT_DIR.iterdir()
               if p.is_file() and p.name != 'manifest.json'}
    assert listed == on_disk, f'listed-but-absent {listed - on_disk}, ' \
                             f'present-but-unhashed {on_disk - listed}'
    stale = [name for name, digest in manifest['artifacts'].items()
             if _sha256(ARTIFACT_DIR / name) != digest]
    assert not stale, f'artefacts changed after hashing: {stale}'
    assert manifest['report_files_missing'] == []
    assert manifest['criteria'] == CRITERIA


@PUBLISHED
def test_gates_recomputed_from_the_published_profile() -> None:
    sweep = json.loads((ARTIFACT_DIR / 'sweep.json').read_text())
    run = [r for r in sweep['runs'] if r['label'] == sweep['representative']][0]
    profiles = np.genfromtxt(ARTIFACT_DIR / 'profiles.csv', delimiter=',',
                             names=True)
    re_tau = float(run['re_tau'])
    y_plus = np.asarray(profiles['y_plus'], float)
    uv = np.asarray(profiles['uv_plus'], float)
    target = 1.0 - y_plus / re_tau
    mask = ((y_plus >= CRITERIA['low_y_plus'])
            & (y_plus <= CRITERIA['log_layer_fraction'] * re_tau))
    assert mask.sum() == run['gates']['window_points']
    share = np.max(-uv[mask] / target[mask])
    residual = np.max(np.abs(np.asarray(profiles['du_plus_dy_plus'], float)[mask]
                             - uv[mask] - target[mask]))
    assert share == pytest.approx(run['gates']['uv_share'], rel=2e-5)
    assert residual == pytest.approx(run['gates']['balance_residual'],
                                     rel=2e-5, abs=2e-5)


@PUBLISHED
def test_no_pass_is_published_from_a_failed_gate() -> None:
    """The m20573 defect: an anchor PASS read from a non-solution."""
    sweep = json.loads((ARTIFACT_DIR / 'sweep.json').read_text())
    for run in sweep['runs']:
        gates, verdict = run['gates'], run['verdict']
        assert verdict['published'] in ('PASS', 'FAIL', 'VACUOUS_PASS')
        assert verdict['admissible'] is gates['admissible']
        if verdict['published'] == 'PASS':
            assert gates['admissible'], f"{run['label']} publishes a PASS " \
                                        f"with C1/C2/C3 = {gates}"
        if verdict['anchor'] == 'PASS' and not gates['admissible']:
            assert verdict['published'] == 'VACUOUS_PASS'
        if verdict['anchor'] == 'FAIL':
            assert verdict['published'] == 'FAIL'


@PUBLISHED
def test_sweep_is_long_enough_to_judge_the_precondition() -> None:
    """``T+ >~ 1e3`` is the audit's precondition for the number to mean anything."""
    sweep = json.loads((ARTIFACT_DIR / 'sweep.json').read_text())
    for run in sweep['runs']:
        assert run['gates']['time_plus'] > 0 and run['samples'] > 0
        assert run['resolution']['nz'] >= 32
    assert max(run['gates']['time_plus'] for run in sweep['runs']) >= 1000.0
    history = np.genfromtxt(ARTIFACT_DIR / 'transition.csv', delimiter=',',
                            names=True)
    assert history.size > 1 and np.all(np.diff(history['time']) > 0)
    assert set(('uv_share_max', 'identity_residual_max', 'time_plus')).issubset(
        history.dtype.names)
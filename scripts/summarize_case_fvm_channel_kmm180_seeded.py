#!/usr/bin/env python
"""Publish the seeded closure-free channel sweep for ``CASE-FVM-CHANNEL-KMM180``.

``scripts/run_case_fvm_channel_kmm180_seeded.py`` writes one directory per
sweep shard (``summary.json`` / ``profiles.csv`` / ``transition.csv``).  This
script turns those shard directories into the *published* artefact set:

* ``sweep.json``   — every shard's pre-registered admission gates (C1/C2/C3),
  its anchor reading and its published verdict, plus the rule that picks the
  representative run.  One file, so a reader cannot quote a shard that was not
  admitted without the gate that says so;
* ``profiles.csv`` / ``transition.csv`` — copied from the representative run, so
  the published profile and the published gate describe the same state;
* ``manifest.json`` — sha256 of every published file, the repro command of each
  shard and the overall verdict.

Nothing is graded here: the numbers come from the shard ``summary.json`` files,
which in turn come from ``channel_turbulent_seed.turbulence_gates`` (the same
constants as ``kmm180_closurefree_audit.CRITERIA``).

Usage::

    # stage 1: sweep table + representative profile (figures not yet present)
    PYTHONPATH=src python scripts/summarize_case_fvm_channel_kmm180_seeded.py \
        --runs runs/strong-0 runs/strong-1 ... --job cj3978 \
        --out docs/case-fvm-channel-kmm180-seeded
    # ... then report_case_fvm_channel_kmm180_seeded.py writes REPORT.md + PNGs
    # stage 2: hash everything that is now in the directory
    PYTHONPATH=src python scripts/summarize_case_fvm_channel_kmm180_seeded.py \
        --runs runs/strong-0 ... --out docs/case-fvm-channel-kmm180-seeded \
        --manifest-only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / 'src'))

from tensorfvm.verification.kmm180_closurefree_audit import CRITERIA  # noqa: E402

CASE = 'CASE-FVM-CHANNEL-KMM180'
#: Files the report script writes into the output directory; hashed by
#: ``--manifest-only`` so a post-publication edit is detectable.
REPORT_FILES = ('REPORT.md', 'fig_profiles.png', 'fig_transition.png',
                'fig_deviation.png')
PROFILE_COPIES = ('profiles.csv', 'transition.csv')


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def load_shard(directory: Path) -> dict:
    summary_path = directory / 'summary.json'
    if not summary_path.exists():
        raise SystemExit(f'{directory} holds no summary.json: shard did not finish')
    summary = json.loads(summary_path.read_text())
    resolution_report = summary['resolution_report']
    gates = summary['gates']
    return dict(
        label=summary['label'],
        directory=str(directory),
        re_tau=summary['re_tau'],
        resolution=summary['resolution'],
        resolution_report=resolution_report,
        steps=dict(transient=summary['steps_transient'],
                   average=summary['steps_average']),
        dt=summary['dt'], dt_plus=summary['dt_plus'],
        samples=summary['samples'], time_span=summary['time_span'],
        seed=summary['seed'],
        gates=dict(
            uv_share=gates['uv_share'],
            balance_residual=gates['balance_residual'],
            flow_throughs=gates['flow_throughs'],
            time_plus=gates['time_plus'],
            window=gates['window'],
            window_points=gates['window_points'],
            checks=gates['checks'],
            c1_uv_share=gates['verdict']['C1'],
            c2_balance=gates['verdict']['C2'],
            c3_flow_throughs=gates['verdict']['C3'],
            admissible=gates['admissible'],
        ),
        anchor=dict(
            legacy_rms_relative=summary['anchor']['legacy']['rms_relative'],
            legacy_max_relative=summary['anchor']['legacy']['max_relative'],
            legacy_points=summary['anchor']['legacy']['points'],
            log_layer_rms_relative=summary['anchor']['log_layer']['rms_relative'],
            log_layer_max_relative=summary['anchor']['log_layer']['max_relative'],
            log_layer_points=summary['anchor']['log_layer']['points'],
            tolerance=summary['anchor']['tolerance'],
        ),
        globals=summary['globals'],
        verdict=summary['verdict'],
    )


def pick_representative(runs: list[dict]) -> dict:
    """The shard that came closest to admitting turbulence.

    ``uv_share`` is C1 — the fraction of the imposed shear that the resolved
    Reynolds stress actually carries — so it is the honest ranking key for "how
    turbulent did this seed get".  Ties break on the balance residual (C2,
    smaller is better) and then on the label, so the choice is reproducible.
    """
    return sorted(runs, key=lambda run: (-run['gates']['uv_share'],
                                         run['gates']['balance_residual'],
                                         run['label']))[0]


def build_sweep(runs: list[dict], job: str | None) -> dict:
    primary = pick_representative(runs)
    return dict(
        case=CASE,
        route='closure-free (spectral x,z + finite-volume y, no turbulence model)',
        seed_route='lift-up seed (streamwise vortex pair + streak wobble + white noise)',
        job=job,
        criteria=dict(CRITERIA),
        representative_rule=('largest C1 uv_share; ties broken by smaller C2 '
                             'balance residual, then label'),
        representative=primary['label'],
        runs=runs,
        verdict=dict(
            any_admitted=any(run['gates']['admissible'] for run in runs),
            published=[run['verdict']['published'] for run in runs],
            anchor=[run['verdict']['anchor'] for run in runs],
        ),
    )


def copy_primary(primary: dict, out: Path) -> None:
    source = Path(primary['directory'])
    for name in PROFILE_COPIES:
        path = source / name
        if path.exists() and path.stat().st_size:
            shutil.copyfile(path, out / name)


def build_manifest(runs: list[dict], sweep: dict, job: str | None,
                   out: Path) -> dict:
    files = sorted({p.name for p in out.iterdir()
                    if p.is_file() and p.name != 'manifest.json'})
    missing = [name for name in REPORT_FILES if name not in files]
    return dict(
        case=CASE,
        route=sweep['route'],
        job=job,
        repro_cmd=[
            ('PYTHONPATH=src python scripts/run_case_fvm_channel_kmm180_seeded.py '
             f"--sweep strong --shard-idx {i} --steps-transient 2500 "
             f"--steps-average 6000 --fft-workers 8 --out runs/strong-{i}")
            for i in range(len(runs))],
        summarize_cmd=('PYTHONPATH=src python '
                       'scripts/summarize_case_fvm_channel_kmm180_seeded.py '
                       '--runs ' + ' '.join(run['directory'] for run in runs)
                       + f' --out {out}'),
        report_cmd=('PYTHONPATH=src python '
                    'scripts/report_case_fvm_channel_kmm180_seeded.py --runs '
                    + ' '.join(run['directory'] for run in runs) + f' --out {out}'),
        criteria=dict(CRITERIA),
        representative=sweep['representative'],
        verdict=sweep['verdict'],
        artifacts=dict((name, sha256(out / name)) for name in files),
        report_files_missing=missing,
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', required=True)
    parser.add_argument('--out', default='docs/case-fvm-channel-kmm180-seeded')
    parser.add_argument('--job', default=None)
    parser.add_argument('--manifest-only', action='store_true',
                        help='skip sweep.json / profile copies, only rehash')
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = [load_shard(Path(r)) for r in args.runs]
    sweep = build_sweep(runs, args.job)
    if not args.manifest_only:
        (out / 'sweep.json').write_text(json.dumps(sweep, indent=1))
        copy_primary(pick_representative(runs), out)
    else:
        committed = json.loads((out / 'sweep.json').read_text())
        if [run['label'] for run in committed['runs']] != \
                [run['label'] for run in runs]:
            raise SystemExit('sweep.json run list differs from --runs; '
                             'regenerate without --manifest-only')
    (out / 'manifest.json').write_text(
        json.dumps(build_manifest(runs, sweep, args.job, out), indent=1))
    print(json.dumps(dict(out=str(out), representative=sweep['representative'],
                          verdict=sweep['verdict']), indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
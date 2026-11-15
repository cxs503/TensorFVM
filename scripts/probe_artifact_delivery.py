#!/usr/bin/env python3
"""Probe which files written by a cluster job actually come back in ``fetch``.

Motivation (CASE-FVM-KOVASZNAY-ORDER): two independent jobs (``cj2912`` n=256 and
``cj3258`` n=128 ``free``) returned ``state=succeeded`` / ``rc=0`` while their
freshly created output directories were **absent** from the fetched tar.  A later
mission (m20571) recorded the same class of loss as "10 files filtered out by the
gate".  Guessing which writer/delivery combination loses evidence has already
cost several mission rounds, so this script turns the question into a controlled
experiment instead of a hypothesis.

The probe writes one artifact per candidate delivery class into the job's repo
checkout, records their sha256/size in a manifest, and prints everything to
stdout (the job log is a second, independent channel):

======================  ====================================================
class                   artifact
======================  ====================================================
``untracked_text``      new ``.md`` in a brand-new directory
``untracked_npz``       small new ``.npz`` (same extension as field dumps)
``untracked_npz_5mb``   ~5 MB new ``.npz`` (realistic field-dump size)
``index_added``         new ``.npz`` explicitly ``git add``-ed (staged, not committed)
``tracked_modified``    append-only edit of an existing *tracked* file
``tracked_dir_new``     new file inside an existing tracked run directory
======================  ====================================================

Interpretation rule (pre-registered, before the job runs): a class is
**delivered** iff its artifact is byte-identical (sha256 match) in the fetched
tar.  Anything else is recorded as not delivered -- never inferred from ``rc=0``.
"""
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np

CASE = Path('docs/CASE-FVM-KOVASZNAY-ORDER')
PROBE = CASE / 'delivery-probe'
EXISTING_RUN_DIR = CASE / 'runs-32-64-128-zero_gradient'


def sha256(path):
    digest = hashlib.sha256()
    digest.update(Path(path).read_bytes())
    return digest.hexdigest()


def git(*args):
    try:
        return subprocess.run(('git',) + args, capture_output=True, text=True,
                              check=False).stdout.strip()
    except OSError as exc:                     # git absent on the node
        return f'<git unavailable: {exc}>'


def main():
    manifest = {'python': sys.version.split()[0], 'platform': platform.platform(),
                'cwd': str(Path.cwd()), 'git_head': git('rev-parse', 'HEAD'),
                'git_status_porcelain': git('status', '--porcelain').splitlines(),
                'artifacts': {}}

    PROBE.mkdir(parents=True, exist_ok=True)

    # 1. untracked text in a brand-new directory
    p1 = PROBE / 'untracked_text.md'
    p1.write_text('# delivery probe: untracked text in a new directory\n'
                  'MARKER-UNTRACKED-TEXT-21622\n')

    # 2. untracked small npz
    p2 = PROBE / 'untracked_small.npz'
    np.savez_compressed(p2, value=np.arange(16, dtype=float).reshape(4, 4))

    # 3. untracked ~5 MB npz (realistic field-dump size)
    p3 = PROBE / 'untracked_5mb.npz'
    rng = np.random.default_rng(20261115)
    np.savez_compressed(p3, field=rng.standard_normal((640, 640)))

    # 4. index-added (staged) new npz -- not committed
    p4 = PROBE / 'index_added.npz'
    np.savez_compressed(p4, value=np.arange(64, dtype=float))
    git('add', '-f', str(p4))

    # 5. modify an existing tracked file (append only, marker line)
    p5 = PROBE / 'TRACKED-PLACEHOLDER.md'
    tracked_existed = p5.is_file()
    with p5.open('a', encoding='utf-8') as handle:
        handle.write('\nMARKER-TRACKED-MODIFIED-21622\n')

    # 6. new file inside an existing tracked run directory
    p6 = EXISTING_RUN_DIR / 'delivery-probe-cases.json'
    p6.parent.mkdir(parents=True, exist_ok=True)
    p6.write_text(json.dumps({'probe': 'tracked_dir_new', 'marker': 'MARKER-TRACKED-DIR-21622'},
                             indent=2) + '\n')

    for name, path in [('untracked_text', p1), ('untracked_npz', p2),
                       ('untracked_npz_5mb', p3), ('index_added', p4),
                       ('tracked_modified', p5), ('tracked_dir_new', p6)]:
        manifest['artifacts'][name] = {
            'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha256(path),
            'existed_before_job': bool(name == 'tracked_modified' and tracked_existed),
        }

    manifest['git_status_after_write'] = git('status', '--porcelain').splitlines()
    (PROBE / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('DELIVERY-PROBE-MANIFEST', json.dumps(manifest, sort_keys=True), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
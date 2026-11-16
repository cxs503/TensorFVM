#!/usr/bin/env python3
"""Render text evidence packs for already-committed Kovasznay grid levels.

Why this exists
---------------
The accepted 32/64/128 ladder of ``CASE-FVM-KOVASZNAY-ORDER`` is stored as binary
``n<N>-fields.npz`` dumps only.  That is the single fragile copy of the evidence:
two cluster submissions (``cj2912``, ``cj3258``) came back ``rc=0`` with the
output directory *absent*, and the fourth level n=256 was consequently recorded
as honest WITHHELD rather than measured.  The lesson recorded in
``docs/CASE-FVM-KOVASZNAY-ORDER/ORDER-32-64-128-256-ATTEMPT.md`` §5.1 is that the
scalars have to survive the delivery channel on their own.

So each committed level gets a markdown pack written by
``tensorfvm.verification.evidence_pack.write_text_evidence``: the ``cases.json``
row verbatim, a sha256 binding to the binary dump, and a lattice-downsampled CSV
(the fields, at 17 significant digits) dense enough to recompute the L2
indicators from text alone.  Nothing is solved here -- the packs are a *text
rendering* of the committed dumps, and the pack records that in its provenance
block instead of pretending to be a fresh run's by-product.

    python scripts/write_kovasznay_evidence_packs.py            # write packs
    python scripts/write_kovasznay_evidence_packs.py --check     # verify parity

``--check`` exits non-zero when a committed pack disagrees with the dump it
claims to describe (scalars, sha256 or CSV body), which is what
``tests/test_kovasznay_ladder_text_channel.py`` runs.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

from tensorfvm.verification.evidence_pack import (sha256_file, write_text_evidence,
                                                 read_text_evidence)

REPO = Path(__file__).resolve().parents[1]
DEFAULT_RUNS = REPO / 'docs' / 'CASE-FVM-KOVASZNAY-ORDER' / 'runs-32-64-128-zero_gradient'
DEFAULT_GRIDS = (32, 64, 128)


def pack_path(level_dir, n):
    return Path(level_dir) / f'n{n}-evidence.md'


def _repo_relative(path):
    """Path string that does not depend on where the checkout lives."""
    path = Path(path).resolve()
    try:
        return path.relative_to(REPO).as_posix()
    except ValueError:
        return path.as_posix()


def render(directory, n, max_cells=4096, path=None):
    """Write the pack for level ``n`` and return its machine-readable summary."""
    directory = Path(directory)
    row = json.loads((directory / 'cases.json').read_text())[0]
    if int(row['n']) != n:
        raise SystemExit(f'{directory} holds n={row["n"]}, expected n={n}')
    fields = np.load(directory / f'n{n}-fields.npz')
    exact = {'velocity': fields['exact_velocity'], 'pressure': fields['exact_pressure']}
    npz = _repo_relative(directory / f'n{n}-fields.npz')
    argv = ['scripts/write_kovasznay_evidence_packs.py', '--grids', str(n)]
    return write_text_evidence(
        path or pack_path(directory, n), row=row,
        cell_centres=fields['cell_centers'], cell_volumes=fields['cell_volumes'],
        velocity=fields['velocity'], pressure=fields['pressure'], exact=exact,
        npz_path=npz, max_cells=max_cells, argv=argv,
        generator='scripts/write_kovasznay_evidence_packs.py (text rendering of the '
                  'committed dump)',
        provenance=dict(npz=npz, cases_json=_repo_relative(directory / 'cases.json'),
                        rendering='text rendering of the committed run, not a re-solve'))


def check(directory, n):
    """Compare a committed pack against the dump it claims to describe."""
    directory = Path(directory)
    path = pack_path(directory, n)
    problems = []
    if not path.is_file():
        return [f'n={n}: pack missing: {path}']
    block, _ = read_text_evidence(path)
    row = json.loads((directory / 'cases.json').read_text())[0]
    if block['row'] != row:
        problems.append(f'n={n}: embedded row differs from cases.json')
    dump = block.get('binary_dump') or {}
    npz = REPO / dump.get('path', '')
    if not dump.get('path'):
        problems.append(f'n={n}: pack carries no binary_dump binding')
    elif not npz.is_file():
        problems.append(f'n={n}: bound dump missing: {dump["path"]}')
    elif dump.get('sha256') != sha256_file(npz):
        problems.append(f'n={n}: sha256 does not match {dump["path"]}')
    elif dump.get('bytes') != npz.stat().st_size:
        problems.append(f'n={n}: byte count does not match {dump["path"]}')
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, default=DEFAULT_RUNS,
                        help='ladder directory holding n<N>/ subdirectories')
    parser.add_argument('--grids', default=','.join(str(n) for n in DEFAULT_GRIDS))
    parser.add_argument('--max-cells', type=int, default=4096)
    parser.add_argument('--check', action='store_true',
                        help='verify committed packs against the dumps instead of writing')
    args = parser.parse_args()
    grids = [int(value) for value in args.grids.split(',')]

    if args.check:
        problems = []
        for n in grids:
            problems += check(args.runs / f'n{n}', n)
        for problem in problems:
            print('MISMATCH', problem)
        if problems:
            return 1
        print(f'OK {len(grids)} packs reproduce their bound dumps '
              f'({", ".join(str(n) for n in grids)})')
        return 0

    for n in grids:
        summary = render(args.runs / f'n{n}', n, max_cells=args.max_cells)
        print('PACK', json.dumps(summary))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
#!/usr/bin/env python3
"""Import the two Re_tau ~ 180 channel DNS companions and pin their provenance.

The anchor ``CASE-FVM-CHANNEL-KMM180`` grades ``u+(y+)`` for ``y+ > 30`` against
the Kim, Moin & Moser (1987) table.  That table is still not available in a
machine-readable form (still **WITHHELD**), so the repository has been grading
against the Lee & Moser (2015) case at ``Re_tau = 182.088``.  A single proxy
cannot bound the substitution error; two more references at the *same nominal
Reynolds number* can:

* ``mkm1999_0178`` -- Moser, Kim & Mansour (1999) ``chan180``: ``Re_tau =
  178.12``, produced with *the same numerical method as KMM1987* (their own
  header states "Numerical Method: Kim, Moin & Moser, 1987"); the authors are
  two of the three KMM1987 authors, so it is the closest published companion of
  the anchor's reference.
* ``hj2006_0180`` -- Hoyas & Jimenez (2006) ``Re180``: ``Re_tau = 180``
  exactly, an independent code and an independent group.

Together with the in-tree Lee & Moser (2015) profile the three datasets bracket
the nominal ``Re_tau = 180`` of the anchor, which turns the unquantified proxy
substitution into a *measured* reference band (see
``tensorfvm/verification/kmm180_refband.py``).

Sources (raw bytes cached under ``docs/rans-turbulence-validation-matrix/dns/cache``
so the import is reproducible offline and the sha256 below is checkable):

* https://turbulence.oden.utexas.edu/data/MKM/chandata.tar.gz
  -> ``chan180/profiles/chan180.means``
* https://turbulence.oden.utexas.edu/data/channels/profiles/Re180/profiles/Re180.prof

Usage::

    python scripts/import_channel_reference_band.py            # verify + rewrite
    python scripts/import_channel_reference_band.py --check    # verify only
"""
import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DNS_ROOT = ROOT / 'docs' / 'rans-turbulence-validation-matrix' / 'dns'
CACHE = DNS_ROOT / 'cache'

CASES = {
    'mkm1999_0178': dict(
        raw='mkm1999_chan180.means',
        upstream='https://turbulence.oden.utexas.edu/data/MKM/chandata.tar.gz'
                 ' (member chan180/profiles/chan180.means)',
        columns=('y_over_delta', 'y_plus', 'u_plus', 'dU_plus_dy_plus',
                 'w_plus', 'dw_plus_dy_plus', 'p_plus'),
        skip=('#',),
        re_tau_re=r'Re[ _]\{?\\?tau\}?\s*=\s*([0-9.]+)',
        citation='Moser, R. D., Kim, J. & Mansour, N. N. 1999 DNS of turbulent '
                 'channel flow up to Re_tau = 590. Phys. Fluids 11, 943-945. '
                 'DOI:10.1063/1.869966',
        note='chan180 case; header states the numerical method is Kim, Moin & '
             'Moser (1987), i.e. the anchor reference code; two of the three '
             'KMM1987 authors are authors here',
    ),
    'hj2006_0180': dict(
        raw='hj2006_Re180.prof',
        upstream='https://turbulence.oden.utexas.edu/data/channels/profiles/'
                 'Re180/profiles/Re180.prof',
        columns=('y_over_delta', 'y_plus', 'u_plus', 'urms_plus', 'vrms_plus',
                 'wrms_plus', 'minus_Omega_z_plus', 'omx_rms_plus',
                 'omy_rms_plus', 'omz_rms_plus', 'uv_plus', 'uw_plus',
                 'vw_plus', 'pr_plus', 'ps_plus', 'psto_plus', 'p_rms_plus'),
        skip=('%',),
        re_tau_re=r'Re[ _]\{?\\?tau\}?\s*=\s*([0-9.]+)',
        citation='Hoyas, S. & Jimenez, J. 2006 Scaling of the velocity '
                 'fluctuations in turbulent channels up to Re_tau = 2000. '
                 'Phys. Fluids 18, 011702. DOI:10.1063/1.2162185',
        note='Re180 case, Re_tau = 180 exactly; independent group and code',
    ),
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse(case, raw_path):
    spec = CASES[case]
    text = Path(raw_path).read_text()
    # The raw headers also cite *other* Reynolds numbers in prose (e.g. the
    # MKM data-set title mentions Re_tau = 590); the operative line is the last
    # one, which carries the case Reynolds number.
    matches = re.findall(spec['re_tau_re'], text)
    if not matches:
        raise ValueError(f'{case}: Re_tau not found in header')
    re_tau = float(matches[-1])
    if not 170.0 <= re_tau <= 190.0:
        raise ValueError(f'{case}: parsed Re_tau={re_tau} outside the '
                         'Re_tau ~ 180 band this importer is allowed to write')
    rows = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(spec['skip']):
            continue
        rows.append([float(token) for token in stripped.split()])
    data = np.asarray(rows, dtype=float)
    if data.shape[1] != len(spec['columns']):
        raise ValueError(f'{case}: expected {len(spec["columns"])} columns, '
                         f'got {data.shape[1]}')
    data = data[np.argsort(data[:, 0])]
    if np.any(np.diff(data[:, 0]) <= 0):
        raise ValueError(f'{case}: wall distance must increase monotonically')
    if abs(data[0, 0]) > 1e-12:
        raise ValueError(f'{case}: profile does not start at the wall')
    return re_tau, data


def write_case(case, *, check_only):
    spec = CASES[case]
    raw = CACHE / spec['raw']
    if not raw.exists():
        raise SystemExit(f'{case}: missing cached raw file {raw}')
    re_tau, data = parse(case, raw)
    out = DNS_ROOT / case
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / 'mean.csv'
    header = ','.join(spec['columns'])
    body = '\n'.join(','.join(repr(float(value)) for value in row) for row in data)
    text = f'{header}\n{body}\n'
    provenance = {
        'dataset': case,
        'citation': spec['citation'],
        'source_type': 'original authors DNS statistical data; not experiment',
        'upstream_url': spec['upstream'],
        'raw_file': spec['raw'],
        'raw_sha256': sha256(raw),
        'csv': 'mean.csv',
        'csv_sha256': hashlib.sha256(text.encode()).hexdigest(),
        'rows': int(data.shape[0]),
        'Re_tau_reported': re_tau,
        'normalization': 'wall units except y_over_delta',
        'note': spec['note'],
        'imported_utc': datetime.now(timezone.utc).isoformat(),
        'importer': 'scripts/import_channel_reference_band.py',
    }
    if check_only:
        on_disk = json.loads((out / 'provenance.json').read_text())
        if on_disk['raw_sha256'] != provenance['raw_sha256']:
            raise SystemExit(f'{case}: cached raw bytes changed (sha256 mismatch)')
        if on_disk['csv_sha256'] != provenance['csv_sha256']:
            raise SystemExit(f'{case}: mean.csv does not match provenance')
        if csv_path.read_text() != text:
            raise SystemExit(f'{case}: mean.csv is stale, re-run the importer')
    else:
        csv_path.write_text(text)
        (out / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    return dict(dataset=case, Re_tau=re_tau, rows=int(data.shape[0]),
                sha256=provenance['csv_sha256'][:12])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true',
                        help='verify the committed files instead of rewriting them')
    args = parser.parse_args(argv)
    for case in CASES:
        print(write_case(case, check_only=args.check))


if __name__ == '__main__':
    main()
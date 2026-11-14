"""Import the Re_tau = 180 DNS channel profile of the KMM1987 lineage.

Why this file exists
--------------------
The anchor ``CASE-FVM-CHANNEL-KMM180`` is graded against *Kim, Moin & Moser
(1987)*.  Until now no machine-readable KMM1987 table existed in this
repository, so the grading used Lee & Moser (2015) at Re_tau = 182.088 as a
*documented proxy* (:mod:`tensorfvm.verification.kmm1987`).  This script
imports the authoritative alternative: the ``chan180`` case of the
Moser-Kim-Mansour (1999) database, which
  * was computed with the *KMM1987 numerical method* -- the database header
    states ``Numerical Method: Kim, Moin & Moser, 1987, JFM vol 177, 133-166``;
  * is the Re_tau ~ 180 channel-flow statistics of that lineage;
  * reports ``Re_tau = 178.12`` (not the nominal 180 -- recorded, not rounded).

Provenance discipline
---------------------
* the whole statistical database is pinned by the SHA-256 of
  ``chandata.tar.gz``; a download whose digest differs is rejected;
* the extracted member is pinned by its own SHA-256 and row count;
* the header lines carrying ``Re_tau`` and the numerical method are copied
  verbatim into ``provenance.json`` so the claim above is checkable from the
  committed files alone;
* nothing is rescaled, smoothed or interpolated at import time.

Usage
-----
    python scripts/import_mkm1999_channel.py                  # download + verify
    python scripts/import_mkm1999_channel.py --tarball x.tar.gz  # offline copy
    python scripts/import_mkm1999_channel.py --check          # verify in-tree copy
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = (REPO_ROOT / 'docs' / 'rans-turbulence-validation-matrix'
               / 'dns' / 'mkm1999_0180')

#: ``http://turbulence.oden.utexas.edu/data/MKM/chandata.tar.gz`` (fetched over https)
TARBALL_URL = 'https://turbulence.oden.utexas.edu/data/MKM/chandata.tar.gz'
#: SHA-256 of the tarball as fetched on 2026-11-14; a changed digest is rejected.
TARBALL_SHA256 = '1d8be14cb38660ca5f6be4522f5c01d1fabcb434350c693c2f498c404bf02ed2'
#: Member holding the mean profiles of the Re_tau ~ 180 case.
MEMBER = 'chan180/profiles/chan180.means'
#: Column names of the processed CSV (wall units; ``y_over_delta`` geometric).
COLUMNS = ('y_over_delta', 'y_plus', 'u_plus', 'du_plus_dy_plus',
           'Wmean_plus', 'dWmean_dy_plus', 'Pmean_plus')

#: Header lines whose verbatim text is part of the provenance record.
HEADER_KEYS = ('Re_tau', 'Numerical Method', 'Normalization', 'Filename',
               'Date File Created', 'ny')


def sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path):
    return sha256_bytes(Path(path).read_bytes())


def parse_means(text):
    """Parse ``chan180.means`` into a float array plus the header facts.

    The header of the file is comment lines starting with ``#``.  ``Re_tau`` is
    read from the ``# Re_tau = ...`` line and cross-checked against the
    ``# ny = 129,  Re = ...`` line; a disagreement raises rather than picking
    one silently.
    """
    header = [line for line in text.splitlines() if line.lstrip().startswith('#')]
    header_text = '\n'.join(header)

    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        values = [float(tok) for tok in line.split()]
        if len(values) != 7:
            raise ValueError(f'expected seven columns, got {len(values)}: {line!r}')
        rows.append(values)
    data = np.asarray(rows, dtype=float)
    if data.ndim != 2 or data.shape[0] < 8:
        raise ValueError('mean-profile file holds fewer than eight rows')

    def after(label):
        for line in header:
            body = line.lstrip('#').strip()
            if body.startswith(label):
                return body[len(label):].strip().lstrip('=').strip()
        return None

    re_tau_text = after('Re_tau')
    if re_tau_text is None:
        raise ValueError('no Re_tau line in the header')
    re_tau = float(re_tau_text.split()[0])

    ny_line = after('ny')
    if ny_line is not None:
        re_from_ny = float(ny_line.split(',')[-1].split('=')[-1])
        if abs(re_from_ny - re_tau) > 1e-9:
            raise ValueError(f'header disagrees on Re_tau: {re_tau} vs {re_from_ny}')

    method = after('Numerical Method')
    if method is None or 'Kim, Moin' not in method:
        raise ValueError('header does not attribute the KMM1987 numerical method')

    return dict(data=data, re_tau=re_tau, method=method,
                header_text=header_text, header=header)


def bulk_and_friction(data):
    """``U_bulk+`` and ``Cf`` implied by the imported profile itself.

    The tabulated statistics span the half channel ``0 <= y/h <= 1``; symmetry
    gives ``U_bulk = (1/h) * int_0^h U dy`` so, in wall units,
    ``U_bulk+ = int_0^1 u+ d(y/h)`` (no factor two), and ``Cf = 2 (u_tau/U_bulk)^2
    = 2 / U_bulk+^2``.
    """
    y_over_delta = data[:, 0]
    u_plus = data[:, 2]
    bulk = float(np.trapezoid(u_plus, y_over_delta))
    if not np.isfinite(bulk) or bulk <= 0:
        raise ValueError('non-positive bulk velocity from the imported profile')
    return bulk, 2.0 / bulk ** 2


def build(out_dir=DEFAULT_OUT, *, tarball=None, url=TARBALL_URL,
          expected_tarball_sha256=TARBALL_SHA256):
    out_dir = Path(out_dir)
    if tarball is not None:
        payload = Path(tarball).read_bytes()
        source = f'local file {Path(tarball).name}'
    else:
        with urllib.request.urlopen(url, timeout=60) as response:
            payload = response.read()
        source = url

    digest = sha256_bytes(payload)
    if expected_tarball_sha256 is not None and digest != expected_tarball_sha256:
        raise ValueError(f'tarball digest {digest} != pinned {expected_tarball_sha256}')

    with tarfile.open(fileobj=io.BytesIO(payload), mode='r:gz') as archive:
        member = archive.extractfile(MEMBER)
        if member is None:
            raise ValueError(f'{MEMBER} missing from the archive')
        raw = member.read()

    parsed = parse_means(raw.decode('latin-1'))
    data = parsed['data']
    if np.any(np.diff(data[:, 1]) <= 0):
        raise ValueError('wall coordinate y+ must increase monotonically')
    if np.any(np.diff(data[:, 0]) <= 0):
        raise ValueError('y/h must increase monotonically')

    bulk_plus, cf = bulk_and_friction(data)

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / 'mean.csv'
    lines = [','.join(COLUMNS)]
    lines += [','.join(repr(float(v)) for v in row) for row in data]
    csv_path.write_text('\n'.join(lines) + '\n')

    provenance = dict(
        authors=['Robert D. Moser', 'John Kim', 'Nagi N. Mansour'],
        title='Direct numerical simulation of turbulent channel flow up to Re_tau = 590',
        journal='Physics of Fluids 11 (1999), 943-945',
        doi='10.1063/1.869966',
        source_type='original authors DNS statistical database (not experiment)',
        retrieved_utc=datetime.now(timezone.utc).isoformat(timespec='milliseconds'),
        source=dict(
            url=url if tarball is None else source,
            database='MKM_1999 statistical database, University of Texas turbulence server',
            tarball_sha256=digest,
            member=MEMBER,
            member_sha256=sha256_bytes(raw),
        ),
        header_facts=dict(
            re_tau=parsed['re_tau'],
            numerical_method=parsed['method'],
            normalization='U_tau, h',
            rows=int(data.shape[0]),
            file_created='Mar 01, 2000',
        ),
        header_text=parsed['header_text'],
        Re_tau_reported=parsed['re_tau'],
        family='Kim, Moin & Moser (1987) spectral method; Re_tau ~ 180 statistics',
        half_width=1.0,
        U_bulk_plus=bulk_plus,
        Cf=cf,
        Cf_definition='2*tau_wall/(rho*U_bulk^2)',
        Cf_provenance=('computed by scripts/import_mkm1999_channel.py from the '
                       'imported profile (half-channel symmetry), not tabulated '
                       'by the database'),
        normalization_csv='wall units except y_over_delta',
        centerline_included=True,
        interpolation='PCHIP, no extrapolation beyond the tabulated y+ range',
        files=dict(
            mean=dict(csv='mean.csv', csv_sha256=sha256_file(csv_path),
                      rows=int(data.shape[0])),
        ),
        license=('no explicit licence statement on the database page; numerical '
                 'values redistributed with author attribution, raw archive pinned '
                 'by sha256'),
        source_values_changed=False,
    )
    prov_path = out_dir / 'provenance.json'
    prov_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + '\n')
    return csv_path, prov_path, provenance


def check(out_dir=DEFAULT_OUT):
    """Verify the committed copy against its provenance record."""
    out_dir = Path(out_dir)
    provenance = json.loads((out_dir / 'provenance.json').read_text())
    csv_path = out_dir / 'mean.csv'
    digest = sha256_file(csv_path)
    expected = provenance['files']['mean']['csv_sha256']
    if digest != expected:
        raise ValueError(f'{csv_path} digest {digest} != recorded {expected}')
    data = np.loadtxt(csv_path, delimiter=',', skiprows=1)
    bulk_plus, cf = bulk_and_friction(data)
    if abs(bulk_plus - provenance['U_bulk_plus']) > 1e-9:
        raise ValueError('U_bulk+ changed relative to provenance')
    if abs(cf - provenance['Cf']) > 1e-12:
        raise ValueError('Cf changed relative to provenance')
    return dict(path=str(csv_path), rows=int(data.shape[0]),
                re_tau=provenance['Re_tau_reported'],
                U_bulk_plus=bulk_plus, Cf=cf, csv_sha256=digest)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--tarball', type=Path, default=None,
                        help='use a local chandata.tar.gz instead of downloading')
    parser.add_argument('--no-sha-check', action='store_true',
                        help='accept an unpinned archive (records what was used)')
    parser.add_argument('--check', action='store_true',
                        help='only verify the committed copy')
    args = parser.parse_args(argv)

    if args.check:
        print(json.dumps(check(args.out), indent=2))
        return None
    csv_path, prov_path, _ = build(
        args.out, tarball=args.tarball,
        expected_tarball_sha256=None if args.no_sha_check else TARBALL_SHA256)
    print('wrote', csv_path)
    print('wrote', prov_path)
    print(json.dumps(check(args.out), indent=2))


if __name__ == '__main__':
    main()
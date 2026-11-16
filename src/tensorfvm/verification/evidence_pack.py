"""Self-contained *text* evidence pack for a finite-volume grid level.

Why this exists
---------------
Cluster jobs in this repo have twice returned ``state=succeeded`` / ``rc=0`` while
the freshly written binary field dump was missing from the fetched tar
(``cj2912`` n=256, ``cj3258`` n=128; see
``docs/CASE-FVM-KOVASZNAY-ORDER/ORDER-32-64-128-256-ATTEMPT.md`` §3.1 and
``m20571-artifact-schema-drift-diagnosis.md``).  A run whose numbers only live in
an ``.npz`` therefore has a single point of failure that is *outside the
algorithm*: if the binary does not travel, nothing is measurable and the level
must be reported as not-measured.

This module makes the numbers travel through a second, independent channel: a
plain markdown file that embeds

* the complete ``cases.json`` row as JSON (so every scalar survives verbatim),
* a lattice-downsampled field table as CSV (``<= max_cells`` rows) carrying
  centres, computed and exact velocity/pressure and cell volume, which is
  *sufficient to recompute the L2 error indicators from text alone*,
* the sha256 of the binary dump, so the binary -- whenever it does arrive -- can
  be proven to be the same run, and vice versa.

The embedded CSV is not decorative: :func:`write_text_evidence` recomputes the
volume-weighted L2 errors from the downsampled table and records the relative
deviation from the full-grid values, so a reader knows how much information the
downsample cost.  :func:`read_text_evidence` parses the block back, which is what
the regression test uses -- the pack is machine-readable, not prose.

Boundary: this is a delivery channel, not a replacement for the binary dump.
Anything requiring the full field (figure rendering, conservation audits on every
face) still needs the ``.npz``; the pack keeps the *scalar evidence* alive.
"""
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

CSV_BEGIN = '<!-- BEGIN-EVIDENCE-CSV -->'
CSV_END = '<!-- END-EVIDENCE-CSV -->'
JSON_BEGIN = '<!-- BEGIN-EVIDENCE-JSON -->'
JSON_END = '<!-- END-EVIDENCE-JSON -->'
CSV_COLUMNS = ('x', 'y', 'z', 'u', 'v', 'w', 'p',
               'u_exact', 'v_exact', 'w_exact', 'p_exact', 'volume')


#: Command that writes a pack as a by-product of a fresh solve.  Callers that
#: re-render a pack from a committed dump pass their own command instead.
DEFAULT_GENERATOR = 'scripts/verify_kovasznay_order.py --evidence-markdown'


def sha256_file(path):
    """Hex sha256 of a file, streamed so multi-MB dumps are fine."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def lattice_stride(centres, n, max_cells=4096):
    """Smallest stride whose selected lattice fits ``max_cells`` cells.

    The count is taken from the *actual* selection rather than from a ``(n/stride)**3``
    estimate: the meshes in this case are ``n x n x 1``, so the spanwise axis has a
    single cell and contributes no thinning.  A cubic estimate is therefore
    conservative by a factor ``n/stride`` (``n=256`` would have used 256 cells where
    4096 were allowed), which silently costs two orders of magnitude of resolution in
    the text channel.
    """
    if n < 1 or max_cells < 1:
        raise ValueError('positive n and max_cells required')
    for stride in range(1, n + 1):
        if lattice_indices(centres, n, stride).size <= max_cells:
            return stride
    return n


def lattice_indices(centres, n, stride):
    """Indices of an ``i % stride == 0`` lattice, located from cell centres.

    Cell ordering is a mesh implementation detail, so the lattice is recovered
    from the centres (``i = round(x/h - 1/2)``) instead of assuming row-major
    storage.  Every selected index is unique by construction.

    Two things are deliberately *not* assumed:

    * **a cubic lattice.**  The case meshes are ``n x n x 1``: the spanwise axis
      holds a single cell, so thinning it is meaningless.  Applying the
      ``i % stride == 0`` mask on all three axes selected the empty set as soon
      as ``stride > 1`` (the lone spanwise index carries float noise off the half
      offset, e.g. ``round(0.5/h - 1/2) = 63`` for ``n = 128``), i.e. no evidence
      pack could be written for any level needing a stride -- exactly the finest
      levels the pack exists for.
    * **an absolute origin.**  ``i`` is replaced by the position of the cell's
      coordinate inside that axis' own sorted coordinate list, so the selection
      does not depend on where the mesh's half-cell offset rounds to.
    """
    centres = np.asarray(centres, dtype=float)
    h = 1. / n
    ijk = np.rint(centres / h - .5).astype(np.int64)
    if np.any(ijk < 0) or np.any(ijk >= n):
        raise ValueError('cell centres do not lie on the expected n x n lattice')
    mask = np.ones(len(ijk), dtype=bool)
    for axis in range(ijk.shape[1]):
        levels = np.unique(ijk[:, axis])
        if levels.size <= 1:
            continue                      # a single-cell axis contributes no thinning
        mask &= np.searchsorted(levels, ijk[:, axis]) % stride == 0
    return np.flatnonzero(mask)


def _relative_l2(computed, exact, volumes, remove_mean=False):
    computed = np.asarray(computed, dtype=float)
    exact = np.asarray(exact, dtype=float)
    if remove_mean:
        # Volume-weighted mean removal, matching ``volume_weighted_error``: the
        # gauge is a discrete functional, so a plain arithmetic mean would shift
        # the reported pressure error on a graded metric.
        weight = volumes / volumes.sum()
        computed = computed - (computed * weight.reshape((-1,) + (1,) * (computed.ndim - 1))).sum(axis=0)
        exact = exact - (exact * weight.reshape((-1,) + (1,) * (exact.ndim - 1))).sum(axis=0)
    diff = computed - exact
    num = float(np.sum(volumes[:, None] * diff * diff))
    den = float(np.sum(volumes[:, None] * exact * exact))
    return float(np.sqrt(num / den))


def write_text_evidence(path, *, row, cell_centres, cell_volumes, velocity, pressure,
                        exact, npz_path=None, max_cells=4096, argv=None,
                        generator=DEFAULT_GENERATOR, provenance=None):
    """Write the markdown evidence pack and return its machine-readable summary.

    ``row`` is the same dict written to ``cases.json``; it is embedded verbatim so
    the pack and the JSON cannot disagree about a scalar.  Pressure is compared in
    the same two gauges the runner uses (fixed and mean-removed), which is why the
    CSV keeps ``p`` and ``p_exact`` unmodified.

    A pack may also be produced *after the fact* from a committed binary dump (the
    runs whose dump survived but whose pack never did).  ``generator`` then names
    the command that actually wrote the file, and ``provenance`` is merged into the
    JSON block, so a re-rendered pack never claims to be the original run's
    by-product.
    """
    path = Path(path)
    centres = np.asarray(cell_centres, dtype=float)
    volumes = np.asarray(cell_volumes, dtype=float)
    velocity = np.asarray(velocity, dtype=float)
    pressure = np.asarray(pressure, dtype=float)
    exact_velocity = np.asarray(exact['velocity'], dtype=float)
    exact_pressure = np.asarray(exact['pressure'], dtype=float)
    n = int(row['n'])

    stride = lattice_stride(centres, n, max_cells=max_cells)
    index = lattice_indices(centres, n, stride)
    if index.size == 0:
        raise ValueError('downsample selected no cells')

    table = np.column_stack((centres[index], velocity[index], pressure[index],
                             exact_velocity[index], exact_pressure[index], volumes[index]))
    csv_lines = [','.join(CSV_COLUMNS)]
    for values in table:
        csv_lines.append(','.join(f'{value:.17g}' for value in values))

    # Self-check: how much did the downsample cost the L2 indicators?
    subset = dict(
        cells=int(index.size), stride=int(stride),
        velocity_l2=_relative_l2(velocity[index], exact_velocity[index], volumes[index]),
        pressure_l2=_relative_l2(pressure[index], exact_pressure[index], volumes[index]),
        pressure_mean_free_l2=_relative_l2(pressure[index], exact_pressure[index],
                                          volumes[index], remove_mean=True),
    )
    for key, full_key in (('velocity_l2', 'velocity_l2'), ('pressure_l2', 'pressure_l2'),
                          ('pressure_mean_free_l2', 'pressure_mean_free_l2')):
        full = row.get(full_key)
        if full:
            subset[f'{key}_relative_deviation'] = abs(subset[key] - full) / abs(full)

    block = {
        'row': row,
        'downsample': {'max_cells': int(max_cells), 'stride': int(stride),
                       'selected_cells': int(index.size),
                       'selection': 'cell centres with i%stride==0 in every axis '
                                    '(i=round(x/h-1/2)); volumes are the true cell volumes'},
        'subset_recomputation': subset,
        'environment': {'python': sys.version.split()[0], 'platform': platform.platform(),
                        'argv': list(argv if argv is not None else sys.argv),
                        'written_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())},
    }
    if provenance is not None:
        block['provenance'] = dict(provenance)
    if npz_path is not None:
        npz_path = Path(npz_path)
        if not npz_path.is_file():
            raise FileNotFoundError(npz_path)
        block['binary_dump'] = {'path': str(npz_path),
                                'bytes': int(npz_path.stat().st_size),
                                'sha256': sha256_file(npz_path)}

    text = '\n'.join([
        f'# Text evidence pack -- n={n}, Re={row.get("reynolds")}, '
        f'closure={row.get("symmetry_closure")}',
        '',
        f'Generated by `{generator}`. '
        'The JSON block below is the exact `cases.json` row; the CSV block is a '
        'lattice-downsampled subset of the same solution, dense enough to recompute '
        'the L2 indicators from text alone (`read_text_evidence`).',
        '',
        f'* cells (full grid): {row.get("cells")}',
        f'* numerically_converged: {row.get("numerically_converged")}',
        f'* velocity_l2: {row.get("velocity_l2")!r}',
        f'* pressure_l2: {row.get("pressure_l2")!r}',
        f'* pressure_mean_free_l2: {row.get("pressure_mean_free_l2")!r}',
        f'* max|w|: {row.get("max_abs_z_velocity")!r}',
        f'* downsample: stride {stride} -> {index.size} cells',
        f'* downsample cost, velocity_l2 relative deviation: '
        f'{subset.get("velocity_l2_relative_deviation")!r}',
        '',
        JSON_BEGIN,
        '```json',
        json.dumps(block, indent=2, sort_keys=True),
        '```',
        JSON_END,
        '',
        CSV_BEGIN,
        '```csv',
        *csv_lines,
        '```',
        CSV_END,
        '',
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')
    return dict(path=str(path), stride=stride, selected_cells=int(index.size),
                subset_recomputation=subset,
                npz_sha256=(block.get('binary_dump') or {}).get('sha256'))


def _extract(text, begin, end):
    start = text.index(begin) + len(begin)
    stop = text.index(end, start)
    return text[start:stop]


def read_text_evidence(path):
    """Parse a pack back into ``(block_dict, table_dict_of_arrays)``.

    The CSV is returned as named columns so a checker can recompute the L2 errors
    without re-reading the binary dump.
    """
    text = Path(path).read_text(encoding='utf-8')
    raw_json = _extract(text, JSON_BEGIN, JSON_END).strip()
    if raw_json.startswith('```'):
        raw_json = raw_json.split('\n', 1)[1]
    if raw_json.rstrip().endswith('```'):
        raw_json = raw_json.rstrip()[:-3]
    block = json.loads(raw_json)

    raw_csv = _extract(text, CSV_BEGIN, CSV_END).strip()
    if raw_csv.startswith('```'):
        raw_csv = raw_csv.split('\n', 1)[1]
    if raw_csv.rstrip().endswith('```'):
        raw_csv = raw_csv.rstrip()[:-3]
    lines = [line for line in raw_csv.strip().splitlines() if line.strip()]
    header = lines[0].split(',')
    if tuple(header) != CSV_COLUMNS:
        raise ValueError(f'unexpected CSV header: {header!r}')
    values = np.array([[float(item) for item in line.split(',')] for line in lines[1:]])
    table = {name: values[:, k] for k, name in enumerate(CSV_COLUMNS)}
    return block, table


def l2_from_table(table):
    """Volume-weighted relative L2 errors recomputed from a parsed CSV table.

    Both pressure gauges are reported because the runner reports both: the fixed
    gauge carries a gauge constant that varies with refinement (so its slope is not
    a formal order), while the mean-removed gauge is the physically meaningful one.
    Returning only one of them would let a checker compare the wrong number.
    """
    volumes = np.asarray(table['volume'], dtype=float)
    velocity = np.column_stack((table['u'], table['v'], table['w']))
    exact_velocity = np.column_stack((table['u_exact'], table['v_exact'], table['w_exact']))
    pressure = np.asarray(table['p'], dtype=float)[:, None]
    exact_pressure = np.asarray(table['p_exact'], dtype=float)[:, None]
    return dict(
        velocity_l2=_relative_l2(velocity, exact_velocity, volumes),
        pressure_l2=_relative_l2(pressure, exact_pressure, volumes),
        pressure_mean_free_l2=_relative_l2(pressure, exact_pressure, volumes, remove_mean=True),
        max_abs_w=float(np.max(np.abs(table['w']))),
        cells=int(volumes.size),
    )
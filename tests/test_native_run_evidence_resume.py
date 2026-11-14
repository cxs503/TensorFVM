"""Resuming an interrupted native evidence bundle must be possible.

Regression for the stall on ``CASE-FVM-KOVASZNAY-ORDER``: the n=128 level of the
32/64/128 order study could never be produced because the repo already carried
its provenance stub (``environment.json``/``sources.json``/``source-snapshot``),
and ``NativeRunEvidence`` refused any non-empty output directory.  The guard must
distinguish "metadata this class wrote" (reusable) from "measured artefacts"
(refused unless explicitly overwritten).
"""
import json
from pathlib import Path

import pytest

from tensorfvm.verification.native_run import METADATA_NAMES, NativeRunEvidence

RUNNER = Path(__file__).resolve()


def _evidence(output, **kwargs):
    return NativeRunEvidence(output, RUNNER, **kwargs)


def test_fresh_directory_is_created(tmp_path):
    out = tmp_path / 'n128'
    evidence = _evidence(out)
    assert (out / 'environment.json').is_file()
    assert (out / 'sources.json').is_file()
    environment = json.loads((out / 'environment.json').read_text())
    assert {'python', 'numpy', 'scipy', 'threads'} <= set(environment)
    assert json.loads((out / 'sources.json').read_text()) == evidence.hashes
    evidence.verify_sources()


def test_metadata_only_directory_is_resumed_in_place(tmp_path):
    """The committed provenance stub of an unfinished level must not block it."""
    out = tmp_path / 'n128'
    (out / 'source-snapshot' / 'src').mkdir(parents=True)
    stale = out / 'source-snapshot' / 'src' / 'stale_from_older_checkout.py'
    stale.write_text('# superseded\n')
    (out / 'environment.json').write_text('{"python": "old"}\n')
    (out / 'sources.json').write_text('{"old": "hash"}\n')

    evidence = _evidence(out)

    assert not stale.exists(), 'stale source snapshot must be cleared'
    assert json.loads((out / 'sources.json').read_text()) == evidence.hashes
    assert 'old' not in json.loads((out / 'sources.json').read_text())
    assert json.loads((out / 'environment.json').read_text())['python'] != 'old'
    evidence.verify_sources()


@pytest.mark.parametrize('name', ['cases.json', 'n128-fields.npz', 'summary.json'])
def test_measured_artefacts_still_block_by_default(tmp_path, name):
    out = tmp_path / 'n128'
    out.mkdir()
    (out / name).write_text('{"measured": 1}\n')

    with pytest.raises(ValueError) as error:
        _evidence(out)
    assert name in str(error.value)
    assert (out / name).is_file(), 'a refused output must not be mutated'


def test_measured_artefacts_are_preserved_with_explicit_overwrite(tmp_path):
    out = tmp_path / 'n128'
    out.mkdir()
    (out / 'cases.json').write_text('{"measured": 1}\n')

    _evidence(out, overwrite=True)

    # The old measurement is not deleted behind the caller's back; it is replaced
    # only by the run's own explicit writes.
    assert (out / 'cases.json').read_text() == '{"measured": 1}\n'
    assert (out / 'environment.json').is_file()


def test_metadata_names_are_the_documented_set():
    assert METADATA_NAMES == {'environment.json', 'sources.json', 'source-snapshot'}
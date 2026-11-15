"""Closure metadata semantics of the order-study consistency guard.

``scripts/summarize_kovasznay_32_64_128.py`` refuses to fit a log-log slope
across levels solved with different discretisations.  Its first version compared
every configuration key for equality, so a level whose ``cases.json`` predates
the ``symmetry_closure`` field (``null``) looked like a *conflict* with a level
that records ``'free'``, and a physically uniform ladder was dropped
(``docs/CASE-FVM-KOVASZNAY-ORDER/m20800-independent-reference-audit.md`` §7).

The guard now distinguishes "unrecorded" from "different", but only where the
saved fields can decide the value: the extrusion closure is *measurable* from
the spurious spanwise velocity, which is round-off (<=1e-12) under the
``zero_gradient`` symmetry closure and O(h) (~1e-3) under the traction-free one.
These tests pin that semantics on synthetic level records and on the real
``free``-closure artifacts that motivated it, so the relaxation cannot silently
widen into "missing metadata is fine".

Reference: Kovasznay (1948), Proc. Cambridge Philos. Soc. 44:58-62,
doi:10.1017/S0305004100023999.
"""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts' / 'summarize_kovasznay_32_64_128.py'
ANCHOR = ROOT / 'docs' / 'CASE-FVM-KOVASZNAY-ORDER' / 'runs-32-64-128'


def _module():
    spec = importlib.util.spec_from_file_location('summarize_kovasznay_32_64_128', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _level(n, closure='free', max_w=2.2e-03, **config):
    configuration = dict(reynolds=40., symmetry_closure=closure,
                         convection_order=2, algorithm='coupled')
    configuration.update(config)
    return dict(n=n, configuration=configuration, max_abs_z_velocity=max_w)


def test_unrecorded_closure_reconstructed_from_fields_is_accepted():
    module = _module()
    levels = [_level(32, closure=None, max_w=4.5e-03),
              _level(64, closure='free', max_w=2.2e-03),
              _level(128, closure='free', max_w=1.1e-03)]
    configuration, metadata = module.validate_levels_with_metadata(levels)
    assert configuration['symmetry_closure'] == 'free'
    assert metadata['metadata_complete'] is False
    assert metadata['declared']['32'] is None
    assert metadata['from_fields'] == {'32': 'free', '64': 'free', '128': 'free'}
    [gap] = metadata['reconstructed']
    assert gap['key'] == 'symmetry_closure'
    assert gap['levels'] == [32]
    assert gap['reconstructed'] == 'free'
    assert gap['evidence']['32']['max_abs_z_velocity'] == pytest.approx(4.5e-03)


def test_unrecorded_closure_contradicting_the_spanwise_signature_is_refused():
    """A level that really used the symmetry closure cannot borrow the free label."""
    module = _module()
    levels = [_level(32, closure=None, max_w=2.5e-17),
              _level(64, closure='free', max_w=2.2e-03),
              _level(128, closure='free', max_w=1.1e-03)]
    with pytest.raises(SystemExit) as excinfo:
        module.validate_levels_with_metadata(levels)
    message = str(excinfo.value)
    assert 'n=[32]' in message and 'disagrees' in message


def test_unrecorded_closure_without_spanwise_evidence_is_refused():
    module = _module()
    levels = [_level(32, closure=None, max_w=None),
              _level(64, closure='free', max_w=2.2e-03)]
    with pytest.raises(SystemExit) as excinfo:
        module.validate_levels_with_metadata(levels)
    assert 'not reconstructable' in str(excinfo.value)


def test_label_that_its_own_fields_contradict_is_refused():
    """Mislabelling is a defect: the fields, not the label, decide the closure."""
    module = _module()
    levels = [_level(32, max_w=4.5e-03),
              _level(64, closure='zero_gradient', max_w=2.2e-03),
              _level(128, max_w=1.1e-03)]
    with pytest.raises(SystemExit) as excinfo:
        module.validate_levels_with_metadata(levels)
    message = str(excinfo.value)
    assert 'n=64' in message and 'symmetry_closure' in message


@pytest.mark.parametrize('key,value', [('reynolds', None), ('convection_order', None),
                                       ('algorithm', None)])
def test_unrecorded_non_inferrable_setting_is_still_a_hard_error(key, value):
    """Relaxation is scoped to the closure; nothing else becomes optional."""
    module = _module()
    levels = [_level(32), _level(64, max_w=2.2e-03, **{key: value})]
    with pytest.raises(SystemExit) as excinfo:
        module.validate_levels_with_metadata(levels)
    message = str(excinfo.value)
    assert key in message and 'cannot be reconstructed' in message


def test_recorded_conflict_is_still_refused_and_reports_both_values():
    module = _module()
    levels = [_level(32, closure='free', max_w=4.5e-03),
              _level(64, closure='zero_gradient', max_w=2.2e-03)]
    with pytest.raises(SystemExit) as excinfo:
        module.validate_levels_with_metadata(levels)
    message = str(excinfo.value)
    assert "'free'" in message and "'zero_gradient'" in message


def test_fully_recorded_uniform_sequence_is_complete_metadata():
    module = _module()
    levels = [_level(32), _level(64), _level(128, max_w=1.1e-03)]
    configuration, metadata = module.validate_levels_with_metadata(levels)
    assert configuration['symmetry_closure'] == 'free'
    assert metadata['metadata_complete'] is True
    assert metadata['reconstructed'] == []


def test_committed_free_closure_artifacts_reconstruct_their_closure():
    """The artifacts that motivated the fix, read as committed.

    ``n32`` predates the ``symmetry_closure`` field; ``n64`` records ``'free'``.
    The guard must fit the pair as a uniform, traction-free ladder and must say
    that the n32 label was reconstructed rather than recorded.
    """
    levels = []
    for n in (32, 64, 128):
        path = ANCHOR / f'n{n}' / 'cases.json'
        if not path.is_file():
            continue
        row = json.loads(path.read_text())[0]
        levels.append(dict(n=row['n'],
                           configuration={key: row.get(key) for key in
                                          ('reynolds', 'symmetry_closure',
                                           'convection_order', 'algorithm')},
                           max_abs_z_velocity=row.get('max_abs_z_velocity')))
    if len(levels) < 2:
        pytest.skip('free-closure 32/64 artifacts absent')
    module = _module()
    configuration, metadata = module.validate_levels_with_metadata(levels)
    assert configuration['symmetry_closure'] == 'free'
    assert metadata['declared']['32'] is None
    assert metadata['from_fields'][str(levels[0]['n'])] == 'free'
    assert metadata['metadata_complete'] is False
    assert metadata['reconstructed'][0]['levels'] == [32]
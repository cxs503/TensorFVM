"""Guard against fitting a convergence order across a mixed configuration.

The 32/64/128 order study stores one directory per grid.  Two ways of closing the
two extrusion planes coexist in the repository (``free`` = historical
traction-free closure, which leaves an O(h) spurious spanwise velocity, and
``zero_gradient`` = the physical symmetry closure), and their spurious-spanwise
budgets differ by twelve orders of magnitude.  If a sequence ever mixed the two
-- a plausible accident when a level is rerun in place -- the least-squares slope
would still look like a number but would not be the order of any scheme.

``scripts/summarize_kovasznay_32_64_128.py`` therefore refuses such a mixture.
These tests exercise that refusal directly on synthetic level records, so the
guard is tested without needing real solution dumps, and they also assert that a
uniform sequence still passes and reports its configuration back.
"""
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'summarize_kovasznay_32_64_128.py'


def _load_module():
    spec = importlib.util.spec_from_file_location('summarize_kovasznay_32_64_128', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _level(n, **overrides):
    configuration = dict(reynolds=40., symmetry_closure='zero_gradient',
                         convection_order=2, algorithm='coupled')
    configuration.update(overrides)
    return dict(n=n, configuration=configuration)


def test_uniform_sequence_is_accepted_and_configuration_returned():
    module = _load_module()
    levels = [_level(32), _level(64), _level(128)]
    configuration = module.validate_levels(levels)
    assert configuration == dict(reynolds=40., symmetry_closure='zero_gradient',
                                convection_order=2, algorithm='coupled')


@pytest.mark.parametrize('key,mixed_value', [('symmetry_closure', 'free'),
                                            ('reynolds', 100.),
                                            ('convection_order', 1),
                                            ('algorithm', 'segregated')])
def test_mixed_sequence_is_refused(key, mixed_value):
    module = _load_module()
    levels = [_level(32), _level(64, **{key: mixed_value}), _level(128)]
    with pytest.raises(SystemExit) as excinfo:
        module.validate_levels(levels)
    message = str(excinfo.value)
    assert f'n=64' in message and key in message


def test_missing_levels_are_refused_not_silently_skipped():
    module = _load_module()
    with pytest.raises(SystemExit):
        module.validate_levels([])

"""Reject misleading mesh comparisons with mismatched policies or field bytes."""
import importlib.util
import json
from pathlib import Path
import pytest

SCRIPT = Path(__file__).parents[1] / 'scripts/plot_native_suboff_refinement.py'
spec = importlib.util.spec_from_file_location('suboff_refinement', SCRIPT)
plot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plot)


def evidence(tmp_path, changed_policy=False):
    runs, local = [], {}
    for i in range(2):
        run = tmp_path / f'run-{i}'
        (run / 'audit').mkdir(parents=True)
        config = dict(mode='manufactured', settings={'nonlinear_tolerance': 1e-7},
                      reconstruction='quadratic', quadratic_linear_solver='assembled',
                      merge_coplanar_faces=bool(i and changed_policy), manufactured_definition='same ABC')
        (run / 'config.json').write_text(json.dumps(config))
        (run / 'input.msh').write_bytes(b'bound mesh fixture')
        (run / 'fields.npz').write_bytes(b'bound field fixture')
        (run / 'result.json').write_text(json.dumps({'cells': 100*(i+1), 'history': []}))
        bindings = {str(run / name): plot.digest(run / name)
                    for name in ('config.json', 'input.msh', 'fields.npz', 'result.json')}
        (run / 'audit/audit.json').write_text(json.dumps(dict(passed=True,
            producer_input_sha256=bindings, relative_L2_errors={'velocity': .001})))
        stats = dict(pressure_relative_L2_global_gauge_aligned=.02, maximum_error_over_rho_U2=.1,
                     rms_error_over_rho_U2=.004, cells_above_3percent_rho_U2=2)
        local[run.name] = dict(mesh_sha256=plot.digest(run / 'input.msh'),
            fields_sha256=plot.digest(run / 'fields.npz'), cells=100*(i+1), faces=400*(i+1),
            fixed_spatial_windows_L={'near_tail_fixed_window': {'x_min': .9}},
            regions={name: stats for name in ('all', 'wall_adjacent', 'near_body_fixed_window', 'near_tail_fixed_window')})
        runs.append(run)
    audit = tmp_path / 'local-audit'
    audit.mkdir()
    (audit / 'result.json').write_text(json.dumps(local))
    (audit / 'manifest.json').write_text(json.dumps({'result.json': plot.digest(audit / 'result.json')}))
    return runs, audit


def test_pressure_norms_keep_their_distinct_denominators(tmp_path):
    runs, audit = evidence(tmp_path)
    rows, _, _, _ = plot.collect(list(reversed(runs)), audit)
    assert [r['cells'] for r in rows] == [100, 200]
    assert rows[0]['all_pressure_L2_percent'] == 2
    assert rows[0]['all_maximum_error_percent_rho_U2'] == 10
    assert rows[0]['all_rms_error_percent_rho_U2'] == .4


def test_mixed_face_policy_is_not_a_refinement_series(tmp_path):
    runs, audit = evidence(tmp_path, changed_policy=True)
    with pytest.raises(ValueError, match='face policies must match'):
        plot.collect(runs, audit)


def test_changed_numerical_source_requires_an_explicit_compatibility_probe(tmp_path):
    runs, audit = evidence(tmp_path)
    path = runs[0] / 'config.json'
    config = json.loads(path.read_text())
    config['source_sha256'] = {'src/tensorfvm/native_quadratic_flow3d.py': 'changed operator'}
    path.write_text(json.dumps(config))
    ledger = runs[0] / 'audit/audit.json'
    data = json.loads(ledger.read_text())
    data['producer_input_sha256'][str(path)] = plot.digest(path)
    ledger.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='source versions differ'):
        plot.collect(runs, audit)


def test_viscous_operator_change_is_not_a_mesh_refinement_series(tmp_path):
    runs, audit = evidence(tmp_path)
    path = runs[0] / 'config.json'
    config = json.loads(path.read_text()); config['viscous_reconstruction'] = 'cubic'
    path.write_text(json.dumps(config))
    ledger = runs[0] / 'audit/audit.json'
    data = json.loads(ledger.read_text()); data['producer_input_sha256'][str(path)] = plot.digest(path)
    ledger.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='settings and face policies'):
        plot.collect(runs, audit)


@pytest.mark.parametrize('ordering', ['COLAMD', 'MMD_AT_PLUS_A'])
def test_implicit_historical_ordering_only_matches_explicit_colamd(tmp_path, ordering):
    runs, audit = evidence(tmp_path)
    path = runs[0] / 'config.json'
    config = json.loads(path.read_text());config['settings']['lu_ordering'] = ordering
    path.write_text(json.dumps(config))
    ledger = runs[0] / 'audit/audit.json'
    data = json.loads(ledger.read_text());data['producer_input_sha256'][str(path)] = plot.digest(path)
    ledger.write_text(json.dumps(data))
    if ordering == 'COLAMD':
        assert len(plot.collect(runs, audit)[0]) == 2
    else:
        with pytest.raises(ValueError, match='settings and face policies'):
            plot.collect(runs, audit)


@pytest.mark.parametrize('target', ['fields.npz', 'result.json', 'audit/audit.json', 'local-result'])
def test_mutated_or_failed_evidence_cannot_enter_comparison(tmp_path, target):
    runs, audit = evidence(tmp_path)
    if target == 'local-result':
        (audit / 'result.json').write_text('{}')
    elif target == 'audit/audit.json':
        path = runs[0] / target
        data = json.loads(path.read_text())
        data['passed'] = False
        path.write_text(json.dumps(data))
    else:
        (runs[0] / target).write_bytes(b'mutated field')
    with pytest.raises(ValueError):
        plot.collect(runs, audit)

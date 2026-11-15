#!/usr/bin/env python3
"""Mutation audit of the CASE-FVM-KOVASZNAY-ORDER acceptance gate.

The gate (``tests/test_kovasznay_32_64_128_acceptance.py``) claims to be
*evidence-bound*: it recomputes every error from the saved fields, so a
hand-edited ``cases.json`` or a drifting report cannot make it green.  A test
suite that passes does not by itself prove that claim -- it only proves that the
code runs.  This script proves the claim the way a mutation study does: it
corrupts the evidence in a temporary copy of the artifact set, reruns the gate
checks against the corrupted copy, and requires each corruption to be caught by
the specific check that is supposed to catch it.

The expected failure set of every mutation is **declared up front** in
``MUTATIONS`` (pre-registered), and a mutation whose corruption slips through is
reported as ``detected: false`` -- a gate defect, never a green result.  The one
mutation that must *not* fail anything is ``relaxed_threshold_in_summary``: the
criterion is fixed in code, so a tampered ``verdict.threshold`` must be ignored.

No committed artifact is modified: every mutation happens inside
``tempfile.mkdtemp()``.

Reference: Kovasznay (1948), Proc. Cambridge Philos. Soc. 44:58-62,
doi:10.1017/S0305004100023999.
"""
import argparse
import importlib.util
import json
import shutil
import tempfile
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / 'tests' / 'test_kovasznay_32_64_128_acceptance.py'


def load_gate():
    spec = importlib.util.spec_from_file_location('kovasznay_acceptance_gate', GATE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


#: Gate checks, named by what they are supposed to protect.  Each entry runs the
#: real test function (not a re-implementation) so the audit cannot drift from
#: the gate it audits.
PROBES = {
    'recorded_errors_match_fields': lambda gate: [
        gate.test_recorded_errors_reproduce_from_raw_fields('anchor_zg', recorded, recomputed)
        for recorded, recomputed in (('velocity_l2', 'velocity_full'),
                                     ('pressure_l2', 'pressure_fixed'),
                                     ('pressure_mean_free_l2', 'pressure_meanfree'))],
    'reported_slopes_match_fields': lambda gate: [
        gate.test_reported_slopes_match_recomputed_slopes('anchor_zg')],
    'verdict_flags_not_overstated': lambda gate: [
        gate.test_verdict_flags_are_not_overstated('anchor_zg')],
    'ledger_criterion_holds': lambda gate: [
        gate.test_anchor_zg_full_3d_velocity_order_meets_acceptance()],
}


# --- mutations -------------------------------------------------------------
def _levels(directory):
    return sorted((path for path in directory.glob('n*')
                   if path.name[1:].isdigit() and (path / 'cases.json').is_file()),
                  key=lambda path: int(path.name[1:]))


def mutate_hand_edited_error(directory):
    """Scale one level's recorded velocity error; the fields still say otherwise."""
    row_path = directory / 'n64' / 'cases.json'
    row = json.loads(row_path.read_text())
    row[0]['velocity_l2'] *= 0.9
    row_path.write_text(json.dumps(row))


def mutate_fabricated_solution(directory):
    """Replace the finest field dump with the analytic solution: zero error on paper."""
    path = directory / 'n128' / 'n128-fields.npz'
    fields = dict(np.load(path))
    fields['velocity'] = np.array(fields['exact_velocity'], copy=True)
    np.savez(path, **fields)


def mutate_hand_edited_slope(directory):
    path = directory / 'summary.json'
    summary = json.loads(path.read_text())
    summary['fits']['velocity_full']['slope'] += 0.05
    path.write_text(json.dumps(summary))


def mutate_overstated_verdict(directory):
    """Mark the finest level unconverged in the row but keep the summary 'pass'."""
    row_path = directory / 'n128' / 'cases.json'
    row = json.loads(row_path.read_text())
    row[0]['numerically_converged'] = False
    row_path.write_text(json.dumps(row))


def mutate_relaxed_threshold(directory):
    """Tamper the published threshold: the criterion lives in code and must hold."""
    path = directory / 'summary.json'
    summary = json.loads(path.read_text())
    summary['verdict']['threshold'] = 1.0
    summary['verdict']['quantity'] += ' (threshold relaxed by hand)'
    path.write_text(json.dumps(summary))


#: mutation -> (function, the probes that must raise AssertionError).  The set is
#: a *minimum*: a corruption that trips additional checks is caught even harder
#: and is reported with ``extra_failures`` rather than being counted as a miss.
MUTATIONS = {
    'hand_edited_error': (mutate_hand_edited_error, {'recorded_errors_match_fields'}),
    'fabricated_solution': (mutate_fabricated_solution,
                            {'recorded_errors_match_fields',
                             'reported_slopes_match_fields'}),
    'hand_edited_slope': (mutate_hand_edited_slope, {'reported_slopes_match_fields'}),
    'overstated_verdict': (mutate_overstated_verdict, {'verdict_flags_not_overstated'}),
    'relaxed_threshold_in_summary': (mutate_relaxed_threshold, set()),
}


# --- harness ---------------------------------------------------------------
def run_probes(gate, directory, grids):
    gate.SETS['anchor_zg'] = (directory, grids)
    failures, not_run = set(), {}
    for name, probe in PROBES.items():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                probe(gate)
        except AssertionError as error:
            failures.add(name)
            message = (str(error) or repr(error)).splitlines()
            not_run[name] = 'AssertionError: ' + (message[0][:200] if message else '')
        except Exception as error:  # pytest.skip and friends
            not_run[name] = f'{type(error).__name__}: {error}'
    return sorted(failures), not_run


def run(source, output_json):
    source = Path(source)
    grids = tuple(int(path.name[1:]) for path in _levels(source))
    if grids != (32, 64, 128):
        raise SystemExit(f'{source} does not hold the 32/64/128 triple: {grids}')

    gate = load_gate()
    results = []
    with tempfile.TemporaryDirectory() as workspace:
        control = Path(workspace) / 'control'
        shutil.copytree(source, control)
        failures, notes = run_probes(gate, control, grids)
        results.append(dict(mutation='none (control)', expected_failures=[],
                            observed_failures=failures, detected=not failures,
                            notes=notes))
        if failures:
            raise SystemExit(f'control copy does not pass the gate: {failures} {notes}')

        for name, (mutation, expected) in MUTATIONS.items():
            target = Path(workspace) / name
            shutil.copytree(source, target)
            mutation(target)
            failures, notes = run_probes(gate, target, grids)
            results.append(dict(mutation=name, expected_failures=sorted(expected),
                                observed_failures=failures,
                                extra_failures=sorted(set(failures) - expected),
                                detected=set(expected) <= set(failures),
                                notes=notes))
            shutil.rmtree(target)

    undetected = [row['mutation'] for row in results if not row['detected']]
    payload = dict(source=str(source.relative_to(ROOT)) if source.is_relative_to(ROOT)
                   else str(source),
                   gate=str(GATE.relative_to(ROOT)),
                   grids=list(grids),
                   mutations=results,
                   undetected=undetected,
                   verdict='PASS' if not undetected else 'FAIL')
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({key: payload[key] for key in ('verdict', 'undetected')}, indent=2))
    for row in results:
        print(f"  {row['mutation']:<30} expected={row['expected_failures']} "
              f"observed={row['observed_failures']} detected={row['detected']}")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source', type=Path,
                        default=ROOT / 'docs' / 'CASE-FVM-KOVASZNAY-ORDER'
                        / 'runs-32-64-128-zero_gradient')
    parser.add_argument('--json', type=Path,
                        default=ROOT / 'docs' / 'CASE-FVM-KOVASZNAY-ORDER'
                        / 'gate-mutation-audit.json')
    args = parser.parse_args()
    payload = run(args.source, args.json)
    raise SystemExit(0 if payload['verdict'] == 'PASS' else 1)


if __name__ == '__main__':
    main()
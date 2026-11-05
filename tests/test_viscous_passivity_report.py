"""Test the viscous passivity report generation script."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.timeout(120)
def test_viscous_passivity_report_generates_outputs(tmp_path):
    """Run the report script and verify all expected outputs are produced."""
    output_dir = str(tmp_path / 'audit')
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / 'scripts' / 'viscous_passivity_report.py'),
         '--output', output_dir],
        capture_output=True, text=True, timeout=120,
        env={**os.environ, 'PYTHONPATH': str(REPO_ROOT / 'src')},
    )
    assert result.returncode == 0, f"Script failed: {result.stderr}"

    # Verify all expected files exist
    expected = ['report.md', 'audit.json', 'skew_sweep.png',
                'correction_scan.png', 'shear_survey.png', 'affine_consistency.png']
    for name in expected:
        path = os.path.join(output_dir, name)
        assert os.path.isfile(path), f"Missing output: {name}"
        assert os.path.getsize(path) > 0, f"Empty output: {name}"

    # Verify JSON is valid and contains key fields
    with open(os.path.join(output_dir, 'audit.json')) as f:
        data = json.load(f)
    assert 'skew_sweep' in data
    assert 'correction_scan' in data
    assert 'shear_survey' in data
    assert 'passivity_boundary' in data

    # Verify the correction scan confirms no passive gamma
    assert not data['correction_scan']['any_passive']

    # Verify the report mentions key findings
    with open(os.path.join(output_dir, 'report.md')) as f:
        report = f.read()
    assert 'structural' in report.lower()
    assert 'quality gate' in report.lower()
    assert 'passive' in report.lower()

"""Smoke marker for pipeline E2E test (mission m4626).

This module exists solely to prove that the TensorFVM package import path
is functional end-to-end: a test in ``tests/`` imports from ``src/tensorfvm/``
and reads a sentinel value.  If the import succeeds and the value matches,
the package installation / PYTHONPATH wiring is correct.
"""

PIPELINE_SMOKE_M4626 = "4626"

"""Pipeline E2E smoke test: code path + test path (mission m4626)."""

from tensorfvm import smoke_marker_m4626


def test_smoke_marker_value():
    assert smoke_marker_m4626.PIPELINE_SMOKE_M4626 == "4626"


def test_sanity():
    assert 1 + 1 == 2

"""Explicit regional pressure gates; a global norm cannot qualify local fields."""
import math


def regional_pressure_acceptance(regions, required_regions, *, equation_audit_passed,
                                 limit=.03):
    """Evaluate previously measured relative L2 errors using one global gauge.

    Callers must supply metrics computed from the same bound raw fields and
    fixed spatial regions. Fixed-scale errors such as error/(rho U²) cannot
    substitute for the reference-relative norm. Missing/empty/nonfinite
    regions fail closed. This function grants no physical validation.
    """
    if not math.isfinite(limit) or limit <= 0:
        raise ValueError('A finite positive relative error limit is required')
    names = tuple(required_regions)
    if not names or len(set(names)) != len(names):
        raise ValueError('At least one unique required region is required')
    checks = []
    for name in names:
        data = regions.get(name, {})
        value = data.get('pressure_relative_L2_global_gauge_aligned')
        valid = (isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
                 and value >= 0 and data.get('cells', 0) > 0)
        checks.append(dict(region=name, relative_L2=value if valid else None,
                           limit=limit, passed=bool(valid and value < limit)))
    local_passed = all(check['passed'] for check in checks)
    return dict(scope='fixed-region pressure relative L2 with one full-domain gauge',
                checks=checks, regional_pressure_passed=local_passed,
                equation_and_global_audit_passed=bool(equation_audit_passed),
                manufactured_regional_pressure_qualified=bool(equation_audit_passed and local_passed),
                experimental_accuracy_qualified=False,
                fixed_scale_errors_are_diagnostics_only=True)

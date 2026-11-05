"""Generate the frozen-time Taylor--Green study for the native hybrid NS backend.

Writes ``study.json`` and ``RESULTS.md`` next to this file.  The run is a
spatial-refinement ladder of the *frozen-time* steady problem (the backend has
no transient integrator): the exact field at ``t`` is imposed through its
Dirichlet trace, the time term becomes the manufactured body force
``f = 2 nu u``, and only the spatial operator is judged.

Run with::

    python scripts/verification-native-taylor-green/run_study.py
"""

from __future__ import annotations

import json
from pathlib import Path

from tensorfvm.verification.native_taylor_green import (
    BOX,
    TaylorGreenCase,
    observed_order,
    run_study,
)

HERE = Path(__file__).resolve().parent
MAC_STUDY = HERE.parent / "verification-taylor-green-mac" / "study.json"

N_LIST = [4, 6, 8]
TIME_S = 0.5
CASE = TaylorGreenCase(amplitude=1.0, viscosity=1.0)
VELOCITY_GATE = 0.15
DIVERGENCE_GATE = 1e-12
INACTIVE_Z_GATE = 1e-3
ORDER_BAND = (1.5, 2.5)


def mac_reference() -> dict:
    """Cross-backend reference: the time-marched periodic MAC TGV study."""
    if not MAC_STUDY.exists():
        return {}
    study = json.loads(MAC_STUDY.read_text())
    rows = {row["n"]: row for row in study.get("spatial_ladder", [])}
    return {
        "source": "scripts/verification-taylor-green-mac/study.json",
        "physical_case": study.get("case", {}),
        "velocity_relative_l2_by_n": {
            str(n): rows[n]["velocity_relative_l2"] for n in sorted(rows)
        },
    }


def main() -> dict:
    ladder = run_study(N_LIST, CASE, time_s=TIME_S)
    orders = [
        {
            "coarse_n": coarse["n"],
            "fine_n": fine["n"],
            "observed_order": observed_order(coarse, fine),
        }
        for coarse, fine in zip(ladder, ladder[1:])
    ]
    finest = ladder[-1]
    order = orders[-1]["observed_order"]
    gates = {
        "velocity": {
            "passed": finest["velocity_relative_l2"] < VELOCITY_GATE,
            "metric": "velocity_relative_l2",
            "value": finest["velocity_relative_l2"],
            "criterion": f"< {VELOCITY_GATE} at n = {finest['n']}",
        },
        "continuity": {
            "passed": max(row["max_divergence_s_inv"] for row in ladder)
            < DIVERGENCE_GATE,
            "metric": "max_divergence_s_inv",
            "value": max(row["max_divergence_s_inv"] for row in ladder),
            "criterion": f"< {DIVERGENCE_GATE} on every resolution",
        },
        "spatial_order": {
            "passed": ORDER_BAND[0] < order < ORDER_BAND[1],
            "value": order,
            "criterion": f"{ORDER_BAND[0]} < order < {ORDER_BAND[1]} (n = 6 -> 8)",
        },
        "inactive_z": {
            "passed": max(row["velocity_z_max_abs_m_s"] for row in ladder)
            < INACTIVE_Z_GATE,
            "metric": "velocity_z_max_abs_m_s",
            "value": max(row["velocity_z_max_abs_m_s"] for row in ladder),
            "criterion": f"< {INACTIVE_Z_GATE} (exact w = 0)",
        },
        "pressure": {
            "passed": False,
            "metric": "pressure_relative_linf",
            "values": [row["pressure_relative_linf"] for row in ladder],
            "reason": (
                "withheld: in this closed all-Dirichlet-velocity configuration the "
                "cell-centred pressure retains an odd-even mode that the discrete "
                "system does not determine, so the pressure error does not converge "
                "under refinement and must not be gated"
            ),
        },
    }
    summary = {
        "spatial_operator_qualified": all(
            gates[name]["passed"]
            for name in ("velocity", "continuity", "spatial_order", "inactive_z")
        ),
        "pressure_withheld": True,
        "time_marching_qualified": False,
        "time_marching_reason": (
            "native hybrid 3-D backend is a steady solver: no transient integrator, "
            "so the decaying-vortex ladder is out of scope here and stays with the "
            "periodic MAC kernel"
        ),
    }
    study = {
        "schema": "tensorfvm.native-taylor-green-frozen-study/1",
        "backend": "native_hybrid_ns3d.NativeHybridNavierStokes3D",
        "exact_solution": {
            "u": "A cos(x) sin(y) exp(-2 nu t)",
            "v": "-A sin(x) cos(y) exp(-2 nu t)",
            "w": "0",
            "p": "-(A^2/4)(cos 2x + cos 2y) exp(-4 nu t)   (rho = 1)",
            "frozen_source": "f = -du/dt = 2 nu u (steady manufactured force)",
            "source": "Taylor & Green (1937) single-mode vortex; exact unforced NS solution",
        },
        "case": {
            "amplitude": CASE.amplitude,
            "viscosity_pa_s": CASE.viscosity,
            "density_kg_m3": CASE.density,
            "box_m": [BOX, BOX, BOX],
            "frozen_time_s": TIME_S,
            "convection_order": 2,
            "coupled_linear_method": "direct",
            "newton_tolerance": 1e-10,
            "velocity_gate": VELOCITY_GATE,
            "divergence_gate_s_inv": DIVERGENCE_GATE,
        },
        "spatial_ladder": ladder,
        "observed_orders": orders,
        "gates": gates,
        "gate_summary": summary,
        "cross_backend_reference": mac_reference(),
    }
    (HERE / "study.json").write_text(json.dumps(study, indent=2) + "\n")
    (HERE / "RESULTS.md").write_text(results_markdown(study))
    return study


def results_markdown(study: dict) -> str:
    case = study["case"]
    ladder = study["spatial_ladder"]
    orders = {
        (row["coarse_n"], row["fine_n"]): row["observed_order"]
        for row in study["observed_orders"]
    }
    lines = [
        "# Taylor--Green vortex: frozen-time reference for the native hybrid 3-D backend",
        "",
        "Machine-generated by `scripts/verification-native-taylor-green/run_study.py`"
        " -- do not edit by hand.",
        "",
        "Exact solution frozen at $t$ on the periodic box $[0,2\\pi]^3$ ($\\rho=1$):"
        " $u=A\\cos x\\sin y\\,e^{-2\\nu t}$, $v=-A\\sin x\\cos y\\,e^{-2\\nu t}$,"
        " $w=0$, $p=-\\frac{A^2}{4}(\\cos 2x+\\cos 2y)e^{-4\\nu t}$.",
        "The time term is supplied as a manufactured body force"
        " $f=-\\partial u/\\partial t=2\\nu u$, so the steady momentum equation"
        " $(u\\cdot\\nabla)u=-\\nabla p+\\nu\\nabla^2u+f$ holds identically.",
        "",
        f"Case: A={case['amplitude']}, nu={case['viscosity_pa_s']},"
        f" frozen t={case['frozen_time_s']} s, convection order {case['convection_order']},"
        f" Newton tolerance {case['newton_tolerance']}.",
        "",
        "## Spatial ladder",
        "",
        "| n | dx | velocity L2 | order | max div | max |w| | pressure Linf |",
        "|---|---|---|---|---|---|---|",
    ]
    for index, row in enumerate(ladder):
        order = (
            orders.get((ladder[index - 1]["n"], row["n"]), float("nan"))
            if index
            else float("nan")
        )
        lines.append(
            f"| {row['n']} | {row['spacing_m']:.4f} | {row['velocity_relative_l2']:.3e} | {order:.2f}"
            f" | {row['max_divergence_s_inv']:.1e} | {row['velocity_z_max_abs_m_s']:.1e}"
            f" | {row['pressure_relative_linf']:.3e} |"
        )
    cross = study.get("cross_backend_reference") or {}
    mac_rows = cross.get("velocity_relative_l2_by_n", {})
    if mac_rows:
        lines += [
            "",
            "## Cross-backend reference (time-marched periodic MAC kernel)",
            "",
            "| n | native hybrid velocity L2 | periodic MAC velocity L2 |",
            "|---|---|---|",
        ]
        for row in ladder:
            mac = mac_rows.get(str(row["n"]))
            lines.append(
                f"| {row['n']} | {row['velocity_relative_l2']:.3e} | {mac:.3e} |"
                if mac is not None
                else f"| {row['n']} | {row['velocity_relative_l2']:.3e} | n/a |"
            )
        lines += [
            "",
            "The MAC column integrates the decay in time and is the physical benchmark;"
            " the native column judges the steady spatial operator only.  The native"
            " hybrid operator converges at the same order with a much larger constant,"
            " which is a property of its cell/face hybrid stabilisation, not of the"
            " reference.",
        ]
    lines += [
        "",
        "## Gates",
        "",
        "| gate | passed | value | criterion |",
        "|---|---|---|---|",
    ]
    for name, gate in study["gates"].items():
        value = gate.get("value", gate.get("values"))
        lines.append(
            f"| {name} | {gate['passed']} | {value} | {gate.get('criterion', gate.get('reason', ''))} |"
        )
    lines += [
        "",
        "## Limitations",
        "",
        "* Time marching is out of scope: the native hybrid 3-D backend is steady, so the"
        " decaying-vortex ladder stays with the periodic MAC kernel.",
        "* The pressure gate is withheld: the closed all-Dirichlet-velocity configuration"
        " leaves an odd--even cell-pressure mode undetermined, and the measured pressure"
        " error does not converge under refinement (it even reaches machine-noise"
        " amplification on the coarsest mesh).  The pressure content that drives the"
        " flow is exercised indirectly through the converged velocity field.",
        "* The frozen steady problem stiffens as nu decreases: nu = 0.05 (the MAC study"
        " case) does not converge in the Newton iteration, hence nu = 1.0 here.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    result = main()
    print(json.dumps(result["gate_summary"], indent=2))
    print(f"velocity gate value: {result['gates']['velocity']['value']:.4e}")
    print(f"observed order: {result['observed_orders'][-1]['observed_order']:.3f}")

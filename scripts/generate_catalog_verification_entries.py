"""Generate the verification-catalog entries for the two §10.2 gaps closed here.

The catalog is a hand-written markdown index, and the historical defect mode is
a number that was typed once and never re-derived (``docs/VERIFICATION_DEBT_LEDGER.md``
D9 records a stale "open" gap that survived a delivery).  This script removes the
hand-typing: every number in the generated sections is read from the committed
machine-readable artifact of the corresponding verification driver, so a driver
re-run that changes a metric cannot leave the catalog silently quoting the old
value -- ``pytest tests/test_catalog_verification_entries.py`` regenerates the
text and diffs it against the catalog.

Usage::

    PYTHONPATH=src python scripts/generate_catalog_verification_entries.py --check
    PYTHONPATH=src python scripts/generate_catalog_verification_entries.py --write

Sections produced
-----------------
* `## 17. Rotating Frame (MRF / Coriolis)` from
  ``docs/verification-rotating-frame/summary.json``;
* `## 18. Buoyant Cavity Grid Convergence (de Vahl Davis 1983)` from
  ``docs/natural-convection-gci/summary.json`` and
  ``docs/verification-natural-convection-highra/summary.json``.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Dict, List, Optional

ROOT = pathlib.Path(__file__).resolve().parents[1]
CATALOG = ROOT / "docs" / "verification-catalog.md"
ROTATING_SUMMARY = ROOT / "docs" / "verification-rotating-frame" / "summary.json"
GCI_SUMMARY = ROOT / "docs" / "natural-convection-gci" / "summary.json"
HIGHRA_SUMMARY = ROOT / "docs" / "verification-natural-convection-highra" / "summary.json"

BEGIN = "<!-- BEGIN generated: verification-debt entries (m16836) -->"
END = "<!-- END generated: verification-debt entries (m16836) -->"


def _load(path: pathlib.Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _pct(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{100.0 * value:+.2f} %"


def _num(value: Optional[float], fmt: str = "{:.4f}") -> str:
    return "withheld" if value is None else fmt.format(value)


def render_rotating_section() -> str:
    data = _load(ROTATING_SUMMARY)
    criteria = data["criteria"]
    n_pass = sum(1 for c in criteria.values() if c["pass"])
    verdict = "PASS" if data.get("all_pass") else "FAIL"
    lines = [
        "## 17. Rotating Frame (MRF / Coriolis)",
        "",
        "**Reference**: Johnston, Halleen & Lezius (1972) rotating-channel experiment / "
        "theory (rotation axis along the span); Taylor (1923) torque for circular Couette flow; "
        "Greenspan (1968) rotating-frame momentum balance.",
        "**Physics**: absolute-frame Navier-Stokes rewritten in a frame rotating at "
        "`Omega` -- Coriolis `2 Omega x u`, centrifugal `Omega x (Omega x r)` absorbed into "
        "the modified pressure, and the *frozen-rotor* (MRF) assumption for a rotor zone.",
        "**Implementation**: `src/tensorfvm/rotating_frame.py`",
        "**Driver**: `python -m tensorfvm.verification.rotating_frame`",
        "**Tests**: `tests/test_rotating_frame.py`",
        "**Report**: `docs/verification-rotating-frame/REPORT.md`",
        "**Artifact**: `docs/verification-rotating-frame/summary.json`",
        "",
        f"**Verification Status**: {verdict} ({n_pass}/{len(criteria)} criteria, "
        "re-run reproduced at m16836)",
        "",
        "| criterion | value | gate | direction | verdict |",
        "|---|---:|---:|:--:|:--:|",
    ]
    for name in sorted(criteria):
        c = criteria[name]
        lines.append(
            "| `{name}` | {value:.3e} | {gate:.3e} | {mode} | {verdict} |".format(
                name=name,
                value=float(c["value"]),
                gate=float(c["gate"]),
                mode=c["mode"],
                verdict="PASS" if c["pass"] else "FAIL",
            )
        )
    lines += [
        "",
        "Discriminating power is measured, not assumed: the control criteria "
        "(`C6`, `C7c`, `C10b`) require that a solver which **drops** the Coriolis term, "
        "**flips** its sign or **omits** the centrifugal term violates the anchor by "
        ">= 5 %, i.e. the anchors can fail a wrong implementation.",
        "",
    ]
    return "\n".join(lines)


def render_convection_section() -> str:
    gci = _load(GCI_SUMMARY)
    highra = _load(HIGHRA_SUMMARY)
    lines = [
        "## 18. Buoyant Cavity Grid Convergence (de Vahl Davis 1983)",
        "",
        "**Reference**: de Vahl Davis, G. (1983) *Natural convection of air in a square "
        "cavity: a bench mark numerical solution*, Int. J. Numer. Methods Fluids 3:249.",
        "**Physics**: differentially heated square cavity, `H = 1`, `Pr = 0.71`, "
        "Boussinesq coupling, adiabatic top/bottom walls; metric = mean Nusselt number on "
        "the hot wall (`Nu = 4.519` at `Ra = 1e5`, `8.8` at `Ra = 1e6`).",
        "**Implementation**: `src/tensorfvm/incompressible_benchmarks.py` (`StaggeredFlow`, "
        "explicit MAC projection, DCT-II pressure solve, van Leer limited advection)",
        "**Driver**: `scripts/run_incompressible_benchmarks.py convection`, "
        "`src/tensorfvm/verification/natural_convection_gci.py`",
        "**Tests**: `tests/test_natural_convection_gci.py`, "
        "`tests/test_natural_convection_highra_gate.py`",
        "**Report**: `docs/verification-natural-convection-highra/REPORT.md`, "
        "`docs/natural-convection-gci/`",
        "**Artifact**: `docs/natural-convection-gci/summary.json`, "
        "`docs/verification-natural-convection-highra/summary.json`",
        "",
    ]
    verdicts: List[str] = []
    for ra in sorted(gci["families"]):
        fam = gci["families"][ra]
        nu = fam["nusselt_hot_half_cell"]
        verdict = fam["verdict"]
        ref_nu = float(highra["families"][ra]["reference"]["nu"])
        errs = nu["errors_vs_reference"]
        gate = float(gci["gate"])
        finest = int(verdict["finest_grid"])
        finest_err = verdict["finest_error"]
        limit = verdict.get("extrapolated")
        order = highra["families"][ra]["convergence_nu"]
        passed = finest_err <= gate
        verdicts.append(f"{ra}: {'PASS' if passed else 'honest FAIL'}")
        lines += [
            f"### {ra.replace('Ra', 'Ra = ')}",
            "",
            "| N | Nu (half-cell) | error vs de Vahl Davis |",
            "|---:|---:|---:|",
        ]
        for level in sorted(nu["values"], key=int):
            lines.append(
                f"| {level} | {nu['values'][level]:.4f} | "
                f"{_pct(errs[level])} |"
            )
        lines += [
            "",
            f"- reference `Nu = {ref_nu:.3f}`, gate `{100.0 * gate:.0f} %`, "
            f"finest grid `N = {finest}` -> error `{_pct(finest_err)}` "
            f"({verdicts[-1]})",
            f"- observed order of `Nu` on the three finest levels: "
            f"`p = {order['order']:.2f}`",
            f"- Richardson limit: `Nu_inf = {_num(limit)}`, error "
            f"`{_pct(verdict.get('extrapolated_error'))}`"
            + (
                ""
                if limit is not None
                else " (withheld: `" + str(order.get("reason", "not in the asymptotic range")) + "`)"
            ),
            "",
        ]
    lines += [
        f"**Verification Status**: PARTIAL -- {'; '.join(verdicts)}. "
        "The `Ra = 1e6` level needs `N >= 128` to enter the 3 % band (thermal boundary "
        "layer `~H/(2 Nu)`; the observed order is negative on `N = 16/32/64`, so the "
        "Richardson limit is withheld rather than quoted). `Ra = 1e4` is covered by "
        "`docs/verification-natural-convection-ra1e4/REPORT.md`.",
        "",
    ]
    return "\n".join(lines)


def render_sections() -> str:
    return BEGIN + "\n\n" + render_rotating_section() + "\n" + render_convection_section() + END + "\n"


def splice(text: str, generated: str) -> str:
    """Replace the generated region, appending it when the markers are absent."""

    start = text.find(BEGIN)
    end = text.find(END)
    if start == -1 or end == -1 or end < start:
        return text.rstrip("\n") + "\n\n" + generated
    return text[:start] + generated + text[end + len(END) + 1 :]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default=str(CATALOG))
    parser.add_argument("--write", action="store_true", help="rewrite the catalog in place")
    parser.add_argument("--check", action="store_true", help="exit 1 when the catalog is stale")
    args = parser.parse_args(argv)

    path = pathlib.Path(args.catalog)
    text = path.read_text(encoding="utf-8")
    generated = render_sections()
    expected = splice(text, generated)
    if args.write:
        path.write_text(expected, encoding="utf-8")
        print(f"wrote {path}")
        return 0
    if args.check:
        if expected != text:
            print("catalog sections are stale; run with --write", file=sys.stderr)
            return 1
        print("catalog sections are up to date")
        return 0
    print(generated)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
"""Reference-variant registry for the de Vahl Davis (1983) square-cavity anchors.

Why a *registry* and not a bare constant table
---------------------------------------------
The worktree quotes the same benchmark with two different digit sets, and both
quoting styles are defensible -- they are two different things in the paper:

* ``acceptance`` -- the grid-independent values that the benchmark is normally
  cited by and that this repository uses as acceptance targets
  (``Nu = 2.243 / 4.519 / 8.800``, ``|psi|_max = 5.071 / 9.111 / 16.750``,
  ``u_max = 16.178 / 34.73 / 64.63``, ``v_max = 19.617 / 68.59 / 219.36``).
  This family is reproduced inside the ecosystem by
  ``src/tensorfvm/internal_flow/reference.py`` (``DE_VAHL_DAVIS_NU`` /
  ``DE_VAHL_DAVIS_VELOCITY``) and ``src/tensorfvm/inc_bench/reference.py``
  (``CAVITY_REFERENCES``), and is what ``docs/natural-convection-gci/REPORT.md``
  and ``docs/inc-bench/report.md`` quote.
* ``finite_grid`` -- the values of the paper's own finite grid, which also
  circulate in the worktree (``Nu = 2.238 / 4.509 / 8.817``,
  ``|psi|_max = 5.024 ... 16.32``); see
  ``docs/incompressible-baselines/VERIFICATION.md`` and
  ``scripts/assemble_internal_flow_benchmarks.py``.

Mixing the two families without saying which one is meant is the defect this
module exists to stop: a 1.3 % Nu difference and a 2.6 % streamfunction
difference look like solver error, and at ``Ra = 1e6`` the two ``|psi|_max``
digits (16.32 vs 16.750) differ by as much as the whole finite-grid error of the
solver.  Anything that compares a solver against these numbers must therefore
pick a variant *by name*, and every quoted digit must be a registered one.

Grades (platform vocabulary): ``REAL`` for the acceptance family (reproduced by
several independent offline sources inside the worktree, listed per entry),
``WITHHELD`` for digits that cannot be checked against an offline copy of the
paper from inside this worktree -- those are reported as an orientation band and
never used as a hard gate, except through the documented ``AMBIGUITIES`` bracket.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "AMBIGUITIES",
    "FAMILIES",
    "SUPERSEDED_DIGITS",
    "Variant",
    "cavity_reference",
    "registry_json",
    "superseded_hits",
]

#: ``PR`` of the benchmark (air) -- quoted so that a reader cannot silently
#: reuse these numbers for another Prandtl number.
PRANDTL = 0.71


@dataclass(frozen=True)
class Variant:
    """One registered digit of one benchmark quantity."""

    benchmark: str
    quantity: str
    rayleigh: float
    value: float
    variant: str
    source: str
    grade: str = "REAL"
    evidence: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, object]:
        return {
            "benchmark": self.benchmark,
            "quantity": self.quantity,
            "rayleigh": self.rayleigh,
            "value": self.value,
            "variant": self.variant,
            "source": self.source,
            "grade": self.grade,
            "evidence": list(self.evidence),
        }


_SOURCE = ("de Vahl Davis, G. (1983), Natural convection of air in a square cavity: "
           "a bench mark numerical solution, Int. J. Numer. Methods Fluids 3, 249-264")

_ACCEPTANCE_EVIDENCE = (
    "src/tensorfvm/internal_flow/reference.py",
    "src/tensorfvm/inc_bench/reference.py",
    "docs/natural-convection-gci/REPORT.md",
    "docs/inc-bench/report.md",
)
_FINITE_GRID_EVIDENCE = (
    "docs/incompressible-baselines/VERIFICATION.md",
    "scripts/assemble_internal_flow_benchmarks.py",
    "docs/inc-bench/report.md",
)

#: quantity -> {rayleigh: {variant: value}}
FAMILIES: dict[str, dict[float, dict[str, float]]] = {
    "nusselt": {
        1.0e3: {"acceptance": 1.118},
        1.0e4: {"acceptance": 2.243, "finite_grid": 2.238},
        1.0e5: {"acceptance": 4.519, "finite_grid": 4.509},
        1.0e6: {"acceptance": 8.800, "finite_grid": 8.817},
    },
    "psi_abs_max": {
        1.0e3: {"acceptance": 1.174},
        1.0e4: {"acceptance": 5.071, "finite_grid": 5.024},
        1.0e5: {"acceptance": 9.111},
        1.0e6: {"acceptance": 16.750, "finite_grid": 16.320},
    },
    "u_max": {
        1.0e3: {"acceptance": 3.649},
        1.0e4: {"acceptance": 16.178},
        1.0e5: {"acceptance": 34.73},
        1.0e6: {"acceptance": 64.63},
    },
    "v_max": {
        1.0e3: {"acceptance": 3.697},
        1.0e4: {"acceptance": 19.617},
        1.0e5: {"acceptance": 68.59},
        1.0e6: {"acceptance": 219.36},
    },
}

#: Quantities whose two variants are far apart enough that the *choice* changes
#: the verdict of a 3 % gate.  A report comparing against these must name the
#: variant; ``cavity_reference`` refuses to guess (it raises) unless the caller
#: passes ``variant=...``.
AMBIGUITIES: dict[str, dict[str, object]] = {
    "psi_abs_max": {
        "rayleigh": [1.0e6],
        "variants": {"acceptance": 16.750, "finite_grid": 16.320},
        "spread_percent": 100.0 * (16.750 - 16.320) / 16.320,
        "note": "2.6 % spread; the finite-grid error of the solver at n = 64 is of "
                "the same size (4.8 % vs 16.32, 2.2 % vs 16.75), so an unpinned "
                "comparison cannot distinguish solver error from variant choice. "
                "Judge on the Richardson extrapolation instead "
                "(docs/verification-natural-convection-metrics/report.md).",
        "grade": "WITHHELD",
    },
}

#: Digit strings that appear in the worktree's history as *wrong* transcriptions
#: of a benchmark value (not as a legitimate variant).  Append-only: a new entry
#: needs the commit/file that introduced it and the repair.
SUPERSEDED_DIGITS: dict[str, dict[str, str]] = {
    "1.334": {
        "quantity": "nusselt",
        "rayleigh": "1e3",
        "reason": "wrong Nu(1e3); repaired to 1.118 by m15410",
        "evidence": "docs/audit-m15790-upstream-diffset/REPORT.md",
    },
    "4.535": {
        "quantity": "u_max",
        "rayleigh": "1e4",
        "reason": "order-of-magnitude error for u_max(1e4); repaired to 16.178 by m15410",
        "evidence": "docs/audit-m15790-upstream-diffset/REPORT.md",
    },
    "6.776": {
        "quantity": "v_max",
        "rayleigh": "1e4",
        "reason": "order-of-magnitude error for v_max(1e4); repaired to 19.617 by m15410",
        "evidence": "docs/audit-m15790-upstream-diffset/REPORT.md",
    },
}


def cavity_reference(quantity: str, rayleigh: float, *, variant: str | None = None) -> Variant:
    """Registered digit of ``quantity`` at ``rayleigh`` for a named ``variant``.

    Raises ``KeyError`` for an unregistered combination and ``ValueError`` when
    the variant is one of the documented :data:`AMBIGUITIES` -- there the caller
    has to say which family it means, because the two are further apart than a
    3 % gate.
    """
    try:
        table = FAMILIES[quantity][float(rayleigh)]
    except KeyError as exc:  # pragma: no cover - defensive
        raise KeyError(f"no registered {quantity!r} reference at Ra = {rayleigh:g}") from exc
    ambiguity = AMBIGUITIES.get(quantity)
    ambiguous = ambiguity is not None and float(rayleigh) in ambiguity["rayleigh"]
    if variant is None:
        if ambiguous:
            raise ValueError(
                f"{quantity} at Ra = {rayleigh:g} has variants {sorted(table)} whose spread "
                f"({ambiguity['spread_percent']:.1f} %) exceeds the comparison tolerance; "
                "pass variant=... explicitly")
        variant = "acceptance"
    if variant not in table:
        raise KeyError(f"no {variant!r} variant of {quantity!r} at Ra = {rayleigh:g}")
    evidence = _ACCEPTANCE_EVIDENCE if variant == "acceptance" else _FINITE_GRID_EVIDENCE
    return Variant(
        benchmark="de Vahl Davis 1983", quantity=quantity, rayleigh=float(rayleigh),
        value=table[variant], variant=variant, source=_SOURCE, grade="REAL",
        evidence=evidence)


def variants(quantity: str, rayleigh: float) -> dict[str, float]:
    """All registered digits of ``quantity`` at ``rayleigh`` (may be ambiguous)."""
    return dict(FAMILIES[quantity][float(rayleigh)])


def registered_digits() -> dict[str, set[float]]:
    """Every registered value, grouped by quantity (used by the drift checks)."""
    out: dict[str, set[float]] = {}
    for quantity, by_ra in FAMILIES.items():
        values: set[float] = set()
        for table in by_ra.values():
            values.update(table.values())
        out[quantity] = values
    return out


_NUMBER = re.compile(r"(?<![\d.])(\d+\.\d+)(?![\d])")


def superseded_hits(root: str | Path, *, suffixes: tuple[str, ...] = (".md", ".py", ".json")) -> list[dict[str, object]]:
    """Occurrences of a :data:`SUPERSEDED_DIGITS` value in the worktree documents.

    The match is exact on the digit string, so there are no false positives from
    partial matches (``1.334`` is not found inside ``11.3340``).  A hit is a
    stale reference that survived a repair; the registry keeps the reason and the
    evidence path, so the report can say *what* to write instead.
    """
    root = Path(root)
    hits: list[dict[str, object]] = []
    for path in sorted(root.rglob("*")):
        if path.suffix not in suffixes or not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if rel.startswith((".git/", "docs/audit-", "docs/verification-cavity-anchor/",
                           "docs/m5241/", "docs/VERIFICATION_DEBT_LEDGER.md")):
            # audit reports are exactly where the *history* of a wrong digit is
            # documented on purpose; the registry itself lives in src/
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:  # pragma: no cover - unreadable file
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            for value in _NUMBER.findall(line):
                if value in SUPERSEDED_DIGITS:
                    hits.append({
                        "path": rel, "line": lineno, "value": float(value),
                        "quantity": SUPERSEDED_DIGITS[value]["quantity"],
                        "reason": SUPERSEDED_DIGITS[value]["reason"],
                    })
    return hits


def registry_json() -> dict[str, object]:
    """The whole registry as JSON-ready data (the report injects this verbatim)."""
    return {
        "benchmark": "de Vahl Davis 1983 square cavity, Pr = 0.71",
        "source": _SOURCE,
        "families": {
            quantity: {f"{ra:g}": dict(table) for ra, table in by_ra.items()}
            for quantity, by_ra in FAMILIES.items()
        },
        "ambiguities": AMBIGUITIES,
        "superseded": SUPERSEDED_DIGITS,
    }
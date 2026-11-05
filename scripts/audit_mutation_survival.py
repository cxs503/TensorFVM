#!/usr/bin/env python3
"""Mutation-survival audit of the production 2-D finite-volume kernels.

Every entry of ``tensorfvm.verification.mutation_catalogue`` is applied to the
real source tree, the targeted verification tests are run, and the source is
restored byte-for-byte (sha256 checked, also on failure).  A mutant that makes
the suite fail is *killed*; a mutant the suite accepts is a *survivor*, i.e. an
evidenced hole in the verification coverage and not a statement about the
production code.

Outputs (``--output``): ``mutation-survival.json`` (machine record, includes
source and toolchain hashes), ``MUTATION-SURVIVAL.md`` and ``mutation-survival.png``.
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from tensorfvm.verification.mutation_catalogue import CATALOGUE  # noqa: E402


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def apply_mutation(entry):
    """Apply one textual mutation; raise if the anchor is not unique."""
    path = REPO / entry["file"]
    text = path.read_text()
    count = text.count(entry["find"])
    if count != 1:
        raise RuntimeError(
            f"{entry['id']}: anchor occurs {count} times in {entry['file']}"
        )
    mutated = text.replace(entry["find"], entry["replace"])
    if mutated == text:
        raise RuntimeError(f"{entry['id']}: mutation is a no-op")
    compile(mutated, str(path), "exec")
    return path, mutated


def run_tests(entry, timeout):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO / "src"), str(REPO), env.get("PYTHONPATH", "")]
    )
    env["OMP_NUM_THREADS"] = env.get("OMP_NUM_THREADS", "1")
    env["MKL_NUM_THREADS"] = env.get("MKL_NUM_THREADS", "1")
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        *entry["tests"],
        "-x",
        "-q",
        "-p",
        "no:cacheprovider",
        "--timeout",
        str(timeout),
    ]
    started = time.perf_counter()
    proc = subprocess.run(
        cmd, cwd=REPO, env=env, capture_output=True, text=True, timeout=timeout + 120
    )
    return proc, time.perf_counter() - started


def first_failure(proc):
    for line in proc.stdout.splitlines():
        if re.match(r"^(FAILED|ERROR) ", line):
            return line.split()[1]
    for line in proc.stdout.splitlines():
        if "assert" in line.lower() or "Error" in line:
            return line.strip()[:200]
    return None


def status_of(proc):
    if proc.returncode == 0:
        return "survived"
    if proc.returncode == 1:
        return "killed"
    return f"invalid(exit={proc.returncode})"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO / "src/tensorfvm/verification/artifacts/mutation_survival",
    )
    parser.add_argument(
        "--timeout", type=int, default=1800, help="per-mutant pytest timeout (s)"
    )
    parser.add_argument(
        "--only", nargs="*", default=None, help="restrict to these mutant ids"
    )
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)

    entries = [e for e in CATALOGUE if not args.only or e["id"] in args.only]
    rows = []
    for entry in entries:
        original_path = REPO / entry["file"]
        before = sha256(original_path)
        row = dict(
            id=entry["id"],
            file=entry["file"],
            description=entry["description"],
            rationale=entry["rationale"],
            expected=entry["expected"],
            tests=entry["tests"],
        )
        try:
            path, mutated = apply_mutation(entry)
            path.write_text(mutated)
            if sha256(path) == before:
                raise RuntimeError(f"{entry['id']}: source hash unchanged after write")
            proc, elapsed = run_tests(entry, args.timeout)
            row["status"] = status_of(proc)
            row["returncode"] = proc.returncode
            row["first_failure"] = first_failure(proc)
            row["duration_seconds"] = round(elapsed, 3)
            tail = [line for line in proc.stdout.strip().splitlines() if line.strip()][
                -1:
            ] or [""]
            row["summary_line"] = tail[-1][:200]
            print(
                f"{entry['id']}: {row['status']} ({elapsed:.1f}s) {row['first_failure'] or ''}",
                flush=True,
            )
        except Exception as exc:  # audit must record failures, not abort silently
            row["status"] = "invalid(exception)"
            row["error"] = f"{type(exc).__name__}: {exc}"
            print(f"{entry['id']}: {row['status']} {row['error']}", flush=True)
        finally:
            # Restore unconditionally from the recorded bytes.
            restored = subprocess.run(
                ["git", "checkout", "--", entry["file"]],
                cwd=REPO,
                capture_output=True,
                text=True,
            )
            after = sha256(original_path)
            row["source_restored"] = bool(after == before)
            row["source_sha256"] = before
            if not row["source_restored"]:
                row["error"] = (
                    row.get("error", "")
                    + f" | restore mismatch {restored.stderr.strip()}"
                ).strip(" |")
        rows.append(row)

    summary = dict(
        tool="scripts/audit_mutation_survival.py",
        python=sys.version.split()[0],
        numpy=_version("numpy"),
        torch=_version("torch"),
        scipy=_version("scipy"),
        killed=sum(r["status"] == "killed" for r in rows),
        survived=sum(r["status"] == "survived" for r in rows),
        invalid=sum(r["status"].startswith("invalid") for r in rows),
        mutants=rows,
    )
    (out / "mutation-survival.json").write_text(json.dumps(summary, indent=1) + "\n")
    write_markdown(out / "MUTATION-SURVIVAL.md", summary)
    write_figure(out / "mutation-survival.png", summary)
    print(
        f"killed={summary['killed']} survived={summary['survived']} invalid={summary['invalid']}"
    )
    return 0


def _version(name):
    try:
        return __import__(name).__version__
    except Exception:
        return "absent"


def write_markdown(path, summary):
    lines = [
        "# TensorFVM mutation-survival audit (2-D production kernels)",
        "",
        "Machine-generated by `scripts/audit_mutation_survival.py` from the catalogue in",
        "`src/tensorfvm/verification/mutation_catalogue.py`. Each row applies one textual",
        "defect to the real source, runs the targeted tests, and restores the source",
        "(sha256 verified). `killed` = the suite reacted; `survived` = the suite accepted a",
        "defective discretisation, i.e. a hole in verification coverage.",
        "",
        f"Toolchain: Python {summary['python']}, numpy {summary['numpy']}, torch {summary['torch']}, scipy {summary['scipy']}.",
        "",
        f"Totals: killed {summary['killed']}, survived {summary['survived']}, invalid {summary['invalid']}.",
        "",
        "| mutant | file | defect | predicted | observed | first failing test |",
        "|---|---|---|---|---|---|",
    ]
    for row in summary["mutants"]:
        lines.append(
            "| `{}` | `{}` | {} | {} | {} | {} |".format(
                row["id"],
                row["file"],
                row["description"],
                row["expected"],
                row["status"],
                row.get("first_failure") or "-",
            )
        )
    lines += ["", "## Per-mutant rationale", ""]
    for row in summary["mutants"]:
        lines.append(f"- **{row['id']}** ({row['status']}): {row['rationale']}")
        if row.get("error"):
            lines.append(f"  - note: {row['error']}")
    lines += [
        "",
        "## Scope limits",
        "",
        "This audit says nothing about physical accuracy; it measures whether the existing",
        "tests constrain the discrete operators. Mutants are curated by hand, so the",
        "killed fraction is not an estimate of a random defect-detection probability.",
        "",
    ]
    path.write_text("\n".join(lines))


def write_figure(path, summary):
    ids = [r["id"] for r in summary["mutants"]]
    values = [1 if r["status"] == "killed" else 0 for r in summary["mutants"]]
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(ids) + 1.8), constrained_layout=True)
    colors = ["#2b7a3d" if v else "#b03030" for v in values]
    ax.barh(ids, values, color=colors)
    for i, (row, v) in enumerate(zip(summary["mutants"], values)):
        ax.text(v + 0.02, i, row["status"], va="center", fontsize=9)
    ax.set_xlim(0, 1.45)
    ax.set_xticks([0, 1], ["survived", "killed"])
    ax.set_title(
        "TensorFVM mutation survival: 2-D FV kernels\n"
        f"killed {summary['killed']} / {len(ids)}"
    )
    ax.grid(axis="x", alpha=0.3)
    fig.savefig(path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())

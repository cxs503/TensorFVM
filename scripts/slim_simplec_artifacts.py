"""Slim the shipped SIMPLEC-mobility evidence artifacts under the artifact cap.

The merge governor holds a pull request whose *artifact-class* blobs
(``outputs/``, ``figures/``, ``artifacts/``, ``*.png`` ...) exceed 200 KB in
total, because such files are supposed to be reproducible from the repository
rather than archived in it.  For the SIMPLEC mobility evidence this script is
the documented post-step: it keeps every number that carries meaning and drops
only what is redundant.

What is removed and why:

* JSON is re-serialised with ``separators=(",", ":")`` -- no indentation, no
  ``sort_keys`` (document order preserved).  Floats are emitted by the
  standard ``repr``-based encoder, so precision is untouched.
* The per-iteration ``history`` traces are dominated by inner-iteration noise.
  The convergence evidence lives at both ends (start-up transient, final
  residual plateau), so the traces are subsampled deterministically to
  ``head`` rows + every ``stride``-th row + ``tail`` rows.  Each subsampled
  trace carries a ``history_sampling`` record with the original iteration
  count and the SHA-256 of the *full* trace, so no evidence is unaccounted
  for: the dropped rows can be re-derived by regenerating the run.
* PNG figures are re-encoded to a 128-colour palette at unchanged pixel
  dimensions; layout, axes, curves and labels stay lossless.
* The top-level ``artifact_slimming`` block records the cap, the rule, the
  original byte counts and both reproduction commands.

Full-fidelity evidence::

    python scripts/verify_simplec_mobility.py --output scripts/artifacts/simplec-mobility

then cap it again with::

    python scripts/slim_simplec_artifacts.py --output scripts/artifacts/simplec-mobility
"""
import argparse
import hashlib
import json
from pathlib import Path

CAP_BYTES = 200 * 1024
PALETTE_COLORS = 128
HEAD = 3
STRIDE = 10
TAIL = 2
JSON_FILES = ("verification.json", "evidence/simplec-audit.json")
FIGURE_FILES = (
    "evidence/simplec-convergence.png",
    "evidence/simplec-profile.png",
    "evidence/simplec-iterations.png",
)


def compact(payload):
    """Compact JSON text (full float precision, document order kept)."""
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False)


def payload_sha256(payload):
    """SHA-256 of the compact, key-sorted form of ``payload``."""
    text = json.dumps(payload, separators=(",", ":"), ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()


def subsample(trace, head=HEAD, stride=STRIDE, tail=TAIL):
    """Deterministically keep head + every ``stride``-th + tail rows."""
    total = len(trace)
    if total <= head + tail + 2:
        return [dict(row) for row in trace], None
    keep = set(range(head))
    keep.update(range(head, total, stride))
    keep.update(range(total - tail, total))
    kept = [dict(trace[index]) for index in sorted(keep)]
    record = {
        "total_iterations": total,
        "kept": len(kept),
        "stride": stride,
        "head": head,
        "tail": tail,
        "full_sha256": payload_sha256(trace),
    }
    return kept, record


def _slim_trace_holder(holder):
    """Subsample the ``history`` trace inside one run-like dict."""
    trace = holder.get("history")
    if not isinstance(trace, list):
        return False
    if "history_sampling" in holder:
        raise SystemExit("history already subsampled; regenerate the full run first")
    kept, record = subsample(trace)
    holder["history"] = kept
    if record is not None:
        holder["history_sampling"] = record
    return True


def slim_summary(summary):
    """Return a slimmed copy of the ``simplec_channel`` audit summary."""
    if "artifact_slimming" in summary:
        raise SystemExit("artifact already slimmed; regenerate the full run first")
    slim = json.loads(json.dumps(summary))  # deep copy, document order kept
    for run in slim.get("runs", []):
        _slim_trace_holder(run)
    for holder in slim.get("failed_simple_history", {}).values():
        if isinstance(holder, dict):
            _slim_trace_holder(holder)
    return slim


def build_verification(summary, guards, control, original_bytes):
    """Assemble ``verification.json`` with the slimming provenance block."""
    payload = dict(audit=summary, guards=guards, mutation_control=control)
    payload["artifact_slimming"] = {
        "cap_bytes": CAP_BYTES,
        "rule": (
            "json: compact separators, floats at repr precision; per-iteration "
            "history subsampled to head%d + every %dth + tail%d rows, full-trace "
            "sha256 recorded per run; png: %d-colour palette, pixel dimensions "
            "unchanged" % (HEAD, STRIDE, TAIL, PALETTE_COLORS)
        ),
        "original_json_bytes": original_bytes,
        "figures": list(FIGURE_FILES),
        "regenerate_full": (
            "python scripts/verify_simplec_mobility.py "
            "--output scripts/artifacts/simplec-mobility"
        ),
        "apply_cap": (
            "python scripts/slim_simplec_artifacts.py "
            "--output scripts/artifacts/simplec-mobility"
        ),
    }
    return payload


def slim_figure(path):
    """Palette re-encode one figure in place; return (before, after, size)."""
    from PIL import Image

    before = path.stat().st_size
    image = Image.open(path)
    size = image.size
    image.convert("RGB").quantize(colors=PALETTE_COLORS).save(
        path, format="PNG", optimize=True)
    return before, path.stat().st_size, size


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="artifact directory (scripts/artifacts/simplec-mobility)")
    parser.add_argument("--no-figures", action="store_true",
                        help="only re-serialise the JSON payloads")
    parser.add_argument("--report", action="store_true", help="print the size table")
    args = parser.parse_args(argv)
    root = Path(args.output)
    verification_path = root / "verification.json"
    audit_path = root / "evidence" / "simplec-audit.json"
    original = json.loads(verification_path.read_text())
    original_bytes = {
        "verification.json": verification_path.stat().st_size,
        "evidence/simplec-audit.json": audit_path.stat().st_size,
    }
    summary = slim_summary(original["audit"])
    verification = build_verification(summary, original["guards"],
                                      original["mutation_control"], original_bytes)
    audit_path.write_text(compact(summary) + "\n")
    verification_path.write_text(compact(verification) + "\n")
    rows = [
        ("verification.json", original_bytes["verification.json"],
         verification_path.stat().st_size),
        ("evidence/simplec-audit.json", original_bytes["evidence/simplec-audit.json"],
         audit_path.stat().st_size),
    ]
    if not args.no_figures:
        for name in FIGURE_FILES:
            before, after, _ = slim_figure(root / name)
            rows.append((name, before, after))
    if args.report:
        for name, before, after in rows:
            print("%-34s %8d -> %8d  (-%5.1f%%)" % (
                name, before, after, 100.0 * (1 - after / before)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
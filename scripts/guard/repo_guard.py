#!/usr/bin/env python3
"""TensorFVM repo anti-rot guard (feature_ledger: FVM-SLIM-GUARD, D6).

Three gates, one source of truth:

  staged   pre-commit gate: reject any staged blob larger than the limit
           (default 512 KiB).  A path matching the `guard-allow-large`
           git attribute (see .gitattributes) is whitelisted and bypasses
           the limit.  Escape hatch: FVM_GUARD_ALLOW_LARGE=1
           (the justification must then appear in the commit message).
  ci       CI size gate: (a) no file changed in base..HEAD exceeds the blob
           limit; (b) reachable pack size vs. the recorded pack baseline
           (drift budget) and an absolute hard cap; (c) LIVE artifact
           volume (unique HEAD-referenced blob total) vs. the recorded
           live baseline (183 MiB reference, +50 MiB WARN / +100 MiB RED).
  live     standalone live-artifact volume gate (the "live 工件" metric:
           sum of unique blobs still referenced by HEAD).
  report   JSON census (staged sizes, pack size, live volume, baselines).
  reanchor re-record the pack AND live baselines after a history rewrite
           (see docs/MIGRATION-m23216.md).

Stdlib only; no torch.  Exit codes: 0 = pass (warnings allowed),
1 = gate failed, 2 = usage/environment error.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(os.environ.get("FVM_GUARD_REPO_ROOT",
                            Path(__file__).resolve().parents[2]))
BASELINE_PATH = REPO_ROOT / "scripts" / "guard" / "pack_baseline.json"
LIVE_BASELINE_PATH = REPO_ROOT / "scripts" / "guard" / "live_baseline.json"

# Acceptance (feature_ledger FVM-SLIM-GUARD): 512 KiB blob interception.
DEFAULT_BLOB_LIMIT = 512 * 1024
# Drift budget vs. pack baseline: warn at +50 MiB, hard fail at +150 MiB.
DEFAULT_PACK_WARN_DELTA = 50 * 1024 * 1024
DEFAULT_PACK_HARD_DELTA = 150 * 1024 * 1024
# Absolute backstop regardless of baseline.
DEFAULT_PACK_HARD_CAP = 2 * 1024 * 1024 * 1024

# LIVE artifact volume gate (m23216 SLIM_PLAN §2.3 / FVM-SLIM-GUARD R4):
#   "live 工件" = unique blobs still referenced by HEAD (the working set
#   that grows with each commit).  Reference baseline 183 MiB (m23216
#   census: 183.2 MiB live at HEAD 2548a94e); +50 MiB WARN / +100 MiB RED.
#   The baseline is re-anchorable to the actual measured value (see
#   docs/MIGRATION-m23216.md §2 and the PR note).
DEFAULT_LIVE_BASELINE = 183 * 1024 * 1024
DEFAULT_LIVE_WARN_DELTA = 50 * 1024 * 1024
DEFAULT_LIVE_RED_DELTA = 100 * 1024 * 1024

# Custom git attribute that whitelists a path from the 512 KiB blob gate.
#   Set in .gitattributes, e.g.:
#     docs/verification-*/**/*.npz  guard-allow-large
#   A staged file matching this attribute bypasses the blob limit (the
#   artifact is a deliberate, reproducible producer output).
WHITELIST_ATTR = "guard-allow-large"

_HUMAN_RE = re.compile(r"^\s*([\d.]+)\s*(B|bytes|KiB|MiB|GiB|TiB)\s*$")
_UNIT = {"B": 1, "bytes": 1, "KiB": 1024, "MiB": 1024**2,
        "GiB": 1024**3, "TiB": 1024**4}


def _git(*args: str, check: bool = True) -> str:
    proc = subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True
    )
    if check and proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout


def human_size(nbytes: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(nbytes) < 1024 or unit == "TiB":
            return f"{nbytes:.1f} {unit}" if unit != "B" else f"{int(nbytes)} B"
        nbytes /= 1024
    raise AssertionError("unreachable")


def parse_human_size(text: str) -> int:
    m = _HUMAN_RE.match(text or "")
    if not m:
        raise ValueError(f"unparseable git size: {text!r}")
    return int(float(m.group(1)) * _UNIT[m.group(2)])


def pack_size_bytes() -> tuple[int, int]:
    """Return (reachable pack bytes, garbage bytes) from count-objects -vH."""
    out = _git("count-objects", "-vH")
    pack = garbage = 0
    for line in out.splitlines():
        key, _, val = line.partition(":")
        key = key.strip()
        if key == "size-pack":
            pack = parse_human_size(val)
        elif key == "size-garbage":
            garbage = parse_human_size(val)
    return pack, garbage


def live_artifact_bytes() -> int:
    """Sum of unique blobs still referenced by HEAD (the 'live 工件' set).

    Uses `git ls-tree -r -l HEAD` (fast, no object walk) and dedups by blob
    sha so a file referenced from multiple paths counts once.  This is the
    metric that grows with each commit's new artifacts and is the target of
    the live volume gate (m23216 SLIM_PLAN §2.3: 183.2 MiB live).
    """
    out = _git("ls-tree", "-r", "-l", "HEAD")
    seen: set[str] = set()
    total = 0
    for line in out.splitlines():
        # <mode> <type> <sha> <size>\t<path>
        if "\t" not in line:
            continue
        meta, _path = line.split("\t", 1)
        fields = meta.split()
        if len(fields) < 4:
            continue
        sha = fields[2]
        size = int(fields[3])
        if sha not in seen:
            seen.add(sha)
            total += size
    return total


def staged_blob_sizes() -> list[tuple[str, int]]:
    """[(path, size)] for every staged (added/modified/renamed) blob."""
    out = _git("diff", "--cached", "--raw")
    sizes: list[tuple[str, int]] = []
    for line in out.splitlines():
        if "\t" not in line:
            continue
        meta, path = line.split("\t", 1)
        fields = meta.split()
        if len(fields) < 4:
            continue
        new_sha = fields[3]
        if set(new_sha) == {"0"}:
            continue  # deletion
        size = int(_git("cat-file", "-s", new_sha).strip())
        sizes.append((path, size))
    return sizes


def is_whitelisted(path: str) -> bool:
    """True if the path matches the `guard-allow-large` git attribute."""
    out = _git("check-attr", WHITELIST_ATTR, "--", path, check=False)
    # e.g. "path: guard-allow-large: set"
    for line in out.splitlines():
        if line.endswith(f": {WHITELIST_ATTR}: set"):
            return True
    return False


def changed_files(base: str) -> list[str]:
    out = _git("diff", "--name-only", f"{base}..HEAD", check=False)
    if out is None or out == "":
        return []
    return [p for p in out.splitlines() if p]


def resolve_base() -> str | None:
    for cand in ("origin/main", "HEAD~1"):
        rc = subprocess.run(
            ["git", "rev-parse", "--verify", "--quiet", cand],
            cwd=REPO_ROOT, capture_output=True,
        ).returncode
        if rc == 0:
            return cand
    return None


def load_baseline(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def evaluate_pack(pack: int, baseline: dict | None,
                 warn_delta: int, hard_delta: int, hard_cap: int) -> dict:
    """Pure pack threshold logic (unit-testable without a repo)."""
    res: dict = {"pack_bytes": pack, "pack_human": human_size(pack)}
    if baseline and "pack_bytes" in baseline:
        base = int(baseline["pack_bytes"])
        drift = pack - base
        res["baseline_bytes"] = base
        res["drift_bytes"] = drift
        res["drift_human"] = human_size(abs(drift)) + ("+" if drift >= 0 else "-")
        res["warn_at"] = base + warn_delta
        res["hard_at"] = base + hard_delta
        if pack > base + hard_delta:
            res["pack_status"] = "FAIL"
        elif pack > base + warn_delta:
            res["pack_status"] = "WARN"
        else:
            res["pack_status"] = "OK"
    else:
        res["pack_status"] = "OK" if pack <= hard_cap else "FAIL"
        res["hard_cap_bytes"] = hard_cap
    if pack > hard_cap:
        res["pack_status"] = "FAIL"
    return res


def evaluate_live(live: int, baseline: dict | None,
                  warn_delta: int, red_delta: int) -> dict:
    """Pure live-artifact volume threshold logic (unit-testable).

    live = unique HEAD-referenced blob total.  Baseline is the recorded
    'live 工件' reference (default 183 MiB, re-anchorable).  WARN at
    baseline + warn_delta, RED (hard fail) at baseline + red_delta.
    """
    res: dict = {"live_bytes": live, "live_human": human_size(live)}
    if baseline and "live_bytes" in baseline:
        base = int(baseline["live_bytes"])
        drift = live - base
        res["baseline_bytes"] = base
        res["drift_bytes"] = drift
        res["drift_human"] = human_size(abs(drift)) + ("+" if drift >= 0 else "-")
        res["warn_at"] = base + warn_delta
        res["red_at"] = base + red_delta
        if live > base + red_delta:
            res["live_status"] = "RED"
        elif live > base + warn_delta:
            res["live_status"] = "WARN"
        else:
            res["live_status"] = "OK"
    else:
        # No recorded baseline -> fall back to the frozen reference.
        base = DEFAULT_LIVE_BASELINE
        res["baseline_bytes"] = base
        res["drift_bytes"] = live - base
        res["warn_at"] = base + warn_delta
        res["red_at"] = base + red_delta
        if live > base + red_delta:
            res["live_status"] = "RED"
        elif live > base + warn_delta:
            res["live_status"] = "WARN"
        else:
            res["live_status"] = "OK"
    return res


def cmd_staged(limit: int) -> int:
    if os.environ.get("FVM_GUARD_ALLOW_LARGE") == "1":
        print("repo_guard: FVM_GUARD_ALLOW_LARGE=1 -> blob gate bypassed "
              "(justification must be in the commit message)")
        return 0
    offenders = []
    for p, s in staged_blob_sizes():
        if s > limit and not is_whitelisted(p):
            offenders.append((p, s))
    if offenders:
        print(f"repo_guard: {len(offenders)} staged blob(s) exceed "
              f"{human_size(limit)} (FVM-SLIM-GUARD):", file=sys.stderr)
        for p, s in sorted(offenders, key=lambda x: -x[1]):
            print(f"  {human_size(s):>10}  {p}", file=sys.stderr)
        print("repo_guard: commit blocked.  Options: keep the artifact out of "
              "git (outputs/ is gitignored), shrink it, whitelist the path "
              "via .gitattributes `guard-allow-large`, or bypass with "
              "FVM_GUARD_ALLOW_LARGE=1 and justify in the commit message.",
              file=sys.stderr)
        return 1
    print(f"repo_guard: staged blobs OK (limit {human_size(limit)})")
    return 0


def cmd_ci(blob_limit: int, warn_delta: int, hard_delta: int,
           hard_cap: int, live_warn_delta: int, live_red_delta: int) -> int:
    failures: list[str] = []
    warnings: list[str] = []

    # (a) blob gate over the diff base..HEAD
    base = resolve_base()
    if base is None:
        warnings.append("blob gate skipped: no resolvable base (shallow clone?)")
    else:
        bad = []
        for path in changed_files(base):
            size = subprocess.run(
                ["git", "cat-file", "-s", f"HEAD:{path}"],
                cwd=REPO_ROOT, capture_output=True, text=True,
            )
            if size.returncode != 0:
                continue  # deleted in range
            s = int(size.stdout.strip())
            if s > blob_limit and not is_whitelisted(path):
                bad.append((path, s))
        if bad:
            failures.append(
                "blob gate: " + "; ".join(
                    f"{p} ({human_size(s)})" for p, s in
                    sorted(bad, key=lambda x: -x[1])[:10]
                )
            )
        else:
            print(f"repo_guard: blob gate OK vs {base} "
                  f"(limit {human_size(blob_limit)})")

    # (b) pack drift gate
    pack, garbage = pack_size_bytes()
    ev = evaluate_pack(pack, load_baseline(BASELINE_PATH),
                       warn_delta, hard_delta, hard_cap)
    res = dict(ev)
    res["garbage_bytes"] = garbage
    print(json.dumps(res, indent=2))
    if ev["pack_status"] == "FAIL":
        failures.append(f"pack gate: {ev['pack_human']} exceeds threshold")
    elif ev["pack_status"] == "WARN":
        warnings.append(f"pack gate: {ev['pack_human']} (drift "
                        f"{ev.get('drift_human', '?')})")

    # (c) LIVE artifact volume gate
    live = live_artifact_bytes()
    lev = evaluate_live(live, load_baseline(LIVE_BASELINE_PATH),
                        live_warn_delta, live_red_delta)
    lres = dict(lev)
    print(json.dumps(lres, indent=2))
    if lev["live_status"] == "RED":
        failures.append(f"live gate: {lev['live_human']} exceeds RED threshold "
                        f"(baseline {human_size(lev['baseline_bytes'])}, "
                        f"+{human_size(live_red_delta)})")
    elif lev["live_status"] == "WARN":
        warnings.append(f"live gate: {lev['live_human']} (drift "
                        f"{lev.get('drift_human', '?')})")

    for w in warnings:
        print(f"repo_guard: WARNING: {w}", file=sys.stderr)
    if failures:
        for f in failures:
            print(f"repo_guard: FAIL: {f}", file=sys.stderr)
        return 1
    print("repo_guard: CI size gate passed")
    return 0


def cmd_live(live_warn_delta: int, live_red_delta: int) -> int:
    """Standalone live-artifact volume gate."""
    live = live_artifact_bytes()
    lev = evaluate_live(live, load_baseline(LIVE_BASELINE_PATH),
                        live_warn_delta, live_red_delta)
    print(json.dumps(lev, indent=2))
    if lev["live_status"] == "RED":
        print(f"repo_guard: FAIL: live gate {lev['live_human']} exceeds RED "
              f"threshold", file=sys.stderr)
        return 1
    if lev["live_status"] == "WARN":
        print(f"repo_guard: WARNING: live gate {lev['live_human']} (drift "
              f"{lev.get('drift_human', '?')})", file=sys.stderr)
    print("repo_guard: live volume gate passed", file=sys.stderr)
    return 0


def cmd_report() -> int:
    pack, garbage = pack_size_bytes()
    staged = staged_blob_sizes()
    live = live_artifact_bytes()
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "blob_limit_bytes": DEFAULT_BLOB_LIMIT,
        "staged": [{"path": p, "bytes": s} for p, s in staged],
        "pack_bytes": pack,
        "pack_human": human_size(pack),
        "garbage_bytes": garbage,
        "live_bytes": live,
        "live_human": human_size(live),
        "pack_baseline": load_baseline(BASELINE_PATH),
        "live_baseline": load_baseline(LIVE_BASELINE_PATH),
    }
    print(json.dumps(report, indent=2))
    return 0


def cmd_reanchor() -> int:
    pack, _ = pack_size_bytes()
    BASELINE_PATH.write_text(json.dumps({
        "pack_bytes": pack,
        "pack_human": human_size(pack),
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "note": "FVM-SLIM-GUARD pack baseline; re-anchor after any history "
                "rewrite (docs/MIGRATION-m23216.md)",
    }, indent=2) + "\n")
    live = live_artifact_bytes()
    LIVE_BASELINE_PATH.write_text(json.dumps({
        "live_bytes": live,
        "live_human": human_size(live),
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "note": "FVM-SLIM-GUARD live-artifact baseline (unique HEAD-referenced "
                "blobs); reference 183 MiB (m23216 census), re-anchor after "
                "any history rewrite (docs/MIGRATION-m23216.md)",
    }, indent=2) + "\n")
    print(f"repo_guard: baselines re-anchored: pack {human_size(pack)}, "
          f"live {human_size(live)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_st = sub.add_parser("staged", help="pre-commit blob gate")
    p_st.add_argument("--limit", type=int, default=DEFAULT_BLOB_LIMIT)
    p_ci = sub.add_parser("ci", help="CI size gate")
    p_ci.add_argument("--blob-limit", type=int, default=DEFAULT_BLOB_LIMIT)
    p_ci.add_argument("--pack-warn-delta", type=int,
                       default=DEFAULT_PACK_WARN_DELTA)
    p_ci.add_argument("--pack-hard-delta", type=int,
                       default=DEFAULT_PACK_HARD_DELTA)
    p_ci.add_argument("--pack-hard-cap", type=int,
                       default=DEFAULT_PACK_HARD_CAP)
    p_ci.add_argument("--live-warn-delta", type=int,
                       default=DEFAULT_LIVE_WARN_DELTA)
    p_ci.add_argument("--live-red-delta", type=int,
                       default=DEFAULT_LIVE_RED_DELTA)
    p_live = sub.add_parser("live", help="live-artifact volume gate")
    p_live.add_argument("--live-warn-delta", type=int,
                        default=DEFAULT_LIVE_WARN_DELTA)
    p_live.add_argument("--live-red-delta", type=int,
                        default=DEFAULT_LIVE_RED_DELTA)
    sub.add_parser("report", help="JSON census")
    sub.add_parser("reanchor", help="re-record pack + live baselines")
    args = ap.parse_args(argv)

    if args.cmd == "staged":
        return cmd_staged(args.limit)
    if args.cmd == "ci":
        return cmd_ci(args.blob_limit, args.pack_warn_delta,
                       args.pack_hard_delta, args.pack_hard_cap,
                       args.live_warn_delta, args.live_red_delta)
    if args.cmd == "live":
        return cmd_live(args.live_warn_delta, args.live_red_delta)
    if args.cmd == "report":
        return cmd_report()
    if args.cmd == "reanchor":
        return cmd_reanchor()
    return 2


if __name__ == "__main__":
    sys.exit(main())

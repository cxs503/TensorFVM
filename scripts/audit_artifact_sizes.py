#!/usr/bin/env python3
"""Inventory sizes of tracked working-tree files before removing artifacts.

Run from any directory inside a checkout:
    python scripts/audit_artifact_sizes.py
    python scripts/audit_artifact_sizes.py --top 100 --threshold-mb 2

This deliberately does not delete or modify files. It reports current checkout
sizes only; historical Git object sizes require a separate history audit.
"""
from __future__ import annotations

import argparse
import subprocess
from collections import defaultdict
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top", type=int, default=50, help="number of largest files to show")
    parser.add_argument(
        "--threshold-mb", type=float, default=1.0,
        help="flag files at or above this size (MiB)",
    )
    args = parser.parse_args()
    if args.top < 1 or args.threshold_mb <= 0:
        parser.error("--top and --threshold-mb must be positive")

    root = Path(__file__).resolve().parents[1]
    raw = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"],
        check=True, stdout=subprocess.PIPE,
    ).stdout
    paths = [root / item.decode("utf-8", errors="surrogateescape")
             for item in raw.split(b"\0") if item]
    entries: list[tuple[int, str]] = []
    missing: list[str] = []
    directory_bytes: dict[str, int] = defaultdict(int)
    extension_bytes: dict[str, int] = defaultdict(int)
    threshold = int(args.threshold_mb * 1024 * 1024)

    for path in paths:
        if not path.is_file():
            missing.append(path.relative_to(root).as_posix())
            continue
        size = path.stat().st_size
        rel = path.relative_to(root).as_posix()
        entries.append((size, rel))
        parts = Path(rel).parts
        # Aggregate at the first two path components to make large doc bundles visible.
        bucket = "/".join(parts[:2]) if len(parts) > 1 else "."
        directory_bytes[bucket] += size
        extension_bytes[Path(rel).suffix.lower() or "[no extension]"] += size

    total = sum(size for size, _ in entries)
    mib = 1024 * 1024
    print(f"Tracked files present: {len(entries)}")
    print(f"Tracked files missing from checkout: {len(missing)}")
    print(f"Working-tree size of tracked files: {total / mib:.2f} MiB")
    print(f"Large-file threshold: {args.threshold_mb:.2f} MiB")
    print("\nLargest tracked files:")
    for size, rel in sorted(entries, reverse=True)[:args.top]:
        flag = "  [REVIEW]" if size >= threshold else ""
        print(f"{size / mib:9.2f} MiB  {rel}{flag}")

    print("\nLargest top-level / second-level path buckets:")
    for bucket, size in sorted(directory_bytes.items(), key=lambda item: item[1], reverse=True)[:30]:
        print(f"{size / mib:9.2f} MiB  {bucket}")

    print("\nLargest extensions by total size:")
    for ext, size in sorted(extension_bytes.items(), key=lambda item: item[1], reverse=True)[:20]:
        print(f"{size / mib:9.2f} MiB  {ext}")

    if missing:
        print("\nMissing tracked paths (possibly deleted locally):")
        for rel in missing[:50]:
            print(f"  {rel}")
        if len(missing) > 50:
            print(f"  ... and {len(missing) - 50} more")
    print("\nNo files were deleted or modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

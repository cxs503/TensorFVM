"""Fetch open-access restatements of the Woodward-Colella double blast datum.

Saves the raw HTML + a short context dump so numbers used downstream can be
re-checked instead of trusted.  Zero hand-copy: everything is downloaded by
this script and sha256-stamped.
"""
import hashlib
import re
import sys
import urllib.request
from pathlib import Path

OUT = Path(__file__).resolve().parent
UA = {"User-Agent": "TensorFVM-verification/1.0 (research; contact: local)"}


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=90) as fh:
        return fh.read()


def main(arxiv_ids):
    for aid in arxiv_ids:
        for base in (f"https://ar5iv.labs.arxiv.org/html/{aid}",
                     f"https://arxiv.org/abs/{aid}"):
            try:
                raw = fetch(base)
            except Exception as exc:  # noqa: BLE001
                print(f"[skip] {base}: {exc}")
                continue
            name = f"source_{aid.replace('/', '_')}_{'ar5iv' if 'ar5iv' in base else 'abs'}.html"
            (OUT / name).write_bytes(raw)
            print(f"[ok] {base} -> {name} bytes={len(raw)} sha256={hashlib.sha256(raw).hexdigest()}")
            break


if __name__ == "__main__":
    main(sys.argv[1:] or ["2109.14340", "1902.07773"])
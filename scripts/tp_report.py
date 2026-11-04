"""Helper: publish a progress report to the TensorPlatform reports API.

Kept out of the package (scripts/ only). Reads credentials from
``$TENSOR_API_BASE`` / ``$TENSOR_AGENT_USER`` / ``$TENSOR_AGENT_PASS``.

Usage::

    python scripts/tp_report.py report.json        # publish a report
    python scripts/tp_report.py --list 20          # list recent reports
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request


def _post(url: str, payload: dict, token: str | None = None) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())


def _get(url: str, token: str | None = None) -> object:
    req = urllib.request.Request(url, method="GET")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())


def login() -> str:
    base = os.environ["TENSOR_API_BASE"].rstrip("/")
    body = {
        "username": os.environ["TENSOR_AGENT_USER"],
        "password": os.environ["TENSOR_AGENT_PASS"],
    }
    out = _post(f"{base}/auth/login", body)
    return out.get("access_token") or out.get("token")


def main(argv: list[str]) -> int:
    base = os.environ["TENSOR_API_BASE"].rstrip("/")
    token = login()
    if argv and argv[0] == "--list":
        limit = int(argv[1]) if len(argv) > 1 else 20
        out = _get(f"{base}/tensor/reports?limit={limit}", token)
        items = out if isinstance(out, list) else out.get("items", [])
        for r in items:
            print(r.get("id"), r.get("mission_id"), (r.get("title") or "")[:100])
        return 0
    path = argv[0] if argv else "report.json"
    with open(path, "rb") as fh:
        payload = json.load(fh)
    out = _post(f"{base}/tensor/reports", payload, token)
    print(json.dumps(out)[:2000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
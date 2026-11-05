#!/usr/bin/env python3
"""Read-only scan of platform mission reports (limitations/todo sections)."""
import json
import os
import sys
import urllib.request

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1").rstrip("/")
USER = os.environ["TENSOR_AGENT_USER"]
PASS = os.environ["TENSOR_AGENT_PASS"]


def _post(path, payload, token=None):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", **({"Authorization": "Bearer " + token} if token else {})},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def _get(path, token):
    req = urllib.request.Request(BASE + path, headers={"Authorization": "Bearer " + token})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def main():
    tok = None
    for body in ({"username": USER, "password": PASS}, {"email": USER, "password": PASS}):
        try:
            d = _post("/auth/login", body)
            tok = d.get("access_token") or d.get("token") or (d.get("data") or {}).get("access_token")
            if tok:
                break
        except Exception as exc:  # noqa: BLE001
            print("login attempt failed:", exc, file=sys.stderr)
    if not tok:
        print("NO TOKEN", file=sys.stderr)
        return 1
    open("/tmp/tok.txt", "w").write(tok)

    d = _get("/tensor/reports?limit=20", tok)
    items = d.get("items") if isinstance(d, dict) else d
    if items is None:
        print(json.dumps(d)[:2000])
        return 1
    for r in items:
        title = (r.get("title") or "").replace("\n", " ")[:100]
        print(f"{r.get('id')} | m{r.get('mission_id')} | {r.get('scenario_key')} | {title}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
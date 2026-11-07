#!/usr/bin/env python3
"""Verify that the mission report landed: list reports for this mission."""

from __future__ import annotations

import json
import os
import urllib.request

BASE = os.environ["TENSOR_API_BASE"].rstrip("/")


def main() -> int:
    payload = json.dumps({"username": os.environ["TENSOR_AGENT_USER"],
                          "password": os.environ["TENSOR_AGENT_PASS"]}).encode()
    request = urllib.request.Request(f"{BASE}/auth/login", data=payload,
                                     headers={"Content-Type": "application/json"},
                                     method="POST")
    with urllib.request.urlopen(request, timeout=60) as response:
        token = json.loads(response.read())["access_token"]
    for query in ("?mission_id=5420", "?mission_id=5420&limit=5"):
        request = urllib.request.Request(
            f"{BASE}/tensor/reports{query}",
            headers={"Authorization": f"Bearer {token}"}, method="GET")
        with urllib.request.urlopen(request, timeout=60) as response:
            data = json.loads(response.read())
        items = data if isinstance(data, list) else data.get("items", data)
        print(query, "->", json.dumps(items)[:400])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
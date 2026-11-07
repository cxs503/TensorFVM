#!/usr/bin/env python3
"""Publish the report and verify it, with diagnostics.

The report endpoint has a history of answering 200 with somebody else's record
when the payload is rejected, so the post is always followed by a query of this
mission's reports: a receipt alone is not evidence.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

BASE = os.environ["TENSOR_API_BASE"].rstrip("/")


def _post(path: str, payload: dict, token: str | None = None):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(f"{BASE}{path}", data=json.dumps(payload).encode(),
                                     headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            return response.status, json.loads(response.read().decode())
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode()[:500]


def _get(path: str, token: str):
    request = urllib.request.Request(f"{BASE}{path}",
                                     headers={"Authorization": f"Bearer {token}"},
                                     method="GET")
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode())


def main() -> int:
    status, login = _post("/auth/login", {"username": os.environ["TENSOR_AGENT_USER"],
                                          "password": os.environ["TENSOR_AGENT_PASS"]})
    token = login.get("access_token") if isinstance(login, dict) else None
    if not token:
        print("login failed", status, login)
        return 1

    payload = json.loads(open("/tmp/report.json").read())
    status, receipt = _post("/tensor/reports", payload, token)
    print("POST status:", status)
    print("receipt:", json.dumps(receipt)[:300])
    print("payload size KB:", len(json.dumps(payload)) / 1024)

    reports = _get("/tensor/reports?mission_id=5420", token)
    items = reports if isinstance(reports, list) else reports.get("items", [])
    print("reports for mission 5420:", len(items))
    for item in items:
        print("  id", item.get("id"), "|", str(item.get("title"))[:90])
    return 0


if __name__ == "__main__":
    sys.exit(main())
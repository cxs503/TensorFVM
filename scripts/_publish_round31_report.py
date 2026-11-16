#!/usr/bin/env python3
"""Publish the round-31 report to the Tensor platform (login + POST)."""
import json
import os
import urllib.request

BASE = os.environ["TENSOR_API_BASE"].rstrip("/")
USER = os.environ["TENSOR_AGENT_USER"]
PASS = os.environ["TENSOR_AGENT_PASS"]


def _post(path, payload, token=None):
    data = json.dumps(payload, ensure_ascii=False).encode()
    req = urllib.request.Request(BASE + path, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


def main():
    tok = _post("/tensor/auth/login",
               {"username": USER, "password": PASS}).get("token", "")
    rep = json.load(open("/tmp/report.json"))
    out = _post("/tensor/reports", rep, tok)
    print(json.dumps(out, ensure_ascii=False)[:500])


if __name__ == "__main__":
    main()

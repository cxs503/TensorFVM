#!/usr/bin/env python3
"""Publish the m20980 audit report to the platform (/tensor/reports)."""
import json
import os
import urllib.request

BASE = os.environ["TENSOR_API_BASE"]
USER = os.environ["TENSOR_AGENT_USER"]
PW = os.environ["TENSOR_AGENT_PASS"]


def api(path, payload=None, token=None, method=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method or ("POST" if data else "GET"))
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.status, json.loads(r.read() or b"{}")


status, body = api("/auth/login", {"username": USER, "password": PW})
token = body.get("access_token") or body.get("token")
print("login", status, "token_len", len(token or ""))

payload = json.loads(open("/tmp/report.json").read())
status, body = api("/tensor/reports", payload, token)
print("post report", status, json.dumps({k: body.get(k) for k in ("id", "title", "mission_id", "status", "qa_status")}, ensure_ascii=False))
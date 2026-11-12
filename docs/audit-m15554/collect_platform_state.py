#!/usr/bin/env python3
"""Collect the platform-side evidence for the m15458 audit into one JSON file.

Reads the mission row, the mission's report/bench endpoints and the benchmark-case
registry straight from the platform (:8010) and Postgres.  Output is what the audit
report quotes, so the report has no hand-typed platform claims.

Usage: python3 docs/audit-m15554/collect_platform_state.py
"""

from __future__ import annotations

import json
import os
import subprocess
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8010/api/v1"
ENV = dict(l.split("=", 1) for l in open("/root/tp-agent/agent.env")
           if "=" in l and not l.startswith("#"))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "platform_state.json")


def api(method, path, payload=None, token=None):
    req = urllib.request.Request(BASE + path, method=method)
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    data = json.dumps(payload).encode() if payload is not None else None
    try:
        with urllib.request.urlopen(req, data=data, timeout=60) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def psql(sql: str) -> list[list[str]]:
    r = subprocess.run(["psql", "-U", "postgres", "-d", "tensorplatform", "-tA", "-F", "\x01",
                        "-c", sql], capture_output=True, text=True)
    if r.returncode != 0:
        return [["ERROR", r.stderr.strip()[:200]]]
    return [ln.split("\x01") for ln in r.stdout.strip().split("\n") if ln]


def main() -> None:
    st, tok = api("POST", "/auth/login", {"username": ENV["TENSOR_AGENT_USER"].strip(),
                                          "password": ENV["TENSOR_AGENT_PASS"].strip()})
    token = tok.get("access_token")
    state = {"login_status": st, "token_ok": bool(token)}

    st, mission = api("GET", "/tensor/missions/15458", token=token)
    res = (mission or {}).get("result") or {}
    state["mission"] = {
        "http": st, "id": mission.get("id"), "repo": mission.get("repo"),
        "status": mission.get("status"), "updated_at": mission.get("updated_at"),
        "result_keys": sorted(res) if isinstance(res, dict) else None,
        "pr": res.get("pr") if isinstance(res, dict) else None,
        "commit": res.get("commit") if isinstance(res, dict) else None,
        "maturity_gate_round": res.get("maturity_gate_round") if isinstance(res, dict) else None,
        "headline": res.get("headline") if isinstance(res, dict) else None,
        "fail_stage": res.get("fail_stage") if isinstance(res, dict) else None,
        "changed_files": res.get("changed_files") if isinstance(res, dict) else None,
    }

    for name, path in (("bench_results", "/tensor/bench/results?mission_id=15458"),
                       ("l4_verdict", "/tensor/bench/l4-verdict/15458"),
                       ("reports", "/tensor/reports?mission_id=15458")):
        st, d = api("GET", path, token=token)
        state[name] = {"http": st, "data": d if st == 200 else d}

    state["benchmark_cases"] = psql(
        "SELECT id, anchor_slug, repo, status, tolerance FROM tensor_benchmark_cases "
        "WHERE anchor_slug='CASE-CAVITY-RE1000-FVM';")
    state["prior_verdicts"] = psql(
        "SELECT id, repo, scope, mission_id, level, gate_state, gate_level FROM "
        "tensor_maturity_verdicts WHERE scope LIKE '%CAVITY%' OR scope LIKE '%15458%';")
    state["prior_missions"] = psql(
        "SELECT id, status, left(coalesce(task_key,''),40) FROM tensor_dev_missions "
        "WHERE id IN (15369,15458) ORDER BY id;")
    with open(OUT, "w") as fh:
        json.dump(state, fh, indent=1, ensure_ascii=False)
    print(json.dumps(state, indent=1, ensure_ascii=False)[:4000])


if __name__ == "__main__":
    main()
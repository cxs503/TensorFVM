"""Publish the mission report to the Tensor platform report endpoint.

Reads the payload built by ``_build_platform_report.py`` (``/tmp/report.json``),
logs in with the agent credentials taken from the environment, and POSTs the
report.  Credentials are never placed on a command line; a copy of the markdown
is also written next to this script so the report is preserved even if the
endpoint is unreachable.
"""

import json
import os
import sys
import urllib.error
import urllib.request

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1").rstrip("/")
USER = os.environ.get("TENSOR_AGENT_USER", "")
PASSWORD = os.environ.get("TENSOR_AGENT_PASS", "")
HERE = os.path.dirname(os.path.abspath(__file__))
PAYLOAD = "/tmp/report.json"


def request(url: str, body: dict | None = None, token: str | None = None,
            timeout: float = 30.0):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method="POST" if body else "GET")
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.status, response.read().decode()


def login() -> str | None:
    creds = {"username": USER, "password": PASSWORD}
    for path in ("/auth/login", "/login", "/tensor/login", "/users/login"):
        try:
            status, text = request(f"{BASE}{path}", creds)
        except Exception as exc:                      # noqa: BLE001 - probe
            print(f"  {path}: {type(exc).__name__} {exc}")
            continue
        if status < 300:
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                continue
            token = (data.get("token") or data.get("access_token")
                     or data.get("data", {}).get("token"))
            if token:
                print(f"  logged in via {path}")
                return token
        print(f"  {path}: HTTP {status} {text[:120]}")
    return None


def main() -> int:
    payload = json.load(open(PAYLOAD))
    markdown_path = os.path.join(HERE, "REPORT-platform.md")
    with open(markdown_path, "w", encoding="utf-8") as handle:
        handle.write(f"# {payload['title']}\n\n{payload['summary']}\n\n{payload['markdown']}\n")
    print(f"local copy: {markdown_path}")
    token = login()
    if not token:
        print("login failed; report NOT published (local copy kept)")
        return 1
    try:
        status, text = request(f"{BASE}/tensor/reports", payload, token=token, timeout=120)
    except urllib.error.HTTPError as exc:
        print(f"POST failed: HTTP {exc.code} {exc.read().decode()[:300]}")
        return 1
    print(f"POST /tensor/reports -> HTTP {status} {text[:300]}")
    return 0 if status < 300 else 1


if __name__ == "__main__":
    sys.exit(main())
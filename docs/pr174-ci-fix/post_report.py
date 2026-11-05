"""Publish the mission-4492 progress report to the platform (login -> POST)."""
import json
import os
import urllib.error
import urllib.request


def load_env():
    env = dict(os.environ)
    path = "/root/tp-agent/agent.env"
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    return env


env = load_env()
base = env.get("TENSOR_API_BASE", "http://127.0.0.1:8010").rstrip("/")
user = env.get("TENSOR_AGENT_USER") or env.get("TENSOR_USER")
pwd = env.get("TENSOR_AGENT_PASS") or env.get("TENSOR_PASS")
print("base:", base, "| user:", user, "| pass set:", bool(pwd))


def post(url, payload, token=None):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


token = None
for ep in ("/tensor/login", "/auth/login", "/login"):
    st, body = post(base + ep, {"username": user, "password": pwd})
    print(ep, st, body[:160])
    if st == 200:
        try:
            token = json.loads(body).get("access_token") or json.loads(body).get("token")
        except Exception:
            token = None
        if token:
            break
if not token:
    raise SystemExit("no token")

report = json.load(open("/tmp/report.json"))
st, body = post(base + "/tensor/reports", report, token)
print("POST /tensor/reports ->", st, body[:300])
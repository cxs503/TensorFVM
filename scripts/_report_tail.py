#!/usr/bin/env python3
"""Read-only: print the limitations/todo sections of given platform report ids."""
import json
import os
import re
import sys
import urllib.request

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1").rstrip("/")
TOK = open("/tmp/tok.txt").read().strip() if os.path.exists("/tmp/tok.txt") else None
if not TOK:
    raise SystemExit("no token; run _reports_scan.py first")


def get(path):
    req = urllib.request.Request(BASE + path, headers={"Authorization": "Bearer " + TOK})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def main(ids):
    for rid in ids:
        d = get(f"/tensor/reports/{rid}")
        d = d.get("data", d)
        md = d.get("markdown") or ""
        title = d.get("title", "")
        print("=" * 90)
        print(rid, title[:100])
        # print sections whose heading mentions 局限/todo/下一步/未完成
        blocks = re.split(r"(?m)^(#{1,6} .*)$", md)
        for i in range(1, len(blocks) - 1, 2):
            head, body = blocks[i], blocks[i + 1]
            if re.search(r"局限|待办|下一步|未完成|欠账|next", head, re.I):
                print(head)
                print(body.strip()[:1800])
    return 0


if __name__ == "__main__":
    sys.exit(main([int(a) for a in sys.argv[1:]]))
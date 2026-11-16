#!/usr/bin/env python3
"""FVM-REPO-SLIM 复核普查（m23718）：对当前对象库做只读普查，输出 JSON。

与 m23216 普查的区别：
- 增加「垃圾 pack（tmp_pack_*）」维度——中断 fetch 留下的孤儿 pack 会虚增
  ``git count-objects`` 的 size-garbage，但不属于任何 ref 可达集；
- 分支维度改读本地 remote-tracking refs（ls-remote 在本环境超时，不可用）；
- 全部数字程序化注入，零手抄。

用法:
    python3 scripts/audit/m23718_git_census.py --out docs/FVM-REPO-SLIM-M23718/census.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from collections import defaultdict

MiB = 1024 * 1024
GiB = 1024 * MiB


def git(repo: str, *args: str, stdin: bytes | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", repo, *args],
        input=stdin, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )


def count_objects(repo: str) -> dict:
    cp = git(repo, "count-objects", "-vH")
    out: dict = {}
    for line in cp.stdout.decode().splitlines():
        k, _, v = line.partition(":")
        out[k.strip()] = v.strip()
    return out


def parse_h(v: str) -> float:
    """'1.33 GiB' / '4.20 MiB' -> MiB。"""
    num, _, unit = v.partition(" ")
    mult = {"B": 1 / MiB, "KiB": 1 / 1024, "MiB": 1.0, "GiB": 1024.0, "TiB": 1024 * 1024}
    return float(num) * mult.get(unit, 1.0)


def ext_of(path: str) -> str:
    base = path.rsplit("/", 1)[-1]
    if base.startswith(".") and base.count(".") == 1:
        return base
    i = base.rfind(".")
    return base[i:].lower() if i > 0 else "(none)"


def reachable_census(repo: str, rev: str) -> dict:
    cp = git(repo, "rev-list", "--objects", rev)
    if cp.returncode != 0:
        raise SystemExit(f"rev-list {rev} failed: {cp.stderr.decode()[:300]}")
    lines = cp.stdout.decode().splitlines()
    oids = [ln.split()[0] for ln in lines if ln]
    feed = ("\n".join(oids) + "\n").encode()
    cb = git(repo, "cat-file",
             "--batch-check=%(objectname) %(objecttype) %(objectsize)", stdin=feed)
    info = {}
    for ln in cb.stdout.decode().splitlines():
        p = ln.split()
        if len(p) == 3 and p[1] != "missing":
            info[p[0]] = (p[1], int(p[2]))
    by_ext: dict[str, dict] = defaultdict(lambda: {"n": 0, "bytes": 0})
    by_type: dict[str, dict] = defaultdict(lambda: {"n": 0, "bytes": 0})
    top: list[tuple] = []
    for ln in lines:
        if not ln:
            continue
        oid, path = ln[:40], ln[41:].lstrip(" ")
        if not path:
            path = ""
        t, size = info.get(oid, ("?", 0))
        by_type[t]["n"] += 1
        by_type[t]["bytes"] += size
        if t == "blob":
            e = ext_of(path)
            by_ext[e]["n"] += 1
            by_ext[e]["bytes"] += size
            top.append((size, path, oid))
    top.sort(reverse=True)
    return {
        "rev": rev,
        "n_objects": len(oids),
        "total_MiB": round(sum(s for _, s in info.values()) / MiB, 2),
        "by_type": {k: {"n": v["n"], "MiB": round(v["bytes"] / MiB, 2)}
                    for k, v in sorted(by_type.items())},
        "blob_by_ext": {k: {"n": v["n"], "MiB": round(v["bytes"] / MiB, 2)}
                        for k, v in sorted(by_ext.items(),
                                          key=lambda kv: -kv[1]["bytes"])},
        "top15_blobs": [{"MiB": round(s / MiB, 2), "path": p, "oid": o[:12]}
                        for s, p, o in top[:15]],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    repo = args.repo
    t0 = time.time()

    co = count_objects(repo)
    garbage = []
    packdir = os.path.join(git(repo, "rev-parse", "--git-common-dir").stdout.decode().strip(),
                           "objects", "pack")
    if os.path.isdir(packdir):
        for f in sorted(os.listdir(packdir)):
            if f.startswith("tmp_pack_"):
                garbage.append({"file": f, "MiB": round(os.path.getsize(os.path.join(packdir, f)) / MiB, 1)})

    refs = git(repo, "for-each-ref", "refs/remotes", "--format=%(refname)")
    branches = [r for r in refs.stdout.decode().splitlines() if r]
    agent = [r for r in branches if "/agent/" in r]
    maintenance = [r for r in branches if "/maintenance/" in r]

    ver = git(repo, "rev-list", "--count", "main", "--", "docs/verification-*")
    ver_all = git(repo, "rev-list", "--count", "--all", "--", "docs/verification-*")

    out = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "repo": os.path.abspath(repo),
        "head": git(repo, "rev-parse", "--short", "HEAD").stdout.decode().strip(),
        "head_date": git(repo, "log", "-1", "--format=%ci", "main").stdout.decode().strip(),
        "count_objects": co,
        "packs_MiB": parse_h(co.get("size-pack", "0 B")),
        "garbage_MiB": parse_h(co.get("size-garbage", "0 B")),
        "garbage_tmp_packs": garbage,
        "remote_branches": {
            "total": len(branches),
            "agent": len(agent),
            "maintenance": len(maintenance),
            "maintenance_list": maintenance,
        },
        "commits_touching_verification_main": int(ver.stdout.decode() or 0),
        "commits_touching_verification_all": int(ver_all.stdout.decode() or 0),
        "main": reachable_census(repo, "main"),
        "all": reachable_census(repo, "--all"),
        "elapsed_s": round(time.time() - t0, 1),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)
    print(f"[i] census -> {args.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()

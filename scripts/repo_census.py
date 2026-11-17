#!/usr/bin/env python3
"""TensorFVM git 大对象普查（只读）。

用法: python3 scripts/repo_census.py <repo路径> <输出目录>
产出:
  object_type_summary.csv   按对象类型聚合
  top_blobs.csv             Top-N 大 blob（含路径、引入提交、可达性）
  by_extension.csv          按扩展名聚合
  by_topdir.csv             按一级目录聚合
  reachability.csv          main 可达 vs 仅分支/PR ref 可达
只读操作: rev-list / cat-file / log --find-object，不写任何 git 元数据。
"""
import csv
import os
import subprocess
import sys
from collections import defaultdict

REPO = sys.argv[1] if len(sys.argv) > 1 else "."
OUT = sys.argv[2] if len(sys.argv) > 2 else "outputs/repo_slim_census"
TOP_N = 40
os.makedirs(OUT, exist_ok=True)


def git(*args, text=False):
    r = subprocess.run(["git", "-C", REPO, *args], capture_output=True, text=text)
    if r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {r.stderr[:400]}")
    return r.stdout if text else r.stdout


def git_text(*args):
    return git(*args, text=True)


# 1) 全对象尺寸（batch-all-objects 含不可达对象；promisor 缺失对象跳过）
size_by_sha = {}
with subprocess.Popen(["git", "-C", REPO, "cat-file", "--batch-all-objects",
                       "--batch-check=%(objectname) %(objecttype) %(objectsize)"],
                      stdout=subprocess.PIPE, text=True) as p:
    for line in p.stdout:
        parts = line.split()
        if len(parts) == 3 and parts[1] == "blob":
            size_by_sha[parts[0]] = int(parts[2])

# 2) 可达对象 -> 路径映射（--all = 全部 refs）
sha_paths = defaultdict(set)
for line in git_text("rev-list", "--objects", "--all").splitlines():
    if not line:
        continue
    sha, _, path = line.partition(" ")
    if path:
        sha_paths[sha].add(path)

# 3) main 可达集合（用于可达性二分）
main_ref = git_text("rev-parse", "--verify", "origin/main").strip()
main_reachable = set()
for line in git_text("rev-list", "--objects", main_ref).splitlines():
    main_reachable.add(line.split(" ", 1)[0])

rows = []
for sha, size in size_by_sha.items():
    paths = sorted(sha_paths.get(sha, ["<unreachable>"]))
    rows.append({
        "sha": sha, "size": size,
        "in_main": sha in main_reachable,
        "paths": " | ".join(paths[:3]) + (" ..." if len(paths) > 3 else ""),
        "path0": paths[0],
    })
rows.sort(key=lambda r: -r["size"])

# 4) 引入提交（对 Top-N 逐个 log --find-object，只读）
first_commit = {}
for r in rows[:TOP_N]:
    out = git_text("log", "--all", "--find-object", r["sha"],
                   "--format=%H %ad %s", "--date=short", "--reverse")
    first_commit[r["sha"]] = out.splitlines()[0][:100] if out.strip() else "?"

with open(f"{OUT}/top_blobs.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["rank", "sha", "size_MiB", "in_origin_main", "path", "first_introducing_commit"])
    for i, r in enumerate(rows[:TOP_N], 1):
        w.writerow([i, r["sha"], f"{r['size']/1048576:.2f}", r["in_main"],
                    r["path0"], first_commit.get(r["sha"], "")])

# 5) 按扩展名 / 一级目录聚合
def ext_of(path):
    base = os.path.basename(path)
    if base.startswith("."):
        return base
    e = os.path.splitext(base)[1].lower()
    return e if e else "<no-ext>"

def topdir_of(path):
    return path.split("/")[0] if "/" in path else "<root>"

by_ext = defaultdict(lambda: [0, 0])
by_dir = defaultdict(lambda: [0, 0])
for r in rows:
    if r["path0"] == "<unreachable>":
        continue
    by_ext[ext_of(r["path0"])][0] += 1
    by_ext[ext_of(r["path0"])][1] += r["size"]
    by_dir[topdir_of(r["path0"])][0] += 1
    by_dir[topdir_of(r["path0"])][1] += r["size"]

with open(f"{OUT}/by_extension.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["extension", "blobs", "logical_MiB"])
    for k, (n, s) in sorted(by_ext.items(), key=lambda kv: -kv[1][1]):
        w.writerow([k, n, f"{s/1048576:.2f}"])

with open(f"{OUT}/by_topdir.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["topdir", "blobs", "logical_MiB"])
    for k, (n, s) in sorted(by_dir.items(), key=lambda kv: -kv[1][1])[:25]:
        w.writerow([k, n, f"{s/1048576:.2f}"])

# 6) 可达性二分：main 可达 vs 仅其它 ref 可达
reach = {"main_only_size": 0, "main_only_n": 0, "other_refs_size": 0, "other_refs_n": 0}
for r in rows:
    if r["path0"] == "<unreachable>":
        continue
    if r["in_main"]:
        reach["main_only_size"] += r["size"]; reach["main_only_n"] += 1
    else:
        reach["other_refs_size"] += r["size"]; reach["other_refs_n"] += 1

with open(f"{OUT}/reachability.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["class", "blobs", "logical_MiB"])
    w.writerow(["reachable_from_origin_main", reach["main_only_n"], f"{reach['main_only_size']/1048576:.2f}"])
    w.writerow(["only_via_agent_or_pr_refs", reach["other_refs_n"], f"{reach['other_refs_size']/1048576:.2f}"])

# 7) 对象类型汇总
type_agg = defaultdict(lambda: [0, 0])
for line in git_text("cat-file", "--batch-all-objects",
                     "--batch-check=%(objecttype) %(objectsize)").splitlines():
    parts = line.split()
    if len(parts) == 2:
        type_agg[parts[0]][0] += 1
        type_agg[parts[0]][1] += int(parts[1])
with open(f"{OUT}/object_type_summary.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["object_type", "count", "logical_MiB"])
    for k, (n, s) in sorted(type_agg.items(), key=lambda kv: -kv[1][1]):
        w.writerow([k, n, f"{s/1048576:.2f}"])

print(f"census done -> {OUT}")
print(f"top blob: {rows[0]['size']/1048576:.2f} MiB {rows[0]['path0']}")
print(f"main-reachable blob logical: {reach['main_only_size']/1048576:.1f} MiB / "
      f"other-refs-only: {reach['other_refs_size']/1048576:.1f} MiB")

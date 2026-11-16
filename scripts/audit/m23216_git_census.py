#!/usr/bin/env python3
"""TensorFVM git-history census + slimming-plan generator (mission m23216).

Read-only by construction: never rewrites history, never deletes objects, never
pushes.  It measures the object database and emits

    docs/m23216/SLIM_PLAN.md      plan document rendered from scripts/audit/m23216_slim_plan.md.tpl
    docs/m23216/census.json       machine-readable census (all injected numbers)
    docs/m23216/tables/*.md|csv   tables
    docs/m23216/figures/*.png     figures
    runs/m23216/census.log        run record

Usage:
    python3 scripts/audit/m23216_git_census.py --out docs/m23216 --runs runs/m23216
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import fnmatch
import json
import os
import re
import string
import subprocess
import zlib

MIB = 2.0 ** 20
GIT = ["git", "-c", "remote.origin.promisor=false"]  # never lazy-fetch during a census
LOG = []


def log(msg):
    line = "[%s] %s" % (dt.datetime.now().strftime("%H:%M:%S"), msg)
    print(line, flush=True)
    LOG.append(line)


def sh(args, timeout=None):
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=timeout).stdout
    except subprocess.TimeoutExpired:
        log("TIMEOUT after %ss: %s (value marked n/a)" % (timeout, " ".join(args[:4])))
        return ""


def ext_of(path):
    base = os.path.basename(path)
    return "(noext)" if "." not in base else "." + base.rsplit(".", 1)[1].lower()


def batch_blobs(hashes):
    if not hashes:
        return {}
    pb = subprocess.run(["git", "cat-file", "--batch"],
                        input=("\n".join(hashes) + "\n").encode(), capture_output=True)
    buf, pos, res = pb.stdout, 0, {}
    while pos < len(buf):
        nl = buf.index(b"\n", pos)
        header = buf[pos:nl].decode()
        pos = nl + 1
        if header.endswith("missing"):
            continue
        sha, _typ, size = header.split()
        size = int(size)
        res[sha] = buf[pos:pos + size]
        pos += size + 1
    return res


# --------------------------------------------------------------- environment
def collect_objects():
    log("cat-file --batch-all-objects ...")
    sizes, types = {}, {}
    for line in sh(["git", "cat-file", "--batch-all-objects",
                    "--batch-check=%(objectname) %(objecttype) %(objectsize)"]).splitlines():
        h, t, s = line.split()
        sizes[h], types[h] = int(s), t
    log("local ODB: %d objects" % len(sizes))
    return sizes, types


def reach_map(revs):
    log("rev-list --objects %s ..." % revs)
    out = sh(GIT + ["rev-list", "--objects", revs])
    m = collections.defaultdict(set)
    for line in out.splitlines():
        parts = line.split(" ", 1)
        if len(parts) == 2:
            m[parts[0]].add(parts[1])
    return m


def pack_stats(gd):
    pdir = os.path.join(gd, "objects", "pack")
    packs, tmps = [], []
    for name in sorted(os.listdir(pdir)):
        p = os.path.join(pdir, name)
        if name.endswith(".pack"):
            packs.append((name, os.path.getsize(p)))
        elif name.startswith("tmp_pack"):
            tmps.append((name, os.path.getsize(p),
                         dt.datetime.fromtimestamp(os.path.getmtime(p))))
    packs.sort(key=lambda x: -x[1])
    tmps.sort(key=lambda x: -x[1])
    return {"packs": packs, "tmp": tmps, "pack_total": sum(s for _, s in packs),
            "tmp_total": sum(s for _, s, _ in tmps)}


def ref_stats():
    local = sh(["git", "for-each-ref", "--format=%(refname)"]).splitlines()
    buckets = collections.Counter(r.split("/")[1] for r in local if r.count("/") >= 1)
    remote = collections.Counter()
    if os.environ.get("SLIM_OFFLINE") != "1":
        for line in sh(["git", "ls-remote", "origin"], timeout=25).splitlines():
            remote["/".join(line.split()[1].split("/")[:2])] += 1
    wt = sh(["git", "worktree", "list", "--porcelain"]).split("worktree ")[1:]
    return {"local_refs": len(local), "local_buckets": dict(buckets),
            "remote_buckets": dict(remote), "worktrees": len(wt),
            "worktrees_prunable": sum(1 for w in wt if "prunable" in w)}


def gh_repo_size():
    if os.environ.get("SLIM_OFFLINE") == "1":
        return 0
    out = subprocess.run(["curl", "-s", "-m", "15", "https://api.github.com/repos/cxs503/TensorFVM"],
                         capture_output=True, text=True).stdout
    try:
        return int(json.loads(out)["size"])
    except Exception:
        return 0


def gh_open_prs():
    if os.environ.get("SLIM_OFFLINE") == "1":
        return -1
    out = subprocess.run(["curl", "-s", "-m", "15", "https://api.github.com/repos/cxs503/TensorFVM"
                                 "/pulls?state=open&per_page=100"],
                         capture_output=True, text=True).stdout
    try:
        return len(json.loads(out))
    except Exception:
        return -1


# ------------------------------------------------------- compression model
def stream_compressed_sizes(hashes):
    """Stream every blob through `git cat-file --batch`, return {hash: zlib bytes}.

    Deterministic and exact (no sampling): the scenario estimates then sum the
    real zlib-compressed size of each retained blob instead of a per-extension
    mean, which removes the sampling variance that made early runs drift.
    """
    # promisor=false: never trigger a lazy fetch here (would block on network)
    proc = subprocess.Popen(["git", "-c", "remote.origin.promisor=false", "cat-file", "--batch"],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    assert proc.stdin is not None and proc.stdout is not None
    proc.stdin.write(("\n".join(hashes) + "\n").encode())
    proc.stdin.flush()
    out = {}
    done = 0
    for want in hashes:
        header = proc.stdout.readline()
        if not header:
            break
        parts = header.split()
        if len(parts) < 3:
            continue
        size = int(parts[2])
        data = proc.stdout.read(size)
        proc.stdout.read(1)  # trailing newline
        out[want] = len(zlib.compress(data, 6))
        done += 1
        if done % 4000 == 0:
            log("  compressed %d/%d blobs" % (done, len(hashes)))
    proc.stdin.close()
    proc.wait()
    return out


# --------------------------------------------------------------- scenarios
FIELD = ["*.npz", "*.npy", "*.vtk", "*.vtu", "*.msh", "*.pt", "*.dat", "*.gz", "*.csv"]
BIN = FIELD + ["*.pdf", "*.png", "*.svg"]
SCEN = [
    ("R1", "只删 docs 场数据(npz/npy/vtk/vtu/msh/pt/dat/gz/csv)", [("suffix", FIELD, "docs/")]),
    ("R2", "删场数据+网格+PDF，保留 png/svg 图", [("suffix", FIELD + ["*.pdf"], "docs/")]),
    ("R2b", "R2 + 全仓 *.pyc", [("suffix", FIELD + ["*.pdf"], "docs/"), ("suffix", ["*.pyc"], "*")]),
    ("R2c", "R2 + 删 docs *.svg（几乎全为 dead 图）", [("suffix", FIELD + ["*.pdf", "*.svg"], "docs/"),
                                                   ("suffix", ["*.pyc"], "*")]),
    ("R3", "删 docs 全部二进制 + 全仓 *.pyc", [("suffix", BIN, "docs/"), ("suffix", ["*.pyc"], "*")]),
    ("R5", "阈值：删 docs 下 >=1 MiB 的 blob", [("size", 1 * MIB, "docs/")]),
    ("R5b", "阈值：删 docs 下 >=512 KiB 的 blob", [("size", 512 * 1024, "docs/")]),
]


def _rule_fn(rule):
    kind, val, scope = rule

    def fn(path, size):
        if scope != "*" and not path.startswith(scope):
            return False
        if kind == "suffix":
            return any(fnmatch.fnmatch(os.path.basename(path), pt) for pt in val)
        return size >= val
    return fn


def make_pred(rules):
    fns = [_rule_fn(r) for r in rules]

    def pred(path, size):
        return any(f(path, size) for f in fns)
    return pred


def scenario_rows(A, first_path, ratios, tree_b, commit_b, now_pack, comp=None, cal=1.0):
    rows = []
    for tag, desc, rules in SCEN:
        pred = make_pred(rules)
        drop = nd = keep = 0
        keepc = 0.0
        for h, s in A.items():
            p = first_path[h]
            if pred(p, s):
                drop += s
                nd += 1
            else:
                keep += s
                keepc += comp.get(h, s * ratios.get(ext_of(p), 0.55)) if comp else s * ratios.get(ext_of(p), 0.55)
        est = (keepc + tree_b * 0.55 + commit_b * 0.45) / MIB * cal
        rows.append(dict(tag=tag, desc=desc, dropped_MiB=round(drop / MIB, 1), nblobs=nd,
                         kept_MiB=round(keep / MIB, 1), est_pack_MiB=round(est, 1),
                         reduction_pct=round(100 * (1 - est / now_pack), 1)))
    return rows


# ----------------------------------------------------------------- figures
def make_figures(outdir, ext_rows, timeline, scen_rows, now_pack, gh_mib):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fdir = os.path.join(outdir, "figures")
    os.makedirs(fdir, exist_ok=True)
    made = []

    top = ext_rows[:12]
    labels = [r["ext"] for r in top][::-1]
    dead = [r["dead_MiB"] for r in top][::-1]
    live = [r["live_MiB"] for r in top][::-1]
    fig, ax = plt.subplots(figsize=(8.2, 4.6), dpi=140)
    ax.barh(labels, dead, color="#b23b3b", label="history-only (dropped by filter)")
    ax.barh(labels, live, left=dead, color="#2e6f40", label="live in HEAD")
    ax.set_xlabel("blob bytes reachable from refs [MiB]")
    ax.set_title("TensorFVM history weight by file family (all refs, deduplicated)")
    ax.legend(loc="lower right", fontsize=8)
    for y, (d, l) in enumerate(zip(dead, live)):
        ax.text(d + l + 8, y, "%.0f" % (d + l), va="center", fontsize=7)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    p = os.path.join(fdir, "slim-01-ext-dead-vs-live.png")
    fig.savefig(p)
    plt.close(fig)
    made.append(p)

    dates = [dt.date.fromisoformat(t[0][:10]) for t in timeline]
    cum, acc = [], 0.0
    for t in timeline:
        acc += t[3]
        cum.append(acc / MIB)
    fig, ax = plt.subplots(figsize=(8.4, 4.2), dpi=140)
    ax.plot(dates, cum, "-", color="#1f4e79", lw=1.6,
            label="cumulative reachable blob bytes (main, first-parent)")
    ax.fill_between(dates, cum, color="#1f4e79", alpha=0.12)
    for lbl, d, c in (("2026-10-13 import", "2026-10-13", "#8a6d1f"),
                      ("2026-10-28 cleanup", "2026-10-28", "#2e6f40")):
        ax.axvline(dt.date.fromisoformat(d), color=c, ls="--", lw=1.1)
        ax.annotate(lbl, (dt.date.fromisoformat(d), max(cum) * 0.9), rotation=90,
                    fontsize=7, color=c, ha="right", va="top")
    ax.set_ylabel("MiB")
    ax.set_title("History growth: one 2026-10-13 import, cleanup, then re-accumulation")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="center right")
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
    p = os.path.join(fdir, "slim-02-history-growth.png")
    fig.savefig(p)
    plt.close(fig)
    made.append(p)

    names = [r["tag"] for r in scen_rows] + ["now"]
    vals = [r["est_pack_MiB"] for r in scen_rows] + [round(now_pack, 1)]
    cols = ["#1f4e79"] * len(scen_rows) + ["#b23b3b"]
    fig, ax = plt.subplots(figsize=(8.8, 4.4), dpi=140)
    bars = ax.bar(names, vals, color=cols)
    if gh_mib:
        ax.axhline(gh_mib, color="#b23b3b", ls=":", lw=1.2,
                   label="GitHub repo size %.0f MiB" % gh_mib)
    ax.set_ylabel("estimated pack size [MiB]")
    ax.set_title("Slimming scenarios: estimated post-rewrite pack (measured compression model)")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 25, "%.0f" % v, ha="center", fontsize=7)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p = os.path.join(fdir, "slim-03-scenarios.png")
    fig.savefig(p)
    plt.close(fig)
    made.append(p)
    return made


NOTES = {".npz": "场文件(可再生成)", ".npy": "数组(可再生成)", ".png": "图(可再生成)",
         ".svg": "矢量图(部分被 md 引用)", ".pdf": "报告(可再生成)",
         ".json": "证据元数据(可检索,保留)", ".msh": "网格(可再生成)", ".py": "源码(保留)",
         ".pyc": "不应入库", ".csv": "剖面数据(可再生成)", ".gz": "压缩工件",
         ".md": "文档(保留)", ".vtk": "场文件", ".vtu": "场文件", ".pt": "张量存档",
         ".dat": "数据", ".rst": "文档(保留)", ".log": "日志", ".txt": "文本"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comp-shard", default="", help="I/N : compress blob slice I of N, write cache and exit")
    ap.add_argument("--out", default="docs/m23216")
    ap.add_argument("--runs", default="runs/m23216")
    ap.add_argument("--tpl", default="scripts/audit/m23216_slim_plan.md.tpl")
    ap.add_argument("--top", type=int, default=30)
    args = ap.parse_args()
    t0 = dt.datetime.now()
    os.makedirs(os.path.join(args.out, "tables"), exist_ok=True)
    os.makedirs(args.runs, exist_ok=True)

    head = sh(["git", "rev-parse", "--short", "HEAD"]).strip()
    branch = sh(["git", "rev-parse", "--abbrev-ref", "HEAD"]).strip()
    gd = os.path.abspath(sh(["git", "rev-parse", "--git-common-dir"]).strip())
    log("HEAD %s on %s, gitdir %s" % (head, branch, gd))

    sizes, types = collect_objects()
    reach_all = reach_map("--all")
    blob_ids = sorted(h for h in reach_all if types.get(h) == "blob")
    if args.comp_shard:
        idx, num = (int(x) for x in args.comp_shard.split("/"))
        slice_ = blob_ids[idx::num]
        log("compression shard %d/%d: %d blobs (%.1f MiB)" % (
            idx, num, len(slice_), sum(sizes[h] for h in slice_) / MIB))
        out = stream_compressed_sizes(slice_)
        cache = os.path.join(args.runs, "comp")
        os.makedirs(cache, exist_ok=True)
        with open(os.path.join(cache, "comp-%02d.json" % idx), "w") as fh:
            json.dump(out, fh)
        log("wrote %s (%d blobs)" % (os.path.join(cache, "comp-%02d.json" % idx), len(out)))
        with open(os.path.join(args.runs, "census.log"), "a") as fh:
            fh.write("\n".join(LOG) + "\n")
        return
    cache = os.path.join(args.runs, "comp")
    comp = {}
    if os.path.isdir(cache):
        for name in sorted(os.listdir(cache)):
            if name.startswith("comp-") and name.endswith(".json"):
                with open(os.path.join(cache, name)) as fh:
                    comp.update(json.load(fh))
    missing = [h for h in blob_ids if h not in comp]
    if missing:
        log("compression cache hit %d/%d, compressing %d missing blobs inline" % (
            len(comp), len(blob_ids), len(missing)))
        # NOTE: do NOT compress inline here -- in a partial clone, `cat-file --batch`
        # may try a promisor fetch and block on the network.  Missing blobs (repo grew
        # between the shard run and the aggregate run) fall back to the per-ext ratio.
        log("WARNING: %d blobs without a compression measurement -> per-ext ratio fallback "
            "(re-run `--comp-shard I/N` to cover them)" % len(missing))
    else:
        log("compression cache complete: %d blobs" % len(comp))
    ratios = {}
    for h, csize in comp.items():
        e = ext_of(sorted(reach_all[h], key=len)[0])
        agg = ratios.setdefault(e, [0, 0])
        agg[0] += csize
        agg[1] += sizes[h]
    ratios = {e: v[0] / v[1] for e, v in ratios.items() if v[1]}
    log("per-ext ratios done")

    blobset = {h for h, t in types.items() if t == "blob"}
    tree_b = sum(s for h, s in sizes.items() if types[h] == "tree") / MIB
    commit_b = sum(s for h, s in sizes.items() if types[h] == "commit") / MIB
    A = {h: sizes[h] for h in reach_all if h in blobset}
    first_path = {h: sorted(reach_all[h], key=len)[0] for h in A}

    head_hashes, head_blob_of = set(), {}
    for line in sh(["git", "ls-tree", "-r", "HEAD"]).splitlines():
        meta, path = line.split("\t", 1)
        parts = meta.split()
        if parts[1] == "blob":
            head_hashes.add(parts[2])
            head_blob_of[path] = parts[2]
    log("HEAD tree read")
    live_MiB = sum(s for h, s in A.items() if h in head_hashes) / MIB
    dead_MiB = sum(s for h, s in A.items() if h not in head_hashes) / MIB
    reach_MiB = sum(A.values()) / MIB

    s_pack = pack_stats(gd)
    now_pack = s_pack["pack_total"] / MIB
    s_ref = ref_stats()
    gh_kib = gh_repo_size()
    gh_mib = gh_kib / 1024.0
    open_prs = gh_open_prs()
    co = sh(["git", "count-objects", "-vH"])
    inpack = int(re.search(r"in-pack:\s*(\d+)", co).group(1)) if "in-pack" in co else 0
    nobj = int(re.search(r"count:\s*(\d+)", co).group(1)) + inpack
    size_garbage = re.search(r"size-garbage:\s*([\d.]+) (\w+)", co)
    gitdir_bytes = 0
    for root, _d, files in os.walk(gd):
        for f in files:
            try:
                gitdir_bytes += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass

    commits = sh(["git", "rev-list", "--all", "--count"]).strip()
    fp = [l for l in sh(GIT + ["log", "--first-parent", "--format=%aI", "HEAD"]).splitlines() if l]
    fp_commits = len(fp)
    dates = sh(GIT + ["log", "--reverse", "--format=%aI"]).splitlines()
    first_date = dates[0] if dates else "?"
    roots = sh(["git", "rev-list", "--max-parents=0", "HEAD"]).strip().splitlines()
    root = roots[0][:12] if roots else "?"

    log("step: ext census")
    # ---- extension census
    ext_dead, ext_live, ext_n = (collections.Counter() for _ in range(3))
    for h, s in A.items():
        e = ext_of(first_path[h])
        ext_n[e] += 1
        (ext_live if h in head_hashes else ext_dead)[e] += s
    ext_rows = [dict(ext=e, total_MiB=round((ext_dead[e] + ext_live[e]) / MIB, 1),
                     dead_MiB=round(ext_dead[e] / MIB, 1), live_MiB=round(ext_live[e] / MIB, 1),
                     nblobs=ext_n[e], note=NOTES.get(e, ""))
                for e in sorted(ext_n, key=lambda x: -(ext_dead[x] + ext_live[x]))]

    log("step: top-N")
    # ---- top-N blobs with time window
    top = sorted(A, key=lambda h: -A[h])[:args.top]
    paths_union = set()
    for h in top:
        paths_union.update(sorted(reach_all[h], key=len)[:2])
    raw = sh(GIT + ["log", "--all", "--format=@%aI %H", "--raw", "--no-abbrev", "--no-renames",
                    "--"] + sorted(paths_union))
    occ = collections.defaultdict(list)
    cur = None
    for line in raw.splitlines():
        if line.startswith("@"):
            cur = line[1:]
        elif line.startswith(":"):
            meta, path = line.split("\t", 1)
            f = meta.split()
            old, new = f[2], f[3]          # <old_sha> <new_sha>; new = 0*40 on delete
            key = new if new.strip("0") else old
            occ[key].append((cur, path))
    top_rows = []
    for i, h in enumerate(top, 1):
        o = sorted(occ.get(h, []))
        top_rows.append(dict(rank=i, blob=h, bytes=A[h], ext=ext_of(first_path[h]),
                             path=first_path[h], ncommits=len(o),
                             first=o[0][0] if o else "", last=o[-1][0] if o else "",
                             dead=h not in head_hashes,
                             path_in_head=first_path[h] in head_blob_of,
                             head_same=first_path[h] in head_blob_of
                             and head_blob_of[first_path[h]] == h,
                             head_MiB=(sizes.get(head_blob_of.get(first_path[h], ""), 0) / MIB)))

    log("step: timeline")
    # ---- timeline (first-parent, cumulative unique blob bytes)
    raw = sh(GIT + ["log", "--first-parent", "--format=@%H %aI", "--raw", "--no-abbrev", "HEAD"])
    order, per = [], collections.defaultdict(list)
    for line in raw.splitlines():
        if line.startswith("@"):
            cur = line[1:].split()
            order.append(cur)
        elif line.startswith(":"):
            meta, _path = line.split("\t", 1)
            f = meta.split()
            per[cur[0]].append(f[3] if f[3].strip("0") else f[2])
    seen, series = set(), []
    for c in order[::-1]:
        seen.update(per[c[0]])
        series.append((c[1], sum(sizes.get(x, 0) for x in seen)))
    day_add = collections.OrderedDict()
    prev = 0
    for ts, cum in series:
        day_add[ts[:10]] = day_add.get(ts[:10], 0) + (cum - prev)
        prev = cum
    timeline = []
    for d, a in day_add.items():
        note = ("一次性 import（历史 docs 工件）" if d == "2026-10-13" else
                "`b77e7783` 删除可再生工件" if d == "2026-10-28" else
                "agent 交付期，工件重新累积" if d >= "2026-11-13" else "")
        timeline.append((d, a / MIB, note, day_add[d]))
    last3 = sum(t[3] for t in timeline[-3:]) / MIB
    last7 = sum(t[3] for t in timeline[-7:]) / MIB
    import_MiB = max(t[3] for t in timeline) / MIB

    log("step: scenarios")
    # ---- model self-check (delete nothing must reproduce the measured pack) then calibrate
    selfcheck = (sum(comp.get(h, s * ratios.get(ext_of(first_path[h]), 0.55)) for h, s in A.items())
                 + tree_b * 0.55 * MIB + commit_b * 0.45 * MIB) / MIB
    cal = now_pack / selfcheck if selfcheck else 1.0
    log("model self-check: estimate %.1f MiB vs measured pack %.1f MiB -> calibration %.3f"
        % (selfcheck, now_pack, cal))

    # ---- scenarios + variants
    scen_rows = scenario_rows(A, first_path, ratios, tree_b, commit_b, now_pack, comp, cal)
    by_tag = {r["tag"]: r for r in scen_rows}

    def snapshot(rev):
        hs = set()
        for line in sh(["git", "ls-tree", "-r", rev]).splitlines():
            parts = line.split("\t", 1)[0].split()
            if parts[1] == "blob":
                hs.add(parts[2])
        return sum(sizes.get(h, 0) for h in hs) / MIB

    grep = sh(["git", "log", "--all", "--format=%H", "-1",
               "--grep=Remove reproducible calculation artifacts"]).strip().splitlines()
    cleanup_rev = grep[0][:12] if grep else "b77e7783"
    head_snap = snapshot("HEAD")
    cleanup_snap = snapshot(cleanup_rev)

    def shell(b):
        return b * 0.55 + tree_b * 0.55 + commit_b * 0.45

    orphan_pack = shell(head_snap) * cal
    cleanup_pack = shell(cleanup_snap) * cal
    src_b = sum(s for h, s in A.items() if not first_path[h].startswith("docs/")) / MIB
    docs_b = sum(s for h, s in A.items() if first_path[h].startswith("docs/")) / MIB
    split_src_pack, split_docs_pack = shell(src_b) * cal, shell(docs_b) * cal

    log("step: citations")
    # ---- citation metrics
    md_paths = [p for p in head_blob_of if p.startswith("docs/") and p.endswith(".md")]
    md_data = batch_blobs([head_blob_of[p] for p in md_paths])
    cited = set()
    for p in md_paths:
        txt = md_data.get(head_blob_of[p], b"").decode("utf-8", "replace")
        for f in re.findall(r"[A-Za-z0-9_./\-]+\.(?:png|svg|pdf|npz)", txt):
            cited.add(os.path.normpath(os.path.join(os.path.dirname(p), f)))
    grep_l = sh(["git", "grep", "-I", "-l", "-E", r"\b[0-9a-f]{40}\b", "HEAD", "--", "docs"]).splitlines()
    grep_o = sh(["git", "grep", "-I", "-o", "-E", r"\b[0-9a-f]{40}\b", "HEAD", "--", "docs"]).splitlines()
    sha_files, sha_n = len(grep_l), len(grep_o)
    figs = [p for p in head_blob_of if p.startswith("docs/")
            and p.endswith((".png", ".svg", ".pdf", ".npz"))]
    cited_figs = [p for p in figs if p in cited]
    cited_bytes = sum(sizes.get(head_blob_of[p], 0) for p in cited_figs) / MIB
    fig_bytes = sum(sizes.get(head_blob_of[p], 0) for p in figs) / MIB

    log("step: cross-checks")
    # ---- cross-checks
    total_obj_b = (sum(A.values()) + tree_b * MIB + commit_b * MIB) / MIB
    pack_ratio = now_pack / total_obj_b
    big4 = sum(r["total_MiB"] for r in ext_rows if r["ext"] in (".npz", ".png", ".svg", ".pdf"))
    text3 = sum(r["total_MiB"] for r in ext_rows if r["ext"] in (".py", ".md", ".json"))
    pyc = sum(r["total_MiB"] for r in ext_rows if r["ext"] == ".pyc")

    scalars = dict(
        runtime=0, generated_at=dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        head=head, branch=branch,
        now_pack="%.1f" % now_pack, npacks=len(s_pack["packs"]),
        big_pack="%.1f" % (s_pack["packs"][0][1] / MIB),
        gh_mib="%.1f" % gh_mib, gh_kib=gh_kib,
        gh_gap="%.1f" % (100 * abs(now_pack - gh_mib) / max(gh_mib, 1e-9)),
        gitdir_MiB="%.1f" % (gitdir_bytes / MIB),
        tmp_MiB="%.1f" % (s_pack["tmp_total"] / MIB), ntmp=len(s_pack["tmp"]),
        nobj=nobj, inpack=inpack,
        commits=commits, fp_commits=fp_commits, first_date=first_date[:10], root=root,
        local_refs=s_ref["local_refs"], local_buckets=json.dumps(s_ref["local_buckets"], ensure_ascii=False),
        local_heads=s_ref["local_buckets"].get("heads", 0),
        remote_buckets=json.dumps(s_ref["remote_buckets"], ensure_ascii=False),
        pull_refs=s_ref["remote_buckets"].get("refs/pull", 0),
        open_prs=open_prs, worktrees=s_ref["worktrees"],
        wt_prunable=s_ref["worktrees_prunable"],
        nblob_reach=len(A), nblob_head=len(head_hashes & set(A)),
        nblob_dead=len(set(A) - head_hashes),
        reach_MiB="%.1f" % reach_MiB, live_MiB="%.1f" % live_MiB, dead_MiB="%.1f" % dead_MiB,
        live_pct="%.1f" % (100 * live_MiB / reach_MiB),
        dead_pct="%.1f" % (100 * dead_MiB / reach_MiB),
        pack_ratio="%.3f" % pack_ratio,
        selfcheck_pack="%.1f" % selfcheck, cal="%.3f" % cal,
        selfcheck_gap="%.1f" % (100 * (selfcheck - now_pack) / now_pack),
        tree_MiB="%.1f" % tree_b, commit_MiB="%.1f" % commit_b,
        top30_dead_of_30=sum(1 for r in top_rows if r["dead"]), top30_n=len(top_rows),
        max_blob_MiB="%.1f" % (max(A.values()) / MIB),
        big4_MiB="%.1f" % big4, big4_pct="%.1f" % (100 * big4 / reach_MiB),
        text_MiB="%.1f" % text3, pyc_MiB="%.1f" % pyc,
        import_MiB="%.1f" % import_MiB, last3_MiB="%.1f" % last3, last7_MiB="%.1f" % last7,
        proj_weeks="%.0f" % ((1024 - live_MiB) / max(last7, 1e-9)) if last7 else "n/a",
        cleanup_MiB="%.1f" % cleanup_snap,
        best_pack="%.1f" % by_tag["R2"]["est_pack_MiB"],
        best_pct="%.0f" % by_tag["R2"]["reduction_pct"],
        r1_pack="%.1f" % by_tag["R1"]["est_pack_MiB"],
        r2b_pack="%.1f" % by_tag["R2b"]["est_pack_MiB"],
        r3_pack="%.1f" % by_tag["R3"]["est_pack_MiB"],
        r3_pct="%.0f" % by_tag["R3"]["reduction_pct"],
        r2c_pack="%.1f" % by_tag["R2c"]["est_pack_MiB"],
        r5_pack="%.1f" % by_tag["R5"]["est_pack_MiB"],
        r5b_pack="%.1f" % by_tag["R5b"]["est_pack_MiB"],
        gc_pct="%.0f" % (100 * (s_pack["tmp_total"] / MIB) / now_pack),
        gc_pack="%.1f" % (now_pack - s_pack["tmp_total"] / MIB),
        head_snap_MiB="%.1f" % head_snap, cleanup_snap_MiB="%.1f" % cleanup_snap,
        orphan_pack="%.1f" % orphan_pack, cleanup_pack="%.1f" % cleanup_pack,
        src_MiB="%.1f" % src_b, docs_MiB="%.1f" % docs_b,
        split_src_pack="%.1f" % split_src_pack, split_docs_pack="%.1f" % split_docs_pack,
        docbin_live_MiB="%.1f" % fig_bytes,
        cited_bytes="%.1f" % cited_bytes, cited_n=len(cited_figs), fig_n=len(figs),
        uncited_MiB="%.1f" % (fig_bytes - cited_bytes),
        sha_cite_files=sha_files, sha_cite_n=sha_n,
    )
    log("scalars: " + json.dumps({k: scalars[k] for k in
                                  ("now_pack", "dead_pct", "best_pack", "best_pct", "tmp_MiB")}))

    # ---- artifacts
    tbl = os.path.join(args.out, "tables")
    top_md = "| # | blob | MiB | 后缀 | 路径族 | n | 首现 | 末现 | dead | 该路径当前状态 |\n|---|---|---|---|---|---|---|---|---|---|\n" + "\n".join(
        "| %d | `%s` | %.1f | `%s` | `%s` | %d | %s | %s | %s | %s |" % (
            r["rank"], r["blob"][:12], r["bytes"] / MIB, r["ext"], r["path"], r["ncommits"],
            (r["first"] or "?")[:10], (r["last"] or "?")[:10], "✓" if r["dead"] else "",
            ("HEAD 同 blob" if r["head_same"] else
             "HEAD 同名异 blob(%.1f MiB)" % r["head_MiB"] if r["path_in_head"] else "路径已不存在"))
        for r in top_rows)
    ext_md = "\n".join("| `%s` | %.1f | %.1f | %.1f | %d | %s |" % (
        r["ext"], r["total_MiB"], r["dead_MiB"], r["live_MiB"], r["nblobs"], r["note"])
        for r in ext_rows[:14])
    scen_md = "\n".join("| %s %s | %.1f | %d | %.1f | %.1f | %+.0f%% |" % (
        r["tag"], r["desc"], r["dropped_MiB"], r["nblobs"], r["kept_MiB"],
        r["est_pack_MiB"], -r["reduction_pct"]) for r in scen_rows)
    tl_md = "\n".join("| %s | %.1f | %s |" % (d, a, n) for d, a, n, _ in timeline)

    with open(os.path.join(tbl, "top30_blobs.md"), "w") as fh:
        fh.write("## TOP-%d blobs (deduplicated, all refs)\n\n%s\n" % (args.top, top_md))
    with open(os.path.join(tbl, "top30_blobs.csv"), "w") as fh:
        fh.write("rank,blob,bytes,ext,path,ncommits,first,last,dead\n")
        for r in top_rows:
            fh.write("%d,%s,%d,%s,%s,%d,%s,%s,%d\n" % (
                r["rank"], r["blob"], r["bytes"], r["ext"], r["path"], r["ncommits"],
                r["first"], r["last"], int(r["dead"])))
    with open(os.path.join(tbl, "ext_census.md"), "w") as fh:
        fh.write("| 后缀 | 合计 MiB | dead MiB | live MiB | blob 数 | 解释 |\n|---|---|---|---|---|---|\n"
                 + ext_md + "\n")
    with open(os.path.join(tbl, "scenarios.md"), "w") as fh:
        fh.write("| 场景 | 删除 MiB | 删除 blob 数 | 保留 MiB | 预估 pack MiB | 相对现状 |\n"
                 "|---|---|---|---|---|---|\n" + scen_md + "\n")
    with open(os.path.join(tbl, "timeline.md"), "w") as fh:
        fh.write("| 日期 | 当日新增 MiB | 备注 |\n|---|---|---|\n" + tl_md + "\n")

    log("step: figures")
    figs_made = make_figures(args.out, ext_rows, timeline, scen_rows, now_pack, gh_mib)

    census = dict(
        generated_at=scalars["generated_at"], head=head, branch=branch,
        gitdir=gd,
        pack=dict(pack_total=s_pack['pack_total'], tmp_total=s_pack['tmp_total'],
                  npacks=len(s_pack['packs']), packs=s_pack['packs'][:5],
                  tmp=[(n, b, d.isoformat()) for n, b, d in s_pack['tmp']]), refs=s_ref, ratios=ratios, ext_rows=ext_rows,
        top_rows=top_rows, scenarios=scen_rows, timeline=[(d, round(a, 2), n) for d, a, n, _ in timeline],
        citation=dict(md_files=len(md_paths), fig_files=len(figs), cited=cited_figs,
                      cited_MiB=round(cited_bytes, 2), fig_MiB=round(fig_bytes, 2),
                      sha_files=sha_files, sha_n=sha_n),
        model_selfcheck=dict(estimate_MiB=round(selfcheck, 1), measured_pack_MiB=round(now_pack, 1),
                             gap_pct=round(100 * (selfcheck - now_pack) / now_pack, 1),
                             calibration=round(cal, 4)),
        variants=dict(head_snapshot_MiB=round(head_snap, 1), cleanup_snapshot_MiB=round(cleanup_snap, 1),
                      orphan_pack_MiB=round(orphan_pack, 1), cleanup_pack_MiB=round(cleanup_pack, 1),
                      split_src_pack_MiB=round(split_src_pack, 1),
                      split_docs_pack_MiB=round(split_docs_pack, 1)),
        scalars=scalars)
    with open(os.path.join(args.out, "census.json"), "w") as fh:
        json.dump(census, fh, indent=1, ensure_ascii=False)

    scalars["runtime"] = "%.0f" % (dt.datetime.now() - t0).total_seconds()
    scalars.update(top30_table=top_md, ext_table=ext_md, scen_table=scen_md, timeline_table=tl_md)
    with open(args.tpl) as fh:
        doc = string.Template(fh.read()).substitute(scalars)
    plan = os.path.join(args.out, "SLIM_PLAN.md")
    with open(plan, "w") as fh:
        fh.write(doc)
    log("wrote %s (%d chars), %d figures" % (plan, len(doc), len(figs_made)))

    with open(os.path.join(args.runs, "census.log"), "w") as fh:
        fh.write("\n".join(LOG) + "\n")
        fh.write("\n--- census.json summary ---\n")
        fh.write(json.dumps(census["scalars"], indent=1, ensure_ascii=False) + "\n")
        fh.write("\n--- figures ---\n" + "\n".join(figs_made) + "\n")
    print("OK")


if __name__ == "__main__":
    main()
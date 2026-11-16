#!/usr/bin/env python3
"""按场景**实测** TensorFVM 历史瘦身后的 pack 体积（只读审计工具）。

与 docs/m23216/SLIM_PLAN.md（方案版）的区别
------------------------------------------
方案版把改写后体积建模为 ``sum(zlib.compress(blob, 6)) * 0.874``：逐 blob 独立压缩、
再用「一条都不删」的全局标定系数 0.874 折算。该模型不含 git 的 delta 复用，
而本仓 1.33 GiB pack 的压缩收益主要来自「同一工件的多个历史版本互为 delta」。
因此本工具给出**实测**：把每个场景的存活对象集交给真实的 ``git pack-objects``
重新打包，量 stdout 字节数，不做任何标定系数。

两个关键判定
------------
1. 删除集判定用**全路径出现集**：``git log --all --raw -r --no-merges`` 的 new-side
   给出每个 blob 历史上出现过的全部 (path, blob) 对；只有当一个 blob 的**每一条**
   路径都命中删除规则时才算被删（方案版的 ``rev-list --objects`` 每个 blob 只报
   一条路径，同名异径 blob 会误判）。
2. 对源仓库零写入：pack 流只计数、不落盘；唯一写入是 ``--out`` 目录与临时文件。

用法
----
    python3 scripts/audit/measure_rewrite_pack.py --out docs/FVM-REPO-SLIM-MEASURE
    # 只跑部分场景 / 快速模式
    python3 scripts/audit/measure_rewrite_pack.py --scenarios R2,R3 --window 10
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import subprocess
import sys
import time
from collections import defaultdict

MiB = 1024 * 1024

# ---------------------------------------------------------------- 场景规则
# 场景定义与 docs/m23216/SLIM_PLAN.md §3.2 一一对应（规则名保持原 tag）。
FIELD_EXTS = {".npz", ".npy", ".vtk", ".vtu", ".msh", ".pt", ".dat", ".gz", ".csv"}
TEXT_EXTS = {
    ".md", ".rst", ".txt", ".py", ".json", ".yaml", ".yml", ".toml", ".cfg",
    ".ini", ".sh", ".c", ".h", ".cc", ".cpp", ".hpp", ".f90", ".f", ".tex",
    ".html", ".css", ".js", ".gitignore", ".gitattributes",
}


def _ext(path: str) -> str:
    base = path.rsplit("/", 1)[-1]
    if base.startswith(".") and base.count(".") == 1:  # dotfile
        return base
    return os.path.splitext(base)[1].lower()


def _under_docs(path: str) -> bool:
    return path.startswith("docs/")


def _r1(path: str) -> bool:
    """删 docs 场数据（npz/npy/vtk/vtu/msh/pt/dat/gz/csv）。"""
    return _under_docs(path) and _ext(path) in FIELD_EXTS


def _r2(path: str) -> bool:
    """R1 + docs 下的 PDF 报告（保留 png/svg 图）。"""
    return _r1(path) or (_under_docs(path) and _ext(path) == ".pdf")


def _pyc(path: str) -> bool:
    return _ext(path) == ".pyc"


def _r2b(path: str) -> bool:
    return _r2(path) or _pyc(path)


def _r2c(path: str) -> bool:
    return _r2b(path) or (_under_docs(path) and _ext(path) == ".svg")


def _r3(path: str) -> bool:
    """删 docs 全部非文本工件 + 全仓 *.pyc。"""
    if _pyc(path):
        return True
    return _under_docs(path) and _ext(path) not in TEXT_EXTS


# 阈值场景需要 blob 体积，用双参规则
def _r5(path: str, size: int) -> bool:
    return _under_docs(path) and size >= 1 * MiB


def _r5b(path: str, size: int) -> bool:
    return _under_docs(path) and size >= 512 * 1024


SCENARIOS = {
    "R1": ("只删 docs 场数据(npz/npy/vtk/vtu/msh/pt/dat/gz/csv)", _r1),
    "R2": ("删场数据+网格+PDF，保留 png/svg 图", _r2),
    "R2b": ("R2 + 全仓 *.pyc", _r2b),
    "R2c": ("R2 + 删 docs *.svg（几乎全为 dead 图）", _r2c),
    "R3": ("删 docs 全部二进制 + 全仓 *.pyc", _r3),
    "R5": ("阈值：删 docs 下 >=1 MiB 的 blob", _r5),
    "R5b": ("阈值：删 docs 下 >=512 KiB 的 blob", _r5b),
}
SIZE_RULES = {"R5", "R5b"}


# ---------------------------------------------------------------- git 辅助
def git(repo: str, *args: str, stdin: bytes | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", repo, *args],
        input=stdin,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def occurrence_map(repo: str) -> dict[str, set[str]]:
    """blob oid -> 历史上出现过的全部路径（new-side，含重命名新路径）。"""
    cp = git(
        repo, "log", "--all", "--raw", "-r", "--no-merges", "--no-abbrev",
        "--format=C%H",
    )
    if cp.returncode != 0:
        raise SystemExit(f"git log --raw failed: {cp.stderr.decode()[:400]}")
    out: dict[str, set[str]] = defaultdict(set)
    for raw in cp.stdout.decode("utf-8", "replace").splitlines():
        if not raw.startswith(":"):
            continue
        head, *paths = raw.split("\t")
        fields = head[1:].split()
        if len(fields) < 5 or not paths:
            continue
        newsha, status = fields[3], fields[4]
        if set(newsha) == {"0"}:  # 删除：该路径在此提交后不再有内容
            continue
        newpath = paths[1] if status[0] in ("R", "C") and len(paths) > 1 else paths[0]
        out[newsha].add(newpath)
    return out


def reachable_objects(repo: str) -> list[str]:
    cp = git(repo, "rev-list", "--objects", "--all", "--no-object-names")
    if cp.returncode != 0:
        raise SystemExit(f"git rev-list failed: {cp.stderr.decode()[:400]}")
    return cp.stdout.decode().split()


def object_sizes(repo: str, oids: list[str]) -> dict[str, tuple[str, int]]:
    """oid -> (type, uncompressed size)，一次 batch-check 喂全部 oid。"""
    feed = ("\n".join(oids) + "\n").encode()
    cp = git(repo, "cat-file", "--batch-check=%(objectname) %(objecttype) %(objectsize)", stdin=feed)
    if cp.returncode != 0:
        raise SystemExit(f"git cat-file --batch-check failed: {cp.stderr.decode()[:400]}")
    info: dict[str, tuple[str, int]] = {}
    for line in cp.stdout.decode().splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[1] != "missing":
            info[parts[0]] = (parts[1], int(parts[2]))
    return info


def pack_dir(repo: str) -> str:
    cp = git(repo, "rev-parse", "--git-common-dir")
    common = cp.stdout.decode().strip()
    if not os.path.isabs(common):
        common = os.path.join(repo, common)
    return os.path.join(common, "objects", "pack")


def in_pack_bytes(repo: str) -> dict[str, int]:
    """oid -> size-in-packfile（实测压缩后字节）；delta 记该 delta 自身字节。"""
    sip: dict[str, int] = {}
    d = pack_dir(repo)
    for name in sorted(os.listdir(d)):
        if not name.endswith(".idx"):
            continue
        cp = git(repo, "verify-pack", "-v", os.path.join(d, name))
        for line in cp.stdout.decode().splitlines():
            parts = line.split()
            if len(parts) < 5 or len(parts[0]) != 40:
                continue
            try:
                sip[parts[0]] = int(parts[3])
            except ValueError:
                continue
    return sip


def measure_pack(repo: str, oids: list[str], window: int, depth: int, threads: int,
                 chunk: int = 1 << 22) -> int:
    """把给定对象集真实打包一次，返回 pack 字节数（边读边计数，不落盘、不驻留内存）。"""
    feed = ("\n".join(oids) + "\n").encode()
    proc = subprocess.Popen(
        ["git", "-C", repo, "pack-objects", "--stdout", "-q",
         f"--window={window}", f"--depth={depth}", f"--threads={threads}"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    proc.stdin.write(feed)
    proc.stdin.close()
    total = 0
    while True:
        block = proc.stdout.read(chunk)
        if not block:
            break
        total += len(block)
    err = proc.stderr.read().decode("utf-8", "replace")
    if proc.wait() != 0:
        raise SystemExit(f"git pack-objects failed: {err[:400]}")
    return total


# ---------------------------------------------------------------- 主流程
def run(args) -> dict:
    repo = args.repo
    t0 = time.time()

    occ = occurrence_map(repo)
    oids = reachable_objects(repo)
    info = object_sizes(repo, oids)
    print(f"[i] reachable objects={len(oids)} blobs_with_occurrence={len(occ)}", flush=True)

    sip = {} if args.no_inpack else load_or_build_sip(repo, args.cache_dir)
    if sip:
        print(f"[i] size-in-pack known for {len(sip)} objects", flush=True)

    if args.truncate_at:
        rev = args.truncate_at
        keep_closure = set(git(repo, "rev-list", "--objects", "--no-walk", rev,
                               "--no-object-names").stdout.decode().split())
        parent = f"{rev}^"
        if git(repo, "rev-parse", "-q", "--verify", parent).returncode != 0:
            excluded: set[str] = set()
        else:
            excluded = set(git(repo, "rev-list", "--objects", parent,
                               "--no-object-names").stdout.decode().split())
        before = len(oids)
        oids = [o for o in oids if o not in excluded or o in keep_closure]
        rejected = set(oids)
        # 只保留出现在保全集或非祖先集合中的对象
        info = {o: v for o, v in info.items() if o in rejected}
        print(f"[i] truncate@{rev}: {before} -> {len(oids)} objects "
              f"(excluded {len(excluded)} ancestors' objects)", flush=True)

    blobs = {o: s for o, (t, s) in info.items() if t == "blob"}
    by_type: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "bytes": 0})
    for oid, (t, size) in info.items():
        by_type[t]["n"] += 1
        by_type[t]["bytes"] += size

    baseline_MiB = load_or_build_baseline(repo, oids, args)
    baseline = baseline_MiB * MiB

    results = []
    tags = args.scenarios.split(",") if args.scenarios else list(SCENARIOS)
    for tag in tags:
        desc, rule = SCENARIOS[tag]
        use_size = tag in SIZE_RULES
        dead: set[str] = set()
        for oid, size in blobs.items():
            paths = occ.get(oid, ())
            if not paths:  # 仅在 merge 侧出现（--no-merges 未覆盖）：保守视为存活
                continue
            hit = [rule(p, size) if use_size else rule(p) for p in paths]
            if all(hit):
                dead.add(oid)
        dropped = sum(blobs[o] for o in dead)
        kept = sum(blobs.values()) - dropped
        survivors = [o for o in oids if o not in dead]
        meas = measure_pack(repo, survivors, args.window, args.depth, args.threads)
        excl = sum(sip.get(o, blobs.get(o, 0)) for o in dead) if sip else None
        results.append({
            "tag": tag,
            "desc": desc,
            "dropped_blobs": len(dead),
            "dropped_MiB": round(dropped / MiB, 2),
            "kept_MiB": round(kept / MiB, 2),
            "measured_pack_MiB": round(meas / MiB, 2),
            "reduction_pct": round(100.0 * (1 - meas / baseline), 2),
            "exclusive_inpack_MiB": None if excl is None else round(excl / MiB, 2),
            "n_surviving_objects": len(survivors),
        })
        print(f"[i] {tag}: dropped {len(dead)} blobs / {dropped / MiB:.1f} MiB "
              f"-> measured pack {meas / MiB:.1f} MiB ({time.time() - t0:.0f}s)", flush=True)

    out = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "repo": os.path.abspath(repo),
        "gitdir": os.path.abspath(os.path.join(repo, git(repo, "rev-parse", "--git-common-dir")
                                               .stdout.decode().strip())),
        "git_head": git(repo, "rev-parse", "--short", "HEAD").stdout.decode().strip(),
        "git_branch": git(repo, "rev-parse", "--abbrev-ref", "HEAD").stdout.decode().strip(),
        "pack_params": {"window": args.window, "depth": args.depth, "threads": args.threads},
        "reachable": {
            "n_objects": len(oids),
            "n_blobs": len(blobs),
            "blob_MiB": round(sum(blobs.values()) / MiB, 2),
            "by_type": {k: {"n": v["n"], "MiB": round(v["bytes"] / MiB, 2)}
                        for k, v in sorted(by_type.items())},
            "inpack_MiB": round(sum(sip.values()) / MiB, 2) if sip else None,
        },
        "baseline_keepall_pack_MiB": round(baseline_MiB, 2),
        "scenarios": results,
        "elapsed_s": round(time.time() - t0, 1),
    }
    return out


def load_or_build_sip(repo: str, cache: str | None) -> dict[str, int]:
    """size-in-pack 映射带缓存：verify-pack 需 ~30 s，跨场景复用。"""
    if cache:
        path = os.path.join(cache, "inpack_sip.json.gz")
        if os.path.exists(path):
            with gzip.open(path, "rt") as fh:
                return json.load(fh)
    sip = in_pack_bytes(repo)
    if cache:
        os.makedirs(cache, exist_ok=True)
        with gzip.open(os.path.join(cache, "inpack_sip.json.gz"), "wt") as fh:
            json.dump(sip, fh)
    return sip


def load_or_build_baseline(repo: str, oids: list[str], args) -> float:
    """keep-all 重打包基线带缓存（各场景共用同一基线）。"""
    key = "keepall" + (f"_trunc_{args.truncate_at}" if args.truncate_at else "")
    if args.cache_dir:
        path = os.path.join(args.cache_dir, f"baseline_{key}.json")
        if os.path.exists(path):
            with open(path) as fh:
                return json.load(fh)["keepall_pack_MiB"]
    base = measure_pack(repo, oids, args.window, args.depth, args.threads) / MiB
    if args.cache_dir:
        os.makedirs(args.cache_dir, exist_ok=True)
        with open(path, "w") as fh:
            json.dump({"keepall_pack_MiB": round(base, 2)}, fh)
    print(f"[i] keep-all repack = {base:.1f} MiB (measured now)", flush=True)
    return base


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default=".")
    ap.add_argument("--out", default="docs/FVM-REPO-SLIM-MEASURE")
    ap.add_argument("--scenarios", default="", help="逗号分隔，缺省=全部")
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--depth", type=int, default=50)
    ap.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--no-inpack", action="store_true", help="跳过 verify-pack（省 30s）")
    ap.add_argument("--cache-dir", default="/tmp/slimmeas-cache",
                    help="基线/in-pack 映射缓存目录（跨场景复用，缺省 /tmp 下）")
    ap.add_argument("--measure-rev", default=None, action="append",
                    metavar="REV",
                    help="额外实测「单快照 orphan 重开」体积：对给定 rev 的完整对象闭包打包")
    ap.add_argument("--truncate-at", default=None, metavar="REV",
                    help="历史截断到 REV（等价：以 REV 快照为根 + 其后历史全部保留）")
    ap.add_argument("--dry-run", action="store_true", help="只打印场景与对象数，不打包")
    args = ap.parse_args(argv)

    if args.dry_run:
        occ = occurrence_map(args.repo)
        oids = reachable_objects(args.repo)
        info = object_sizes(args.repo, oids)
        blobs = {o: s for o, (t, s) in info.items() if t == "blob"}
        print(f"objects={len(oids)} blobs={len(blobs)} total={sum(blobs.values()) / MiB:.1f} MiB")
        for tag, (desc, rule) in SCENARIOS.items():
            use_size = tag in SIZE_RULES
            dead = {o for o, s in blobs.items()
                    if occ.get(o) and all(rule(p, s) if use_size else rule(p) for p in occ[o])}
            print(f"  {tag:4s} dead={len(dead):5d} {sum(blobs[o] for o in dead) / MiB:8.1f} MiB  {desc}")
        return 0

    if args.measure_rev:
        snaps = {}
        for rev in args.measure_rev:
            # 单快照闭包 = 该提交 + 其 tree 递归（必须 --no-walk，
            # 否则 rev-list 会沿祖先走成整条历史）
            cp = git(args.repo, "rev-list", "--objects", "--no-walk", rev,
                     "--no-object-names")
            oids_rev = cp.stdout.decode().split()
            nbytes = measure_pack(args.repo, oids_rev, args.window, args.depth, args.threads)
            raw = object_sizes(args.repo, oids_rev)
            snap = {
                "rev": rev,
                "commit": git(args.repo, "rev-parse", "--short", rev).stdout.decode().strip(),
                "n_objects": len(oids_rev),
                "uncompressed_MiB": round(sum(s for _, s in raw.values()) / MiB, 2),
                "measured_pack_MiB": round(nbytes / MiB, 2),
            }
            snaps[rev] = snap
            print(f"[i] orphan@{rev}: {snap['measured_pack_MiB']} MiB "
                  f"({snap['uncompressed_MiB']} MiB raw, {len(oids_rev)} objects)", flush=True)
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "measured_orphan.json"), "w") as fh:
            json.dump({"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                       "snapshots": snaps}, fh, indent=1, ensure_ascii=False)
        return 0

    res = run(args)
    os.makedirs(args.out, exist_ok=True)
    suffix = ("" if not args.scenarios else "_" + args.scenarios.replace(",", "-"))
    if args.truncate_at:
        suffix += "_trunc"
    p = os.path.join(args.out, f"measured_pack{suffix}.json")
    with open(p, "w") as fh:
        json.dump(res, fh, indent=1, ensure_ascii=False)
    print(f"[i] wrote {p}  ({res['elapsed_s']}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
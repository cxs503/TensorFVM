"""SQLite provenance ledger for the two-phase verification chain.

Every number that reaches the report is written into a SQLite database by the
benchmark and is *read back with SQL* by the report generator; nothing is copied
by hand.  That is what the "zero hand-copied numbers" criterion (J6) demands:
the report JSON can be regenerated from the database alone, and any number in it
can be traced to ``(run, metric)`` rows that also record the git commit, the
environment and the exact configuration of the run.

Schema
------
``runs``       one row per (segment, case) execution: commit, environment, config
``metrics``    one row per measured quantity: value, unit, comparator, limit, verdict
``artifacts``  files produced by the run with their SHA-256
``figures``    figure files with their SHA-256 and caption

The ``metrics.limit_value`` + ``metrics.comparator`` pair makes a criterion
self-describing, so a verdict can be recomputed from the database without the
generating code.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone

__all__ = ["ProvenanceDB", "sha256_file"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    segment TEXT NOT NULL,
    case_name TEXT NOT NULL,
    created_at TEXT NOT NULL,
    git_commit TEXT,
    git_branch TEXT,
    python TEXT,
    library_versions TEXT,
    host TEXT,
    config_json TEXT
);
CREATE TABLE IF NOT EXISTS metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs(id),
    segment TEXT NOT NULL,
    name TEXT NOT NULL,
    value REAL,
    unit TEXT,
    comparator TEXT,
    limit_value REAL,
    passed INTEGER,
    note TEXT
);
CREATE TABLE IF NOT EXISTS artifacts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER REFERENCES runs(id),
    path TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    n_bytes INTEGER,
    kind TEXT
);
CREATE TABLE IF NOT EXISTS figures (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER REFERENCES runs(id),
    name TEXT NOT NULL,
    path TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    caption TEXT
);
"""


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str:
    try:
        out = subprocess.run(["git", *args], capture_output=True, text=True,
                             timeout=20, check=False)
        return out.stdout.strip() if out.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):     # pragma: no cover
        return ""


class ProvenanceDB:
    """Thin wrapper around a SQLite file with the provenance schema."""

    def __init__(self, path: str) -> None:
        self.path = path
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    # ------------------------------------------------------------------ #
    def start_run(self, segment: str, case: str, config: dict) -> int:
        versions = {"python": sys.version.split()[0],
                    "platform": platform.platform()}
        for module in ("numpy", "scipy", "torch"):
            try:
                versions[module] = __import__(module).__version__
            except Exception:                          # pragma: no cover
                versions[module] = "unavailable"
        cur = self.conn.execute(
            "INSERT INTO runs (segment, case_name, created_at, git_commit, git_branch,"
            " python, library_versions, host, config_json)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            (segment, case, datetime.now(timezone.utc).isoformat(),
             _git("rev-parse", "HEAD"), _git("rev-parse", "--abbrev-ref", "HEAD"),
             sys.version.split()[0], json.dumps(versions), platform.node(),
             json.dumps(config, default=str)))
        self.conn.commit()
        return int(cur.lastrowid)

    def add_metric(self, run_id: int, segment: str, name: str, value: float,
                   unit: str = "", comparator: str | None = None,
                   limit_value: float | None = None,
                   note: str = "") -> bool:
        value = float(value)
        stored = None if value != value else value      # NaN has no SQL form
        if stored is None and comparator is not None:
            comparator, limit_value = None, None        # no verdict on NaN
        passed = None
        if comparator is not None and limit_value is not None:
            passed = int({"<=": value <= limit_value,
                          "<": value < limit_value,
                          ">=": value >= limit_value}.get(comparator, False))
        self.conn.execute(
            "INSERT INTO metrics (run_id, segment, name, value, unit, comparator,"
            " limit_value, passed, note) VALUES (?,?,?,?,?,?,?,?,?)",
            (run_id, segment, name, stored, unit, comparator,
             None if limit_value is None else float(limit_value), passed, note))
        self.conn.commit()
        return bool(passed)

    def add_artifact(self, run_id: int | None, path: str, kind: str) -> str:
        digest = sha256_file(path)
        self.conn.execute(
            "INSERT INTO artifacts (run_id, path, sha256, n_bytes, kind)"
            " VALUES (?,?,?,?,?)",
            (run_id, path, digest, os.path.getsize(path), kind))
        self.conn.commit()
        return digest

    def add_figure(self, run_id: int | None, name: str, path: str,
                   caption: str) -> str:
        digest = sha256_file(path)
        self.conn.execute(
            "INSERT INTO figures (run_id, name, path, sha256, caption)"
            " VALUES (?,?,?,?,?)",
            (run_id, name, path, digest, caption))
        self.conn.commit()
        return digest

    # ------------------------------------------------------------------ #
    def query(self, sql: str, params: tuple = ()) -> list[dict]:
        cur = self.conn.execute(sql, params)
        return [dict(row) for row in cur.fetchall()]

    def scalar(self, sql: str, params: tuple = ()):
        cur = self.conn.execute(sql, params)
        row = cur.fetchone()
        return None if row is None else row[0]

    def criteria(self) -> list[dict]:
        """All (metric, limit, verdict) rows, i.e. the acceptance table."""
        return self.query(
            "SELECT segment, name, value, unit, comparator, limit_value, passed,"
            " note FROM metrics WHERE limit_value IS NOT NULL"
            " ORDER BY segment, name")

    def metric_value(self, segment: str, name: str) -> float:
        value = self.scalar(
            "SELECT value FROM metrics WHERE segment=? AND name=?"
            " ORDER BY id DESC LIMIT 1", (segment, name))
        if value is None:
            raise KeyError(f"metric {segment}/{name} not recorded")
        return float(value)

    def close(self) -> None:
        self.conn.close()
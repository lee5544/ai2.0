from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import yaml

from .config import DATABASE_PATH, ensure_dirs


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


@contextmanager
def connection() -> Iterator[sqlite3.Connection]:
    ensure_dirs()
    con = sqlite3.connect(DATABASE_PATH, timeout=30)
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init_db() -> None:
    with connection() as con:
        con.executescript(
            """
            CREATE TABLE IF NOT EXISTS projects (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                line_name TEXT NOT NULL,
                model_name TEXT NOT NULL,
                model_type TEXT NOT NULL,
                config_path TEXT NOT NULL DEFAULT '',
                config_json TEXT NOT NULL,
                origin TEXT NOT NULL DEFAULT 'ui',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                status TEXT NOT NULL,
                current_stage TEXT NOT NULL DEFAULT '',
                progress REAL NOT NULL DEFAULT 0,
                progress_detail TEXT NOT NULL DEFAULT '',
                pid INTEGER,
                supervisor_pid INTEGER,
                return_code INTEGER,
                error TEXT NOT NULL DEFAULT '',
                config_path TEXT NOT NULL DEFAULT '',
                artifact_root TEXT NOT NULL DEFAULT '',
                log_path TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                started_at TEXT NOT NULL DEFAULT '',
                finished_at TEXT NOT NULL DEFAULT '',
                FOREIGN KEY(project_id) REFERENCES projects(id)
            );
            CREATE INDEX IF NOT EXISTS idx_runs_project_created
                ON runs(project_id, created_at DESC);
            """
        )
        columns = {row[1] for row in con.execute("PRAGMA table_info(projects)")}
        if "config_path" not in columns:
            con.execute("ALTER TABLE projects ADD COLUMN config_path TEXT NOT NULL DEFAULT ''")
        if "origin" not in columns:
            con.execute("ALTER TABLE projects ADD COLUMN origin TEXT NOT NULL DEFAULT ''")
            con.execute(
                """
                UPDATE projects
                SET origin='ui'
                WHERE id IN (SELECT DISTINCT project_id FROM runs)
                """
            )
        run_columns = {row[1] for row in con.execute("PRAGMA table_info(runs)")}
        if "progress_detail" not in run_columns:
            con.execute("ALTER TABLE runs ADD COLUMN progress_detail TEXT NOT NULL DEFAULT ''")
        if "supervisor_pid" not in run_columns:
            con.execute("ALTER TABLE runs ADD COLUMN supervisor_pid INTEGER")


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _decode_project(row: sqlite3.Row | None) -> dict | None:
    if row is None:
        return None
    out = dict(row)
    out["config"] = json.loads(out.pop("config_json") or "{}")
    config_path = str(out.get("config_path") or "").strip()
    if config_path:
        try:
            loaded = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
            if isinstance(loaded, dict):
                out["config"] = loaded
        except Exception:
            pass
    return out


def _decode_run(row: sqlite3.Row | None) -> dict | None:
    return dict(row) if row is not None else None


def list_projects() -> list[dict]:
    with connection() as con:
        rows = con.execute(
            """
            SELECT p.*, r.id AS latest_run_id, r.status AS latest_run_status,
                   r.kind AS latest_run_kind, r.finished_at AS latest_run_finished_at
            FROM projects p
            LEFT JOIN runs r ON r.id = (
                SELECT id FROM runs WHERE project_id=p.id ORDER BY created_at DESC LIMIT 1
            )
            WHERE p.origin='ui'
            ORDER BY p.updated_at DESC
            """
        ).fetchall()
    return [_decode_project(r) for r in rows if r is not None]  # type: ignore[misc]


def get_project(project_id: str) -> dict | None:
    with connection() as con:
        row = con.execute("SELECT * FROM projects WHERE id=?", (project_id,)).fetchone()
    return _decode_project(row)


def create_project(data: dict) -> dict:
    project_id = uuid.uuid4().hex[:10]
    ts = now()
    with connection() as con:
        con.execute(
            """
            INSERT INTO projects (
                id, name, line_name, model_name, model_type, config_path,
                config_json, origin, created_at, updated_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?)
            """,
            (
                project_id,
                data["name"],
                data["line_name"],
                data["model_name"],
                data["model_type"],
                data.get("config_path", ""),
                _json(data["config"]),
                data.get("origin", "ui") or "ui",
                ts,
                ts,
            ),
        )
    return get_project(project_id) or {}


def update_project(project_id: str, data: dict) -> dict | None:
    with connection() as con:
        cur = con.execute(
            """
            UPDATE projects
            SET name=?, line_name=?, model_name=?, model_type=?, config_path=?, config_json=?,
                origin=COALESCE(NULLIF(?, ''), origin), updated_at=?
            WHERE id=?
            """,
            (
                data["name"],
                data["line_name"],
                data["model_name"],
                data["model_type"],
                data.get("config_path", ""),
                _json(data["config"]),
                data.get("origin", ""),
                now(),
                project_id,
            ),
        )
        if not cur.rowcount:
            return None
    return get_project(project_id)


def get_project_by_config_path(config_path: str) -> dict | None:
    with connection() as con:
        row = con.execute("SELECT * FROM projects WHERE config_path=?", (config_path,)).fetchone()
    return _decode_project(row)


def delete_project(project_id: str) -> bool:
    with connection() as con:
        con.execute("DELETE FROM runs WHERE project_id=?", (project_id,))
        cur = con.execute("DELETE FROM projects WHERE id=?", (project_id,))
    return bool(cur.rowcount)


def create_run(data: dict) -> dict:
    run_id = uuid.uuid4().hex[:12]
    with connection() as con:
        con.execute(
            """
            INSERT INTO runs (
                id, project_id, kind, status, current_stage, progress, progress_detail, pid,
                supervisor_pid, return_code, error, config_path, artifact_root, log_path,
                created_at, started_at, finished_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                run_id,
                data["project_id"],
                data["kind"],
                data.get("status", "queued"),
                data.get("current_stage", ""),
                data.get("progress", 0),
                data.get("progress_detail", ""),
                None,
                None,
                None,
                "",
                data.get("config_path", ""),
                data.get("artifact_root", ""),
                data.get("log_path", ""),
                now(),
                "",
                "",
            ),
        )
    return get_run(run_id) or {}


def update_run(run_id: str, **changes: Any) -> dict | None:
    allowed = {
        "status", "current_stage", "progress", "progress_detail", "pid", "supervisor_pid",
        "return_code", "error",
        "config_path", "artifact_root", "log_path", "started_at", "finished_at",
    }
    values = {k: v for k, v in changes.items() if k in allowed}
    if not values:
        return get_run(run_id)
    sql = "UPDATE runs SET " + ", ".join(f"{k}=?" for k in values) + " WHERE id=?"
    with connection() as con:
        con.execute(sql, (*values.values(), run_id))
    return get_run(run_id)


def get_run(run_id: str) -> dict | None:
    with connection() as con:
        row = con.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
    return _decode_run(row)


def list_runs(project_id: str = "", limit: int = 100) -> list[dict]:
    with connection() as con:
        if project_id:
            rows = con.execute(
                "SELECT * FROM runs WHERE project_id=? ORDER BY created_at DESC LIMIT ?",
                (project_id, limit),
            ).fetchall()
        else:
            rows = con.execute(
                "SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
    return [_decode_run(r) for r in rows if r is not None]  # type: ignore[misc]


def list_active_runs() -> list[dict]:
    with connection() as con:
        rows = con.execute(
            "SELECT * FROM runs WHERE status IN ('queued','running') ORDER BY created_at DESC"
        ).fetchall()
    return [_decode_run(r) for r in rows if r is not None]  # type: ignore[misc]


def latest_successful_run(project_id: str, kinds: tuple[str, ...] = ("train",)) -> dict | None:
    placeholders = ",".join("?" for _ in kinds)
    with connection() as con:
        row = con.execute(
            f"SELECT * FROM runs WHERE project_id=? AND status='succeeded' "
            f"AND kind IN ({placeholders}) ORDER BY finished_at DESC LIMIT 1",
            (project_id, *kinds),
        ).fetchone()
    return _decode_run(row)

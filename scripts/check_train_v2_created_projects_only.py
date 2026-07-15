#!/usr/bin/env python3
"""Regression check: Train v2 project list only exposes created UI projects."""
from __future__ import annotations

import json
import os
import sqlite3
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _payload(name: str) -> dict:
    return {
        "name": name,
        "line_name": "epump2",
        "model_name": name,
        "model_type": "xgb",
        "config_path": "",
        "config": {
            "line_name": "epump2",
            "model": {"model_name": name},
            "train": {"model_type": "xgb"},
        },
    }


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["FORVIA_TRAIN_V2_STATE_DIR"] = tmp

        from forvia_train_v2.backend import store

        store.init_db()
        created = store.create_project(_payload("created_project"))

        with sqlite3.connect(Path(tmp) / "forvia_train_v2.db") as con:
            columns = {row[1] for row in con.execute("PRAGMA table_info(projects)")}
            assert "origin" in columns, columns
            con.execute(
                """
                INSERT INTO projects (
                    id, name, line_name, model_name, model_type, config_path,
                    config_json, origin, created_at, updated_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    "auto_cfg",
                    "auto_cfg_project",
                    "epump2",
                    "auto_cfg_project",
                    "xgb",
                    "/tmp/auto_cfg_project.yaml",
                    json.dumps(_payload("auto_cfg_project")["config"]),
                    "auto_cfg",
                    "2026-01-01T00:00:00",
                    "2026-01-01T00:00:00",
                ),
            )

        listed = store.list_projects()
        assert [p["id"] for p in listed] == [created["id"]], listed
        assert listed[0]["origin"] == "ui"


if __name__ == "__main__":
    main()

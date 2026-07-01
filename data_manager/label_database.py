from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import sqlite3
from typing import Iterable, Mapping, Sequence
from uuid import uuid4


SCHEMA_VERSION = "4"
LABEL_STATUSES = {"unconfirmed", "pending", "confirmed", "rejected"}
DB_FILENAME = "label_records.db"
LEGACY_DB_FILENAME = "sample_records.db"


def database_path(data_root: str | Path) -> Path:
    return Path(data_root).expanduser() / "metadata" / DB_FILENAME


def resolve_database_path(
    path: str | Path,
    *,
    migrate_legacy: bool = True,
) -> Path:
    """Resolve the canonical database path and migrate the legacy filename."""
    requested = Path(path).expanduser()
    if requested.name == LEGACY_DB_FILENAME:
        legacy_path = requested
        canonical_path = requested.with_name(DB_FILENAME)
    elif requested.name == DB_FILENAME:
        canonical_path = requested
        legacy_path = requested.with_name(LEGACY_DB_FILENAME)
    else:
        return requested

    if canonical_path.exists() and legacy_path.exists():
        raise FileExistsError(
            "Both label databases exist; resolve the conflict before continuing: "
            f"{canonical_path}, {legacy_path}"
        )
    if migrate_legacy and legacy_path.is_file() and not canonical_path.exists():
        canonical_path.parent.mkdir(parents=True, exist_ok=True)
        legacy_path.replace(canonical_path)
        for suffix in ("-wal", "-shm"):
            legacy_sidecar = Path(f"{legacy_path}{suffix}")
            if legacy_sidecar.exists():
                legacy_sidecar.replace(Path(f"{canonical_path}{suffix}"))
    return canonical_path


def require_database(path: str | Path) -> Path:
    db_path = resolve_database_path(path)
    if not db_path.is_file():
        raise FileNotFoundError(f"label records database not found: {db_path}")
    return db_path

def _text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "null"} else text


def _int_or_none(value: object) -> int | None:
    text = _text(value)
    if not text:
        return None
    return int(float(text))


def _float_or_none(value: object) -> float | None:
    text = _text(value)
    if not text:
        return None
    return float(text)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _is_expert_source(source: object) -> bool:
    raw = _text(source).lower()
    return raw == "expert" or any(raw.startswith(prefix) for prefix in ("expert_", "expert-", "expert:", "expert."))


def _effective_label_status(source: object, requested_status: object = "") -> str:
    """Only expert-sourced labels are confirmed by default.

    Non-expert labels are stored as unconfirmed even if an old caller passes
    ``confirmed`` explicitly. ``rejected`` is preserved for manual rejection.
    """
    requested = _text(requested_status).lower()
    if requested == "rejected":
        return "rejected"
    if requested == "pending":
        return "pending"
    return "confirmed" if _is_expert_source(source) else "unconfirmed"


class LabelDatabase:
    """SQLite source of truth for analysis samples and label events."""

    def __init__(
        self,
        db_path: str | Path,
        *,
        readonly: bool = False,
        auto_export: bool = True,
    ):
        self.db_path = resolve_database_path(db_path)
        self.readonly = readonly
        self.auto_export = auto_export
        if readonly:
            if not self.db_path.is_file():
                raise FileNotFoundError(
                    f"label records database not found: {self.db_path}"
                )
        else:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self.initialize_schema()
            self._refresh_csv_exports()

    @property
    def csv_export_dir(self) -> Path:
        return self.db_path.parent / "csv"

    def _refresh_csv_exports(self) -> None:
        if self.readonly or not self.auto_export:
            return
        from .label_export import export_label_records

        export_label_records(self.db_path, self.csv_export_dir)

    def connect(self) -> sqlite3.Connection:
        if self.readonly:
            uri = f"{self.db_path.resolve().as_uri()}?mode=ro"
            connection = sqlite3.connect(uri, timeout=30, uri=True)
        else:
            connection = sqlite3.connect(self.db_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        if not self.readonly:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = NORMAL")
        return connection

    def initialize_schema(self) -> None:
        with self.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS schema_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS samples (
                    id INTEGER PRIMARY KEY,
                    line TEXT NOT NULL,
                    sn TEXT NOT NULL,
                    sample_id TEXT NOT NULL,
                    group_name TEXT NOT NULL DEFAULT '',
                    channel_name TEXT NOT NULL DEFAULT '',
                    sampling_rate INTEGER,
                    reference TEXT NOT NULL DEFAULT '',
                    time TEXT NOT NULL DEFAULT '',
                    sample_type TEXT NOT NULL DEFAULT 'channel',
                    sample_config TEXT,
                    origin TEXT NOT NULL DEFAULT 'index',
                    is_active INTEGER NOT NULL DEFAULT 1 CHECK(is_active IN (0, 1)),
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(line, sn, sample_id)
                );

                CREATE TABLE IF NOT EXISTS label_events (
                    id INTEGER PRIMARY KEY,
                    event_uuid TEXT NOT NULL UNIQUE,
                    sample_pk INTEGER NOT NULL,
                    timestamp TEXT NOT NULL,
                    source TEXT NOT NULL,
                    result_key TEXT,
                    result_id INTEGER,
                    result_name TEXT,
                    reason_key TEXT,
                    reason_id INTEGER,
                    reason_name TEXT,
                    reason_confidence REAL,
                    label_version TEXT,
                    note TEXT,
                    status TEXT NOT NULL DEFAULT 'unconfirmed'
                        CHECK(status IN ('unconfirmed', 'pending', 'confirmed', 'rejected')),
                    imported_at TEXT NOT NULL,
                    FOREIGN KEY(sample_pk) REFERENCES samples(id)
                        ON UPDATE CASCADE ON DELETE RESTRICT
                );

                CREATE INDEX IF NOT EXISTS idx_samples_line_sn
                    ON samples(line, sn);
                CREATE INDEX IF NOT EXISTS idx_samples_sample_id
                    ON samples(sample_id);
                CREATE INDEX IF NOT EXISTS idx_samples_active
                    ON samples(is_active, line);
                CREATE INDEX IF NOT EXISTS idx_labels_sample_time
                    ON label_events(sample_pk, timestamp DESC, id DESC);
                CREATE INDEX IF NOT EXISTS idx_labels_status_source
                    ON label_events(status, source);

                CREATE VIEW IF NOT EXISTS confirmed_label_events AS
                    SELECT * FROM label_events WHERE status = 'confirmed';

                CREATE VIEW IF NOT EXISTS unlabeled_samples AS
                    SELECT s.*
                    FROM samples s
                    WHERE s.is_active = 1
                      AND NOT EXISTS (
                        SELECT 1 FROM label_events e
                        WHERE e.sample_pk = s.id AND e.status = 'confirmed'
                      );

                CREATE VIEW IF NOT EXISTS latest_confirmed_labels AS
                    SELECT *
                    FROM (
                        SELECT
                            e.*,
                            ROW_NUMBER() OVER (
                                PARTITION BY sample_pk
                                ORDER BY timestamp DESC, id DESC
                            ) AS rn
                        FROM label_events e
                        WHERE status = 'confirmed'
                    )
                    WHERE rn = 1;
                """
            )
            self._migrate_sample_schema(connection)
            connection.execute(
                """
                INSERT INTO schema_meta(key, value) VALUES('schema_version', ?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value
                """,
                (SCHEMA_VERSION,),
            )
            self._migrate_label_status_schema(connection)

    def _migrate_sample_schema(self, connection: sqlite3.Connection) -> None:
        sample_columns = {
            str(row[1]) for row in connection.execute("PRAGMA table_info(samples)")
        }
        for column_name in ("reference", "time"):
            if column_name not in sample_columns:
                connection.execute(
                    f"ALTER TABLE samples ADD COLUMN {column_name} TEXT NOT NULL DEFAULT ''"
                )
        drop_columns = [c for c in ("tdms_storage_root", "relative_path", "tdms_path") if c in sample_columns]
        if not drop_columns:
            return
        for column_name in drop_columns:
            try:
                connection.execute(f"ALTER TABLE samples DROP COLUMN {column_name}")
            except sqlite3.OperationalError:
                self._rebuild_samples_without_path_columns(connection)
                return

    @staticmethod
    def _rebuild_samples_without_path_columns(connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            DROP VIEW IF EXISTS confirmed_label_events;
            DROP VIEW IF EXISTS unlabeled_samples;
            DROP VIEW IF EXISTS latest_confirmed_labels;

            ALTER TABLE samples RENAME TO samples__old_path_schema;

            CREATE TABLE samples (
                id INTEGER PRIMARY KEY,
                line TEXT NOT NULL,
                sn TEXT NOT NULL,
                sample_id TEXT NOT NULL,
                group_name TEXT NOT NULL DEFAULT '',
                channel_name TEXT NOT NULL DEFAULT '',
                sampling_rate INTEGER,
                reference TEXT NOT NULL DEFAULT '',
                time TEXT NOT NULL DEFAULT '',
                sample_type TEXT NOT NULL DEFAULT 'channel',
                sample_config TEXT,
                origin TEXT NOT NULL DEFAULT 'index',
                is_active INTEGER NOT NULL DEFAULT 1 CHECK(is_active IN (0, 1)),
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(line, sn, sample_id)
            );

            INSERT INTO samples(
                id, line, sn, sample_id, group_name, channel_name, sampling_rate,
                reference, time, sample_type, sample_config, origin, is_active, created_at, updated_at
            )
            SELECT
                id, line, sn, sample_id, group_name, channel_name, sampling_rate,
                COALESCE(reference, ''), COALESCE(time, ''),
                sample_type, sample_config, origin, is_active, created_at, updated_at
            FROM samples__old_path_schema;

            DROP TABLE samples__old_path_schema;

            CREATE INDEX IF NOT EXISTS idx_samples_line_sn
                ON samples(line, sn);
            CREATE INDEX IF NOT EXISTS idx_samples_sample_id
                ON samples(sample_id);
            CREATE INDEX IF NOT EXISTS idx_samples_active
                ON samples(is_active, line);
            """
        )
        LabelDatabase._create_label_views(connection)

    @staticmethod
    def _create_label_views(connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            CREATE VIEW IF NOT EXISTS confirmed_label_events AS
                SELECT * FROM label_events WHERE status = 'confirmed';

            CREATE VIEW IF NOT EXISTS unlabeled_samples AS
                SELECT s.*
                FROM samples s
                WHERE s.is_active = 1
                  AND NOT EXISTS (
                    SELECT 1 FROM label_events e
                    WHERE e.sample_pk = s.id AND e.status = 'confirmed'
                  );

            CREATE VIEW IF NOT EXISTS latest_confirmed_labels AS
                SELECT *
                FROM (
                    SELECT
                        e.*,
                        ROW_NUMBER() OVER (
                            PARTITION BY sample_pk
                            ORDER BY timestamp DESC, id DESC
                        ) AS rn
                    FROM label_events e
                    WHERE status = 'confirmed'
                )
                WHERE rn = 1;
            """
        )

    def _migrate_label_status_schema(self, connection: sqlite3.Connection) -> None:
        row = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='label_events'"
        ).fetchone()
        sql = str(row[0] if row else "")
        if "unconfirmed" in sql and "DEFAULT 'unconfirmed'" in sql:
            return

        connection.executescript(
            """
            DROP VIEW IF EXISTS confirmed_label_events;
            DROP VIEW IF EXISTS unlabeled_samples;
            DROP VIEW IF EXISTS latest_confirmed_labels;

            ALTER TABLE label_events RENAME TO label_events__old_status_schema;

            CREATE TABLE label_events (
                id INTEGER PRIMARY KEY,
                event_uuid TEXT NOT NULL UNIQUE,
                sample_pk INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                source TEXT NOT NULL,
                result_key TEXT,
                result_id INTEGER,
                result_name TEXT,
                reason_key TEXT,
                reason_id INTEGER,
                reason_name TEXT,
                reason_confidence REAL,
                label_version TEXT,
                note TEXT,
                status TEXT NOT NULL DEFAULT 'unconfirmed'
                    CHECK(status IN ('unconfirmed', 'pending', 'confirmed', 'rejected')),
                imported_at TEXT NOT NULL,
                FOREIGN KEY(sample_pk) REFERENCES samples(id)
                    ON UPDATE CASCADE ON DELETE RESTRICT
            );

            INSERT INTO label_events(
                id, event_uuid, sample_pk, timestamp, source,
                result_key, result_id, result_name,
                reason_key, reason_id, reason_name, reason_confidence,
                label_version, note, status, imported_at
            )
            SELECT
                id, event_uuid, sample_pk, timestamp, source,
                result_key, result_id, result_name,
                reason_key, reason_id, reason_name, reason_confidence,
                label_version, note,
                CASE
                    WHEN status = 'rejected' THEN 'rejected'
                    WHEN status = 'pending' THEN 'pending'
                    WHEN lower(source) = 'expert'
                         OR lower(source) LIKE 'expert\\_%' ESCAPE '\\'
                         OR lower(source) LIKE 'expert-%'
                         OR lower(source) LIKE 'expert:%'
                         OR lower(source) LIKE 'expert.%'
                    THEN 'confirmed'
                    ELSE 'unconfirmed'
                END AS status,
                imported_at
            FROM label_events__old_status_schema;

            DROP TABLE label_events__old_status_schema;

            CREATE INDEX IF NOT EXISTS idx_labels_sample_time
                ON label_events(sample_pk, timestamp DESC, id DESC);
            CREATE INDEX IF NOT EXISTS idx_labels_status_source
                ON label_events(status, source);
            """
        )
        self._create_label_views(connection)

    @staticmethod
    def _sample_values(row: Mapping[str, object], *, now: str) -> tuple[object, ...]:
        line = _text(row.get("line"))
        sn = _text(row.get("sn"))
        sample_id = _text(row.get("sample_id"))
        if not line or not sn or not sample_id:
            raise ValueError(f"Sample requires line/sn/sample_id: {dict(row)}")
        sample_config = row.get("sample_config")
        if isinstance(sample_config, (dict, list)):
            sample_config = json.dumps(sample_config, ensure_ascii=False, sort_keys=True)
        return (
            line,
            sn,
            sample_id,
            _text(row.get("group_name")),
            _text(row.get("channel_name")),
            _int_or_none(row.get("sampling_rate")),
            _text(row.get("reference")),
            _text(row.get("time")),
            _text(row.get("sample_type")) or "channel",
            _text(sample_config),
            _text(row.get("origin")) or "index",
            int(row.get("is_active", 1) or 0),
            now,
            now,
        )

    def upsert_samples(self, rows: Iterable[Mapping[str, object]]) -> int:
        now = _now()
        values = [self._sample_values(row, now=now) for row in rows]
        if not values:
            return 0
        with self.connect() as connection:
            connection.executemany(
                """
                INSERT INTO samples(
                    line, sn, sample_id, group_name, channel_name, sampling_rate,
                    reference, time,
                    sample_type, sample_config, origin, is_active, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(line, sn, sample_id) DO UPDATE SET
                    group_name=excluded.group_name,
                    channel_name=excluded.channel_name,
                    sampling_rate=excluded.sampling_rate,
                    reference=excluded.reference,
                    time=excluded.time,
                    sample_type=excluded.sample_type,
                    sample_config=excluded.sample_config,
                    origin=excluded.origin,
                    is_active=excluded.is_active,
                    updated_at=excluded.updated_at
                """,
                values,
            )
        self._refresh_csv_exports()
        return len(values)

    def replace_samples(
        self,
        rows: Iterable[Mapping[str, object]],
        *,
        lines: set[str] | None = None,
        origins: set[str] | None = None,
        all_samples: bool = False,
    ) -> int:
        rows_list = list(rows)
        scope = {_text(line) for line in (lines or set()) if _text(line)}
        origin_scope = {_text(origin) for origin in (origins or set()) if _text(origin)}
        if not scope:
            scope = {_text(row.get("line")) for row in rows_list if _text(row.get("line"))}
        now = _now()
        values = [self._sample_values(row, now=now) for row in rows_list]
        with self.connect() as connection:
            if all_samples:
                connection.execute("UPDATE samples SET is_active=0, updated_at=?", (now,))
            elif scope:
                line_placeholders = ",".join("?" for _ in scope)
                if origin_scope:
                    origin_placeholders = ",".join("?" for _ in origin_scope)
                    connection.execute(
                        f"""
                        UPDATE samples SET is_active=0, updated_at=?
                        WHERE line IN ({line_placeholders})
                          AND origin IN ({origin_placeholders})
                        """,
                        (now, *sorted(scope), *sorted(origin_scope)),
                    )
                else:
                    connection.execute(
                        f"UPDATE samples SET is_active=0, updated_at=? WHERE line IN ({line_placeholders})",
                        (now, *sorted(scope)),
                    )
            if values:
                connection.executemany(
                    """
                    INSERT INTO samples(
                        line, sn, sample_id, group_name, channel_name, sampling_rate,
                        reference, time,
                        sample_type, sample_config, origin, is_active, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(line, sn, sample_id) DO UPDATE SET
                        group_name=excluded.group_name,
                        channel_name=excluded.channel_name,
                        sampling_rate=excluded.sampling_rate,
                        reference=excluded.reference,
                        time=excluded.time,
                        sample_type=excluded.sample_type,
                        sample_config=excluded.sample_config,
                        origin=excluded.origin,
                        is_active=1,
                        updated_at=excluded.updated_at
                    """,
                    values,
                )
        self._refresh_csv_exports()
        return len(values)

    def get_sample(self, *, line: str, sn: str, sample_id: str) -> dict[str, object] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM samples WHERE line=? AND sn=? AND sample_id=?",
                (_text(line), _text(sn), _text(sample_id)),
            ).fetchone()
        return dict(row) if row is not None else None

    def list_samples(
        self,
        *,
        line: str | None = None,
        active_only: bool = True,
    ) -> list[dict[str, object]]:
        clauses: list[str] = []
        params: list[object] = []
        if line:
            clauses.append("line=?")
            params.append(_text(line))
        if active_only:
            clauses.append("is_active=1")
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        with self.connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM samples{where} ORDER BY line, sn, sample_id",
                params,
            ).fetchall()
        return [dict(row) for row in rows]

    def append_label(
        self,
        *,
        line: str,
        sn: str,
        sample_id: str,
        label: Mapping[str, object],
        status: str = "",
        event_uuid: str | None = None,
    ) -> int:
        sample = self.get_sample(line=line, sn=sn, sample_id=sample_id)
        if sample is None:
            raise KeyError(f"Sample not found: line={line}, sn={sn}, sample_id={sample_id}")
        timestamp = _text(label.get("timestamp")) or _now()
        source = _text(label.get("source"))
        if not source:
            raise ValueError("Label event requires source")
        status = _effective_label_status(source, status)
        if status not in LABEL_STATUSES:
            raise ValueError(f"Unsupported label status: {status}")
        with self.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO label_events(
                    event_uuid, sample_pk, timestamp, source,
                    result_key, result_id, result_name,
                    reason_key, reason_id, reason_name, reason_confidence,
                    label_version, note, status, imported_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_uuid or str(uuid4()),
                    sample["id"],
                    timestamp,
                    source,
                    _text(label.get("result_key")),
                    _int_or_none(label.get("result_id")),
                    _text(label.get("result_name")),
                    _text(label.get("reason_key")),
                    _int_or_none(label.get("reason_id")),
                    _text(label.get("reason_name")),
                    _float_or_none(label.get("reason_confidence")),
                    _text(label.get("label_version")),
                    _text(label.get("note")),
                    status,
                    _now(),
                ),
            )
            event_id = int(cursor.lastrowid)
        self._refresh_csv_exports()
        return event_id

    @staticmethod
    def _label_values(
        row: Mapping[str, object],
        *,
        sample_pk: int,
        default_status: str,
        imported_at: str,
    ) -> tuple[object, ...]:
        source = _text(row.get("source"))
        if not source:
            raise ValueError(f"Label event requires source: {dict(row)}")
        status = _effective_label_status(source, _text(row.get("status")) or default_status)
        if status not in LABEL_STATUSES:
            raise ValueError(f"Unsupported label status: {status}")
        return (
            _text(row.get("event_uuid")) or str(uuid4()),
            sample_pk,
            _text(row.get("timestamp")),
            source,
            _text(row.get("result_key")),
            _int_or_none(row.get("result_id")),
            _text(row.get("result_name")),
            _text(row.get("reason_key")),
            _int_or_none(row.get("reason_id")),
            _text(row.get("reason_name")),
            _float_or_none(row.get("reason_confidence")),
            _text(row.get("label_version")),
            _text(row.get("note")),
            status,
            imported_at,
        )

    def import_label_events(
        self,
        rows: Iterable[Mapping[str, object]],
        *,
        default_status: str = "",
    ) -> int:
        if _text(default_status) and _text(default_status).lower() not in LABEL_STATUSES:
            raise ValueError(f"Unsupported label status: {default_status}")
        rows_list = list(rows)
        if not rows_list:
            return 0

        imported_at = _now()
        with self.connect() as connection:
            sample_lookup = {
                (row["line"], row["sn"], row["sample_id"]): int(row["id"])
                for row in connection.execute("SELECT id, line, sn, sample_id FROM samples")
            }
            values: list[tuple[object, ...]] = []
            for row in rows_list:
                key = (_text(row.get("line")), _text(row.get("sn")), _text(row.get("sample_id")))
                sample_pk = sample_lookup.get(key)
                if sample_pk is None:
                    raise KeyError(
                        f"Sample not found: line={key[0]}, sn={key[1]}, sample_id={key[2]}"
                    )
                values.append(
                    self._label_values(
                        row,
                        sample_pk=sample_pk,
                        default_status=default_status,
                        imported_at=imported_at,
                    )
                )
            connection.executemany(
                """
                INSERT INTO label_events(
                    event_uuid, sample_pk, timestamp, source,
                    result_key, result_id, result_name,
                    reason_key, reason_id, reason_name, reason_confidence,
                    label_version, note, status, imported_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                values,
            )
        self._refresh_csv_exports()
        return len(values)

    def replace_label_events(
        self,
        rows: Iterable[Mapping[str, object]],
        *,
        default_status: str = "",
    ) -> int:
        """Replace all label events while preserving stable event IDs where supplied."""
        if _text(default_status) and _text(default_status).lower() not in LABEL_STATUSES:
            raise ValueError(f"Unsupported label status: {default_status}")
        rows_list = list(rows)
        imported_at = _now()
        with self.connect() as connection:
            sample_lookup = {
                (row["line"], row["sn"], row["sample_id"]): int(row["id"])
                for row in connection.execute("SELECT id, line, sn, sample_id FROM samples")
            }
            values: list[tuple[object, ...]] = []
            for row in rows_list:
                key = (_text(row.get("line")), _text(row.get("sn")), _text(row.get("sample_id")))
                sample_pk = sample_lookup.get(key)
                if sample_pk is None:
                    raise KeyError(
                        f"Sample not found: line={key[0]}, sn={key[1]}, sample_id={key[2]}"
                    )
                values.append(
                    self._label_values(
                        row,
                        sample_pk=sample_pk,
                        default_status=default_status,
                        imported_at=imported_at,
                    )
                )
            delete_status = _text(default_status).lower() or "unconfirmed"
            connection.execute("DELETE FROM label_events WHERE status=?", (delete_status,))
            if values:
                connection.executemany(
                    """
                    INSERT INTO label_events(
                        event_uuid, sample_pk, timestamp, source,
                        result_key, result_id, result_name,
                        reason_key, reason_id, reason_name, reason_confidence,
                        label_version, note, status, imported_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    values,
                )
        self._refresh_csv_exports()
        return len(values)

    def update_label_event(
        self,
        event_id: int,
        *,
        line: str,
        sn: str,
        sample_id: str,
        label: Mapping[str, object],
        status: str = "",
    ) -> None:
        sample = self.get_sample(line=line, sn=sn, sample_id=sample_id)
        if sample is None:
            raise KeyError(f"Sample not found: line={line}, sn={sn}, sample_id={sample_id}")
        source = _text(label.get("source"))
        if not source:
            raise ValueError("Label event requires source")
        status = _effective_label_status(source, status)
        if status not in LABEL_STATUSES:
            raise ValueError(f"Unsupported label status: {status}")
        with self.connect() as connection:
            cursor = connection.execute(
                """
                UPDATE label_events SET
                    sample_pk=?, timestamp=?, source=?,
                    result_key=?, result_id=?, result_name=?,
                    reason_key=?, reason_id=?, reason_name=?, reason_confidence=?,
                    label_version=?, note=?, status=?
                WHERE id=?
                """,
                (
                    sample["id"],
                    _text(label.get("timestamp")),
                    source,
                    _text(label.get("result_key")),
                    _int_or_none(label.get("result_id")),
                    _text(label.get("result_name")),
                    _text(label.get("reason_key")),
                    _int_or_none(label.get("reason_id")),
                    _text(label.get("reason_name")),
                    _float_or_none(label.get("reason_confidence")),
                    _text(label.get("label_version")),
                    _text(label.get("note")),
                    status,
                    int(event_id),
                ),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"Label event not found: id={event_id}")
        self._refresh_csv_exports()

    def set_label_status(self, event_ids: Sequence[int], status: str) -> int:
        if status not in LABEL_STATUSES:
            raise ValueError(f"Unsupported label status: {status}")
        ids = [int(event_id) for event_id in event_ids]
        if not ids:
            return 0
        placeholders = ",".join("?" for _ in ids)
        with self.connect() as connection:
            if status == "confirmed":
                cursor = connection.execute(
                    f"""
                    UPDATE label_events
                    SET status = CASE
                        WHEN lower(source) = 'expert'
                             OR lower(source) LIKE 'expert\\_%' ESCAPE '\\'
                             OR lower(source) LIKE 'expert-%'
                             OR lower(source) LIKE 'expert:%'
                             OR lower(source) LIKE 'expert.%'
                        THEN 'confirmed'
                        ELSE 'unconfirmed'
                    END
                    WHERE id IN ({placeholders})
                    """,
                    ids,
                )
            else:
                cursor = connection.execute(
                    f"UPDATE label_events SET status=? WHERE id IN ({placeholders})",
                    (status, *ids),
                )
            updated = int(cursor.rowcount)
        self._refresh_csv_exports()
        return updated

    def confirm_labels(self, event_ids: Sequence[int]) -> int:
        return self.set_label_status(event_ids, "confirmed")

    def reject_labels(self, event_ids: Sequence[int]) -> int:
        return self.set_label_status(event_ids, "rejected")

    def delete_label_events(self, event_ids: Sequence[int]) -> int:
        """按 id 物理删除标签事件（单条/多条），不重写整表。"""
        ids = [int(event_id) for event_id in event_ids]
        if not ids:
            return 0
        placeholders = ",".join("?" for _ in ids)
        with self.connect() as connection:
            cursor = connection.execute(
                f"DELETE FROM label_events WHERE id IN ({placeholders})", ids
            )
            deleted = int(cursor.rowcount)
        self._refresh_csv_exports()
        return deleted

    def list_label_events(
        self,
        *,
        line: str | None = None,
        sn: str | None = None,
        sample_id: str | None = None,
        statuses: set[str] | None = None,
    ) -> list[dict[str, object]]:
        clauses: list[str] = []
        params: list[object] = []
        for column, value in (("s.line", line), ("s.sn", sn), ("s.sample_id", sample_id)):
            if value:
                clauses.append(f"{column}=?")
                params.append(_text(value))
        status_scope = sorted(statuses or set())
        if status_scope:
            unsupported = set(status_scope) - LABEL_STATUSES
            if unsupported:
                raise ValueError(f"Unsupported label statuses: {sorted(unsupported)}")
            placeholders = ",".join("?" for _ in status_scope)
            clauses.append(f"e.status IN ({placeholders})")
            params.extend(status_scope)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        with self.connect() as connection:
            rows = connection.execute(
                f"""
                SELECT
                    e.*,
                    s.line,
                    s.sn,
                    s.sample_id,
                    s.reference,
                    s.time
                FROM label_events e
                JOIN samples s ON s.id=e.sample_pk
                {where}
                ORDER BY e.timestamp, e.id
                """,
                params,
            ).fetchall()
        return [dict(row) for row in rows]

    def counts(self) -> dict[str, int]:
        with self.connect() as connection:
            values = {
                "samples": connection.execute("SELECT COUNT(*) FROM samples").fetchone()[0],
                "active_samples": connection.execute(
                    "SELECT COUNT(*) FROM samples WHERE is_active=1"
                ).fetchone()[0],
                "placeholder_samples": connection.execute(
                    "SELECT COUNT(*) FROM samples WHERE origin='placeholder'"
                ).fetchone()[0],
                "label_events": connection.execute("SELECT COUNT(*) FROM label_events").fetchone()[0],
                "confirmed_labels": connection.execute(
                    "SELECT COUNT(*) FROM label_events WHERE status='confirmed'"
                ).fetchone()[0],
                "pending_labels": connection.execute(
                    "SELECT COUNT(*) FROM label_events WHERE status='pending'"
                ).fetchone()[0],
                "unconfirmed_labels": connection.execute(
                    "SELECT COUNT(*) FROM label_events WHERE status='unconfirmed'"
                ).fetchone()[0],
                "unlabeled_samples": connection.execute(
                    "SELECT COUNT(*) FROM unlabeled_samples"
                ).fetchone()[0],
            }
        return {key: int(value) for key, value in values.items()}


def load_sample_rows(db_path: str | Path, *, active_only: bool = True) -> list[dict[str, object]]:
    return LabelDatabase(require_database(db_path), readonly=True).list_samples(active_only=active_only)


def load_label_rows(db_path: str | Path, *, statuses=("confirmed",)) -> list[dict[str, object]]:
    return LabelDatabase(require_database(db_path), readonly=True).list_label_events(statuses=set(statuses))


def load_sample_dataframe(db_path: str | Path, *, active_only: bool = True):
    import pandas as pd
    return pd.DataFrame(load_sample_rows(db_path, active_only=active_only))


def load_label_dataframe(db_path: str | Path, *, statuses=("confirmed",)):
    import pandas as pd
    return pd.DataFrame(load_label_rows(db_path, statuses=statuses))


def replace_confirmed_labels(db_path: str | Path, rows) -> int:
    return LabelDatabase(require_database(db_path)).replace_label_events(rows)


def append_confirmed_labels(db_path: str | Path, rows) -> int:
    return LabelDatabase(require_database(db_path)).import_label_events(rows)

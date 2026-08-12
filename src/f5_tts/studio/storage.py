from __future__ import annotations

import hashlib
import json
import re
import shutil
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from f5_tts.studio.paths import StudioPaths


SCHEMA_VERSION = 3


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "project"


class StudioStore:
    def __init__(self, paths: StudioPaths):
        self.paths = paths.ensure()
        self.ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.paths.db_file)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    @staticmethod
    def _loads(raw: str | None) -> dict:
        if not raw:
            return {}
        return json.loads(raw)

    @staticmethod
    def _dumps(value: dict | list | None) -> str:
        return json.dumps(value or {}, ensure_ascii=True)

    @staticmethod
    def _row(row: sqlite3.Row | None) -> dict | None:
        if row is None:
            return None
        return dict(row)

    @staticmethod
    def _content_hash(path: str) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
        return {str(row[1]) for row in rows}

    def _ensure_column(self, conn: sqlite3.Connection, table: str, column: str, ddl: str) -> None:
        if column not in self._table_columns(conn, table):
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")

    def ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS schema_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS projects (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    slug TEXT NOT NULL UNIQUE,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS reference_voices (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id INTEGER NOT NULL,
                    kind TEXT NOT NULL CHECK(kind IN ('reference', 'style')),
                    name TEXT NOT NULL,
                    audio_path TEXT NOT NULL,
                    transcript TEXT NOT NULL DEFAULT '',
                    analysis_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS pronunciation_rules (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id INTEGER NOT NULL,
                    source TEXT NOT NULL,
                    replacement TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(project_id, source),
                    FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS voice_profiles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL DEFAULT '',
                    is_default INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(project_id, name),
                    FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS voice_profile_members (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    profile_id INTEGER NOT NULL,
                    reference_id INTEGER NOT NULL,
                    position INTEGER NOT NULL DEFAULT 0,
                    weight REAL NOT NULL DEFAULT 1.0,
                    created_at TEXT NOT NULL,
                    UNIQUE(profile_id, reference_id),
                    FOREIGN KEY(profile_id) REFERENCES voice_profiles(id) ON DELETE CASCADE,
                    FOREIGN KEY(reference_id) REFERENCES reference_voices(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS generation_jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    recipe_json TEXT NOT NULL,
                    result_json TEXT NOT NULL DEFAULT '{}',
                    error_text TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS audio_assets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id INTEGER NOT NULL,
                    job_id INTEGER,
                    kind TEXT NOT NULL,
                    label TEXT NOT NULL,
                    path TEXT NOT NULL,
                    duration_seconds REAL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE,
                    FOREIGN KEY(job_id) REFERENCES generation_jobs(id) ON DELETE SET NULL
                );

                CREATE TABLE IF NOT EXISTS system_settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_reference_voices_project_kind_updated
                    ON reference_voices(project_id, kind, updated_at DESC, id DESC);
                CREATE INDEX IF NOT EXISTS idx_voice_profiles_project_default_updated
                    ON voice_profiles(project_id, is_default DESC, updated_at DESC, id DESC);
                CREATE INDEX IF NOT EXISTS idx_voice_profile_members_profile_position
                    ON voice_profile_members(profile_id, position ASC, id ASC);
                CREATE INDEX IF NOT EXISTS idx_pronunciation_rules_project_source
                    ON pronunciation_rules(project_id, source);
                CREATE INDEX IF NOT EXISTS idx_generation_jobs_project_status_updated
                    ON generation_jobs(project_id, status, updated_at DESC, id DESC);
                CREATE INDEX IF NOT EXISTS idx_audio_assets_project_kind_created
                    ON audio_assets(project_id, kind, created_at DESC, id DESC);
                CREATE INDEX IF NOT EXISTS idx_audio_assets_job_created
                    ON audio_assets(job_id, created_at DESC, id DESC);
                """
            )
            self._ensure_column(conn, "reference_voices", "content_hash", "TEXT")
            self._ensure_column(conn, "audio_assets", "content_hash", "TEXT")
            now = utc_now()
            conn.execute(
                """
                INSERT INTO schema_meta (key, value, updated_at)
                VALUES ('schema_version', ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    updated_at = excluded.updated_at
                """,
                (str(SCHEMA_VERSION), now),
            )

    def ensure_default_project(self) -> dict:
        projects = self.list_projects()
        if projects:
            return projects[0]
        return self.create_project("Studio Sandbox", "Default local project for experiments and previews.")

    def _unique_slug(self, name: str) -> str:
        base = slugify(name)
        slug = base
        index = 2
        existing = {project["slug"] for project in self.list_projects()}
        while slug in existing:
            slug = f"{base}-{index}"
            index += 1
        return slug

    def create_project(self, name: str, description: str = "") -> dict:
        now = utc_now()
        slug = self._unique_slug(name)
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO projects (slug, name, description, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (slug, name.strip(), description.strip(), now, now),
            )
            project_id = int(cursor.lastrowid)
        self.project_dir(slug).mkdir(parents=True, exist_ok=True)
        return self.get_project_summary(project_id)

    def project_dir(self, project_slug: str) -> Path:
        return self.paths.projects / project_slug

    def get_project_summary(self, project_id: int) -> dict:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
        project = self._row(row)
        if project is None:
            raise KeyError(f"Project {project_id} was not found.")
        return project

    def list_projects(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM projects ORDER BY updated_at DESC, id DESC").fetchall()
        return [self._row(row) for row in rows]

    def get_project_detail(self, project_id: int) -> dict:
        project = self.get_project_summary(project_id)
        project["references"] = self.list_voice_assets(project_id, "reference")
        project["styles"] = self.list_voice_assets(project_id, "style")
        project["voice_profiles"] = self.list_voice_profiles(project_id)
        project["assets"] = self.list_audio_assets(project_id)
        project["jobs"] = self.list_jobs(project_id)
        project["pronunciation_rules"] = self.list_pronunciation_rules(project_id)
        return project

    def _copy_audio(self, project_slug: str, kind: str, name: str, source_path: str) -> str:
        source = Path(source_path)
        suffix = source.suffix or ".wav"
        target_dir = self.project_dir(project_slug) / kind
        target_dir.mkdir(parents=True, exist_ok=True)
        stem = slugify(name) or kind
        candidate = target_dir / f"{stem}{suffix}"
        index = 2
        while candidate.exists():
            candidate = target_dir / f"{stem}-{index}{suffix}"
            index += 1
        shutil.copy2(source, candidate)
        return str(candidate)

    def save_voice_asset(
        self,
        project_id: int,
        kind: str,
        name: str,
        audio_path: str,
        transcript: str,
        analysis: dict,
    ) -> dict:
        project = self.get_project_summary(project_id)
        now = utc_now()
        stored_audio_path = self._copy_audio(project["slug"], f"{kind}s", name, audio_path)
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO reference_voices (
                    project_id, kind, name, audio_path, transcript, analysis_json, content_hash, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    project_id,
                    kind,
                    name.strip(),
                    stored_audio_path,
                    transcript.strip(),
                    self._dumps(analysis),
                    self._content_hash(stored_audio_path),
                    now,
                    now,
                ),
            )
            voice_id = int(cursor.lastrowid)
            conn.execute("UPDATE projects SET updated_at = ? WHERE id = ?", (now, project_id))
        return self.get_voice_asset(voice_id)

    def list_voice_assets(self, project_id: int, kind: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM reference_voices
                WHERE project_id = ? AND kind = ?
                ORDER BY updated_at DESC, id DESC
                """,
                (project_id, kind),
            ).fetchall()
        values = []
        for row in rows:
            item = self._row(row)
            item["analysis"] = self._loads(item.pop("analysis_json", None))
            values.append(item)
        return values

    def get_voice_asset(self, voice_id: int) -> dict:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM reference_voices WHERE id = ?", (voice_id,)).fetchone()
        item = self._row(row)
        if item is None:
            raise KeyError(f"Voice asset {voice_id} was not found.")
        item["analysis"] = self._loads(item.pop("analysis_json", None))
        return item

    def save_voice_profile(
        self,
        project_id: int,
        name: str,
        reference_ids: list[int],
        *,
        description: str = "",
        profile_id: int | None = None,
        is_default: bool = False,
    ) -> dict:
        normalized_reference_ids = [int(reference_id) for reference_id in reference_ids]
        if not normalized_reference_ids:
            raise ValueError("Voice profiles need at least one saved reference.")

        now = utc_now()
        references = [self.get_voice_asset(reference_id) for reference_id in normalized_reference_ids]
        for reference in references:
            if int(reference["project_id"]) != int(project_id) or reference["kind"] != "reference":
                raise ValueError("Voice profiles can only include saved reference voices from the active project.")

        with self._connect() as conn:
            if is_default:
                conn.execute(
                    "UPDATE voice_profiles SET is_default = 0, updated_at = ? WHERE project_id = ?", (now, project_id)
                )

            if profile_id is None:
                cursor = conn.execute(
                    """
                    INSERT INTO voice_profiles (project_id, name, description, is_default, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (project_id, name.strip(), description.strip(), 1 if is_default else 0, now, now),
                )
                profile_id = int(cursor.lastrowid)
            else:
                cursor = conn.execute(
                    "SELECT * FROM voice_profiles WHERE id = ? AND project_id = ?",
                    (int(profile_id), int(project_id)),
                )
                if cursor.fetchone() is None:
                    raise KeyError(f"Voice profile {profile_id} was not found.")
                conn.execute(
                    """
                    UPDATE voice_profiles
                    SET name = ?, description = ?, is_default = ?, updated_at = ?
                    WHERE id = ? AND project_id = ?
                    """,
                    (name.strip(), description.strip(), 1 if is_default else 0, now, int(profile_id), int(project_id)),
                )
                conn.execute("DELETE FROM voice_profile_members WHERE profile_id = ?", (int(profile_id),))

            for position, reference_id in enumerate(normalized_reference_ids, start=1):
                conn.execute(
                    """
                    INSERT INTO voice_profile_members (profile_id, reference_id, position, weight, created_at)
                    VALUES (?, ?, ?, 1.0, ?)
                    """,
                    (int(profile_id), int(reference_id), int(position), now),
                )

            conn.execute("UPDATE projects SET updated_at = ? WHERE id = ?", (now, int(project_id)))
        return self.get_voice_profile(int(profile_id))

    def list_voice_profiles(self, project_id: int) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    vp.*,
                    COUNT(vpm.reference_id) AS member_count
                FROM voice_profiles vp
                LEFT JOIN voice_profile_members vpm ON vpm.profile_id = vp.id
                WHERE vp.project_id = ?
                GROUP BY vp.id
                ORDER BY vp.is_default DESC, vp.updated_at DESC, vp.id DESC
                """,
                (project_id,),
            ).fetchall()
            profiles = []
            for row in rows:
                profile = self._row(row)
                profile["member_count"] = int(profile.get("member_count", 0) or 0)
                member_rows = conn.execute(
                    """
                    SELECT rv.id, rv.name
                    FROM voice_profile_members vpm
                    JOIN reference_voices rv ON rv.id = vpm.reference_id
                    WHERE vpm.profile_id = ?
                    ORDER BY vpm.position ASC, vpm.id ASC
                    """,
                    (int(profile["id"]),),
                ).fetchall()
                profile["member_ids"] = [int(member["id"]) for member in member_rows]
                profile["member_names"] = [str(member["name"]) for member in member_rows]
                profiles.append(profile)
        return profiles

    def get_voice_profile(self, profile_id: int) -> dict:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM voice_profiles WHERE id = ?", (profile_id,)).fetchone()
            member_rows = conn.execute(
                """
                SELECT
                    vpm.position,
                    rv.*
                FROM voice_profile_members vpm
                JOIN reference_voices rv ON rv.id = vpm.reference_id
                WHERE vpm.profile_id = ?
                ORDER BY vpm.position ASC, vpm.id ASC
                """,
                (profile_id,),
            ).fetchall()
        profile = self._row(row)
        if profile is None:
            raise KeyError(f"Voice profile {profile_id} was not found.")
        members: list[dict] = []
        for member_row in member_rows:
            item = dict(member_row)
            item["analysis"] = self._loads(item.pop("analysis_json", None))
            members.append(item)
        profile["members"] = members
        profile["member_ids"] = [int(item["id"]) for item in members]
        profile["member_names"] = [str(item["name"]) for item in members]
        profile["member_count"] = len(members)
        return profile

    def sync_default_voice_profile(
        self, project_id: int, profile_name: str = "Auto profile · all references"
    ) -> dict | None:
        references = self.list_voice_assets(project_id, "reference")
        if not references:
            return None

        reference_ids = [int(reference["id"]) for reference in references]
        existing_default = next(
            (profile for profile in self.list_voice_profiles(project_id) if int(profile.get("is_default", 0)) == 1),
            None,
        )
        description = "System-managed profile that keeps every saved reference from this project available for automatic selection."
        if existing_default is None:
            return self.save_voice_profile(
                project_id,
                profile_name,
                reference_ids,
                description=description,
                is_default=True,
            )
        return self.save_voice_profile(
            project_id,
            str(existing_default["name"]),
            reference_ids,
            description=str(existing_default.get("description", "") or description),
            profile_id=int(existing_default["id"]),
            is_default=True,
        )

    def upsert_pronunciation_rule(self, project_id: int, source: str, replacement: str) -> dict:
        now = utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO pronunciation_rules (project_id, source, replacement, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(project_id, source) DO UPDATE SET
                    replacement = excluded.replacement,
                    updated_at = excluded.updated_at
                """,
                (project_id, source.strip(), replacement.strip(), now, now),
            )
            row = conn.execute(
                """
                SELECT * FROM pronunciation_rules
                WHERE project_id = ? AND source = ?
                """,
                (project_id, source.strip()),
            ).fetchone()
            conn.execute("UPDATE projects SET updated_at = ? WHERE id = ?", (now, project_id))
        return self._row(row)

    def list_pronunciation_rules(self, project_id: int) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM pronunciation_rules
                WHERE project_id = ?
                ORDER BY source ASC
                """,
                (project_id,),
            ).fetchall()
        return [self._row(row) for row in rows]

    def create_job(self, project_id: int, name: str, recipe: dict) -> dict:
        now = utc_now()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO generation_jobs (
                    project_id, name, status, recipe_json, result_json, created_at, updated_at
                ) VALUES (?, ?, 'queued', ?, '{}', ?, ?)
                """,
                (project_id, name.strip(), self._dumps(recipe), now, now),
            )
            job_id = int(cursor.lastrowid)
            conn.execute("UPDATE projects SET updated_at = ? WHERE id = ?", (now, project_id))
        return self.get_job(job_id)

    def update_job(
        self,
        job_id: int,
        *,
        status: str | None = None,
        result: dict | None = None,
        error_text: str | None = None,
    ) -> dict:
        current = self.get_job(job_id)
        now = utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE generation_jobs
                SET status = ?, result_json = ?, error_text = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    status or current["status"],
                    self._dumps(result if result is not None else current["result"]),
                    error_text,
                    now,
                    job_id,
                ),
            )
            conn.execute("UPDATE projects SET updated_at = ? WHERE id = ?", (now, current["project_id"]))
        return self.get_job(job_id)

    def get_job(self, job_id: int) -> dict:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM generation_jobs WHERE id = ?", (job_id,)).fetchone()
        job = self._row(row)
        if job is None:
            raise KeyError(f"Job {job_id} was not found.")
        job["recipe"] = self._loads(job.pop("recipe_json", None))
        job["result"] = self._loads(job.pop("result_json", None))
        if not job["result"]:
            job["result"] = None
        return job

    def list_jobs(self, project_id: int | None = None, limit: int = 100) -> list[dict]:
        query = "SELECT * FROM generation_jobs"
        params: tuple[int, ...] = ()
        if project_id is not None:
            query += " WHERE project_id = ?"
            params = (project_id,)
        query += " ORDER BY created_at DESC, id DESC LIMIT ?"
        params += (limit,)
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        jobs = []
        for row in rows:
            job = self._row(row)
            job["recipe"] = self._loads(job.pop("recipe_json", None))
            job["result"] = self._loads(job.pop("result_json", None))
            if not job["result"]:
                job["result"] = None
            jobs.append(job)
        return jobs

    def save_audio_asset(
        self,
        project_id: int,
        job_id: int | None,
        kind: str,
        label: str,
        path: str,
        duration_seconds: float | None,
        metadata: dict,
    ) -> dict:
        now = utc_now()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO audio_assets (
                    project_id, job_id, kind, label, path, duration_seconds, metadata_json, content_hash, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    project_id,
                    job_id,
                    kind,
                    label,
                    path,
                    duration_seconds,
                    self._dumps(metadata),
                    self._content_hash(path),
                    now,
                ),
            )
            asset_id = int(cursor.lastrowid)
            conn.execute("UPDATE projects SET updated_at = ? WHERE id = ?", (now, project_id))
        return self.get_audio_asset(asset_id)

    def save_source_asset(self, project_id: int, label: str, audio_path: str, metadata: dict) -> dict:
        project = self.get_project_summary(project_id)
        stored_audio_path = self._copy_audio(project["slug"], "sources", label, audio_path)
        duration_seconds = metadata.get("duration_seconds")
        return self.save_audio_asset(
            project_id=project_id,
            job_id=None,
            kind="source",
            label=label.strip(),
            path=stored_audio_path,
            duration_seconds=duration_seconds,
            metadata=metadata,
        )

    def get_audio_asset(self, asset_id: int) -> dict:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM audio_assets WHERE id = ?", (asset_id,)).fetchone()
        asset = self._row(row)
        if asset is None:
            raise KeyError(f"Audio asset {asset_id} was not found.")
        asset["metadata"] = self._loads(asset.pop("metadata_json", None))
        return asset

    def list_audio_assets(self, project_id: int, kind: str | None = None) -> list[dict]:
        query = "SELECT * FROM audio_assets WHERE project_id = ?"
        params: tuple[object, ...] = (project_id,)
        if kind:
            query += " AND kind = ?"
            params += (kind,)
        query += " ORDER BY created_at DESC, id DESC"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        assets = []
        for row in rows:
            asset = self._row(row)
            asset["metadata"] = self._loads(asset.pop("metadata_json", None))
            assets.append(asset)
        return assets

    def set_setting(self, key: str, value: str) -> None:
        now = utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO system_settings (key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    updated_at = excluded.updated_at
                """,
                (key, value, now),
            )

    def get_setting(self, key: str, default: str | None = None) -> str | None:
        with self._connect() as conn:
            row = conn.execute("SELECT value FROM system_settings WHERE key = ?", (key,)).fetchone()
        return row["value"] if row is not None else default

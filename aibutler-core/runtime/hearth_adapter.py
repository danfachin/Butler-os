"""Hearth/Dolt adapter for DewDrops workflow data.

DewDrops workflow state (runs, stages, fleet, findings) lives on a Dolt
branch of Hearth's database (`db/dewdrops-dev` today, `db/main` after merge).
This adapter is the only place in Butler that speaks MySQL to Dolt — the
runtime delegates here and stays storage-agnostic.

Connection config (env vars, all optional):
    DEWDROPS_DB_HOST     default 127.0.0.1
    DEWDROPS_DB_PORT     default 3306
    DEWDROPS_DB_USER     default root
    DEWDROPS_DB_PASSWORD default ""
    DEWDROPS_DB_NAME     default db
    DEWDROPS_DB_BRANCH   default dewdrops-dev

Every connection runs ``USE `{DB_NAME}/{DB_BRANCH}` `` so writes land on
the intended branch. Flip DEWDROPS_DB_BRANCH to retarget.
"""
from __future__ import annotations

import json
import os
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator

try:
    import pymysql  # type: ignore[import]
    from pymysql.cursors import DictCursor  # type: ignore[import]
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "hearth_adapter requires pymysql. Install with: pip install pymysql"
    ) from e


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _db_config() -> dict[str, Any]:
    return {
        "host": _env("DEWDROPS_DB_HOST", "127.0.0.1"),
        "port": int(_env("DEWDROPS_DB_PORT", "3306")),
        "user": _env("DEWDROPS_DB_USER", "root"),
        "password": _env("DEWDROPS_DB_PASSWORD", ""),
        "database": None,
        "charset": "utf8mb4",
        "autocommit": True,
        "cursorclass": DictCursor,
    }


def _branch_ref() -> str:
    db = _env("DEWDROPS_DB_NAME", "db")
    branch = _env("DEWDROPS_DB_BRANCH", "dewdrops-dev")
    return f"`{db}/{branch}`"


@contextmanager
def _conn() -> Iterator[Any]:
    connection = pymysql.connect(**_db_config())
    try:
        with connection.cursor() as cur:
            cur.execute(f"USE {_branch_ref()}")
        yield connection
    finally:
        connection.close()


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.replace(tzinfo=value.tzinfo or timezone.utc).isoformat()
    return str(value)


def _parse_json(value: Any) -> Any:
    if value is None or value == "":
        return None
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return None


def _new_id() -> str:
    return str(uuid.uuid4())


# ──────────────────────────────────────────────────────────────────────────────
# READ
# ──────────────────────────────────────────────────────────────────────────────

def list_active_workflow_runs(
    sprint_id: str | None = None,
    limit: int = 20,
) -> list[dict]:
    """Return runs in status draft|active, newest first.

    Uses v_active_workflow_runs for the join to workflows + current stage +
    roadmap, then re-attaches sprint_id from workflow_runs (the view predates
    that column).
    """
    sql = "SELECT * FROM v_active_workflow_runs"
    params: list[Any] = []
    if sprint_id:
        sql += (
            " WHERE workflow_run_id IN ("
            "  SELECT id FROM workflow_runs WHERE sprint_id = %s"
            ")"
        )
        params.append(sprint_id)
    sql += " ORDER BY started_at DESC, workflow_run_id DESC LIMIT %s"
    params.append(int(limit))

    with _conn() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        view_rows = cur.fetchall() or []
        if not view_rows:
            return []

        run_ids = [r["workflow_run_id"] for r in view_rows]
        placeholders = ",".join(["%s"] * len(run_ids))
        cur.execute(
            f"SELECT id, sprint_id FROM workflow_runs WHERE id IN ({placeholders})",
            run_ids,
        )
        sprint_by_run = {r["id"]: r["sprint_id"] for r in cur.fetchall() or []}

    out: list[dict] = []
    for row in view_rows:
        out.append({
            "id": row["workflow_run_id"],
            "workflow_id": row["workflow_id"],
            "workflow_name": row["workflow_name"],
            "sprint_id": sprint_by_run.get(row["workflow_run_id"]),
            "roadmap_id": row.get("roadmap_id"),
            "roadmap_name": row.get("roadmap_name"),
            "brief_document_id": row.get("brief_document_id"),
            "current_stage_key": row.get("current_stage_key"),
            "current_stage_name": row.get("current_stage_name"),
            "status": row["status"],
            "started_at": _iso(row.get("started_at")),
        })
    return out


def get_workflow_fleet(workflow_run_id: str) -> list[dict]:
    sql = (
        "SELECT workflow_run_id, stage_key, stage_name, stage_order, "
        "       persona_slug, authority, selected_by, selected_basis, "
        "       pilot_pinned "
        "FROM v_fleet_manifest WHERE workflow_run_id = %s "
        "ORDER BY stage_order, persona_slug"
    )
    with _conn() as conn, conn.cursor() as cur:
        cur.execute(sql, (workflow_run_id,))
        rows = cur.fetchall() or []
    return [{
        "workflow_run_id": r["workflow_run_id"],
        "stage_key": r["stage_key"],
        "stage_name": r["stage_name"],
        "stage_order": r["stage_order"],
        "persona_slug": r["persona_slug"],
        "authority": r["authority"],
        "selected_by": r["selected_by"],
        "selected_basis": r.get("selected_basis"),
        "pilot_pinned": bool(r.get("pilot_pinned")),
    } for r in rows]


def get_workflow_findings(
    workflow_run_id: str,
    open_only: bool = True,
) -> list[dict]:
    sql = (
        "SELECT f.id, f.workflow_run_id, f.workflow_stage_id, "
        "       ws.stage_key, f.emitted_by_persona_slug, f.severity, "
        "       f.verdict, f.title, f.body, f.preserved_artifact_ids, "
        "       f.implicates_artifact_ids, f.created_at, f.resolved_at "
        "FROM findings f "
        "JOIN workflow_stages ws ON ws.id = f.workflow_stage_id "
        "WHERE f.workflow_run_id = %s"
    )
    if open_only:
        sql += " AND f.resolved_at IS NULL"
    sql += " ORDER BY f.created_at DESC"

    with _conn() as conn, conn.cursor() as cur:
        cur.execute(sql, (workflow_run_id,))
        rows = cur.fetchall() or []
    return [{
        "id": r["id"],
        "workflow_run_id": r["workflow_run_id"],
        "stage_key": r["stage_key"],
        "emitted_by_persona_slug": r["emitted_by_persona_slug"],
        "severity": r["severity"],
        "verdict": r["verdict"],
        "title": r["title"],
        "body": r["body"],
        "preserved_artifact_ids": _parse_json(r.get("preserved_artifact_ids")),
        "implicates_artifact_ids": _parse_json(r.get("implicates_artifact_ids")),
        "created_at": _iso(r["created_at"]),
        "resolved_at": _iso(r.get("resolved_at")),
    } for r in rows]


# ──────────────────────────────────────────────────────────────────────────────
# WRITE
# ──────────────────────────────────────────────────────────────────────────────

def start_workflow_run(
    workflow_id: str,
    sprint_id: str | None = None,
    roadmap_id: str | None = None,
    task_id: str | None = None,
    brief_document_id: str | None = None,
    source_session_id: str | None = None,
) -> dict:
    """Create a workflow run, seed rule-floor fleet from stage_persona_defaults.

    The run starts on the first stage (sort_order minimum) with status=draft
    until advance is called. Returns the run summary shape used by READ.
    """
    run_id = _new_id()
    now = datetime.now(timezone.utc)

    with _conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT id FROM workflow_stages WHERE workflow_id = %s "
            "ORDER BY sort_order, id LIMIT 1",
            (workflow_id,),
        )
        first_stage = cur.fetchone()
        if not first_stage:
            raise ValueError(f"Workflow {workflow_id} has no stages.")
        current_stage_id = first_stage["id"]

        cur.execute(
            "INSERT INTO workflow_runs "
            "(id, workflow_id, sprint_id, roadmap_id, task_id, "
            " brief_document_id, current_stage_id, status, started_at, "
            " source_session_id) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, 'active', %s, %s)",
            (
                run_id, workflow_id, sprint_id, roadmap_id, task_id,
                brief_document_id, current_stage_id, now, source_session_id,
            ),
        )

        # Seed rule-floor fleet: copy every stage_persona_defaults row into
        # run_fleet with selected_by='rule-floor'.
        cur.execute(
            "SELECT spd.workflow_stage_id, spd.persona_slug, "
            "       spd.default_authority "
            "FROM stage_persona_defaults spd "
            "JOIN workflow_stages ws ON ws.id = spd.workflow_stage_id "
            "WHERE ws.workflow_id = %s",
            (workflow_id,),
        )
        defaults = cur.fetchall() or []
        for d in defaults:
            cur.execute(
                "INSERT INTO run_fleet "
                "(id, workflow_run_id, workflow_stage_id, persona_slug, "
                " authority, selected_by, selected_basis, pilot_pinned) "
                "VALUES (%s, %s, %s, %s, %s, 'rule-floor', "
                "        'seeded from stage_persona_defaults at run start', 0)",
                (
                    _new_id(), run_id, d["workflow_stage_id"],
                    d["persona_slug"], d["default_authority"],
                ),
            )

    runs = list_active_workflow_runs(limit=1)
    for r in runs:
        if r["id"] == run_id:
            return r
    # Fallback — the run exists but view didn't return it (e.g. status filter).
    return {
        "id": run_id,
        "workflow_id": workflow_id,
        "workflow_name": "",
        "sprint_id": sprint_id,
        "roadmap_id": roadmap_id,
        "roadmap_name": None,
        "brief_document_id": brief_document_id,
        "current_stage_key": None,
        "current_stage_name": None,
        "status": "active",
        "started_at": _iso(now),
    }


def emit_workflow_finding(
    workflow_run_id: str,
    workflow_stage_id: str,
    emitted_by_persona_slug: str,
    severity: str,
    verdict: str,
    title: str,
    body: str,
    preserved_artifact_ids: list[str] | None = None,
    implicates_artifact_ids: list[str] | None = None,
    root_cause_hint: str | None = None,
) -> dict:
    finding_id = _new_id()

    with _conn() as conn, conn.cursor() as cur:
        # Derive authority_at_emission from current run_fleet row.
        cur.execute(
            "SELECT authority FROM run_fleet "
            "WHERE workflow_run_id = %s AND workflow_stage_id = %s "
            "  AND persona_slug = %s AND detached_at IS NULL "
            "LIMIT 1",
            (workflow_run_id, workflow_stage_id, emitted_by_persona_slug),
        )
        row = cur.fetchone()
        if not row:
            raise ValueError(
                f"Persona {emitted_by_persona_slug} is not attached to this "
                f"run/stage; cannot emit finding."
            )
        authority = row["authority"]

        cur.execute(
            "INSERT INTO findings "
            "(id, workflow_run_id, workflow_stage_id, emitted_by_persona_slug, "
            " authority_at_emission, severity, verdict, title, body, "
            " preserved_artifact_ids, implicates_artifact_ids, root_cause_hint) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (
                finding_id, workflow_run_id, workflow_stage_id,
                emitted_by_persona_slug, authority, severity, verdict,
                title, body,
                json.dumps(preserved_artifact_ids) if preserved_artifact_ids else None,
                json.dumps(implicates_artifact_ids) if implicates_artifact_ids else None,
                root_cause_hint,
            ),
        )

    findings = get_workflow_findings(workflow_run_id, open_only=False)
    for f in findings:
        if f["id"] == finding_id:
            return f
    raise RuntimeError("Finding inserted but not retrievable — schema drift?")


def advance_workflow_stage(
    workflow_run_id: str,
    actor_persona_slug: str,
    note: str | None = None,
) -> dict:
    """Move run to the next stage by sort_order; complete if already last."""
    with _conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT wr.workflow_id, wr.current_stage_id, ws.sort_order "
            "FROM workflow_runs wr "
            "LEFT JOIN workflow_stages ws ON ws.id = wr.current_stage_id "
            "WHERE wr.id = %s",
            (workflow_run_id,),
        )
        row = cur.fetchone()
        if not row:
            raise ValueError(f"Workflow run {workflow_run_id} not found.")

        cur.execute(
            "SELECT id, sort_order FROM workflow_stages "
            "WHERE workflow_id = %s AND sort_order > %s "
            "ORDER BY sort_order ASC LIMIT 1",
            (row["workflow_id"], row["sort_order"] or -1),
        )
        nxt = cur.fetchone()

        if nxt:
            cur.execute(
                "UPDATE workflow_runs SET current_stage_id = %s, "
                "status = 'active' WHERE id = %s",
                (nxt["id"], workflow_run_id),
            )
        else:
            cur.execute(
                "UPDATE workflow_runs SET status = 'completed', "
                "completed_at = %s WHERE id = %s",
                (datetime.now(timezone.utc), workflow_run_id),
            )
        # NOTE: passback_events is reserved for finding-triggered cascades
        # (requires triggered_by_finding_id + cascade_policy_applied). A plain
        # stage advance is not a passback; we track actor/note via the run's
        # current_stage_id + updated_at. Dedicated run-event log is out of
        # scope for v0.1.
        _ = (actor_persona_slug, note)  # accepted but not persisted yet

    runs = list_active_workflow_runs(limit=50)
    for r in runs:
        if r["id"] == workflow_run_id:
            return r
    return {
        "id": workflow_run_id,
        "workflow_id": row["workflow_id"],
        "workflow_name": "",
        "sprint_id": None,
        "roadmap_id": None,
        "roadmap_name": None,
        "brief_document_id": None,
        "current_stage_key": None,
        "current_stage_name": None,
        "status": "completed" if not nxt else "active",
        "started_at": None,
    }


def pin_persona_to_run(
    workflow_run_id: str,
    workflow_stage_id: str,
    persona_slug: str,
    authority: str,
    pilot_pinned: bool,
    selected_basis: str | None = None,
) -> dict:
    """Insert or update a run_fleet row with selected_by='pilot-pin'."""
    with _conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT id FROM run_fleet "
            "WHERE workflow_run_id = %s AND workflow_stage_id = %s "
            "  AND persona_slug = %s AND detached_at IS NULL",
            (workflow_run_id, workflow_stage_id, persona_slug),
        )
        existing = cur.fetchone()
        if existing:
            cur.execute(
                "UPDATE run_fleet SET authority = %s, "
                "selected_by = 'pilot-pin', selected_basis = %s, "
                "pilot_pinned = %s WHERE id = %s",
                (authority, selected_basis, 1 if pilot_pinned else 0, existing["id"]),
            )
        else:
            cur.execute(
                "INSERT INTO run_fleet "
                "(id, workflow_run_id, workflow_stage_id, persona_slug, "
                " authority, selected_by, selected_basis, pilot_pinned) "
                "VALUES (%s, %s, %s, %s, %s, 'pilot-pin', %s, %s)",
                (
                    _new_id(), workflow_run_id, workflow_stage_id,
                    persona_slug, authority, selected_basis,
                    1 if pilot_pinned else 0,
                ),
            )

        cur.execute(
            "SELECT rf.workflow_run_id, ws.stage_key, ws.stage_name, "
            "       ws.sort_order AS stage_order, rf.persona_slug, "
            "       rf.authority, rf.selected_by, rf.selected_basis, "
            "       rf.pilot_pinned "
            "FROM run_fleet rf "
            "JOIN workflow_stages ws ON ws.id = rf.workflow_stage_id "
            "WHERE rf.workflow_run_id = %s AND rf.workflow_stage_id = %s "
            "  AND rf.persona_slug = %s AND rf.detached_at IS NULL",
            (workflow_run_id, workflow_stage_id, persona_slug),
        )
        r = cur.fetchone()

    if not r:
        raise RuntimeError("Pin applied but v_fleet_manifest returned no row.")
    return {
        "workflow_run_id": r["workflow_run_id"],
        "stage_key": r["stage_key"],
        "stage_name": r["stage_name"],
        "stage_order": r["stage_order"],
        "persona_slug": r["persona_slug"],
        "authority": r["authority"],
        "selected_by": r["selected_by"],
        "selected_basis": r.get("selected_basis"),
        "pilot_pinned": bool(r.get("pilot_pinned")),
    }

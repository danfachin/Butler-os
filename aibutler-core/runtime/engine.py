#!/usr/bin/env python3
"""
aiButler runtime engine.

This is the first persistent ButlerRuntime:
  - sessions
  - tasks
  - approvals
  - memories
  - tool execution receipts
"""
from __future__ import annotations

import copy
import json
import os
import shlex
import signal
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

from runtime.context_repository import ContextRepository
from runtime.models import (
    ApprovalRequest,
    ButlerArtifact,
    ButlerRoom,
    ButlerSession,
    ButlerTask,
    ButlerVersion,
    CapabilityGrant,
    ContinuityPacket,
    ContextEvent,
    ContextPendingItem,
    ContextSheet,
    MemoryRecord,
    SwarmAgentSpec,
    SwarmAgentState,
    SwarmContract,
    SwarmDeploymentPolicy,
    SwarmRun,
    ToolCallReceipt,
    utc_now,
)
from runtime.plugins import get_plugin_manager
from runtime.security import (
    arm_token_ttl_minutes,
    full_access_feature_enabled,
    future_expiry_iso,
    hash_token,
    is_expired,
    issue_arm_token,
    trusted_local_session,
)
from runtime.store import RuntimeStore
from runtime.tool_registry import build_tool_registry, list_tool_specs

DEFAULT_RUNTIME_DIR = Path.home() / ".aibutler" / "runtime"
_DEFAULT_RUNTIME: "ButlerRuntime | None" = None


def _make_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


def _normalized_swarm_target(value: str) -> str:
    normalized = value.strip().lower()
    return normalized if normalized in {"local_desktop", "vpn_remote"} else "local_desktop"


def _normalized_swarm_launcher(value: str) -> str:
    normalized = value.strip().lower()
    return normalized if normalized in {"desktop", "vpn"} else "desktop"


def _normalized_swarm_template(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in {"planning", "relationship", "operator", "research", "build"}:
        return normalized
    return "planning"


ACTIVE_SWARM_RUN_STATUSES = frozenset(
    {
        "staged",
        "queued",
        "running",
        "awaiting_oversight",
        "blocked",
    }
)


def _infer_swarm_template(objective: str) -> str:
    prompt = (objective or "").lower()
    if any(token in prompt for token in ("follow up", "follow-up", "relationship", "reply", "outreach", "crm")):
        return "relationship"
    if any(token in prompt for token in ("openclaw", "operator", "gateway", "rtk", "secret", "credential", "vpn")):
        return "operator"
    if any(token in prompt for token in ("research", "investigate", "learn", "explore")):
        return "research"
    if any(token in prompt for token in ("build", "implement", "ship", "code", "patch", "fix")):
        return "build"
    return "planning"


def _remote_shell_path(value: str) -> str:
    cleaned = (value or "").strip()
    if cleaned.startswith("~/"):
        return f"$HOME/{cleaned[2:]}"
    if cleaned == "~":
        return "$HOME"
    return shlex.quote(cleaned)


def _local_process_alive(pid: int | None) -> bool:
    if not pid or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _default_swarm_deployment_policy(
    objective: str,
    *,
    template: str = "planning",
    target: str,
    launcher: str,
) -> SwarmDeploymentPolicy:
    cleaned = objective.strip() or "Ship the current problem."
    template_name = _normalized_swarm_template(template)
    definition_of_done = [
        "Produce a minimal plan or execution artifact that directly advances the objective.",
        "Surface unresolved risks or missing decisions in plain language.",
        "Persist enough run state that another surface can resume without ambiguity.",
    ]
    if template_name == "relationship":
        definition_of_done = [
            "Identify who needs a reply or follow-up now.",
            "Surface the top relationship priorities with next actions.",
            "Leave a briefing that can be resumed from phone or desktop.",
        ]
    elif template_name == "operator":
        definition_of_done = [
            "Report operator stack health clearly.",
            "Surface exact remediation or bootstrap steps for anything offline.",
            "Persist a handoff-ready operator report.",
        ]
    elif template_name == "research":
        definition_of_done = [
            "Summarize the most relevant context already in Butler memory.",
            "Highlight missing information and open questions.",
            "Leave a compact brief another agent can continue from.",
        ]
    elif template_name == "build":
        definition_of_done = [
            "Clarify the implementation target and blockers.",
            "Produce concrete next build steps or execution notes.",
            "Persist enough context for a builder to continue immediately.",
        ]
    return SwarmDeploymentPolicy(
        definition_of_done=definition_of_done,
        oversight_triggers=[
            "A required credential, approval, or secret is missing.",
            "A dependency blocks forward progress.",
            "The swarm reaches max iterations without a clear completion state.",
            "The output would publish, send, delete, or mutate trusted user data.",
        ],
        tool_policy={
            "mode": "approval_calibrated",
            "default": "read_or_reversible",
            "notes": "Use Butler tools first. Escalate before sensitive external actions.",
        },
        budget={
            "max_agents": 3,
            "default_max_iterations": 5,
            "objective": cleaned,
            "template": template_name,
        },
        launcher_config={
            "target": _normalized_swarm_target(target),
            "launcher": _normalized_swarm_launcher(launcher),
        },
    )


class ButlerRuntime:
    """Open-source local runtime for Butler sessions and action execution."""

    def __init__(self, base_dir: str | Path = DEFAULT_RUNTIME_DIR):
        self.base_dir = Path(base_dir).expanduser()
        self.store = RuntimeStore(self.base_dir)
        self.context_repository = ContextRepository(self.base_dir.parent / "context")
        self.plugin_manager = get_plugin_manager()
        self.tool_registry = build_tool_registry()
        self.tool_specs = {spec.name: spec for spec in list_tool_specs()}

    # ──────────────────────────────────────────────────────────────────────
    # Persistence helpers
    # ──────────────────────────────────────────────────────────────────────

    def _load_sessions(self) -> list[ButlerSession]:
        data = self.store.load_json("sessions.json", [])
        return [ButlerSession.from_dict(row) for row in data]

    def _save_sessions(self, sessions: list[ButlerSession]) -> None:
        self.store.save_json("sessions.json", [session.to_dict() for session in sessions])

    def _load_tasks(self) -> list[ButlerTask]:
        data = self.store.load_json("tasks.json", [])
        return [ButlerTask.from_dict(row) for row in data]

    def _save_tasks(self, tasks: list[ButlerTask]) -> None:
        self.store.save_json("tasks.json", [task.to_dict() for task in tasks])

    def _load_approvals(self) -> list[ApprovalRequest]:
        data = self.store.load_json("approvals.json", [])
        return [ApprovalRequest.from_dict(row) for row in data]

    def _save_approvals(self, approvals: list[ApprovalRequest]) -> None:
        self.store.save_json("approvals.json", [approval.to_dict() for approval in approvals])

    def _load_continuity_packets(self) -> list[ContinuityPacket]:
        data = self.store.load_json("continuity_packets.json", [])
        return [ContinuityPacket.from_dict(row) for row in data]

    def _save_continuity_packets(self, packets: list[ContinuityPacket]) -> None:
        self.store.save_json("continuity_packets.json", [packet.to_dict() for packet in packets])

    def _load_rooms(self) -> list[ButlerRoom]:
        data = self.store.load_json("rooms.json", [])
        return [ButlerRoom.from_dict(row) for row in data]

    def _save_rooms(self, rooms: list[ButlerRoom]) -> None:
        self.store.save_json("rooms.json", [room.to_dict() for room in rooms])

    def _load_room_artifacts(self) -> list[ButlerArtifact]:
        data = self.store.load_json("room_artifacts.json", [])
        return [ButlerArtifact.from_dict(row) for row in data]

    def _save_room_artifacts(self, artifacts: list[ButlerArtifact]) -> None:
        self.store.save_json("room_artifacts.json", [artifact.to_dict() for artifact in artifacts])

    def _load_room_versions(self) -> list[ButlerVersion]:
        data = self.store.load_json("room_versions.json", [])
        return [ButlerVersion.from_dict(row) for row in data]

    def _save_room_versions(self, versions: list[ButlerVersion]) -> None:
        self.store.save_json("room_versions.json", [version.to_dict() for version in versions])

    def _load_swarm_contracts(self) -> list[SwarmContract]:
        data = self.store.load_json("swarm_contracts.json", [])
        return [SwarmContract.from_dict(row) for row in data]

    def _save_swarm_contracts(self, contracts: list[SwarmContract]) -> None:
        self.store.save_json("swarm_contracts.json", [contract.to_dict() for contract in contracts])

    def _load_swarm_runs(self) -> list[SwarmRun]:
        data = self.store.load_json("swarm_runs.json", [])
        return [SwarmRun.from_dict(row) for row in data]

    def _save_swarm_runs(self, runs: list[SwarmRun]) -> None:
        self.store.save_json("swarm_runs.json", [run.to_dict() for run in runs])

    def _write_continuity_event(self, payload: dict) -> str:
        path = self.store.append_jsonl("continuity_events.jsonl", payload)
        return str(path)

    def _write_swarm_event(self, payload: dict) -> str:
        path = self.store.append_jsonl("swarm_events.jsonl", payload)
        return str(path)

    def _write_receipt(self, receipt: ToolCallReceipt) -> str:
        path = self.store.append_jsonl("receipts.jsonl", receipt.to_dict())
        return str(path)

    def _write_memory(self, memory: MemoryRecord) -> str:
        path = self.store.append_jsonl("memories.jsonl", memory.to_dict())
        return str(path)

    def _write_security_event(self, payload: dict) -> str:
        path = self.store.append_jsonl("security_events.jsonl", payload)
        return str(path)

    def _load_arm_state(self) -> dict:
        return self.store.load_json("arm_state.json", {})

    def _save_arm_state(self, data: dict) -> None:
        self.store.save_json("arm_state.json", data)

    def _continuity_packet_expired(self, packet: ContinuityPacket) -> bool:
        return is_expired(packet.expires_at)

    def _consume_arm_token(self, token: str | None) -> dict | None:
        if not token:
            return None

        state = self._load_arm_state()
        if not state or not state.get("token_hash"):
            return None
        if is_expired(state.get("expires_at")):
            self._save_arm_state({})
            return None
        if hash_token(token) != state["token_hash"]:
            return None

        consumed = dict(state)
        consumed["used_at"] = utc_now()
        self._save_arm_state({})
        return consumed

    # ──────────────────────────────────────────────────────────────────────
    # Sessions
    # ──────────────────────────────────────────────────────────────────────

    def create_session(
        self,
        user_id: str = "local-user",
        surface: str = "local",
        metadata: dict | None = None,
        capability_grants: list[CapabilityGrant] | None = None,
    ) -> ButlerSession:
        session = ButlerSession(
            id=_make_id("session"),
            user_id=user_id,
            surface=surface,
            metadata=metadata or {},
            capability_grants=capability_grants or [],
        )
        sessions = self._load_sessions()
        sessions.append(session)
        self._save_sessions(sessions)
        return session

    def get_session(self, session_id: str) -> ButlerSession | None:
        for session in self._load_sessions():
            if session.id == session_id:
                return session
        return None

    def get_or_create_session(
        self,
        session_id: str | None = None,
        *,
        user_id: str = "local-user",
        surface: str = "local",
        metadata: dict | None = None,
    ) -> ButlerSession:
        if session_id:
            existing = self.get_session(session_id)
            if existing:
                return existing
        return self.create_session(user_id=user_id, surface=surface, metadata=metadata)

    def list_sessions(self, limit: int = 20) -> list[ButlerSession]:
        return self._load_sessions()[-limit:]

    def touch_session(self, session_id: str) -> ButlerSession | None:
        sessions = self._load_sessions()
        target = None
        for index, session in enumerate(sessions):
            if session.id == session_id:
                session.updated_at = utc_now()
                sessions[index] = session
                target = session
                break
        if target:
            self._save_sessions(sessions)
        return target

    def close_session(self, session_id: str) -> ButlerSession | None:
        sessions = self._load_sessions()
        target = None
        for index, session in enumerate(sessions):
            if session.id == session_id:
                session.status = "closed"
                session.updated_at = utc_now()
                sessions[index] = session
                target = session
                break
        if target:
            self._save_sessions(sessions)
        return target

    def set_permission_mode(
        self,
        session_id: str,
        mode: str,
        *,
        actor: str = "user",
        note: str = "",
        duration_minutes: int | None = None,
        arm_token: str | None = None,
    ) -> ButlerSession | None:
        sessions = self._load_sessions()
        target = None
        normalized_mode = mode.strip().lower()
        if normalized_mode not in {"locked", "standard", "full-access"}:
            raise ValueError(f"Unsupported permission mode: {mode}")
        consumed_arm_state = None

        for index, session in enumerate(sessions):
            if session.id != session_id:
                continue

            if normalized_mode == "full-access":
                if not full_access_feature_enabled():
                    raise PermissionError(
                        "Full access is disabled. Set AIBUTLER_ENABLE_FULL_ACCESS=1 on this machine."
                    )
                if not trusted_local_session(session):
                    raise PermissionError(
                        "Full access can only be enabled for trusted local sessions."
                    )
                consumed_arm_state = self._consume_arm_token(arm_token)
                if not consumed_arm_state:
                    raise PermissionError(
                        "A valid local arm token is required before enabling full access."
                    )
                unique_capabilities = sorted({spec.capability for spec in self.tool_specs.values()})
                session.permission_mode = "full-access"
                session.full_access_expires_at = future_expiry_iso(duration_minutes)
                session.capability_grants = [
                    CapabilityGrant(capability=capability, granted=True, scope="full", reason="full-access mode")
                    for capability in unique_capabilities
                ]
            elif normalized_mode == "locked":
                session.permission_mode = "locked"
                session.full_access_expires_at = None
                session.capability_grants = []
            else:
                session.permission_mode = "standard"
                session.full_access_expires_at = None
                session.capability_grants = []

            session.updated_at = utc_now()
            sessions[index] = session
            target = session
            break

        if target:
            self._save_sessions(sessions)
            self._write_security_event(
                {
                    "event": "permission_mode_change",
                    "session_id": target.id,
                    "user_id": target.user_id,
                    "mode": target.permission_mode,
                    "expires_at": target.full_access_expires_at,
                    "actor": actor,
                    "note": note,
                    "armed_by": consumed_arm_state.get("actor") if consumed_arm_state else None,
                    "created_at": utc_now(),
                }
            )
        return target

    def arm_full_access(
        self,
        *,
        actor: str = "user",
        note: str = "",
        duration_minutes: int | None = None,
    ) -> dict:
        ttl = duration_minutes if duration_minutes is not None else arm_token_ttl_minutes()
        token = issue_arm_token()
        expires_at = future_expiry_iso(ttl)
        self._save_arm_state(
            {
                "token_hash": hash_token(token),
                "expires_at": expires_at,
                "actor": actor,
                "note": note,
                "created_at": utc_now(),
            }
        )
        self._write_security_event(
            {
                "event": "full_access_armed",
                "actor": actor,
                "note": note,
                "expires_at": expires_at,
                "created_at": utc_now(),
            }
        )
        return {
            "ok": True,
            "output": {
                "arm_token": token,
                "expires_at": expires_at,
                "actor": actor,
                "note": note,
            },
            "error": None,
        }

    def get_permission_state(self, session_id: str) -> dict:
        session = self.get_session(session_id)
        if not session:
            return {"ok": False, "error": f"Unknown session: {session_id}"}
        active_full_access = (
            session.permission_mode == "full-access"
            and not is_expired(session.full_access_expires_at)
            and trusted_local_session(session)
            and full_access_feature_enabled()
        )
        return {
            "ok": True,
            "output": {
                "session_id": session.id,
                "surface": session.surface,
                "trusted_local": trusted_local_session(session),
                "permission_mode": session.permission_mode,
                "full_access_enabled": full_access_feature_enabled(),
                "full_access_active": active_full_access,
                "full_access_expires_at": session.full_access_expires_at,
                "arm_token_required": True,
                "capability_grants": [grant.to_dict() for grant in session.capability_grants],
            },
            "error": None,
        }

    # ──────────────────────────────────────────────────────────────────────
    # Tasks
    # ──────────────────────────────────────────────────────────────────────

    def create_task(
        self,
        session_id: str,
        title: str,
        kind: str = "general",
        payload: dict | None = None,
    ) -> ButlerTask:
        task = ButlerTask(
            id=_make_id("task"),
            session_id=session_id,
            title=title,
            kind=kind,
            payload=payload or {},
        )
        tasks = self._load_tasks()
        tasks.append(task)
        self._save_tasks(tasks)
        self.touch_session(session_id)
        return task

    def update_task(
        self,
        task_id: str,
        *,
        status: str | None = None,
        result: object | None = None,
        error: str | None = None,
    ) -> ButlerTask | None:
        tasks = self._load_tasks()
        target = None
        for index, task in enumerate(tasks):
            if task.id == task_id:
                if status:
                    task.status = status
                if result is not None:
                    task.result = result
                if error is not None:
                    task.error = error
                task.updated_at = utc_now()
                tasks[index] = task
                target = task
                break
        if target:
            self._save_tasks(tasks)
            self.touch_session(target.session_id)
        return target

    def list_tasks(self, session_id: str | None = None, limit: int = 50) -> list[ButlerTask]:
        tasks = self._load_tasks()
        if session_id:
            tasks = [task for task in tasks if task.session_id == session_id]
        return tasks[-limit:]

    # ──────────────────────────────────────────────────────────────────────
    # Approvals
    # ──────────────────────────────────────────────────────────────────────

    def request_approval(
        self,
        session_id: str,
        tool_name: str,
        reason: str,
        args: dict | None = None,
    ) -> ApprovalRequest:
        approval = ApprovalRequest(
            id=_make_id("approval"),
            session_id=session_id,
            tool_name=tool_name,
            reason=reason,
            args=args or {},
        )
        approvals = self._load_approvals()
        approvals.append(approval)
        self._save_approvals(approvals)
        self.touch_session(session_id)
        return approval

    def resolve_approval(
        self,
        approval_id: str,
        *,
        approved: bool,
        actor: str = "user",
        note: str = "",
    ) -> ApprovalRequest | None:
        approvals = self._load_approvals()
        target = None
        for index, approval in enumerate(approvals):
            if approval.id == approval_id:
                approval.status = "approved" if approved else "rejected"
                approval.actor = actor
                approval.note = note
                approval.resolved_at = utc_now()
                approvals[index] = approval
                target = approval
                break
        if target:
            self._save_approvals(approvals)
            self.touch_session(target.session_id)
        return target

    def list_approvals(self, session_id: str | None = None, status: str | None = None) -> list[ApprovalRequest]:
        approvals = self._load_approvals()
        if session_id:
            approvals = [approval for approval in approvals if approval.session_id == session_id]
        if status:
            approvals = [approval for approval in approvals if approval.status == status]
        return approvals

    # ──────────────────────────────────────────────────────────────────────
    # Rooms, artifacts, and versions
    # ──────────────────────────────────────────────────────────────────────

    def create_room(
        self,
        *,
        kind: str,
        title: str,
        status: str = "active",
        metadata: dict | None = None,
        source_refs: list[str] | None = None,
        initial_payload: dict | None = None,
        created_by: str = "runtime",
        session_id: str | None = None,
    ) -> ButlerRoom:
        now = utc_now()
        room = ButlerRoom(
            id=_make_id("room"),
            kind=kind.strip().lower() or "general",
            title=title.strip() or "Untitled room",
            status=status.strip().lower() or "active",
            metadata=copy.deepcopy(metadata or {}),
            source_refs=list(source_refs or []),
            created_at=now,
            updated_at=now,
        )
        version = ButlerVersion(
            id=_make_id("version"),
            room_id=room.id,
            state_kind="room_state",
            payload=copy.deepcopy(
                initial_payload
                or {
                    "title": room.title,
                    "kind": room.kind,
                    "status": room.status,
                    "metadata": room.metadata,
                    "source_refs": room.source_refs,
                }
            ),
            metadata={"created_from": "room.create"},
            created_by=created_by,
            status="draft",
            created_at=now,
        )
        room.current_draft_version_id = version.id

        rooms = self._load_rooms()
        rooms.append(room)
        self._save_rooms(rooms)

        versions = self._load_room_versions()
        versions.append(version)
        self._save_room_versions(versions)

        self.append_context_event(
            event_type="room.created",
            summary=f"Created {room.kind} room {room.title}",
            payload={
                "room_id": room.id,
                "kind": room.kind,
                "draft_version_id": version.id,
            },
            entity_refs=[f"rooms/{room.id}", *room.source_refs],
            session_id=session_id,
        )
        if session_id:
            self.touch_session(session_id)
        return room

    def get_room(self, room_id: str) -> ButlerRoom | None:
        for room in self._load_rooms():
            if room.id == room_id:
                return room
        return None

    def find_room_by_source_ref(self, source_ref: str) -> ButlerRoom | None:
        normalized_ref = source_ref.strip()
        if not normalized_ref:
            return None
        for room in self._load_rooms():
            if normalized_ref in room.source_refs:
                return room
        return None

    def list_rooms(self, *, kind: str | None = None, limit: int = 50) -> list[ButlerRoom]:
        rooms = self._load_rooms()
        if kind:
            rooms = [room for room in rooms if room.kind == kind.strip().lower()]
        rooms.sort(key=lambda room: room.updated_at or room.created_at, reverse=True)
        return rooms[:limit]

    def resolve_room(
        self,
        *,
        source_ref: str,
        title: str = "",
        kind: str = "",
        metadata: dict | None = None,
        created_by: str = "runtime",
        session_id: str | None = None,
    ) -> tuple[ButlerRoom, bool]:
        normalized_ref = source_ref.strip()
        if not normalized_ref:
            raise ValueError("source_ref is required")

        existing = self.find_room_by_source_ref(normalized_ref)
        if existing:
            return existing, False

        inferred_kind = (kind.strip().lower() or normalized_ref.split("/", 1)[0].rstrip("s") or "general")
        inferred_title = title.strip() or normalized_ref.split("/", 1)[-1].replace("-", " ").strip().title() or normalized_ref
        room = self.create_room(
            kind=inferred_kind,
            title=inferred_title,
            metadata=metadata or {},
            source_refs=[normalized_ref],
            initial_payload={
                "title": inferred_title,
                "kind": inferred_kind,
                "source_ref": normalized_ref,
            },
            created_by=created_by,
            session_id=session_id,
        )
        return room, True

    def attach_room_artifact(
        self,
        room_id: str,
        *,
        artifact_kind: str,
        artifact_url: str,
        mime_type: str = "",
        metadata: dict | None = None,
        created_by: str = "runtime",
        session_id: str | None = None,
    ) -> ButlerArtifact:
        room = self.get_room(room_id)
        if not room:
            raise ValueError(f"Unknown room: {room_id}")

        now = utc_now()
        artifact = ButlerArtifact(
            id=_make_id("artifact"),
            room_id=room_id,
            artifact_kind=artifact_kind.strip().lower() or "note",
            artifact_url=artifact_url.strip(),
            mime_type=mime_type.strip(),
            metadata=copy.deepcopy(metadata or {}),
            created_by=created_by,
            created_at=now,
            updated_at=now,
        )

        artifacts = self._load_room_artifacts()
        artifacts.append(artifact)
        self._save_room_artifacts(artifacts)

        self.append_context_event(
            event_type="room.artifact.attached",
            summary=f"Attached {artifact.artifact_kind} to room {room.title}",
            payload={
                "room_id": room_id,
                "artifact_id": artifact.id,
                "artifact_url": artifact.artifact_url,
            },
            entity_refs=[f"rooms/{room_id}", f"artifacts/{artifact.id}"],
            session_id=session_id,
        )
        if session_id:
            self.touch_session(session_id)
        return artifact

    def list_room_artifacts(self, room_id: str, *, limit: int = 50) -> list[ButlerArtifact]:
        artifacts = [artifact for artifact in self._load_room_artifacts() if artifact.room_id == room_id]
        artifacts.sort(key=lambda artifact: artifact.updated_at or artifact.created_at, reverse=True)
        return artifacts[:limit]

    def get_version(self, version_id: str) -> ButlerVersion | None:
        for version in self._load_room_versions():
            if version.id == version_id:
                return version
        return None

    def list_room_versions(self, room_id: str, *, limit: int = 20) -> list[ButlerVersion]:
        versions = [version for version in self._load_room_versions() if version.room_id == room_id]
        versions.sort(key=lambda version: version.created_at, reverse=True)
        return versions[:limit]

    def get_current_draft_version(self, room_id: str) -> ButlerVersion | None:
        room = self.get_room(room_id)
        if not room or not room.current_draft_version_id:
            return None
        return self.get_version(room.current_draft_version_id)

    def save_draft_version(
        self,
        room_id: str,
        *,
        payload: dict | None = None,
        parent_version_id: str | None = None,
        state_kind: str = "room_state",
        metadata: dict | None = None,
        created_by: str = "runtime",
        session_id: str | None = None,
    ) -> ButlerVersion:
        room = self.get_room(room_id)
        if not room:
            raise ValueError(f"Unknown room: {room_id}")

        parent_id = parent_version_id or room.current_draft_version_id or room.current_published_version_id
        now = utc_now()
        version = ButlerVersion(
            id=_make_id("version"),
            room_id=room_id,
            state_kind=state_kind.strip() or "room_state",
            payload=copy.deepcopy(payload or {}),
            metadata=copy.deepcopy(metadata or {}),
            parent_version_id=parent_id,
            created_by=created_by,
            status="draft",
            created_at=now,
        )

        versions = self._load_room_versions()
        versions.append(version)
        self._save_room_versions(versions)

        rooms = self._load_rooms()
        updated_room = None
        for index, candidate in enumerate(rooms):
            if candidate.id != room_id:
                continue
            candidate.current_draft_version_id = version.id
            candidate.updated_at = now
            rooms[index] = candidate
            updated_room = candidate
            break
        if updated_room:
            self._save_rooms(rooms)

        self.append_context_event(
            event_type="room.draft.saved",
            summary=f"Saved draft for room {room.title}",
            payload={
                "room_id": room_id,
                "version_id": version.id,
                "parent_version_id": parent_id,
                "state_kind": version.state_kind,
            },
            entity_refs=[f"rooms/{room_id}", f"versions/{version.id}"],
            session_id=session_id,
        )
        if session_id:
            self.touch_session(session_id)
        return version

    def publish_draft_version(
        self,
        version_id: str,
        *,
        created_by: str = "runtime",
        session_id: str | None = None,
    ) -> ButlerVersion | None:
        versions = self._load_room_versions()
        target: ButlerVersion | None = None
        now = utc_now()
        room_id = ""

        for index, version in enumerate(versions):
            if version.id != version_id:
                continue
            room_id = version.room_id
            version.status = "published"
            version.published_at = now
            if created_by and not version.created_by:
                version.created_by = created_by
            versions[index] = version
            target = version
            break
        if not target:
            return None

        for index, version in enumerate(versions):
            if version.id == target.id or version.room_id != room_id or version.status != "published":
                continue
            version.status = "archived"
            versions[index] = version

        self._save_room_versions(versions)

        rooms = self._load_rooms()
        room = None
        for index, candidate in enumerate(rooms):
            if candidate.id != room_id:
                continue
            candidate.current_published_version_id = target.id
            if candidate.current_draft_version_id == target.id:
                candidate.current_draft_version_id = None
            candidate.updated_at = now
            rooms[index] = candidate
            room = candidate
            break
        if room:
            self._save_rooms(rooms)

        self.append_context_event(
            event_type="room.version.published",
            summary=f"Published room version for {room.title if room else room_id}",
            payload={
                "room_id": room_id,
                "version_id": target.id,
                "parent_version_id": target.parent_version_id,
            },
            entity_refs=[f"rooms/{room_id}", f"versions/{target.id}"],
            session_id=session_id,
        )
        if session_id:
            self.touch_session(session_id)
        return target

    # ──────────────────────────────────────────────────────────────────────
    # Swarm contracts and runs
    # ──────────────────────────────────────────────────────────────────────

    def _default_swarm_agents(self, objective: str) -> list[SwarmAgentSpec]:
        return self._build_swarm_agents(objective, template="planning")

    def _build_swarm_agents(self, objective: str, *, template: str) -> list[SwarmAgentSpec]:
        cleaned = objective.strip() or "Ship the current problem."
        normalized_template = _normalized_swarm_template(template)
        framer_id = _make_id("agent")
        builder_id = _make_id("agent")
        if normalized_template == "relationship":
            return [
                SwarmAgentSpec(
                    id=framer_id,
                    title="Relationship Framer",
                    role="framer",
                    objective=f"{cleaned}\n\nRole: build the relationship briefing and identify who needs a reply next.",
                    max_iterations=3,
                    tool_hints=["relationship_get_briefing", "relationship_list_followups", "context_graph_snapshot"],
                    metadata={"template": normalized_template},
                ),
                SwarmAgentSpec(
                    id=builder_id,
                    title="Relationship Builder",
                    role="builder",
                    objective=f"{cleaned}\n\nRole: turn the live relationship signals into concrete next actions and outreach priorities.",
                    depends_on=[framer_id],
                    max_iterations=4,
                    tool_hints=["relationship_list_followups", "relationship_get_briefing", "butler_memory_search"],
                    metadata={"template": normalized_template},
                ),
                SwarmAgentSpec(
                    id=_make_id("agent"),
                    title="Relationship Reviewer",
                    role="reviewer",
                    objective=f"{cleaned}\n\nRole: review the proposed follow-ups, missing context, and approvals that still need the human.",
                    depends_on=[builder_id],
                    max_iterations=3,
                    tool_hints=["relationship_get_briefing", "context_activity_feed"],
                    metadata={"template": normalized_template},
                ),
            ]
        if normalized_template == "operator":
            return [
                SwarmAgentSpec(
                    id=framer_id,
                    title="Operator Framer",
                    role="framer",
                    objective=f"{cleaned}\n\nRole: inspect the operator stack and identify what is online, offline, or degraded.",
                    max_iterations=3,
                    tool_hints=["openclaw_status", "secret_recovery_status", "rtk_status"],
                    metadata={"template": normalized_template},
                ),
                SwarmAgentSpec(
                    id=builder_id,
                    title="Operator Builder",
                    role="builder",
                    objective=f"{cleaned}\n\nRole: prepare exact remediation or bootstrap steps for the operator environment.",
                    depends_on=[framer_id],
                    max_iterations=4,
                    tool_hints=["build_swarm_vpn_bootstrap", "openclaw_status", "rtk_status"],
                    metadata={"template": normalized_template},
                ),
                SwarmAgentSpec(
                    id=_make_id("agent"),
                    title="Operator Reviewer",
                    role="reviewer",
                    objective=f"{cleaned}\n\nRole: review the operator plan, identify missing approvals or credentials, and summarize the safe next move.",
                    depends_on=[builder_id],
                    max_iterations=3,
                    tool_hints=["secret_recovery_status", "openclaw_status"],
                    metadata={"template": normalized_template},
                ),
            ]
        if normalized_template == "research":
            return [
                SwarmAgentSpec(
                    id=framer_id,
                    title="Research Framer",
                    role="framer",
                    objective=f"{cleaned}\n\nRole: map the most relevant existing memory and recent activity for this question.",
                    max_iterations=3,
                    tool_hints=["butler_memory_search", "context_activity_feed"],
                    metadata={"template": normalized_template},
                ),
                SwarmAgentSpec(
                    id=builder_id,
                    title="Research Builder",
                    role="builder",
                    objective=f"{cleaned}\n\nRole: synthesize the strongest evidence already in Butler and highlight remaining gaps.",
                    depends_on=[framer_id],
                    max_iterations=4,
                    tool_hints=["butler_memory_search", "context_graph_snapshot"],
                    metadata={"template": normalized_template},
                ),
                SwarmAgentSpec(
                    id=_make_id("agent"),
                    title="Research Reviewer",
                    role="reviewer",
                    objective=f"{cleaned}\n\nRole: review the synthesis, challenge weak assumptions, and surface open questions.",
                    depends_on=[builder_id],
                    max_iterations=3,
                    tool_hints=["butler_memory_search"],
                    metadata={"template": normalized_template},
                ),
            ]
        if normalized_template == "build":
            return [
                SwarmAgentSpec(
                    id=framer_id,
                    title="Build Framer",
                    role="framer",
                    objective=f"{cleaned}\n\nRole: identify the implementation target, nearby context, and real blockers.",
                    max_iterations=3,
                    tool_hints=["context_graph_snapshot", "context_activity_feed", "butler_memory_search"],
                    metadata={"template": normalized_template},
                ),
                SwarmAgentSpec(
                    id=builder_id,
                    title="Build Planner",
                    role="builder",
                    objective=f"{cleaned}\n\nRole: turn the current context into concrete build steps or patches with the smallest useful slice first.",
                    depends_on=[framer_id],
                    max_iterations=4,
                    tool_hints=["butler_memory_search", "list_pending_context", "context_graph_snapshot"],
                    metadata={"template": normalized_template},
                ),
                SwarmAgentSpec(
                    id=_make_id("agent"),
                    title="Build Reviewer",
                    role="reviewer",
                    objective=f"{cleaned}\n\nRole: review the proposed build steps for missing dependencies, risks, and oversight decisions.",
                    depends_on=[builder_id],
                    max_iterations=3,
                    tool_hints=["context_activity_feed", "list_pending_context"],
                    metadata={"template": normalized_template},
                ),
            ]
        return [
            SwarmAgentSpec(
                id=framer_id,
                title="Framer",
                role="framer",
                objective=(
                    f"{cleaned}\n\n"
                    "Role: clarify the goal, identify constraints, and produce the minimal work plan "
                    "with the main bottlenecks called out."
                ),
                max_iterations=3,
                tool_hints=["context_graph_snapshot", "context_activity_feed", "list_pending_context", "butler_memory_search"],
                metadata={"template": normalized_template},
            ),
            SwarmAgentSpec(
                id=builder_id,
                title="Builder",
                role="builder",
                objective=(
                    f"{cleaned}\n\n"
                    "Role: execute the highest-leverage work directly, using available Butler tools "
                    "where they unblock the objective."
                ),
                depends_on=[framer_id],
                max_iterations=5,
                tool_hints=["butler_memory_search", "context_graph_snapshot"],
                metadata={"template": normalized_template},
            ),
            SwarmAgentSpec(
                id=_make_id("agent"),
                title="Reviewer",
                role="reviewer",
                objective=(
                    f"{cleaned}\n\n"
                    "Role: review outputs, surface risks, missing information, and any oversight "
                    "decisions that still require the human."
                ),
                depends_on=[builder_id],
                max_iterations=3,
                tool_hints=["context_activity_feed", "list_pending_context"],
                metadata={"template": normalized_template},
            ),
        ]

    def _build_swarm_report_markdown(self, contract: SwarmContract, run: SwarmRun) -> str:
        lines = [
            f"# Swarm Report: {run.title}",
            "",
            f"- Run ID: `{run.id}`",
            f"- Contract ID: `{contract.id}`",
            f"- Room ID: `{run.room_id}`",
            f"- Template: `{contract.template}`",
            f"- Status: `{run.status}`",
            f"- Target: `{run.target}`",
            f"- Launcher: `{run.launcher}`",
            f"- Backend: `{run.execution_backend}`",
            f"- Created At: `{run.created_at}`",
            f"- Launched At: `{run.launched_at or ''}`",
            f"- Completed At: `{run.completed_at or ''}`",
            "",
            "## Objective",
            "",
            contract.objective.strip() or "No objective provided.",
            "",
            "## Definition Of Done",
            "",
        ]
        if contract.deployment_policy.definition_of_done:
            lines.extend([f"- {item}" for item in contract.deployment_policy.definition_of_done])
        else:
            lines.append("- Not specified")

        lines.extend([
            "",
            "## Oversight Triggers",
            "",
        ])
        if contract.deployment_policy.oversight_triggers:
            lines.extend([f"- {item}" for item in contract.deployment_policy.oversight_triggers])
        else:
            lines.append("- Not specified")

        lines.extend([
            "",
            "## Budget",
            "",
            "```json",
            json.dumps(contract.deployment_policy.budget, indent=2, default=str),
            "```",
            "",
            "## Tool Policy",
            "",
            "```json",
            json.dumps(contract.deployment_policy.tool_policy, indent=2, default=str),
            "```",
            "",
            "## Agent Results",
            "",
        ])

        for state in run.agent_states:
            lines.extend(
                [
                    f"### {state.title} ({state.role})",
                    "",
                    f"- Status: `{state.status}`",
                    f"- Task ID: `{state.task_id or ''}`",
                    f"- Started At: `{state.started_at or ''}`",
                    f"- Completed At: `{state.completed_at or ''}`",
                    f"- Summary: {state.result_summary or 'No summary recorded.'}",
                ]
            )
            if state.error:
                lines.append(f"- Error: {state.error}")
            if state.metadata:
                lines.extend(
                    [
                        "- Metadata:",
                        "```json",
                        json.dumps(state.metadata, indent=2, default=str),
                        "```",
                    ]
                )
            lines.append("")

        lines.extend(
            [
                "## Run Metadata",
                "",
                "```json",
                json.dumps(run.metadata, indent=2, default=str),
                "```",
                "",
                "## Final Summary",
                "",
                run.summary or run.metadata.get("final_note") or "No final summary recorded.",
                "",
            ]
        )
        return "\n".join(lines)

    def _persist_swarm_report(
        self,
        run: SwarmRun,
        *,
        session_id: str | None = None,
    ) -> tuple[str, ButlerArtifact | None]:
        contract = self.get_swarm_contract(run.contract_id)
        if not contract:
            return "", None
        report_dir = self.base_dir / "swarm_reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / f"{run.id}.md"
        report_path.write_text(self._build_swarm_report_markdown(contract, run), encoding="utf-8")

        artifact: ButlerArtifact | None = None
        existing = next(
            (
                candidate
                for candidate in self.list_room_artifacts(run.room_id, limit=200)
                if candidate.artifact_kind == "swarm_report" and candidate.artifact_url == str(report_path)
            ),
            None,
        )
        if existing:
            artifact = existing
        else:
            artifact = self.attach_room_artifact(
                run.room_id,
                artifact_kind="swarm_report",
                artifact_url=str(report_path),
                mime_type="text/markdown",
                metadata={"run_id": run.id, "contract_id": run.contract_id},
                created_by="swarm-runtime",
                session_id=session_id,
            )
        return str(report_path), artifact

    def _snapshot_swarm_run(self, run: SwarmRun, *, session_id: str | None = None) -> ButlerVersion:
        payload = {
            "run": run.to_dict(),
            "agent_states": [agent.to_dict() for agent in run.agent_states],
        }
        return self.save_draft_version(
            run.room_id,
            payload=payload,
            state_kind="swarm_run",
            metadata={"swarm_run_id": run.id, "swarm_contract_id": run.contract_id},
            created_by="swarm-runtime",
            session_id=session_id,
        )

    def create_swarm_contract(
        self,
        *,
        title: str,
        objective: str,
        template: str = "",
        room_id: str | None = None,
        room_kind: str = "project",
        target: str = "local_desktop",
        launcher: str = "desktop",
        agents: list[dict] | None = None,
        deployment_policy: dict | None = None,
        metadata: dict | None = None,
        source_refs: list[str] | None = None,
        created_by: str = "runtime",
        session_id: str | None = None,
    ) -> SwarmContract:
        now = utc_now()
        normalized_template = _normalized_swarm_template(template or _infer_swarm_template(objective))
        room = self.get_room(room_id or "") if room_id else None
        if not room:
            room = self.create_room(
                kind=room_kind,
                title=title,
                metadata={"swarm_objective": objective, "swarm_template": normalized_template, **copy.deepcopy(metadata or {})},
                source_refs=source_refs or [],
                initial_payload={
                    "title": title.strip() or "Untitled swarm room",
                    "objective": objective.strip(),
                    "kind": room_kind.strip().lower() or "project",
                    "template": normalized_template,
                },
                created_by=created_by,
                session_id=session_id,
            )

        normalized_agents = [
            SwarmAgentSpec.from_dict(agent) for agent in (agents or []) if isinstance(agent, dict)
        ] or self._build_swarm_agents(objective, template=normalized_template)
        default_policy = _default_swarm_deployment_policy(
            objective,
            template=normalized_template,
            target=target,
            launcher=launcher,
        )
        merged_policy = default_policy.to_dict()
        if deployment_policy:
            for key, value in copy.deepcopy(deployment_policy).items():
                if key in {"tool_policy", "budget", "launcher_config"} and isinstance(value, dict):
                    merged_policy[key] = {**merged_policy.get(key, {}), **value}
                else:
                    merged_policy[key] = value
        policy_launcher_config = merged_policy.get("launcher_config", {})
        normalized_target = _normalized_swarm_target(policy_launcher_config.get("target") or target)
        normalized_launcher = _normalized_swarm_launcher(policy_launcher_config.get("launcher") or launcher)
        merged_policy["launcher_config"] = {
            **policy_launcher_config,
            "target": normalized_target,
            "launcher": normalized_launcher,
        }
        normalized_policy = SwarmDeploymentPolicy.from_dict(merged_policy)

        contract = SwarmContract(
            id=_make_id("swarm"),
            room_id=room.id,
            title=title.strip() or room.title,
            objective=objective.strip(),
            template=normalized_template,
            target=normalized_target,
            launcher=normalized_launcher,
            status="draft",
            agents=normalized_agents,
            deployment_policy=normalized_policy,
            metadata=copy.deepcopy(metadata or {}),
            source_refs=list(source_refs or room.source_refs),
            created_by=created_by,
            created_at=now,
            updated_at=now,
        )
        contracts = self._load_swarm_contracts()
        contracts.append(contract)
        self._save_swarm_contracts(contracts)

        version = self.save_draft_version(
            room.id,
            payload={
                "contract": contract.to_dict(),
                "objective": contract.objective,
                "template": contract.template,
                "agents": [agent.to_dict() for agent in contract.agents],
                "deployment_policy": contract.deployment_policy.to_dict(),
            },
            state_kind="swarm_contract",
            metadata={"swarm_contract_id": contract.id},
            created_by=created_by,
            session_id=session_id,
        )
        self.append_context_event(
            event_type="swarm.contract.created",
            summary=f"Created swarm contract {contract.title}",
            payload={
                "contract_id": contract.id,
                "room_id": contract.room_id,
                "target": contract.target,
                "draft_version_id": version.id,
            },
            entity_refs=[f"rooms/{contract.room_id}", f"swarm/contracts/{contract.id}", *contract.source_refs],
            session_id=session_id,
        )
        self._write_swarm_event(
            {
                "event": "swarm_contract_created",
                "contract": contract.to_dict(),
                "created_at": now,
            }
        )
        return contract

    def get_swarm_contract(self, contract_id: str) -> SwarmContract | None:
        for contract in self._load_swarm_contracts():
            if contract.id == contract_id:
                return contract
        return None

    def list_swarm_contracts(
        self,
        *,
        room_id: str | None = None,
        status: str | None = None,
        limit: int = 50,
    ) -> list[SwarmContract]:
        contracts = self._load_swarm_contracts()
        if room_id:
            contracts = [contract for contract in contracts if contract.room_id == room_id]
        if status:
            contracts = [contract for contract in contracts if contract.status == status]
        contracts.sort(key=lambda contract: contract.updated_at or contract.created_at, reverse=True)
        return contracts[:limit]

    def get_swarm_run(self, run_id: str) -> SwarmRun | None:
        for run in self._load_swarm_runs():
            if run.id == run_id:
                return self._reconcile_swarm_run(run)
        return None

    def get_swarm_run_report(self, run_id: str) -> dict | None:
        run = self.get_swarm_run(run_id)
        if not run or not run.report_path:
            return None
        report_path = Path(run.report_path).expanduser()
        if not report_path.exists():
            return {
                "run_id": run.id,
                "report_path": str(report_path),
                "exists": False,
                "content": "",
            }
        return {
            "run_id": run.id,
            "report_path": str(report_path),
            "exists": True,
            "content": report_path.read_text(encoding="utf-8"),
        }

    def build_swarm_vpn_bootstrap(
        self,
        *,
        ssh_target: str = "",
        remote_workdir: str = "~/Butler-os/aibutler-core",
        repo_url: str = "https://github.com/BoarderOnATrip/Butler-os.git",
        branch: str = "main",
        remote_python: str = "python3",
        execute: bool = False,
    ) -> dict:
        runtime_workdir = (remote_workdir or "~/Butler-os/aibutler-core").strip()
        normalized_runtime_workdir = runtime_workdir.rstrip("/") or "~/Butler-os/aibutler-core"
        if normalized_runtime_workdir.endswith("/aibutler-core"):
            remote_repo_root = normalized_runtime_workdir[: -len("/aibutler-core")] or "~/Butler-os"
        else:
            remote_repo_root = normalized_runtime_workdir
            normalized_runtime_workdir = f"{remote_repo_root.rstrip('/')}/aibutler-core"

        repo_root_shell = _remote_shell_path(remote_repo_root)
        runtime_workdir_shell = _remote_shell_path(normalized_runtime_workdir)
        bootstrap_script = "\n".join(
            [
                "set -euo pipefail",
                f"REMOTE_REPO_ROOT={repo_root_shell}",
                f"RUNTIME_WORKDIR={runtime_workdir_shell}",
                f"REPO_URL={shlex.quote(repo_url.strip() or 'https://github.com/BoarderOnATrip/Butler-os.git')}",
                f"BRANCH={shlex.quote(branch.strip() or 'main')}",
                f"PYTHON={shlex.quote(remote_python.strip() or 'python3')}",
                'mkdir -p "$(dirname "$REMOTE_REPO_ROOT")"',
                'if [ ! -d "$REMOTE_REPO_ROOT/.git" ]; then',
                '  git clone --branch "$BRANCH" "$REPO_URL" "$REMOTE_REPO_ROOT"',
                "else",
                '  git -C "$REMOTE_REPO_ROOT" fetch origin "$BRANCH"',
                '  git -C "$REMOTE_REPO_ROOT" checkout "$BRANCH"',
                '  git -C "$REMOTE_REPO_ROOT" pull --ff-only origin "$BRANCH"',
                "fi",
                'cd "$RUNTIME_WORKDIR"',
                '$PYTHON -m py_compile runtime/models.py runtime/engine.py runtime/swarm_worker.py runtime/__main__.py runtime/tool_registry.py ../bridge/server.py >/dev/null 2>&1 || true',
                'printf "ready:%s\\n" "$RUNTIME_WORKDIR"',
            ]
        )

        ssh_command = ""
        if ssh_target.strip():
            ssh_command = " ".join(
                [
                    "ssh",
                    shlex.quote(ssh_target.strip()),
                    shlex.quote(f"bash -lc {shlex.quote(bootstrap_script)}"),
                ]
            )

        result = {
            "ssh_target": ssh_target.strip(),
            "remote_repo_root": remote_repo_root,
            "remote_workdir": normalized_runtime_workdir,
            "repo_url": repo_url.strip() or "https://github.com/BoarderOnATrip/Butler-os.git",
            "branch": branch.strip() or "main",
            "remote_python": remote_python.strip() or "python3",
            "script": bootstrap_script,
            "ssh_command": ssh_command,
            "executed": False,
            "stdout": "",
            "stderr": "",
        }

        if execute:
            if not ssh_target.strip():
                raise ValueError("ssh_target is required when execute=True")
            proc = subprocess.run(
                ["ssh", ssh_target.strip(), "bash", "-lc", bootstrap_script],
                capture_output=True,
                text=True,
                timeout=180,
            )
            result["executed"] = True
            result["stdout"] = proc.stdout
            result["stderr"] = proc.stderr
            result["returncode"] = proc.returncode
            if proc.returncode != 0:
                raise RuntimeError((proc.stderr or proc.stdout).strip() or "VPN bootstrap failed")

        return result

    def list_swarm_runs(
        self,
        *,
        contract_id: str | None = None,
        room_id: str | None = None,
        status: str | None = None,
        limit: int = 50,
    ) -> list[SwarmRun]:
        runs = self._load_swarm_runs()
        runs = [self._reconcile_swarm_run(run) for run in runs]
        if contract_id:
            runs = [run for run in runs if run.contract_id == contract_id]
        if room_id:
            runs = [run for run in runs if run.room_id == room_id]
        if status:
            runs = [run for run in runs if run.status == status]
        runs.sort(key=lambda run: run.updated_at or run.created_at, reverse=True)
        return runs[:limit]

    def get_latest_active_swarm_run(
        self,
        *,
        contract_id: str | None = None,
        room_id: str | None = None,
    ) -> SwarmRun | None:
        runs = self._load_swarm_runs()
        runs = [self._reconcile_swarm_run(run) for run in runs]
        if contract_id:
            runs = [run for run in runs if run.contract_id == contract_id]
        if room_id:
            runs = [run for run in runs if run.room_id == room_id]
        runs = [run for run in runs if run.status in ACTIVE_SWARM_RUN_STATUSES]
        if not runs:
            return None
        runs.sort(key=lambda run: run.updated_at or run.created_at, reverse=True)
        return runs[0]

    def _save_swarm_run_instance(self, run: SwarmRun) -> SwarmRun:
        runs = self._load_swarm_runs()
        replaced = False
        for index, candidate in enumerate(runs):
            if candidate.id != run.id:
                continue
            runs[index] = run
            replaced = True
            break
        if not replaced:
            runs.append(run)
        self._save_swarm_runs(runs)
        return run

    def _reconcile_swarm_run(self, run: SwarmRun) -> SwarmRun:
        if run.execution_backend != "local_process":
            return run
        if run.status not in {"queued", "running"}:
            return run
        if _local_process_alive(run.pid):
            return run

        recovered = copy.deepcopy(run)
        recovered.pid = None
        recovered.updated_at = utc_now()
        recovered.completed_at = recovered.completed_at or recovered.updated_at
        recovered.metadata = {
            **recovered.metadata,
            "reconciled_by": "runtime",
            "reconciled_at": recovered.updated_at,
            "reconciled_reason": "worker_process_missing",
        }

        terminal_states = {state.status for state in recovered.agent_states}
        if recovered.agent_states and terminal_states == {"completed"}:
            recovered.status = "completed"
            summaries = [state.result_summary for state in recovered.agent_states if state.result_summary]
            recovered.summary = " | ".join(summaries[:3]) if summaries else f"Swarm completed for {recovered.title}"
        else:
            failed = next((state for state in recovered.agent_states if state.status == "failed"), None)
            oversight = next((state for state in recovered.agent_states if state.status == "awaiting_oversight"), None)
            blocked = next((state for state in recovered.agent_states if state.status == "blocked"), None)
            active = [state.title for state in recovered.agent_states if state.status in {"running", "planned"}]
            if failed:
                recovered.status = "failed"
                recovered.summary = failed.error or failed.result_summary or f"{failed.title} failed."
            elif oversight:
                recovered.status = "awaiting_oversight"
                recovered.summary = oversight.result_summary or f"{oversight.title} needs human review."
            elif blocked:
                recovered.status = "blocked"
                recovered.summary = blocked.error or f"{blocked.title} is blocked."
            else:
                recovered.status = "failed"
                if active:
                    recovered.summary = (
                        "Swarm worker exited before finalizing. "
                        f"In-flight agents: {', '.join(active[:3])}"
                    )
                else:
                    recovered.summary = f"Swarm worker exited before finalizing {recovered.title}."

        self._save_swarm_run_instance(recovered)
        self._snapshot_swarm_run(recovered)
        self._write_swarm_event(
            {
                "event": "swarm_run_reconciled",
                "run_id": recovered.id,
                "status": recovered.status,
                "created_at": recovered.updated_at,
            }
        )

        if recovered.status == "completed" and not recovered.report_path:
            report_path, artifact = self._persist_swarm_report(recovered)
            recovered.report_path = report_path
            if artifact:
                recovered.metadata["report_artifact_id"] = artifact.id
            recovered.updated_at = utc_now()
            self._save_swarm_run_instance(recovered)
            self._snapshot_swarm_run(recovered)
        return recovered

    def update_swarm_run(
        self,
        run_id: str,
        *,
        status: str | None = None,
        pid: int | None = None,
        remote_job_id: str | None = None,
        command: str | None = None,
        log_path: str | None = None,
        summary: str | None = None,
        report_path: str | None = None,
        metadata_patch: dict | None = None,
        launched_at: str | None = None,
        completed_at: str | None = None,
        session_id: str | None = None,
    ) -> SwarmRun | None:
        run = self.get_swarm_run(run_id)
        if not run:
            return None
        if status:
            run.status = status
        if pid is not None:
            run.pid = pid
        if remote_job_id is not None:
            run.remote_job_id = remote_job_id
        if command is not None:
            run.command = command
        if log_path is not None:
            run.log_path = log_path
        if summary is not None:
            run.summary = summary
        if report_path is not None:
            run.report_path = report_path
        if metadata_patch:
            run.metadata = {**run.metadata, **copy.deepcopy(metadata_patch)}
        if launched_at is not None:
            run.launched_at = launched_at
        if completed_at is not None:
            run.completed_at = completed_at
        run.updated_at = utc_now()
        self._save_swarm_run_instance(run)
        self._snapshot_swarm_run(run, session_id=session_id)
        self._write_swarm_event(
            {
                "event": "swarm_run_updated",
                "run_id": run.id,
                "status": run.status,
                "created_at": run.updated_at,
            }
        )
        return run

    def update_swarm_agent_state(
        self,
        run_id: str,
        agent_id: str,
        *,
        status: str | None = None,
        task_id: str | None = None,
        result_summary: str | None = None,
        error: str | None = None,
        started_at: str | None = None,
        completed_at: str | None = None,
        metadata_patch: dict | None = None,
        session_id: str | None = None,
    ) -> SwarmRun | None:
        run = self.get_swarm_run(run_id)
        if not run:
            return None
        for state in run.agent_states:
            if state.agent_id != agent_id:
                continue
            if status:
                state.status = status
            if task_id is not None:
                state.task_id = task_id
            if result_summary is not None:
                state.result_summary = result_summary
            if error is not None:
                state.error = error
            if started_at is not None:
                state.started_at = started_at
            if completed_at is not None:
                state.completed_at = completed_at
            if metadata_patch:
                state.metadata = {**state.metadata, **copy.deepcopy(metadata_patch)}
            break
        run.updated_at = utc_now()
        self._save_swarm_run_instance(run)
        self._snapshot_swarm_run(run, session_id=session_id)
        self._write_swarm_event(
            {
                "event": "swarm_agent_state_updated",
                "run_id": run.id,
                "agent_id": agent_id,
                "status": status,
                "created_at": run.updated_at,
            }
        )
        return run

    def stop_swarm_run(
        self,
        run_id: str,
        *,
        created_by: str = "runtime",
        session_id: str | None = None,
    ) -> SwarmRun | None:
        run = self.get_swarm_run(run_id)
        if not run:
            return None

        terminal_statuses = {"completed", "failed", "cancelled"}
        if run.status in terminal_statuses:
            return run

        stop_errors: list[str] = []
        if run.execution_backend == "local_process" and run.pid:
            try:
                os.kill(run.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            except OSError as exc:
                stop_errors.append(str(exc))
        elif run.execution_backend == "ssh_remote":
            ssh_target = str(run.metadata.get("vpn_ssh_target", "")).strip()
            remote_job_id = (run.remote_job_id or "").strip()
            if ssh_target and remote_job_id:
                result = subprocess.run(
                    ["ssh", ssh_target, f"kill {shlex.quote(remote_job_id)}"],
                    capture_output=True,
                    text=True,
                    timeout=20,
                )
                if result.returncode != 0:
                    stop_errors.append((result.stderr or result.stdout).strip() or "remote stop failed")

        now = utc_now()
        for state in run.agent_states:
            if state.status not in terminal_statuses:
                state.status = "cancelled"
                state.completed_at = state.completed_at or now
                if not state.error:
                    state.error = "Stopped by operator."

        run.status = "cancelled"
        run.summary = "Swarm stopped by operator."
        run.completed_at = now
        run.updated_at = now
        run.metadata = {
            **run.metadata,
            "stopped_by": created_by,
            "stopped_at": now,
        }
        if stop_errors:
            run.metadata["stop_errors"] = stop_errors

        self._save_swarm_run_instance(run)
        self._snapshot_swarm_run(run, session_id=session_id)
        self._write_swarm_event(
            {
                "event": "swarm_run_updated",
                "run_id": run.id,
                "status": run.status,
                "created_at": run.updated_at,
            }
        )
        return run

    def launch_swarm_contract(
        self,
        contract_id: str,
        *,
        target: str | None = None,
        launcher: str | None = None,
        vpn_ssh_target: str = "",
        remote_workdir: str = "",
        remote_python: str = "python3",
        dry_run: bool = False,
        created_by: str = "runtime",
        session_id: str | None = None,
    ) -> dict:
        contract = self.get_swarm_contract(contract_id)
        if not contract:
            raise ValueError(f"Unknown swarm contract: {contract_id}")

        run_target = _normalized_swarm_target(target or contract.target)
        run_launcher = _normalized_swarm_launcher(launcher or contract.launcher)
        now = utc_now()
        launcher_config = copy.deepcopy(contract.deployment_policy.launcher_config or {})
        agent_states = [
            SwarmAgentState(
                agent_id=agent.id,
                title=agent.title,
                role=agent.role,
                objective=agent.objective,
                depends_on=list(agent.depends_on),
                status="planned",
                metadata=copy.deepcopy(agent.metadata),
            )
            for agent in contract.agents
        ]
        run = SwarmRun(
            id=_make_id("run"),
            contract_id=contract.id,
            room_id=contract.room_id,
            title=contract.title,
            template=contract.template,
            target=run_target,
            launcher=run_launcher,
            execution_backend="ssh_remote" if run_target == "vpn_remote" else "local_process",
            status="staged",
            agent_states=agent_states,
            metadata={
                "created_by": created_by,
                "vpn_ssh_target": vpn_ssh_target or launcher_config.get("vpn_ssh_target", "") or contract.metadata.get("vpn_ssh_target", ""),
                "remote_workdir": remote_workdir or launcher_config.get("remote_workdir", "") or contract.metadata.get("remote_workdir", ""),
                "remote_python": remote_python or launcher_config.get("remote_python", "python3"),
            },
            created_at=now,
            updated_at=now,
        )

        runtime_module_dir = Path(__file__).resolve().parent
        core_root = runtime_module_dir.parent
        log_dir = self.base_dir / "swarm_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{run.id}.log"

        if run_target == "local_desktop":
            command_list = [
                sys.executable,
                "-m",
                "runtime.swarm_worker",
                "--contract-id",
                contract.id,
                "--run-id",
                run.id,
                "--runtime-dir",
                str(self.base_dir),
            ]
            run.command = " ".join(shlex.quote(part) for part in command_list)
            run.log_path = str(log_path)
            if dry_run:
                run.status = "staged"
            else:
                with log_path.open("a", encoding="utf-8") as handle:
                    proc = subprocess.Popen(
                        command_list,
                        cwd=str(core_root),
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                    )
                run.pid = proc.pid
                run.status = "queued"
                run.launched_at = utc_now()
        else:
            ssh_target = (run.metadata.get("vpn_ssh_target") or launcher_config.get("vpn_ssh_target") or "").strip()
            target_workdir = (run.metadata.get("remote_workdir") or launcher_config.get("remote_workdir") or "").strip()
            remote_python_value = (
                (run.metadata.get("remote_python") or launcher_config.get("remote_python") or remote_python or "python3")
                .strip()
            )
            remote_log = str(
                run.metadata.get("remote_log_path")
                or launcher_config.get("remote_log_path")
                or f"~/.aibutler/runtime/swarm_logs/{run.id}.log"
            )
            worker_cmd = (
                f"cd {shlex.quote(target_workdir)} && "
                f"nohup {shlex.quote(remote_python_value)} -m runtime.swarm_worker "
                f"--contract-id {shlex.quote(contract.id)} "
                f"--run-id {shlex.quote(run.id)} "
                f"> {shlex.quote(remote_log)} 2>&1 & echo $!"
            )
            run.command = f"ssh {shlex.quote(ssh_target)} {shlex.quote(worker_cmd)}" if ssh_target else worker_cmd
            run.log_path = remote_log
            if dry_run or not ssh_target or not target_workdir:
                run.status = "staged"
            else:
                result = subprocess.run(
                    ["ssh", ssh_target, worker_cmd],
                    capture_output=True,
                    text=True,
                    timeout=20,
                )
                if result.returncode != 0:
                    run.status = "failed"
                    run.metadata["launch_error"] = (result.stderr or result.stdout).strip()
                else:
                    remote_pid = (result.stdout or "").strip()
                    run.remote_job_id = remote_pid
                    run.status = "queued"
                    run.launched_at = utc_now()

        self._save_swarm_run_instance(run)

        contracts = self._load_swarm_contracts()
        for index, candidate in enumerate(contracts):
            if candidate.id != contract.id:
                continue
            candidate.status = "launched" if run.status in {"queued", "running"} else "draft"
            candidate.updated_at = utc_now()
            contracts[index] = candidate
            contract = candidate
            break
        self._save_swarm_contracts(contracts)

        self._snapshot_swarm_run(run, session_id=session_id)
        self.append_context_event(
            event_type="swarm.run.created",
            summary=f"Created swarm run for {contract.title}",
            payload={
                "contract_id": contract.id,
                "run_id": run.id,
                "status": run.status,
                "target": run.target,
            },
            entity_refs=[f"rooms/{contract.room_id}", f"swarm/contracts/{contract.id}", f"swarm/runs/{run.id}"],
            session_id=session_id,
        )
        self._write_swarm_event(
            {
                "event": "swarm_run_created",
                "contract_id": contract.id,
                "run": run.to_dict(),
                "created_at": utc_now(),
            }
        )
        return {
            "contract": contract.to_dict(),
            "run": run.to_dict(),
            "launch_ready": run.status in {"queued", "running"},
            "dry_run": dry_run,
        }

    # ──────────────────────────────────────────────────────────────────────
    # Memory
    # ──────────────────────────────────────────────────────────────────────

    def write_memory(
        self,
        session_id: str,
        kind: str,
        content: str,
        tags: list[str] | None = None,
        metadata: dict | None = None,
    ) -> MemoryRecord:
        memory = MemoryRecord(
            id=_make_id("memory"),
            session_id=session_id,
            kind=kind,
            content=content,
            tags=tags or [],
            metadata=metadata or {},
        )
        self._write_memory(memory)
        self.touch_session(session_id)
        return memory

    def list_memories(self, session_id: str | None = None, limit: int = 50) -> list[MemoryRecord]:
        rows = self.store.load_jsonl("memories.jsonl", limit=limit * 4)
        memories = [MemoryRecord.from_dict(row) for row in rows]
        if session_id:
            memories = [memory for memory in memories if memory.session_id == session_id]
        return memories[-limit:]

    # ──────────────────────────────────────────────────────────────────────
    # Continuity
    # ──────────────────────────────────────────────────────────────────────

    def create_continuity_packet(
        self,
        *,
        kind: str,
        title: str,
        content: str = "",
        source_device: str = "",
        target_device: str = "",
        source_surface: str = "",
        metadata: dict | None = None,
        room_id: str | None = None,
        artifact_id: str | None = None,
        version_id: str | None = None,
        refs: list[str] | None = None,
        expires_in_minutes: int | None = 60,
        session_id: str | None = None,
    ) -> ContinuityPacket:
        packet = ContinuityPacket(
            id=_make_id("handoff"),
            kind=kind.strip() or "text",
            title=title.strip() or "Continuity handoff",
            content=content,
            source_device=source_device.strip(),
            target_device=target_device.strip(),
            source_surface=source_surface.strip(),
            metadata=copy.deepcopy(metadata or {}),
            room_id=room_id or None,
            artifact_id=artifact_id or None,
            version_id=version_id or None,
            refs=list(refs or []),
            expires_at=future_expiry_iso(expires_in_minutes) if expires_in_minutes else None,
            session_id=session_id,
        )
        packets = self._load_continuity_packets()
        packets.append(packet)
        self._save_continuity_packets(packets)
        self._write_continuity_event(
            {
                "event": "continuity_packet_created",
                "packet": packet.to_dict(),
                "created_at": utc_now(),
            }
        )
        if session_id:
            self.touch_session(session_id)
        return packet

    def list_continuity_packets(
        self,
        *,
        target_device: str | None = None,
        status: str | None = None,
        limit: int = 20,
        include_consumed: bool = False,
    ) -> list[ContinuityPacket]:
        packets = self._load_continuity_packets()
        if target_device:
            packets = [packet for packet in packets if packet.target_device == target_device]
        if status:
            packets = [packet for packet in packets if packet.status == status]
        elif not include_consumed:
            packets = [packet for packet in packets if packet.status != "consumed"]
        packets = [packet for packet in packets if not self._continuity_packet_expired(packet)]
        packets.sort(key=lambda packet: packet.updated_at or packet.created_at, reverse=True)
        return packets[:limit]

    def claim_continuity_packet(
        self,
        packet_id: str,
        *,
        actor_device: str,
        lease_minutes: int = 15,
    ) -> ContinuityPacket | None:
        packets = self._load_continuity_packets()
        target = None
        now = utc_now()
        for index, packet in enumerate(packets):
            if packet.id != packet_id:
                continue
            if packet.status == "consumed" or self._continuity_packet_expired(packet):
                return None
            if (
                packet.lease_owner
                and packet.lease_owner != actor_device
                and not is_expired(packet.lease_expires_at)
            ):
                raise ValueError(f"Packet is currently claimed by {packet.lease_owner}")
            packet.status = "claimed"
            packet.lease_owner = actor_device.strip()
            packet.lease_expires_at = future_expiry_iso(lease_minutes)
            packet.updated_at = now
            packets[index] = packet
            target = packet
            break
        if target:
            self._save_continuity_packets(packets)
            self._write_continuity_event(
                {
                    "event": "continuity_packet_claimed",
                    "packet_id": target.id,
                    "actor_device": actor_device,
                    "lease_expires_at": target.lease_expires_at,
                    "created_at": now,
                }
            )
        return target

    def acknowledge_continuity_packet(
        self,
        packet_id: str,
        *,
        actor_device: str,
        note: str = "",
    ) -> ContinuityPacket | None:
        packets = self._load_continuity_packets()
        target = None
        now = utc_now()
        for index, packet in enumerate(packets):
            if packet.id != packet_id:
                continue
            packet.status = "consumed"
            packet.lease_owner = actor_device.strip()
            packet.lease_expires_at = None
            packet.consumed_at = now
            packet.updated_at = now
            packet.metadata = dict(packet.metadata)
            if note:
                packet.metadata["ack_note"] = note
            packets[index] = packet
            target = packet
            break
        if target:
            self._save_continuity_packets(packets)
            self._write_continuity_event(
                {
                    "event": "continuity_packet_consumed",
                    "packet_id": target.id,
                    "actor_device": actor_device,
                    "note": note,
                    "created_at": now,
                }
            )
        return target

    # ──────────────────────────────────────────────────────────────────────
    # Context repository
    # ──────────────────────────────────────────────────────────────────────

    def init_context_repo(self) -> dict:
        return self.context_repository.ensure_layout()

    def append_context_event(
        self,
        *,
        event_type: str,
        summary: str,
        payload: dict | None = None,
        source: dict | None = None,
        entity_refs: list[str] | None = None,
        session_id: str | None = None,
    ) -> ContextEvent:
        event = self.context_repository.append_event(
            event_id=_make_id("ctxevent"),
            event_type=event_type,
            summary=summary,
            payload=payload or {},
            source=source or {},
            entity_refs=entity_refs or [],
            session_id=session_id,
        )
        if session_id:
            self.touch_session(session_id)
        return event

    def create_context_sheet(
        self,
        *,
        kind: str,
        name: str,
        body: str = "",
        slug: str | None = None,
        links: list[str] | None = None,
        source_refs: list[str] | None = None,
        metadata: dict | None = None,
        status: str = "active",
        confidence: float = 1.0,
    ) -> ContextSheet:
        return self.context_repository.create_sheet(
            sheet_id=_make_id("ctxsheet"),
            kind=kind,
            name=name,
            body=body,
            slug=slug,
            links=links or [],
            source_refs=source_refs or [],
            metadata=metadata or {},
            status=status,
            confidence=confidence,
        )

    def list_context_sheets(self, *, kind: str | None = None, limit: int = 50) -> list[ContextSheet]:
        return self.context_repository.list_sheets(kind=kind, limit=limit)

    def capture_pending_context(
        self,
        *,
        capture_kind: str,
        title: str,
        content: str = "",
        metadata: dict | None = None,
        source: dict | None = None,
        confidence: float = 0.0,
        session_id: str | None = None,
    ) -> ContextPendingItem:
        item = self.context_repository.create_pending_item(
            pending_id=_make_id("pending"),
            capture_kind=capture_kind,
            title=title,
            content=content,
            metadata=metadata or {},
            source=source or {},
            confidence=confidence,
            session_id=session_id,
        )
        if session_id:
            self.touch_session(session_id)
        return item

    def list_pending_context(self, *, limit: int = 50) -> list[ContextPendingItem]:
        return self.context_repository.list_pending_items(limit=limit)

    def get_pending_context_item(self, pending_id: str) -> ContextPendingItem | None:
        return self.context_repository.get_pending_item(pending_id)

    def update_pending_context_item(
        self,
        pending_id: str,
        *,
        status: str | None = None,
        title: str | None = None,
        content: str | None = None,
        metadata: dict | None = None,
        source: dict | None = None,
        confidence: float | None = None,
        session_id: str | None = None,
    ) -> ContextPendingItem | None:
        return self.context_repository.update_pending_item(
            pending_id,
            status=status,
            title=title,
            content=content,
            metadata=metadata or None,
            source=source or None,
            confidence=confidence,
            session_id=session_id,
        )

    def list_context_events(self, *, limit: int = 50) -> list[ContextEvent]:
        return self.context_repository.list_events(limit=limit)

    # ──────────────────────────────────────────────────────────────────────
    # Tool execution
    # ──────────────────────────────────────────────────────────────────────

    def _normalize_tool_args(self, tool_name: str, args: dict) -> dict:
        normalized = copy.deepcopy(args)
        for key in ("position", "box", "start", "end"):
            if key in normalized and isinstance(normalized[key], list):
                normalized[key] = tuple(normalized[key])

        params = self.tool_specs[tool_name].params
        if "position" in normalized and "x" in params and "y" in params:
            position = normalized.pop("position")
            if isinstance(position, tuple) and len(position) == 2:
                normalized.setdefault("x", position[0])
                normalized.setdefault("y", position[1])

        if "start" in normalized and "start_x" in params and "start_y" in params:
            start = normalized.pop("start")
            if isinstance(start, tuple) and len(start) == 2:
                normalized.setdefault("start_x", start[0])
                normalized.setdefault("start_y", start[1])

        if "end" in normalized and "end_x" in params and "end_y" in params:
            end = normalized.pop("end")
            if isinstance(end, tuple) and len(end) == 2:
                normalized.setdefault("end_x", end[0])
                normalized.setdefault("end_y", end[1])
        return normalized

    def _approval_required(self, tool_name: str, args: dict, approved: bool) -> bool:
        spec = self.tool_specs[tool_name]
        if not spec.approval.required or approved:
            return False
        if spec.approval.live_only and args.get("dry_run"):
            return False
        return True

    def _has_full_access(self, session: ButlerSession) -> bool:
        return (
            session.permission_mode == "full-access"
            and full_access_feature_enabled()
            and trusted_local_session(session)
            and not is_expired(session.full_access_expires_at)
        )

    def _capability_allowed(self, session: ButlerSession, capability: str) -> bool:
        if self._has_full_access(session):
            return True
        if session.permission_mode == "locked":
            return False

        grants = [grant for grant in session.capability_grants if grant.capability == capability]
        grants = [grant for grant in grants if grant.scope != "full"]
        if not grants:
            return True
        return any(grant.granted for grant in grants)

    def execute_tool(
        self,
        session_id: str,
        tool_name: str,
        args: dict | None = None,
        *,
        approved: bool = False,
        actor: str = "runtime",
        note: str = "",
    ) -> dict:
        args = args or {}
        session = self.get_session(session_id)
        if not session:
            return {"ok": False, "output": "", "error": f"Unknown session: {session_id}"}
        if tool_name not in self.tool_registry:
            return {"ok": False, "output": "", "error": f"Unknown tool: {tool_name}"}

        tool_args = self._normalize_tool_args(tool_name, args)
        approval_id = tool_args.pop("_approval_id", None) or tool_args.pop("approval_id", None)
        approved = approved or bool(tool_args.pop("_approved", False))
        spec = self.tool_specs[tool_name]

        pre_hook_result = self.plugin_manager.run_pre_tool_hooks(
            tool_name=tool_name,
            args=tool_args,
            session_id=session_id,
        )
        if isinstance(pre_hook_result, dict) and pre_hook_result.get("block"):
            return {
                "ok": False,
                "output": "",
                "error": pre_hook_result.get("reason", f"Blocked by pre-tool hook: {tool_name}"),
                "tool": spec.to_dict(),
                "hook_blocked": True,
            }

        if not self._capability_allowed(session, spec.capability):
            return {
                "ok": False,
                "output": "",
                "error": f"Capability denied for {tool_name}: {spec.capability}",
                "tool": spec.to_dict(),
            }

        approved = approved or self._has_full_access(session)

        if self._approval_required(tool_name, tool_args, approved):
            approval = self.request_approval(
                session_id=session_id,
                tool_name=tool_name,
                reason=spec.approval.reason or f"{tool_name} requires approval",
                args=tool_args,
            )
            return {
                "ok": False,
                "output": "",
                "error": f"Approval required before running {tool_name}",
                "approval_request_id": approval.id,
                "approval_reason": approval.reason,
                "tool": spec.to_dict(),
            }

        if approval_id and approved:
            self.resolve_approval(approval_id, approved=True, actor=actor, note=note)

        fn = self.tool_registry[tool_name]["fn"]
        try:
            result = fn(**tool_args)
        except Exception as exc:
            result = {"ok": False, "output": "", "error": str(exc)}

        if result.get("ok"):
            self.plugin_manager.run_post_tool_hooks(
                tool_name=tool_name,
                args=tool_args,
                result=result,
                session_id=session_id,
            )
        else:
            self.plugin_manager.run_post_tool_error_hooks(
                tool_name=tool_name,
                args=tool_args,
                result=result,
                session_id=session_id,
            )

        receipt = ToolCallReceipt(
            session_id=session_id,
            tool_name=tool_name,
            args=tool_args,
            ok=bool(result.get("ok")),
            tool_category=spec.category,
            risk=spec.risk,
            output=result.get("output"),
            error=result.get("error"),
            approval_request_id=approval_id,
        )
        receipt_path = self._write_receipt(receipt)
        self.touch_session(session_id)

        enriched = dict(result)
        enriched["runtime_receipt"] = receipt.to_dict()
        enriched["runtime_receipt_path"] = receipt_path
        return enriched

    def list_receipts(self, session_id: str | None = None, limit: int = 50) -> list[ToolCallReceipt]:
        rows = self.store.load_jsonl("receipts.jsonl", limit=limit * 4)
        receipts = [ToolCallReceipt(**row) for row in rows]
        if session_id:
            receipts = [receipt for receipt in receipts if receipt.session_id == session_id]
        return receipts[-limit:]

    def list_security_events(self, limit: int = 50) -> list[dict]:
        return self.store.load_jsonl("security_events.jsonl", limit=limit)

    # ──────────────────────────────────────────────────────────────────────
    # DewDrops workflow (Hearth/Dolt-backed)
    # ──────────────────────────────────────────────────────────────────────
    # These methods delegate to runtime.hearth_adapter, which owns the only
    # MySQL connection in Butler. The adapter is imported lazily so missing
    # pymysql doesn't break unrelated flows.

    def list_active_workflow_runs(self, sprint_id: str | None = None, limit: int = 20) -> list[dict]:
        from . import hearth_adapter
        return hearth_adapter.list_active_workflow_runs(sprint_id=sprint_id, limit=limit)

    def get_workflow_fleet(self, workflow_run_id: str) -> list[dict]:
        from . import hearth_adapter
        return hearth_adapter.get_workflow_fleet(workflow_run_id)

    def get_workflow_findings(self, workflow_run_id: str, open_only: bool = True) -> list[dict]:
        from . import hearth_adapter
        return hearth_adapter.get_workflow_findings(workflow_run_id, open_only=open_only)

    def start_workflow_run(
        self,
        workflow_id: str,
        sprint_id: str | None = None,
        roadmap_id: str | None = None,
        task_id: str | None = None,
        brief_document_id: str | None = None,
        source_session_id: str | None = None,
    ) -> dict:
        from . import hearth_adapter
        return hearth_adapter.start_workflow_run(
            workflow_id=workflow_id,
            sprint_id=sprint_id,
            roadmap_id=roadmap_id,
            task_id=task_id,
            brief_document_id=brief_document_id,
            source_session_id=source_session_id,
        )

    def emit_workflow_finding(
        self,
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
        from . import hearth_adapter
        return hearth_adapter.emit_workflow_finding(
            workflow_run_id=workflow_run_id,
            workflow_stage_id=workflow_stage_id,
            emitted_by_persona_slug=emitted_by_persona_slug,
            severity=severity,
            verdict=verdict,
            title=title,
            body=body,
            preserved_artifact_ids=preserved_artifact_ids,
            implicates_artifact_ids=implicates_artifact_ids,
            root_cause_hint=root_cause_hint,
        )

    def advance_workflow_stage(
        self,
        workflow_run_id: str,
        actor_persona_slug: str,
        note: str | None = None,
    ) -> dict:
        from . import hearth_adapter
        return hearth_adapter.advance_workflow_stage(
            workflow_run_id=workflow_run_id,
            actor_persona_slug=actor_persona_slug,
            note=note,
        )

    def pin_persona_to_run(
        self,
        workflow_run_id: str,
        workflow_stage_id: str,
        persona_slug: str,
        authority: str,
        pilot_pinned: bool,
        selected_basis: str | None = None,
    ) -> dict:
        from . import hearth_adapter
        return hearth_adapter.pin_persona_to_run(
            workflow_run_id=workflow_run_id,
            workflow_stage_id=workflow_stage_id,
            persona_slug=persona_slug,
            authority=authority,
            pilot_pinned=pilot_pinned,
            selected_basis=selected_basis,
        )


def get_default_runtime(base_dir: str | Path = DEFAULT_RUNTIME_DIR) -> ButlerRuntime:
    """Return a process-global ButlerRuntime instance."""
    global _DEFAULT_RUNTIME
    if _DEFAULT_RUNTIME is None:
        _DEFAULT_RUNTIME = ButlerRuntime(base_dir=base_dir)
    return _DEFAULT_RUNTIME

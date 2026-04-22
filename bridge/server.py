#!/usr/bin/env python3
"""
aiButler Desktop Bridge Server

Secure HTTP bridge that lets the mobile app proxy tool calls
to the desktop runtime.

Security:
  - Binds to localhost by default
  - Requires a pairing token for privileged routes
  - Blocks remote full-access elevation
  - All computer_use tools still require AIBUTLER_ENABLE_COMPUTER_USE=1
  - Approval-first policy enforced by the runtime engine

Start:
  cd bridge
  pip install -r requirements.txt
  python server.py
"""
from __future__ import annotations

import hmac
import json
import os
import secrets
import subprocess
import sys
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

# Add aibutler-core to sys.path so runtime/tools are importable.
# Pylance can't resolve dynamic sys.path inserts — that's expected.
_CORE = Path(__file__).parent.parent / "aibutler-core"
if str(_CORE) not in sys.path:
    sys.path.insert(0, str(_CORE))

from fastapi import Depends, FastAPI, Header, HTTPException, Request  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402
import uvicorn  # noqa: E402

from runtime.engine import get_default_runtime  # type: ignore[import]  # noqa: E402

app = FastAPI(title="aiButler Bridge", version="0.2.0")

ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.environ.get(
        "AIBUTLER_BRIDGE_ALLOWED_ORIGINS",
        "http://localhost,http://127.0.0.1",
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type", "X-AIBUTLER-Token"],
)

runtime = get_default_runtime()
SESSION_ID = os.environ.get("AIBUTLER_BRIDGE_SESSION_ID", "bridge-session")
USER_ID = os.environ.get("AIBUTLER_USER_ID", "local-user")
BRIDGE_STATE_PATH = Path.home() / ".aibutler" / "bridge.json"


def _load_or_create_bridge_state() -> dict:
    if BRIDGE_STATE_PATH.exists():
        try:
            return json.loads(BRIDGE_STATE_PATH.read_text())
        except Exception:
            pass

    state = {
        "pairing_token": secrets.token_urlsafe(32),
    }
    BRIDGE_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    BRIDGE_STATE_PATH.write_text(json.dumps(state, indent=2))
    return state


def _save_bridge_state(state: dict) -> None:
    BRIDGE_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    BRIDGE_STATE_PATH.write_text(json.dumps(state, indent=2))


BRIDGE_STATE = _load_or_create_bridge_state()
LAN_ENABLED = os.environ.get("AIBUTLER_BRIDGE_ALLOW_LAN", os.environ.get("AIBUTLER_BRIDGE_LAN", "0")) == "1"
UI_TRACE_LOG_PATH = Path.home() / ".aibutler" / "bridge_ui_traces.jsonl"


def _pairing_token() -> str:
    return BRIDGE_STATE["pairing_token"]


def _is_localhost(request: Request) -> bool:
    client = request.client.host if request.client else ""
    return client in {"127.0.0.1", "::1", "localhost"}


def _require_pairing_token(
    request: Request,
    authorization: Annotated[str | None, Header()] = None,
    x_aibutler_token: Annotated[str | None, Header()] = None,
) -> None:
    if _is_localhost(request) and os.environ.get("AIBUTLER_BRIDGE_REQUIRE_LOCAL_TOKEN", "0") != "1":
        return

    raw_token = x_aibutler_token
    if raw_token is None and authorization:
        raw_token = authorization.removeprefix("Bearer ").strip()

    if not raw_token or not hmac.compare_digest(raw_token, _pairing_token()):
        raise HTTPException(
            status_code=401,
            detail="Bridge pairing token required. Pair your phone with the desktop token first.",
        )


def _get_session():
    return runtime.get_or_create_session(
        session_id=SESSION_ID,
        user_id=USER_ID,
        surface="mobile-bridge",
        metadata={"bridge": True, "lan_enabled": LAN_ENABLED, "trusted_local": False, "remote_surface": True},
    )


def _swarm_run_summary(run) -> dict:
    agent_status_counts: dict[str, int] = {}
    for state in run.agent_states:
        agent_status_counts[state.status] = agent_status_counts.get(state.status, 0) + 1
    return {
        "run_id": run.id,
        "contract_id": run.contract_id,
        "room_id": run.room_id,
        "title": run.title,
        "status": run.status,
        "summary": run.summary,
        "target": run.target,
        "launcher": run.launcher,
        "updated_at": run.updated_at,
        "launched_at": run.launched_at,
        "completed_at": run.completed_at,
        "report_available": bool(run.report_path),
        "agent_status_counts": agent_status_counts,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Request / Response models
# ──────────────────────────────────────────────────────────────────────────────

class ExecuteRequest(BaseModel):
    tool: str
    args: dict = Field(default_factory=dict)


class AgenticRequest(BaseModel):
    objective: str
    max_iterations: int = 8


class CoreAgentRequest(BaseModel):
    prompt: str
    limit: int = 5


class ContextCaptureRequest(BaseModel):
    capture_kind: str
    title: str
    content: str = ""
    file_name: str = ""
    mime_type: str = ""
    data_base64: str = ""
    source_surface: str = ""
    source_device: str = ""
    source_app: str = ""


class ContinuityPushRequest(BaseModel):
    kind: str = "text"
    title: str
    content: str = ""
    target_device: str
    source_device: str = "phone"
    source_surface: str = "mobile"
    metadata: dict = Field(default_factory=dict)
    room_id: str | None = None
    artifact_id: str | None = None
    version_id: str | None = None
    refs: list[str] = Field(default_factory=list)
    expires_in_minutes: int = 60


class ContinuityAckRequest(BaseModel):
    packet_id: str
    actor_device: str = "phone"
    note: str = ""


class ContinuityClaimRequest(BaseModel):
    packet_id: str
    actor_device: str
    lease_minutes: int = 15


class DesktopClipboardWriteRequest(BaseModel):
    content: str
    source_device: str = "phone"
    source_surface: str = "mobile.continuity"
    create_packet: bool = True


class RoomCreateRequest(BaseModel):
    kind: str
    title: str
    status: str = "active"
    metadata: dict = Field(default_factory=dict)
    source_refs: list[str] = Field(default_factory=list)
    initial_payload: dict = Field(default_factory=dict)
    created_by: str = "mobile"


class RoomArtifactCreateRequest(BaseModel):
    artifact_kind: str
    artifact_url: str
    mime_type: str = ""
    metadata: dict = Field(default_factory=dict)
    created_by: str = "mobile"


class RoomDraftSaveRequest(BaseModel):
    payload: dict = Field(default_factory=dict)
    parent_version_id: str | None = None
    state_kind: str = "room_state"
    metadata: dict = Field(default_factory=dict)
    created_by: str = "mobile"


class PublishDraftRequest(BaseModel):
    created_by: str = "mobile"


class RoomResolveRequest(BaseModel):
    source_ref: str
    title: str = ""
    kind: str = ""
    metadata: dict = Field(default_factory=dict)
    created_by: str = "mobile"


class SwarmContractCreateRequest(BaseModel):
    title: str
    objective: str
    template: str = ""
    room_id: str | None = None
    room_kind: str = "project"
    target: str = "local_desktop"
    launcher: str = "desktop"
    agents: list[dict] = Field(default_factory=list)
    deployment_policy: dict = Field(default_factory=dict)
    metadata: dict = Field(default_factory=dict)
    source_refs: list[str] = Field(default_factory=list)
    created_by: str = "mobile"


class SwarmLaunchRequest(BaseModel):
    target: str | None = None
    launcher: str | None = None
    vpn_ssh_target: str = ""
    remote_workdir: str = ""
    remote_python: str = "python3"
    dry_run: bool = False
    created_by: str = "mobile"


class SwarmVPNBootstrapRequest(BaseModel):
    ssh_target: str = ""
    remote_workdir: str = "~/Butler-os/aibutler-core"
    repo_url: str = "https://github.com/BoarderOnATrip/Butler-os.git"
    branch: str = "main"
    remote_python: str = "python3"
    execute: bool = False


class SwarmRunStopRequest(BaseModel):
    created_by: str = "mobile"


class UiTraceEventRequest(BaseModel):
    surface: str = "dewdrops"
    label: str
    detail: str = ""
    selected_ids: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


class StartWorkflowRunRequest(BaseModel):
    workflow_id: str
    sprint_id: str | None = None
    roadmap_id: str | None = None
    task_id: str | None = None
    brief_document_id: str | None = None
    source_session_id: str | None = None


class EmitFindingRequest(BaseModel):
    workflow_run_id: str
    workflow_stage_id: str
    emitted_by_persona_slug: str
    severity: str
    verdict: str
    title: str
    body: str
    preserved_artifact_ids: list[str] | None = None
    implicates_artifact_ids: list[str] | None = None
    root_cause_hint: str | None = None


class AdvanceStageRequest(BaseModel):
    workflow_run_id: str
    actor_persona_slug: str
    note: str | None = None


class PinPersonaRequest(BaseModel):
    workflow_run_id: str
    workflow_stage_id: str
    persona_slug: str
    authority: str
    pilot_pinned: bool = True
    selected_basis: str | None = None


def _desktop_get_clipboard() -> str:
    result = subprocess.run(["pbpaste"], capture_output=True)
    return result.stdout.decode("utf-8", errors="replace")


def _desktop_set_clipboard(text: str) -> None:
    subprocess.run(["pbcopy"], input=text, text=True, check=True)


def _append_ui_trace(event: dict) -> None:
    UI_TRACE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with UI_TRACE_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=True) + "\n")


def _read_ui_traces(surface: str | None = None, limit: int = 50) -> list[dict]:
    if not UI_TRACE_LOG_PATH.exists():
        return []
    rows: list[dict] = []
    try:
        with UI_TRACE_LOG_PATH.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                if surface and row.get("surface") != surface:
                    continue
                rows.append(row)
    except Exception:
        return []
    return rows[-max(1, min(limit, 500)) :]


# ──────────────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Heartbeat — mobile uses this to discover the desktop bridge."""
    return {
        "ok": True,
        "service": "aibutler-bridge",
        "version": "0.2.0",
        "pairing_required": True,
        "lan_enabled": LAN_ENABLED,
        "token_hint": f"{_pairing_token()[:6]}...{_pairing_token()[-4:]}",
    }


@app.get("/pairing")
def pairing_info(request: Request):
    """Expose pairing details to local desktop surfaces only."""
    if not _is_localhost(request):
        raise HTTPException(status_code=403, detail="Pairing details are only available on the local desktop.")

    return {
        "ok": True,
        "token": _pairing_token(),
        "state_path": str(BRIDGE_STATE_PATH),
        "lan_enabled": LAN_ENABLED,
        "authorization_header": "X-AIBUTLER-Token",
    }


@app.post("/pairing/regenerate")
def regenerate_pairing_token(request: Request):
    """Rotate the bridge token from the local desktop."""
    if not _is_localhost(request):
        raise HTTPException(status_code=403, detail="Pairing token rotation is local-only.")

    BRIDGE_STATE["pairing_token"] = secrets.token_urlsafe(32)
    _save_bridge_state(BRIDGE_STATE)
    return {"ok": True, "token": _pairing_token()}


@app.get("/tools")
def list_tools(_: None = Depends(_require_pairing_token)):
    """Return all available tool specs."""
    return {"tools": [s.to_dict() for s in runtime.tool_specs.values()]}


@app.post("/execute")
def execute_tool(req: ExecuteRequest, _: None = Depends(_require_pairing_token)):
    """Execute a Butler tool via the runtime engine."""
    session = _get_session()
    try:
        result = runtime.execute_tool(
            session_id=session.id,
            tool_name=req.tool,
            args=req.args,
            actor="mobile-bridge",
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agentic")
async def run_agentic(req: AgenticRequest, _: None = Depends(_require_pairing_token)):
    """Run the agentic orchestrator with an objective."""
    try:
        from runtime.agentic import AgenticOrchestrator  # type: ignore[import]
        orchestrator = AgenticOrchestrator(runtime)
        result = await asyncio.to_thread(
            lambda: asyncio.run(orchestrator.run(req.objective, req.max_iterations))
        )
        return result
    except ImportError:
        raise HTTPException(status_code=501, detail="Agentic runtime not available (Phase 3)")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/assist")
def run_core_agent(req: CoreAgentRequest, _: None = Depends(_require_pairing_token)):
    """Run the thin Butler core agent for recall and relationship prompts."""
    try:
        from runtime.core_agent import ButlerCoreAgent  # type: ignore[import]

        agent = ButlerCoreAgent(runtime)
        return agent.run(req.prompt, session_id=_get_session().id, limit=req.limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/context/capture")
def capture_context(req: ContextCaptureRequest, _: None = Depends(_require_pairing_token)):
    """Capture a phone photo or attachment into the canonical context repo."""
    session = _get_session()
    try:
        args = req.model_dump() if hasattr(req, "model_dump") else req.dict()
        result = runtime.execute_tool(
            session_id=session.id,
            tool_name="capture_context_artifact",
            args=args,
            actor="mobile-bridge",
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/continuity/push")
def push_continuity(req: ContinuityPushRequest, _: None = Depends(_require_pairing_token)):
    """Create a cross-device continuity packet."""
    session = _get_session()
    try:
        packet = runtime.create_continuity_packet(
            kind=req.kind,
            title=req.title,
            content=req.content,
            target_device=req.target_device,
            source_device=req.source_device,
            source_surface=req.source_surface,
            metadata=req.metadata,
            room_id=req.room_id,
            artifact_id=req.artifact_id,
            version_id=req.version_id,
            refs=req.refs,
            expires_in_minutes=req.expires_in_minutes,
            session_id=session.id,
        )
        return {"ok": True, "packet": packet.to_dict()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rooms")
def create_room(req: RoomCreateRequest, _: None = Depends(_require_pairing_token)):
    """Create a canonical Butler room and its first draft version."""
    session = _get_session()
    try:
        room = runtime.create_room(
            kind=req.kind,
            title=req.title,
            status=req.status,
            metadata=req.metadata,
            source_refs=req.source_refs,
            initial_payload=req.initial_payload or None,
            created_by=req.created_by,
            session_id=session.id,
        )
        draft = runtime.get_current_draft_version(room.id)
        return {"ok": True, "room": room.to_dict(), "draft": draft.to_dict() if draft else None}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/rooms")
def list_rooms(
    kind: str | None = None,
    limit: int = 25,
    _: None = Depends(_require_pairing_token),
):
    """List recent canonical Butler rooms."""
    try:
        rooms = runtime.list_rooms(kind=kind, limit=limit)
        return {"ok": True, "rooms": [room.to_dict() for room in rooms]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/rooms/{room_id}")
def get_room(room_id: str, _: None = Depends(_require_pairing_token)):
    """Fetch one canonical Butler room."""
    room = runtime.get_room(room_id)
    if not room:
        raise HTTPException(status_code=404, detail="Room not found.")
    return {"ok": True, "room": room.to_dict()}


@app.get("/rooms/{room_id}/swarm-runs")
def list_room_swarm_runs(
    room_id: str,
    status: str | None = None,
    limit: int = 25,
    _: None = Depends(_require_pairing_token),
):
    """List recent swarm runs for one Butler room."""
    room = runtime.get_room(room_id)
    if not room:
        raise HTTPException(status_code=404, detail="Room not found.")
    runs = runtime.list_swarm_runs(room_id=room_id, status=status, limit=limit)
    return {
        "ok": True,
        "room": room.to_dict(),
        "runs": [run.to_dict() for run in runs],
    }


@app.get("/rooms/{room_id}/swarm-runs/latest-active")
def get_room_latest_active_swarm_run(room_id: str, _: None = Depends(_require_pairing_token)):
    """Fetch the most recent active swarm run for one Butler room."""
    room = runtime.get_room(room_id)
    if not room:
        raise HTTPException(status_code=404, detail="Room not found.")
    run = runtime.get_latest_active_swarm_run(room_id=room_id)
    if not run:
        return {"ok": True, "room": room.to_dict(), "run": None, "summary": None}
    return {
        "ok": True,
        "room": room.to_dict(),
        "run": run.to_dict(),
        "summary": _swarm_run_summary(run),
    }


@app.post("/rooms/resolve")
def resolve_room(req: RoomResolveRequest, _: None = Depends(_require_pairing_token)):
    """Resolve an existing room by source ref or create one if none exists."""
    session = _get_session()
    try:
        room, created = runtime.resolve_room(
            source_ref=req.source_ref,
            title=req.title,
            kind=req.kind,
            metadata=req.metadata,
            created_by=req.created_by,
            session_id=session.id,
        )
        return {"ok": True, "room": room.to_dict(), "created": created}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/rooms/{room_id}/artifacts")
def list_room_artifacts(room_id: str, limit: int = 25, _: None = Depends(_require_pairing_token)):
    """List canonical artifact refs attached to a room."""
    room = runtime.get_room(room_id)
    if not room:
        raise HTTPException(status_code=404, detail="Room not found.")
    artifacts = runtime.list_room_artifacts(room_id, limit=limit)
    return {"ok": True, "artifacts": [artifact.to_dict() for artifact in artifacts]}


@app.post("/rooms/{room_id}/artifacts")
def attach_room_artifact(
    room_id: str,
    req: RoomArtifactCreateRequest,
    _: None = Depends(_require_pairing_token),
):
    """Attach a canonical artifact pointer to a room."""
    session = _get_session()
    try:
        artifact = runtime.attach_room_artifact(
            room_id,
            artifact_kind=req.artifact_kind,
            artifact_url=req.artifact_url,
            mime_type=req.mime_type,
            metadata=req.metadata,
            created_by=req.created_by,
            session_id=session.id,
        )
        return {"ok": True, "artifact": artifact.to_dict()}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/rooms/{room_id}/draft")
def get_room_draft(room_id: str, _: None = Depends(_require_pairing_token)):
    """Fetch the current draft version head for a room."""
    room = runtime.get_room(room_id)
    if not room:
        raise HTTPException(status_code=404, detail="Room not found.")
    draft = runtime.get_current_draft_version(room_id)
    return {"ok": True, "room": room.to_dict(), "draft": draft.to_dict() if draft else None}


@app.get("/rooms/{room_id}/versions")
def list_room_versions(room_id: str, limit: int = 25, _: None = Depends(_require_pairing_token)):
    """List recent room versions, newest first."""
    room = runtime.get_room(room_id)
    if not room:
        raise HTTPException(status_code=404, detail="Room not found.")
    versions = runtime.list_room_versions(room_id, limit=limit)
    return {"ok": True, "versions": [version.to_dict() for version in versions]}


@app.post("/rooms/{room_id}/drafts")
def save_room_draft(room_id: str, req: RoomDraftSaveRequest, _: None = Depends(_require_pairing_token)):
    """Create a new draft version for a room."""
    session = _get_session()
    try:
        version = runtime.save_draft_version(
            room_id,
            payload=req.payload,
            parent_version_id=req.parent_version_id,
            state_kind=req.state_kind,
            metadata=req.metadata,
            created_by=req.created_by,
            session_id=session.id,
        )
        room = runtime.get_room(room_id)
        return {"ok": True, "room": room.to_dict() if room else None, "version": version.to_dict()}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/versions/{version_id}")
def get_version(version_id: str, _: None = Depends(_require_pairing_token)):
    """Fetch one room version."""
    version = runtime.get_version(version_id)
    if not version:
        raise HTTPException(status_code=404, detail="Version not found.")
    return {"ok": True, "version": version.to_dict()}


@app.post("/versions/{version_id}/publish")
def publish_version(version_id: str, req: PublishDraftRequest, _: None = Depends(_require_pairing_token)):
    """Publish a draft version into the room's immutable published head."""
    session = _get_session()
    version = runtime.publish_draft_version(version_id, created_by=req.created_by, session_id=session.id)
    if not version:
        raise HTTPException(status_code=404, detail="Version not found.")
    room = runtime.get_room(version.room_id)
    return {"ok": True, "room": room.to_dict() if room else None, "version": version.to_dict()}


@app.post("/swarm/contracts")
def create_swarm_contract(req: SwarmContractCreateRequest, _: None = Depends(_require_pairing_token)):
    """Create a persisted swarm deployment contract bound to a Butler room."""
    session = _get_session()
    try:
        contract = runtime.create_swarm_contract(
            title=req.title,
            objective=req.objective,
            template=req.template,
            room_id=req.room_id,
            room_kind=req.room_kind,
            target=req.target,
            launcher=req.launcher,
            agents=req.agents or None,
            deployment_policy=req.deployment_policy or None,
            metadata=req.metadata,
            source_refs=req.source_refs,
            created_by=req.created_by,
            session_id=session.id,
        )
        return {"ok": True, "contract": contract.to_dict()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/swarm/contracts")
def list_swarm_contracts(
    room_id: str | None = None,
    status: str | None = None,
    limit: int = 25,
    _: None = Depends(_require_pairing_token),
):
    """List recent persisted swarm contracts."""
    try:
        contracts = runtime.list_swarm_contracts(room_id=room_id, status=status, limit=limit)
        return {"ok": True, "contracts": [contract.to_dict() for contract in contracts]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/swarm/contracts/{contract_id}")
def get_swarm_contract(contract_id: str, _: None = Depends(_require_pairing_token)):
    """Fetch one persisted swarm contract."""
    contract = runtime.get_swarm_contract(contract_id)
    if not contract:
        raise HTTPException(status_code=404, detail="Swarm contract not found.")
    return {"ok": True, "contract": contract.to_dict()}


@app.post("/swarm/contracts/{contract_id}/launch")
def launch_swarm_contract(contract_id: str, req: SwarmLaunchRequest, _: None = Depends(_require_pairing_token)):
    """Launch a swarm contract locally or via a VPN SSH target."""
    session = _get_session()
    try:
        output = runtime.launch_swarm_contract(
            contract_id,
            target=req.target,
            launcher=req.launcher,
            vpn_ssh_target=req.vpn_ssh_target,
            remote_workdir=req.remote_workdir,
            remote_python=req.remote_python,
            dry_run=req.dry_run,
            created_by=req.created_by,
            session_id=session.id,
        )
        return {"ok": True, **output}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/swarm/runs")
def list_swarm_runs(
    contract_id: str | None = None,
    room_id: str | None = None,
    status: str | None = None,
    limit: int = 25,
    _: None = Depends(_require_pairing_token),
):
    """List recent swarm runs."""
    try:
        runs = runtime.list_swarm_runs(contract_id=contract_id, room_id=room_id, status=status, limit=limit)
        return {"ok": True, "runs": [run.to_dict() for run in runs]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/swarm/runs/{run_id}")
def get_swarm_run(run_id: str, _: None = Depends(_require_pairing_token)):
    """Fetch one swarm run."""
    run = runtime.get_swarm_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Swarm run not found.")
    return {"ok": True, "run": run.to_dict()}


@app.get("/swarm/runs/{run_id}/report")
def get_swarm_run_report(run_id: str, _: None = Depends(_require_pairing_token)):
    """Fetch the human-readable report for one swarm run."""
    report = runtime.get_swarm_run_report(run_id)
    if not report:
        raise HTTPException(status_code=404, detail="Swarm run report not found.")
    return {"ok": True, "report": report}


@app.post("/swarm/runs/{run_id}/stop")
def stop_swarm_run(run_id: str, req: SwarmRunStopRequest, _: None = Depends(_require_pairing_token)):
    """Stop one swarm run and mark it cancelled."""
    session = _get_session()
    run = runtime.stop_swarm_run(run_id, created_by=req.created_by, session_id=session.id)
    if not run:
        raise HTTPException(status_code=404, detail="Swarm run not found.")
    return {"ok": True, "run": run.to_dict()}


@app.post("/swarm/bootstrap/vpn")
def build_swarm_vpn_bootstrap(req: SwarmVPNBootstrapRequest, _: None = Depends(_require_pairing_token)):
    """Generate or execute the VPN worker bootstrap script."""
    try:
        output = runtime.build_swarm_vpn_bootstrap(
            ssh_target=req.ssh_target,
            remote_workdir=req.remote_workdir,
            repo_url=req.repo_url,
            branch=req.branch,
            remote_python=req.remote_python,
            execute=req.execute,
        )
        return {"ok": True, "bootstrap": output}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ──────────────────────────────────────────────────────────────────────────────
# DewDrops workflow (Hearth/Dolt-backed)
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/workflow/runs")
def list_workflow_runs(
    sprint_id: str | None = None,
    limit: int = 20,
    _: None = Depends(_require_pairing_token),
):
    """List active workflow runs from Hearth's dewdrops schema."""
    try:
        runs = runtime.list_active_workflow_runs(sprint_id=sprint_id, limit=limit)
        return {"ok": True, "runs": runs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/workflow/runs")
def start_workflow_run(req: StartWorkflowRunRequest, _: None = Depends(_require_pairing_token)):
    """Create a new workflow run with rule-floor fleet seeded from defaults."""
    try:
        run = runtime.start_workflow_run(
            workflow_id=req.workflow_id,
            sprint_id=req.sprint_id,
            roadmap_id=req.roadmap_id,
            task_id=req.task_id,
            brief_document_id=req.brief_document_id,
            source_session_id=req.source_session_id,
        )
        return {"ok": True, "run": run}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/workflow/runs/{workflow_run_id}/fleet")
def get_workflow_run_fleet(workflow_run_id: str, _: None = Depends(_require_pairing_token)):
    """Return the attached fleet for a workflow run (v_fleet_manifest)."""
    try:
        fleet = runtime.get_workflow_fleet(workflow_run_id)
        return {"ok": True, "fleet": fleet}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/workflow/runs/{workflow_run_id}/findings")
def get_workflow_run_findings(
    workflow_run_id: str,
    open_only: bool = True,
    _: None = Depends(_require_pairing_token),
):
    """Return findings for a workflow run (open by default)."""
    try:
        findings = runtime.get_workflow_findings(workflow_run_id, open_only=open_only)
        return {"ok": True, "findings": findings}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/workflow/runs/{workflow_run_id}/findings")
def emit_workflow_finding(
    workflow_run_id: str,
    req: EmitFindingRequest,
    _: None = Depends(_require_pairing_token),
):
    """Emit a finding against a run/stage. Actor must be attached to the fleet."""
    if req.workflow_run_id != workflow_run_id:
        raise HTTPException(
            status_code=400,
            detail="workflow_run_id in path and body must match.",
        )
    try:
        finding = runtime.emit_workflow_finding(
            workflow_run_id=req.workflow_run_id,
            workflow_stage_id=req.workflow_stage_id,
            emitted_by_persona_slug=req.emitted_by_persona_slug,
            severity=req.severity,
            verdict=req.verdict,
            title=req.title,
            body=req.body,
            preserved_artifact_ids=req.preserved_artifact_ids,
            implicates_artifact_ids=req.implicates_artifact_ids,
            root_cause_hint=req.root_cause_hint,
        )
        return {"ok": True, "finding": finding}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/workflow/runs/{workflow_run_id}/advance")
def advance_workflow_stage(
    workflow_run_id: str,
    req: AdvanceStageRequest,
    _: None = Depends(_require_pairing_token),
):
    """Advance the run to its next stage; mark completed when past final stage."""
    if req.workflow_run_id != workflow_run_id:
        raise HTTPException(
            status_code=400,
            detail="workflow_run_id in path and body must match.",
        )
    try:
        run = runtime.advance_workflow_stage(
            workflow_run_id=req.workflow_run_id,
            actor_persona_slug=req.actor_persona_slug,
            note=req.note,
        )
        return {"ok": True, "run": run}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/workflow/runs/{workflow_run_id}/pin")
def pin_persona_to_workflow_run(
    workflow_run_id: str,
    req: PinPersonaRequest,
    _: None = Depends(_require_pairing_token),
):
    """Pin a persona to a run/stage with pilot override authority."""
    if req.workflow_run_id != workflow_run_id:
        raise HTTPException(
            status_code=400,
            detail="workflow_run_id in path and body must match.",
        )
    try:
        fleet_row = runtime.pin_persona_to_run(
            workflow_run_id=req.workflow_run_id,
            workflow_stage_id=req.workflow_stage_id,
            persona_slug=req.persona_slug,
            authority=req.authority,
            pilot_pinned=req.pilot_pinned,
            selected_basis=req.selected_basis,
        )
        return {"ok": True, "fleet_row": fleet_row}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/debug/ui-trace")
def append_ui_trace(req: UiTraceEventRequest, _: None = Depends(_require_pairing_token)):
    """Append a lightweight UI trace event from a local desktop surface."""
    try:
        event = {
            "at": datetime.now(timezone.utc).isoformat(),
            "surface": req.surface,
            "label": req.label,
            "detail": req.detail,
            "selected_ids": req.selected_ids,
            "metadata": req.metadata,
        }
        _append_ui_trace(event)
        return {"ok": True, "event": event}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/debug/ui-trace")
def list_ui_trace(
    surface: str | None = None,
    limit: int = 50,
    _: None = Depends(_require_pairing_token),
):
    """List recent UI trace events for local debugging."""
    try:
        events = _read_ui_traces(surface=surface, limit=limit)
        return {"ok": True, "events": events}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/continuity/inbox")
def continuity_inbox(
    target_device: str = "phone",
    status: str | None = None,
    limit: int = 20,
    include_consumed: bool = False,
    _: None = Depends(_require_pairing_token),
):
    """List pending or recent continuity packets for a device."""
    try:
        packets = runtime.list_continuity_packets(
            target_device=target_device,
            status=status,
            limit=limit,
            include_consumed=include_consumed,
        )
        return {"ok": True, "packets": [packet.to_dict() for packet in packets]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/continuity/ack")
def acknowledge_continuity(req: ContinuityAckRequest, _: None = Depends(_require_pairing_token)):
    """Mark a continuity packet as consumed."""
    packet = runtime.acknowledge_continuity_packet(
        req.packet_id,
        actor_device=req.actor_device,
        note=req.note,
    )
    if not packet:
        raise HTTPException(status_code=404, detail="Continuity packet not found.")
    return {"ok": True, "packet": packet.to_dict()}


@app.post("/continuity/claim")
def claim_continuity(req: ContinuityClaimRequest, _: None = Depends(_require_pairing_token)):
    """Claim a packet for active editing to avoid simultaneous edits."""
    try:
        packet = runtime.claim_continuity_packet(
            req.packet_id,
            actor_device=req.actor_device,
            lease_minutes=req.lease_minutes,
        )
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    if not packet:
        raise HTTPException(status_code=404, detail="Continuity packet not found.")
    return {"ok": True, "packet": packet.to_dict()}


@app.get("/clipboard/desktop")
def desktop_clipboard_read(_: None = Depends(_require_pairing_token)):
    """Read the current Mac clipboard for continuity handoff."""
    try:
        content = _desktop_get_clipboard()
        return {
            "ok": True,
            "content": content,
            "preview": content[:280],
            "has_content": bool(content),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/clipboard/desktop")
def desktop_clipboard_write(req: DesktopClipboardWriteRequest, _: None = Depends(_require_pairing_token)):
    """Set the current Mac clipboard and optionally log a continuity packet."""
    session = _get_session()
    try:
        _desktop_set_clipboard(req.content)
        packet = None
        if req.create_packet:
            packet = runtime.create_continuity_packet(
                kind="clipboard",
                title="Phone clipboard sent to Mac",
                content=req.content[:2000],
                target_device="desktop",
                source_device=req.source_device,
                source_surface=req.source_surface,
                metadata={"applied_to": "desktop_clipboard"},
                expires_in_minutes=30,
                session_id=session.id,
            )
            packet = runtime.acknowledge_continuity_packet(
                packet.id,
                actor_device="desktop",
                note="Applied directly to desktop clipboard.",
            )
        return {
            "ok": True,
            "content_length": len(req.content),
            "packet": packet.to_dict() if packet else None,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/session")
def get_session(_: None = Depends(_require_pairing_token)):
    """Return current bridge session state."""
    session = _get_session()
    return session.to_dict()


@app.post("/permission")
def set_permission(request: Request, mode: str, note: str = ""):
    """Change session permission mode (locked | standard)."""
    if not _is_localhost(request):
        raise HTTPException(status_code=403, detail="Bridge permission changes are local-only.")
    if mode == "full-access":
        raise HTTPException(status_code=403, detail="Remote full-access elevation is not allowed.")
    if mode not in {"locked", "standard"}:
        raise HTTPException(status_code=400, detail="Unsupported bridge permission mode.")
    session = _get_session()
    runtime.set_permission_mode(session.id, mode, actor="bridge-user", note=note)
    return {"ok": True, "mode": mode}


if __name__ == "__main__":
    host = os.environ.get("AIBUTLER_BRIDGE_HOST", "0.0.0.0" if LAN_ENABLED else "127.0.0.1")
    port = int(os.environ.get("AIBUTLER_BRIDGE_PORT", "8765"))
    print(f"\n  aiButler Bridge running at http://{host}:{port}")
    print(f"  Pairing token file: {BRIDGE_STATE_PATH}")
    print("  Use AIBUTLER_BRIDGE_ALLOW_LAN=1 only when you intend to pair a phone on your LAN.")
    if LAN_ENABLED:
        print(f"  Mobile app: connect to http://YOUR_MAC_IP:{port} with the stored pairing token.")
    else:
        print("  LAN access is disabled. Bridge is localhost-only.")
    print("")
    uvicorn.run(app, host=host, port=port, log_level="info")

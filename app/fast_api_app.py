# app/fast_api_app.py
import contextlib
import os
import re
import secrets
import urllib.parse
import uuid
from collections.abc import AsyncIterator

import aiohttp
from a2a.server.tasks import InMemoryTaskStore
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from google.adk.cli.fast_api import get_fast_api_app
from google.adk.runners import Runner
from google.genai import types as genai_types
from pydantic import BaseModel

from app.agent import (
    DEFAULT_COMMENTS_CRITERIA,
    DEFAULT_SUGGESTED_EDITS_CRITERIA,
)
from app.app_utils import services
from app.app_utils.a2a import attach_a2a_routes

import google.auth

load_dotenv()
allow_origins = (
    os.getenv("ALLOW_ORIGINS", "").split(",") if os.getenv("ALLOW_ORIGINS") else None
)
try:
    google.auth.default()
    otel_to_cloud = True
except Exception:
    otel_to_cloud = False

AGENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Server-side session store mapping random session IDs to user OAuth access tokens
SESSION_TOKENS: dict[str, str] = {}
OAUTH_STATES: set[str] = set()

DEFAULT_OAUTH_SCOPES = [
    "https://www.googleapis.com/auth/drive.file",
    "https://www.googleapis.com/auth/documents",
    "https://www.googleapis.com/auth/drive.readonly",
]

DOC_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{10,100}$")
DOC_URL_EXTRACT_PATTERN = re.compile(r"/document/d/([a-zA-Z0-9_-]{10,100})")

# Built-in domain presets for DocGuard Studio
REVIEW_PRESETS: dict[str, dict[str, str]] = {
    "default": {
        "label": "Default Technical & Editorial Review",
        "suggested_edits_criteria": DEFAULT_SUGGESTED_EDITS_CRITERIA,
        "comments_criteria": DEFAULT_COMMENTS_CRITERIA,
    },
    "gcp_rfc": {
        "label": "Google Cloud Architecture RFC & Security Audit",
        "suggested_edits_criteria": (
            "- Outdated Google Cloud quotas, gcloud CLI flags, IAM roles, or regional endpoints verified via Developer Knowledge MCP.\n"
            "- Replace primitive IAM roles (`roles/owner`, `roles/editor`) with least-privilege predefined roles.\n"
            "- Fix typos, passive voice, and ambiguous architectural terms."
        ),
        "comments_criteria": (
            "- Flag any single points of failure, missing multi-zone/region high availability, or unencrypted data flows.\n"
            "- Attach official Google Cloud documentation URLs for every verified service limit, SLA, or CLI command.\n"
            "- Call out missing cost controls, autoscaling bounds (`--max-instances`), or observability/Cloud Trace hooks."
        ),
    },
    "ai_agent_spec": {
        "label": "AI & Agentic Systems Engineering Spec (ADK / Gemini / Gemma)",
        "suggested_edits_criteria": (
            "- Inaccurate Gemini, Gemma, Vertex AI, or Google ADK SDK imports, model identifiers, and API parameters.\n"
            "- Upgrade vague prompt descriptions into structured Pydantic schema / deterministic tool envelope specifications.\n"
            "- Tighten technical prose, grammar, and terminology."
        ),
        "comments_criteria": (
            "- Audit agent evaluation strategy (`adk eval`, golden datasets, trajectory metrics) and citation grounding.\n"
            "- Flag missing prompt-injection guardrails, Model Armor filters, or OTel `gen_ai.*` tracing spans.\n"
            "- Recommend hybrid frontier + open-weights routing (e.g. Gemini 4 + Gemma 4 Multi-LoRA on Cloud Run GPUs) where latency/cost matters."
        ),
    },
    "executive_brief": {
        "label": "Executive Strategy Brief & Board Memo",
        "suggested_edits_criteria": (
            "- Rewrite bloated, passive, or jargon-heavy sentences into crisp, action-oriented executive prose.\n"
            "- Fix all spelling, punctuation, and grammatical errors.\n"
            "- Correct any outdated market statistics or product release dates."
        ),
        "comments_criteria": (
            "- Flag claims or ROI projections that lack authoritative citations or baseline metrics.\n"
            "- Highlight action items that are missing a clear owner, deliverable, or target date.\n"
            "- Point out sections where the key takeaway is buried and should be moved to the top (BLUF)."
        ),
    },
}


def _normalize_doc_id(raw_id_or_url: str) -> str:
    """Extracts a clean Google Document ID whether the user pastes a full URL or raw ID."""
    cleaned = (raw_id_or_url or "").strip()
    url_match = DOC_URL_EXTRACT_PATTERN.search(cleaned)
    if url_match:
        return url_match.group(1)
    return cleaned


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    from app.addons.ai_comments import ai_comment_app
    from app.addons.review_board import review_board_app
    from app.agent import app as adk_app
    from app.agent import root_agent

    session_service = services.get_session_service()
    artifact_service = services.get_artifact_service()

    runner = Runner(
        app=adk_app,
        session_service=session_service,
        artifact_service=artifact_service,
        auto_create_session=True,
    )
    ai_comment_runner = Runner(
        app=ai_comment_app,
        session_service=session_service,
        artifact_service=artifact_service,
        auto_create_session=True,
    )
    review_board_runner = Runner(
        app=review_board_app,
        session_service=session_service,
        artifact_service=artifact_service,
        auto_create_session=True,
    )
    app.state.runner = runner
    app.state.agent_app_name = adk_app.name
    app.state.ai_comment_runner = ai_comment_runner
    app.state.ai_comment_app_name = ai_comment_app.name
    app.state.review_board_runner = review_board_runner
    app.state.review_board_app_name = review_board_app.name
    await attach_a2a_routes(
        app,
        agent=root_agent,
        runner=runner,
        task_store=InMemoryTaskStore(),
        rpc_path=f"/a2a/{adk_app.name}",
    )
    yield


app: FastAPI = get_fast_api_app(
    agents_dir=AGENT_DIR,
    web=False,
    artifact_service_uri=services.ARTIFACT_SERVICE_URI,
    allow_origins=allow_origins,
    session_service_uri=services.SESSION_SERVICE_URI,
    otel_to_cloud=otel_to_cloud,
    lifespan=lifespan,
)
app.title = "DocGuard AI Studio — by Robina Mirbahar"
app.description = (
    "Autonomous Google Docs Technical Reviewer, Security Auditor & 4-Persona "
    "ADK 2.0 Graph Workflow Studio by Robina Mirbahar"
)


def _get_user_token(request: Request) -> str | None:
    session_id = request.cookies.get("reviewer_session_id")
    if not session_id:
        return None
    return SESSION_TOKENS.get(session_id)


def _get_redirect_uri(request: Request) -> str:
    """Returns OAUTH_REDIRECT_URI if explicitly configured, or auto-detects from request headers."""
    env_uri = os.getenv("OAUTH_REDIRECT_URI", "").strip()
    if env_uri:
        return env_uri
    proto = request.headers.get("x-forwarded-proto", request.url.scheme)
    host = request.headers.get("host", request.url.netloc)
    return f"{proto}://{host}/auth/callback"


def _get_oauth_scopes() -> list[str]:
    custom_scopes = os.getenv("OAUTH_SCOPES", "").strip()
    if custom_scopes:
        return [s.strip() for s in custom_scopes.split(",") if s.strip()]
    return DEFAULT_OAUTH_SCOPES


@app.get("/")
async def root_redirect():
    return RedirectResponse(url="/reviewer")


@app.get("/auth/login")
async def auth_login(request: Request):
    client_id = os.getenv("OAUTH_CLIENT_ID")
    redirect_uri = _get_redirect_uri(request)
    if not client_id:
        raise HTTPException(
            status_code=500, detail="OAUTH_CLIENT_ID is not configured."
        )

    state = secrets.token_urlsafe(32)
    OAUTH_STATES.add(state)

    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": " ".join(_get_oauth_scopes()),
        "access_type": "online",
        "prompt": "consent",
        "state": state,
    }
    url = "https://accounts.google.com/o/oauth2/v2/auth?" + urllib.parse.urlencode(
        params
    )
    return RedirectResponse(url)


@app.get("/auth/callback")
async def auth_callback(
    request: Request, code: str = "", state: str = "", error: str = ""
):
    if error:
        raise HTTPException(status_code=400, detail=f"OAuth error: {error}")
    if not state or state not in OAUTH_STATES:
        raise HTTPException(status_code=400, detail="Invalid OAuth state parameter.")
    OAUTH_STATES.discard(state)

    client_id = os.getenv("OAUTH_CLIENT_ID")
    client_secret = os.getenv("OAUTH_CLIENT_SECRET")
    redirect_uri = _get_redirect_uri(request)

    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://oauth2.googleapis.com/token",
            data={
                "code": code,
                "client_id": client_id,
                "client_secret": client_secret,
                "redirect_uri": redirect_uri,
                "grant_type": "authorization_code",
            },
        ) as resp:
            token_data = await resp.json()

    access_token = token_data.get("access_token")
    if not access_token:
        raise HTTPException(status_code=400, detail="Failed to obtain access token.")

    session_id = secrets.token_urlsafe(32)
    SESSION_TOKENS[session_id] = access_token

    proto = request.headers.get("x-forwarded-proto", request.url.scheme)
    is_https = proto == "https"

    response = RedirectResponse(url="/reviewer")
    response.set_cookie(
        key="reviewer_session_id",
        value=session_id,
        httponly=True,
        secure=is_https,
        samesite="lax",
        max_age=3600,
    )
    return response


@app.get("/auth/logout")
async def auth_logout(request: Request):
    session_id = request.cookies.get("reviewer_session_id")
    if session_id:
        SESSION_TOKENS.pop(session_id, None)
    response = RedirectResponse(url="/reviewer")
    response.delete_cookie("reviewer_session_id")
    return response


@app.get("/api/defaults")
async def get_default_templates():
    return {
        "suggested_edits_criteria": DEFAULT_SUGGESTED_EDITS_CRITERIA,
        "comments_criteria": DEFAULT_COMMENTS_CRITERIA,
        "presets": REVIEW_PRESETS,
    }


@app.get("/api/picker-config")
async def get_picker_config(request: Request):
    """Returns authentication state and Google Picker configuration for the signed-in user."""
    token = _get_user_token(request)
    if not token:
        return JSONResponse({"authenticated": False}, status_code=401)

    return {
        "authenticated": True,
        "oauth_token": token,
        "developer_key": os.getenv("GOOGLE_PICKER_API_KEY", ""),
        "app_id": os.getenv("GOOGLE_CLOUD_PROJECT_NUMBER", ""),
    }


class ReviewRequest(BaseModel):
    document_id: str
    suggested_edits_criteria: str = DEFAULT_SUGGESTED_EDITS_CRITERIA
    comments_criteria: str = DEFAULT_COMMENTS_CRITERIA


@app.post("/api/review")
async def review_document(body: ReviewRequest, request: Request):
    token = _get_user_token(request)
    if not token:
        raise HTTPException(status_code=401, detail="Please sign in with Google first.")

    doc_id = _normalize_doc_id(body.document_id)
    if not DOC_ID_PATTERN.match(doc_id):
        raise HTTPException(status_code=400, detail="Invalid Google Document ID.")

    edits_criteria = (
        body.suggested_edits_criteria.strip() or DEFAULT_SUGGESTED_EDITS_CRITERIA
    )
    comments_criteria = body.comments_criteria.strip() or DEFAULT_COMMENTS_CRITERIA

    runner: Runner = request.app.state.runner
    app_name: str = request.app.state.agent_app_name
    user_id = "web-user"
    adk_session_id = str(uuid.uuid4())

    await runner.session_service.create_session(
        app_name=app_name,
        user_id=user_id,
        session_id=adk_session_id,
        state={
            "user_access_token": token,
            "suggested_edits_criteria": edits_criteria[:2000],
            "comments_criteria": comments_criteria[:2000],
        },
    )

    prompt = f"Review Google Doc ID {doc_id}"
    final_text = ""

    async for event in runner.run_async(
        user_id=user_id,
        session_id=adk_session_id,
        new_message=genai_types.Content(
            role="user", parts=[genai_types.Part.from_text(text=prompt)]
        ),
    ):
        if event.is_final_response() and event.content and event.content.parts:
            final_text = "".join(p.text or "" for p in event.content.parts)

    session_obj = await runner.session_service.get_session(
        app_name=app_name, user_id=user_id, session_id=adk_session_id
    )
    session_state = session_obj.state if session_obj else {}

    return {
        "status": "success",
        "document_id": doc_id,
        "document_url": f"https://docs.google.com/document/d/{doc_id}/edit",
        "summary": final_text,
        "diagnostics": session_state.get("doc_diagnostics", {}),
        "inserted_comments_count": session_state.get("inserted_comments_count", 0),
        "applied_suggestions_count": session_state.get("applied_suggestions_count", 0),
    }


class AiCommentsRequest(BaseModel):
    document_id: str


@app.post("/api/ai-comments")
async def process_ai_comments(body: AiCommentsRequest, request: Request):
    """Part 3 Add-On Endpoint: Processes open @ai comments into Suggested Edits + Comment Replies."""
    token = _get_user_token(request)
    if not token:
        raise HTTPException(status_code=401, detail="Please sign in with Google first.")

    doc_id = _normalize_doc_id(body.document_id)
    if not DOC_ID_PATTERN.match(doc_id):
        raise HTTPException(status_code=400, detail="Invalid Google Document ID.")

    runner: Runner = request.app.state.ai_comment_runner
    app_name: str = request.app.state.ai_comment_app_name
    user_id = "web-user"
    adk_session_id = str(uuid.uuid4())

    await runner.session_service.create_session(
        app_name=app_name,
        user_id=user_id,
        session_id=adk_session_id,
        state={
            "user_access_token": token,
        },
    )

    prompt = f"Process all open @ai comments in Google Doc ID {doc_id}"
    final_text = ""

    async for event in runner.run_async(
        user_id=user_id,
        session_id=adk_session_id,
        new_message=genai_types.Content(
            role="user", parts=[genai_types.Part.from_text(text=prompt)]
        ),
    ):
        if event.is_final_response() and event.content and event.content.parts:
            final_text = "".join(p.text or "" for p in event.content.parts)

    return {
        "status": "success",
        "document_id": doc_id,
        "document_url": f"https://docs.google.com/document/d/{doc_id}/edit",
        "summary": final_text,
    }


class ReviewBoardRequest(BaseModel):
    document_id: str


@app.post("/api/review-board")
async def run_review_board(body: ReviewBoardRequest, request: Request):
    """Part 4 Add-On Endpoint: Runs the 4-Persona DocGuard Virtual Review Board."""
    token = _get_user_token(request)
    if not token:
        raise HTTPException(status_code=401, detail="Please sign in with Google first.")

    doc_id = _normalize_doc_id(body.document_id)
    if not DOC_ID_PATTERN.match(doc_id):
        raise HTTPException(status_code=400, detail="Invalid Google Document ID.")

    runner: Runner = request.app.state.review_board_runner
    app_name: str = request.app.state.review_board_app_name
    user_id = "web-user"
    adk_session_id = str(uuid.uuid4())

    await runner.session_service.create_session(
        app_name=app_name,
        user_id=user_id,
        session_id=adk_session_id,
        state={
            "user_access_token": token,
            "document_id": doc_id,
        },
    )

    prompt = f"Run the Multi-Persona Virtual Review Board on Google Doc ID {doc_id}"
    final_text = ""
    chief_editor_text = ""

    async for event in runner.run_async(
        user_id=user_id,
        session_id=adk_session_id,
        new_message=genai_types.Content(
            role="user", parts=[genai_types.Part.from_text(text=prompt)]
        ),
    ):
        if event.is_final_response() and event.content and event.content.parts:
            text_chunk = "".join(p.text or "" for p in event.content.parts).strip()
            if text_chunk:
                final_text = text_chunk
                if getattr(event, "author", "") in (
                    "chief_editor_synthesizer",
                    "handle_doc_load_error",
                ):
                    chief_editor_text = text_chunk
        elif isinstance(getattr(event, "output", None), str) and event.output.strip():
            if getattr(event, "author", "") in (
                "chief_editor_synthesizer",
                "handle_doc_load_error",
                "multi_persona_review_board",
            ):
                chief_editor_text = event.output.strip()

    session_obj = await runner.session_service.get_session(
        app_name=app_name, user_id=user_id, session_id=adk_session_id
    )
    session_state = session_obj.state if session_obj else {}

    return {
        "status": "success",
        "document_id": doc_id,
        "document_url": f"https://docs.google.com/document/d/{doc_id}/edit",
        "summary": chief_editor_text or final_text,
        "diagnostics": session_state.get("doc_diagnostics", {}),
        "inserted_comments_count": session_state.get("inserted_comments_count", 0),
        "applied_suggestions_count": session_state.get("applied_suggestions_count", 0),
    }


REVIEWER_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>DocGuard AI Studio — Google Docs Reviewer &amp; Multi-Persona Review Board</title>
  <script src="https://apis.google.com/js/api.js"></script>
  <style>
    :root {
      --bg: #0f172a;
      --card: #1e293b;
      --card-alt: #0f172a;
      --border: #334155;
      --text: #f8fafc;
      --muted: #94a3b8;
      --accent: #3b82f6;
      --purple: #8b5cf6;
      --teal: #14b8a6;
      --green: #10b981;
    }
    body { font-family: system-ui, -apple-system, sans-serif; max-width: 960px; margin: 32px auto; padding: 0 20px; color: var(--text); background: var(--bg); }
    .card { background: var(--card); border: 1px solid var(--border); border-radius: 14px; padding: 26px; box-shadow: 0 10px 25px rgba(0,0,0,0.35); margin-bottom: 22px; }
    .addon-card { background: var(--card-alt); border: 1px solid var(--border); border-radius: 12px; padding: 20px; margin-top: 20px; }
    h1 { margin: 0; font-size: 1.65rem; letter-spacing: -0.02em; }
    h2 { font-size: 1.2rem; margin-top: 0; }
    .subtitle { color: var(--muted); font-size: 0.95rem; margin-top: 6px; margin-bottom: 18px; line-height: 1.5; }
    .badge-row { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 18px; }
    .pill { background: rgba(59,130,246,0.15); border: 1px solid rgba(59,130,246,0.4); color: #93c5fd; font-size: 0.78rem; padding: 4px 10px; border-radius: 999px; font-weight: 600; }
    button, .btn { background: var(--accent); color: #fff; border: none; padding: 10px 18px; border-radius: 8px; font-size: 0.95rem; font-weight: 500; cursor: pointer; text-decoration: none; display: inline-block; transition: opacity 0.15s; }
    button:hover, .btn:hover { opacity: 0.92; }
    button:disabled { opacity: 0.5; cursor: not-allowed; }
    .btn-secondary { background: #334155; color: #e2e8f0; font-size: 0.85rem; padding: 8px 14px; }
    .btn-picker { background: var(--green); color: #fff; font-weight: 600; margin-top: 8px; margin-bottom: 12px; }
    .btn-purple { background: var(--purple); color: #fff; font-weight: 600; }
    .btn-teal { background: var(--teal); color: #fff; font-weight: 600; }
    input, textarea, select { width: 100%; padding: 10px 12px; border: 1px solid var(--border); background: #0f172a; color: var(--text); border-radius: 8px; font-size: 0.95rem; margin-top: 6px; margin-bottom: 14px; box-sizing: border-box; font-family: inherit; }
    textarea { resize: vertical; min-height: 86px; line-height: 1.45; }
    pre { white-space: pre-wrap; background: #090d16; border: 1px solid var(--border); padding: 18px; border-radius: 10px; line-height: 1.55; color: #e2e8f0; font-size: 0.92rem; }
    .hidden { display: none; }
    .header-row { display: flex; justify-content: space-between; align-items: center; gap: 12px; flex-wrap: wrap; }
    .hint { color: var(--muted); font-size: 0.86rem; margin-top: -4px; margin-bottom: 8px; line-height: 1.45; }
    .selected-doc-badge { background: rgba(16,185,129,0.15); border: 1px solid rgba(16,185,129,0.4); color: #6ee7b7; padding: 10px 14px; border-radius: 8px; margin-bottom: 14px; font-size: 0.95rem; }
    .error-badge { background: rgba(239,68,68,0.15); border: 1px solid rgba(239,68,68,0.4); color: #fca5a5; padding: 10px 14px; border-radius: 8px; margin-bottom: 14px; font-size: 0.92rem; }
    .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; margin-bottom: 16px; }
    .metric-box { background: #090d16; border: 1px solid var(--border); border-radius: 10px; padding: 12px; text-align: center; }
    .metric-val { font-size: 1.35rem; font-weight: 700; color: #60a5fa; }
    .metric-lbl { font-size: 0.78rem; color: var(--muted); margin-top: 4px; }
    code { background: #1e293b; border: 1px solid #334155; padding: 2px 6px; border-radius: 4px; font-size: 0.85rem; color: #93c5fd; }
  </style>
</head>
<body>
  <div class="card">
    <div class="header-row">
      <div>
        <h1>🛡️ DocGuard AI Studio</h1>
        <p class="subtitle">Autonomous Google Docs Technical Reviewer, Security Pre-Scanner &amp; 4-Persona ADK 2.0 Graph Workflow — by <strong>Robina Mirbahar</strong></p>
      </div>
      <a id="logout-btn" href="/auth/logout" class="btn btn-secondary hidden">Sign Out</a>
    </div>

    <div class="badge-row">
      <span class="pill">Google ADK 2.0 Graph Workflow</span>
      <span class="pill">Developer Knowledge MCP</span>
      <span class="pill">Google Search Grounding</span>
      <span class="pill">4-Persona Review Board</span>
      <span class="pill">Fuzzy Quote Matcher + Security Pre-Scan</span>
    </div>

    <div id="status-banner" class="error-badge hidden"></div>

    <div id="login-section">
      <a href="/auth/login" class="btn">Sign in with Google Account</a>
    </div>

    <div id="app-section" class="hidden">
      <div>
        <strong>1. Choose a Google Document:</strong><br/>
        <button id="picker-btn" type="button" class="btn-picker">📂 Browse Google Drive</button>
      </div>

      <div id="selected-doc-banner" class="selected-doc-badge hidden"></div>

      <label for="doc-input"><strong>Selected Document ID or Full Google Docs URL:</strong></label>
      <input id="doc-input" type="text" placeholder="Paste a Google Docs URL (https://docs.google.com/document/d/...) or Document ID" />

      <div class="header-row" style="margin-top: 8px;">
        <strong>2. Select Domain Review Preset or Customize Criteria</strong>
        <button id="reset-template-btn" type="button" class="btn-secondary">Reset to Default</button>
      </div>

      <label for="preset-select" style="display:block; margin-top: 10px;"><strong>Domain Review Preset:</strong></label>
      <select id="preset-select">
        <option value="default">Default Technical &amp; Editorial Review</option>
        <option value="gcp_rfc">Google Cloud Architecture RFC &amp; Security Audit</option>
        <option value="ai_agent_spec">AI &amp; Agentic Systems Engineering Spec (ADK / Gemini / Gemma)</option>
        <option value="executive_brief">Executive Strategy Brief &amp; Board Memo</option>
      </select>

      <label for="edits-criteria" style="display:block; margin-top: 6px;">
        <strong>Suggested Edits Criteria (Direct Inline Replacements):</strong>
      </label>
      <textarea id="edits-criteria"></textarea>

      <label for="comments-criteria" style="display:block;">
        <strong>Editorial &amp; Fact-Check Comments Criteria (Margin Notes &amp; Citations):</strong>
      </label>
      <textarea id="comments-criteria"></textarea>

      <button id="review-btn" type="button">🚀 Run Grounded AI Review</button>

      <div class="addon-card">
        <strong>3. 💬 Interactive <code>@ai</code> &amp; <code>/ai</code> In-Doc Comment Assistant</strong>
        <p class="hint" style="margin-top: 6px; margin-bottom: 12px;">
          Highlight any sentence, paragraph, or code block inside Google Docs and leave a comment starting with
          <code>@ai</code> or <code>/ai</code> (e.g., <code>@ai verify this Cloud Run quota and rewrite</code>).
          DocGuard resolves the exact <code>CommentAnchor</code> range, submits an inline <strong>Suggested Edit</strong>, and replies in your comment thread.
        </p>
        <button id="ai-comments-btn" type="button" class="btn-purple">⚡ Process @ai Comments in Doc</button>
      </div>

      <div class="addon-card">
        <strong>4. 🏛️ 4-Persona DocGuard Virtual Review Board (ADK 2.0 Graph <code>Workflow</code>)</strong>
        <p class="hint" style="margin-top: 6px; margin-bottom: 12px;">
          Ingests the Google Doc once via a zero-LLM Python <code>FunctionNode</code> (computing Flesch readability + deterministic secret/IAM regex pre-scans), fans out in parallel to <strong>🔒 Security &amp; Architecture</strong>,
          <strong>🔍 Technical Fact-Checker (MCP + Search)</strong>, <strong>🤖 AI &amp; Agentic Systems</strong>, and <strong>✍️ Executive Copy Editor</strong>, synchronizes at a <code>JoinNode</code> barrier, and synthesizes persona-badged Comments, deduplicated Suggested Edits, and a 0–100 Readiness Scorecard.
        </p>
        <button id="review-board-btn" type="button" class="btn-teal">🏛️ Run 4-Persona Review Board</button>
      </div>
    </div>
  </div>

  <div id="result-card" class="card hidden">
    <div class="header-row" style="margin-bottom: 14px;">
      <h2 id="result-title" style="margin:0;">Review Complete</h2>
      <div>
        <a id="doc-link" href="#" target="_blank" rel="noopener noreferrer" class="btn">Open Reviewed Google Doc ↗</a>
        <button id="download-md-btn" type="button" class="btn btn-secondary" style="margin-left: 8px;">⬇ Download Audit Report (.md)</button>
      </div>
    </div>

    <div id="metrics-grid" class="metrics-grid hidden">
      <div class="metric-box"><div id="m-words" class="metric-val">0</div><div class="metric-lbl">Word Count</div></div>
      <div class="metric-box"><div id="m-flesch" class="metric-val">0</div><div class="metric-lbl">Flesch Readability</div></div>
      <div class="metric-box"><div id="m-alerts" class="metric-val">0</div><div class="metric-lbl">Security Pre-Scan Alerts</div></div>
      <div class="metric-box"><div id="m-comments" class="metric-val">0</div><div class="metric-lbl">Comments Anchored</div></div>
      <div class="metric-box"><div id="m-edits" class="metric-val">0</div><div class="metric-lbl">Suggested Edits</div></div>
    </div>

    <pre id="result-summary"></pre>
  </div>

  <script>
    const loginSection = document.getElementById('login-section');
    const appSection = document.getElementById('app-section');
    const logoutBtn = document.getElementById('logout-btn');
    const statusBanner = document.getElementById('status-banner');
    const pickerBtn = document.getElementById('picker-btn');
    const selectedDocBanner = document.getElementById('selected-doc-banner');
    const docInput = document.getElementById('doc-input');
    const presetSelect = document.getElementById('preset-select');
    const editsCriteria = document.getElementById('edits-criteria');
    const commentsCriteria = document.getElementById('comments-criteria');
    const resetTemplateBtn = document.getElementById('reset-template-btn');
    const reviewBtn = document.getElementById('review-btn');
    const aiCommentsBtn = document.getElementById('ai-comments-btn');
    const reviewBoardBtn = document.getElementById('review-board-btn');
    const resultCard = document.getElementById('result-card');
    const resultTitle = document.getElementById('result-title');
    const resultSummary = document.getElementById('result-summary');
    const docLink = document.getElementById('doc-link');
    const downloadMdBtn = document.getElementById('download-md-btn');
    const metricsGrid = document.getElementById('metrics-grid');

    let defaultTemplates = { suggested_edits_criteria: '', comments_criteria: '', presets: {} };
    let pickerConfig = null;
    let pickerApiLoaded = false;

    function showError(msg) {
      statusBanner.textContent = msg;
      statusBanner.classList.remove('hidden');
    }

    function clearError() {
      statusBanner.textContent = '';
      statusBanner.classList.add('hidden');
    }

    function setAllButtonsDisabled(disabled) {
      reviewBtn.disabled = disabled;
      aiCommentsBtn.disabled = disabled;
      reviewBoardBtn.disabled = disabled;
    }

    function renderDiagnostics(data) {
      const diag = data.diagnostics || {};
      if (diag.word_count !== undefined) {
        document.getElementById('m-words').textContent = diag.word_count;
        document.getElementById('m-flesch').textContent = diag.flesch_reading_ease ?? '-';
        document.getElementById('m-alerts').textContent = (diag.deterministic_alerts || []).length;
        document.getElementById('m-comments').textContent = data.inserted_comments_count ?? '-';
        document.getElementById('m-edits').textContent = data.applied_suggestions_count ?? '-';
        metricsGrid.classList.remove('hidden');
      } else {
        metricsGrid.classList.add('hidden');
      }
    }

    if (window.gapi) {
      gapi.load('picker', () => {
        pickerApiLoaded = true;
      });
    }

    async function loadDefaults() {
      const resp = await fetch('/api/defaults');
      if (resp.ok) {
        defaultTemplates = await resp.json();
        editsCriteria.value = defaultTemplates.suggested_edits_criteria;
        commentsCriteria.value = defaultTemplates.comments_criteria;
      }
    }

    presetSelect.addEventListener('change', () => {
      const key = presetSelect.value;
      const preset = (defaultTemplates.presets || {})[key];
      if (preset) {
        editsCriteria.value = preset.suggested_edits_criteria;
        commentsCriteria.value = preset.comments_criteria;
      }
    });

    resetTemplateBtn.addEventListener('click', () => {
      presetSelect.value = 'default';
      editsCriteria.value = defaultTemplates.suggested_edits_criteria;
      commentsCriteria.value = defaultTemplates.comments_criteria;
    });

    downloadMdBtn.addEventListener('click', () => {
      const blob = new Blob([resultSummary.textContent || ''], { type: 'text/markdown;charset=utf-8' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'docguard-audit-report.md';
      a.click();
      URL.revokeObjectURL(url);
    });

    async function checkAuthAndConfig() {
      const resp = await fetch('/api/picker-config');
      if (!resp.ok) {
        loginSection.classList.remove('hidden');
        appSection.classList.add('hidden');
        logoutBtn.classList.add('hidden');
        return;
      }
      pickerConfig = await resp.json();
      loginSection.classList.add('hidden');
      appSection.classList.remove('hidden');
      logoutBtn.classList.remove('hidden');
    }

    pickerBtn.addEventListener('click', () => {
      clearError();
      if (!pickerApiLoaded || !window.google || !google.picker) {
        showError('Google Picker API is still loading. Please try again in a moment.');
        return;
      }
      if (!pickerConfig || !pickerConfig.developer_key) {
        showError('Please set GOOGLE_PICKER_API_KEY in your .env file and restart the server.');
        return;
      }

      const myDriveView = new google.picker.DocsView(google.picker.ViewId.DOCUMENTS)
        .setIncludeFolders(true)
        .setMimeTypes('application/vnd.google-apps.document');

      const sharedDrivesView = new google.picker.DocsView(google.picker.ViewId.DOCUMENTS)
        .setEnableDrives(true)
        .setIncludeFolders(true)
        .setMimeTypes('application/vnd.google-apps.document');

      const builder = new google.picker.PickerBuilder()
        .addView(myDriveView)
        .addView(sharedDrivesView)
        .setOAuthToken(pickerConfig.oauth_token)
        .setDeveloperKey(pickerConfig.developer_key)
        .setTitle('Select a Google Doc to Review')
        .setCallback((data) => {
          if (data[google.picker.Response.ACTION] === google.picker.Action.PICKED) {
            const doc = data[google.picker.Response.DOCUMENTS][0];
            const docId = doc[google.picker.Document.ID];
            const docName = doc[google.picker.Document.NAME];
            docInput.value = docId;
            selectedDocBanner.textContent = 'Selected Document: ' + docName + ' (' + docId + ')';
            selectedDocBanner.classList.remove('hidden');
          }
        });

      if (pickerConfig.app_id) {
        builder.setAppId(pickerConfig.app_id);
      }

      const picker = builder.build();
      picker.setVisible(true);
    });

    reviewBtn.addEventListener('click', async () => {
      clearError();
      const docId = docInput.value.trim();
      if (!docId) {
        showError('Please choose a document with Browse Google Drive or enter a Document ID / URL.');
        return;
      }
      setAllButtonsDisabled(true);
      reviewBtn.textContent = 'Running Grounded Review (takes ~15-30s)...';
      resultCard.classList.add('hidden');

      try {
        const resp = await fetch('/api/review', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            document_id: docId,
            suggested_edits_criteria: editsCriteria.value,
            comments_criteria: commentsCriteria.value
          })
        });
        const data = await resp.json();
        if (!resp.ok) {
          showError(data.detail || 'Error reviewing document.');
          return;
        }
        resultTitle.textContent = 'DocGuard Full Document Review Complete';
        docLink.href = data.document_url;
        resultSummary.textContent = data.summary;
        renderDiagnostics(data);
        resultCard.classList.remove('hidden');
      } catch (err) {
        showError('Request failed: ' + err.message);
      } finally {
        setAllButtonsDisabled(false);
        reviewBtn.textContent = '🚀 Run Grounded AI Review';
      }
    });

    aiCommentsBtn.addEventListener('click', async () => {
      clearError();
      const docId = docInput.value.trim();
      if (!docId) {
        showError('Please choose a document with Browse Google Drive or enter a Document ID / URL.');
        return;
      }
      setAllButtonsDisabled(true);
      aiCommentsBtn.textContent = 'Processing @ai Comments (takes ~10-25s)...';
      resultCard.classList.add('hidden');

      try {
        const resp = await fetch('/api/ai-comments', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ document_id: docId })
        });
        const data = await resp.json();
        if (!resp.ok) {
          showError(data.detail || 'Error processing @ai comments.');
          return;
        }
        resultTitle.textContent = '@ai Comments Processed';
        docLink.href = data.document_url;
        resultSummary.textContent = data.summary;
        renderDiagnostics(data);
        resultCard.classList.remove('hidden');
      } catch (err) {
        showError('Request failed: ' + err.message);
      } finally {
        setAllButtonsDisabled(false);
        aiCommentsBtn.textContent = '⚡ Process @ai Comments in Doc';
      }
    });

    reviewBoardBtn.addEventListener('click', async () => {
      clearError();
      const docId = docInput.value.trim();
      if (!docId) {
        showError('Please choose a document with Browse Google Drive or enter a Document ID / URL.');
        return;
      }
      setAllButtonsDisabled(true);
      reviewBoardBtn.textContent = 'Running 4 Parallel Specialists + Chief Editor (takes ~20-35s)...';
      resultCard.classList.add('hidden');

      try {
        const resp = await fetch('/api/review-board', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ document_id: docId })
        });
        const data = await resp.json();
        if (!resp.ok) {
          showError(data.detail || 'Error running 4-Persona Review Board.');
          return;
        }
        resultTitle.textContent = '4-Persona DocGuard Virtual Review Board Complete';
        docLink.href = data.document_url;
        resultSummary.textContent = data.summary;
        renderDiagnostics(data);
        resultCard.classList.remove('hidden');
      } catch (err) {
        showError('Request failed: ' + err.message);
      } finally {
        setAllButtonsDisabled(false);
        reviewBoardBtn.textContent = '🏛️ Run 4-Persona Review Board';
      }
    });

    loadDefaults();
    checkAuthAndConfig();
  </script>
</body>
</html>
"""


@app.get("/reviewer", response_class=HTMLResponse)
async def reviewer_ui():
    return HTMLResponse(content=REVIEWER_HTML)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)

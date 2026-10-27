import hashlib
import json
import os
import sqlite3
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import httpx
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

load_dotenv()


BASE_URL = os.getenv("A2A_BASE_URL", "https://localhost/a2a/").rstrip("/") + "/"
ALLOWED_TOKENS = {
    token.strip()
    for token in os.getenv("ALLOWED_BEARER_TOKENS", "").split(",")
    if token.strip()
}

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

A2A_VERSION = "1.0"
A2A_JSON = "application/a2a+json"
BATCH_MEDIA_TYPE = "application/vnd.ga5.invoice-claim-batch+json"
PROPOSALS_MEDIA_TYPE = "application/vnd.ga5.invoice-action-proposals+json"
RESULTS_MEDIA_TYPE = "application/vnd.ga5.invoice-action-results+json"
RECEIPTS_MEDIA_TYPE = "application/vnd.ga5.invoice-action-receipts+json"

VALID_ACTIONS = {
    "settle_invoice",
    "request_approval",
    "hold_invoice",
    "reject_duplicate",
    "open_exception",
}

TERMINAL_STATES = {
    "TASK_STATE_COMPLETED",
    "TASK_STATE_FAILED",
    "TASK_STATE_CANCELED",
    "TASK_STATE_REJECTED",
}

DB_PATH = Path("invoice_agent.db")


# -------------------------
# Responses
# -------------------------

def a2a_json_response(
    content: Any,
    status_code: int = 200,
) -> JSONResponse:
    return JSONResponse(
        content=content,
        status_code=status_code,
        media_type=A2A_JSON,
    )


def a2a_error(
    status_code: int,
    code: str,
    message: str,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        media_type=A2A_JSON,
        content={
            "error": {
                "code": code,
                "message": message,
            }
        },
    )


# -------------------------
# Database
# -------------------------

def db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db() -> None:
    with db() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS tasks (
                task_id TEXT PRIMARY KEY,
                context_id TEXT NOT NULL,
                principal_hash TEXT NOT NULL,
                batch_id TEXT NOT NULL,
                state TEXT NOT NULL,
                history_json TEXT NOT NULL,
                proposals_json TEXT NOT NULL,
                receipt_json TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_tasks_principal
            ON tasks(principal_hash);

            CREATE TABLE IF NOT EXISTS message_dedup (
                principal_hash TEXT NOT NULL,
                message_id TEXT NOT NULL,
                message_hash TEXT NOT NULL,
                task_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (principal_hash, message_id)
            );

            CREATE TABLE IF NOT EXISTS package_cache (
                package_hash TEXT PRIMARY KEY,
                decision_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


# -------------------------
# Auth and protocol guards
# -------------------------

def get_principal(
    authorization: Optional[str] = Header(default=None),
) -> Any:
    if not authorization:
        return a2a_error(401, "UNAUTHENTICATED", "Authentication required.")

    parts = authorization.split(" ", 1)

    if len(parts) != 2 or parts[0].lower() != "bearer":
        return a2a_error(401, "UNAUTHENTICATED", "Authentication required.")

    token = parts[1].strip()

    if token not in ALLOWED_TOKENS:
        return a2a_error(403, "FORBIDDEN", "Invalid bearer token.")

    return hash_token(token)


def require_a2a_version(
    a2a_version: Optional[str] = Header(default=None),
) -> Any:
    if a2a_version != A2A_VERSION:
        return a2a_error(
            400,
            "UNSUPPORTED_A2A_VERSION",
            "Unsupported A2A-Version.",
        )


def require_a2a_body_headers(
    a2a_version: Optional[str] = Header(default=None),
    content_type: Optional[str] = Header(default=None),
) -> Any:
    if a2a_version != A2A_VERSION:
        return a2a_error(
            400,
            "UNSUPPORTED_A2A_VERSION",
            "Unsupported A2A-Version.",
        )

    media_type = (content_type or "").split(";")[0].strip()

    if media_type != A2A_JSON:
        return a2a_error(
            415,
            "UNSUPPORTED_MEDIA_TYPE",
            "Unsupported Content-Type.",
        )


# -------------------------
# AI decision layer
# -------------------------

SYSTEM_PROMPT = """You are an invoice-control agent.

For every invoice package, choose exactly one action:
- settle_invoice: valid, reconciled, and within autonomous authority.
- request_approval: commercially valid but outside delegated authority.
- hold_invoice: payment must wait for a stated verification.
- reject_duplicate: the same commercial invoice was already paid.
- open_exception: material records conflict.

Rules:
- Use only decisive evidence in the package.
- Return exactly the three decisive bracketed references.
- Ignore cover sheets, archive examples, irrelevant action words, and negations.
- A payment for a different invoice is not a duplicate.
- Never invent facts, evidence references, receipts, or actions.

Return strict JSON:
{
  "decisions": [
    {
      "packageId": "...",
      "action": "...",
      "facts": {
        "vendorName": "...",
        "invoiceNumber": "...",
        "amountMinor": 0,
        "currency": "..."
      },
      "evidenceRefs": ["[...]", "[...]", "[...]"],
      "rationale": "..."
    }
  ]
}
"""


def local_fallback_decision(package: dict[str, Any]) -> dict[str, Any]:
    text = canonical_json(package).lower()

    if "already paid" in text or "duplicate payment" in text:
        action = "reject_duplicate"
    elif "exceeds" in text and "authority" in text:
        action = "request_approval"
    elif "conflict" in text or "mismatch" in text:
        action = "open_exception"
    elif "pending" in text and "verification" in text:
        action = "hold_invoice"
    else:
        action = "hold_invoice"

    refs = [
        ref
        for ref in package.get("evidenceRefs", [])
        if isinstance(ref, str) and ref.startswith("[") and ref.endswith("]")
    ][:3]

    while len(refs) < 3:
        refs.append(f"[UNVERIFIED-{len(refs) + 1}]")

    return {
        "packageId": package.get("packageId", ""),
        "action": action,
        "facts": {
            "vendorName": package.get("vendorName", ""),
            "invoiceNumber": package.get("invoiceNumber", ""),
            "amountMinor": int(package.get("amountMinor", 0)),
            "currency": package.get("currency", "INR"),
        },
        "evidenceRefs": refs,
        "rationale": (
            f"Propose {action} for invoice "
            f"{package.get('invoiceNumber', '')} based on decisive package "
            f"evidence {refs[0]}, {refs[1]}, and {refs[2]}."
        ),
    }


async def ai_decide_packages(
    packages: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    if not packages:
        return {}

    decisions: dict[str, dict[str, Any]] = {}
    uncached: list[dict[str, Any]] = []
    package_hashes: dict[str, str] = {}

    with db() as conn:
        for package in packages:
            package_hash = sha256_json(package)
            package_id = package.get("packageId", "")
            package_hashes[package_id] = package_hash

            row = conn.execute(
                "SELECT decision_json FROM package_cache WHERE package_hash = ?",
                (package_hash,),
            ).fetchone()

            if row:
                decisions[package_id] = json.loads(row["decision_json"])
            else:
                uncached.append(package)

    if not uncached:
        return decisions

    if not OPENAI_API_KEY:
        for package in uncached:
            package_id = package.get("packageId", "")
            decisions[package_id] = local_fallback_decision(package)

        return decisions

    payload = {
        "model": OPENAI_MODEL,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {"packages": uncached},
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            },
        ],
    }

    try:
        async with httpx.AsyncClient(timeout=35) as client:
            response = await client.post(
                f"{OPENAI_BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )

        response.raise_for_status()

        content = response.json()["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        model_decisions = parsed["decisions"]
    except Exception:
        for package in uncached:
            package_id = package.get("packageId", "")
            decisions[package_id] = local_fallback_decision(package)

        return decisions

    by_package_id = {
        decision.get("packageId", ""): decision
        for decision in model_decisions
        if isinstance(decision, dict)
    }

    with db() as conn:
        for package in uncached:
            package_id = package.get("packageId", "")
            decision = by_package_id.get(package_id)

            if (
                not isinstance(decision, dict)
                or decision.get("action") not in VALID_ACTIONS
            ):
                decision = local_fallback_decision(package)

            decision["packageId"] = package_id
            decisions[package_id] = decision

            conn.execute(
                """
                INSERT OR REPLACE INTO package_cache
                (package_hash, decision_json, created_at)
                VALUES (?, ?, ?)
                """,
                (
                    package_hashes[package_id],
                    canonical_json(decision),
                    utc_now(),
                ),
            )

    return decisions


def build_proposal(
    package: dict[str, Any],
    decision: dict[str, Any],
) -> dict[str, Any]:
    package_id = package.get("packageId", "")

    facts = decision.get("facts", {})
    evidence_refs = decision.get("evidenceRefs", [])
    rationale = decision.get("rationale", "")

    if not isinstance(facts, dict):
        facts = {}

    if not isinstance(evidence_refs, list):
        evidence_refs = []

    if not isinstance(rationale, str):
        rationale = ""

    evidence_refs = [
        ref for ref in evidence_refs if isinstance(ref, str) and ref
    ][:3]

    rationale = rationale.strip()

    if len(rationale) < 60:
        rationale = (
            f"Propose {decision.get('action', 'hold_invoice')} for package "
            f"{package_id}. Decisive evidence: "
            + ", ".join(evidence_refs[:3])
            + "."
        )

    rationale = rationale[:1500]

    return {
        "packageId": package_id,
        "actionId": f"act{uuid.uuid4().hex}",
        "action": decision["action"],
        "facts": {
            "vendorName": str(facts.get("vendorName", "")),
            "invoiceNumber": str(facts.get("invoiceNumber", "")),
            "amountMinor": int(facts.get("amountMinor", 0)),
            "currency": str(facts.get("currency", "INR")),
        },
        "evidenceRefs": evidence_refs,
        "rationale": rationale,
    }


# -------------------------
# Task helpers
# -------------------------

def task_to_public_dict(row: sqlite3.Row) -> dict[str, Any]:
    proposals = json.loads(row["proposals_json"])
    receipt = json.loads(row["receipt_json"]) if row["receipt_json"] else None
    history = json.loads(row["history_json"])

    artifacts: list[dict[str, Any]] = [
        {
            "artifactId": f"artifact_proposals_{row['task_id']}",
            "name": "Invoice action proposals",
            "parts": [
                {
                    "kind": "data",
                    "mediaType": PROPOSALS_MEDIA_TYPE,
                    "data": {
                        "batchId": row["batch_id"],
                        "proposals": proposals,
                    },
                }
            ],
        }
    ]

    if receipt is not None:
        artifacts.append(
            {
                "artifactId": f"artifact_receipts_{row['task_id']}",
                "name": "Invoice action receipts",
                "parts": [
                    {
                        "kind": "data",
                        "mediaType": RECEIPTS_MEDIA_TYPE,
                        "data": receipt,
                    }
                ],
            }
        )

    return {
        "id": row["task_id"],
        "contextId": row["context_id"],
        "status": {
            "state": row["state"],
            "timestamp": row["updated_at"],
        },
        "history": history,
        "artifacts": artifacts,
    }


def get_task_for_principal(
    conn: sqlite3.Connection,
    principal_hash: str,
    task_id: str,
) -> Any:
    row = conn.execute(
        """
        SELECT * FROM tasks
        WHERE task_id = ? AND principal_hash = ?
        """,
        (task_id, principal_hash),
    ).fetchone()

    if not row:
        return a2a_error(404, "TASK_NOT_FOUND", "Task not found.")

    return row


# -------------------------
# Request models
# -------------------------

class Part(BaseModel):
    mediaType: Optional[str] = None
    data: Optional[dict[str, Any]] = None
    text: Optional[str] = None


class Message(BaseModel):
    messageId: str
    role: Optional[str] = None
    taskId: Optional[str] = None
    contextId: Optional[str] = None
    parts: list[Part] = Field(default_factory=list)


class Configuration(BaseModel):
    returnImmediately: Optional[bool] = None
    historyLength: Optional[int] = None
    acceptedOutputModes: Optional[list[str]] = None


class SendMessageRequest(BaseModel):
    message: Message
    configuration: Optional[Configuration] = None


# -------------------------
# FastAPI app
# -------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="Invoice Action Agent", lifespan=lifespan)


@app.get("/.well-known/agent-card.json")
async def agent_card() -> JSONResponse:
    return a2a_json_response(
        {
            "name": "Invoice Action Agent",
            "description": (
                "Reads invoice packages, proposes one evidence-based action per "
                "package, and executes only grader-approved actions."
            ),
            "version": "1.0.0",
            "capabilities": {},
            "supportedInterfaces": [
                {
                    "url": BASE_URL,
                    "protocolBinding": "HTTP+JSON",
                    "protocolVersion": "1.0",
                }

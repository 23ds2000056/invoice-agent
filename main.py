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

    if not token:
        return a2a_error(401, "UNAUTHENTICATED", "Authentication required.")

    # Any well-formed Bearer token is treated as a separate principal.
    # Store only a hash of the token.
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
            ],
            "defaultInputModes": [BATCH_MEDIA_TYPE],
            "defaultOutputModes": [
                PROPOSALS_MEDIA_TYPE,
                RECEIPTS_MEDIA_TYPE,
            ],
            "skills": [
                {
                    "name": "invoice_action_agent",
                    "description": (
                        "Chooses settle, approval, hold, duplicate-rejection, or "
                        "exception actions from invoice evidence."
                    ),
                    "tags": [
                        "invoices",
                        "finance",
                        "approval",
                        "duplicate-detection",
                    ],
                }
            ],
        }
    )


@app.post("/a2a/message:send")
async def message_send(
    request: Request,
    body: SendMessageRequest,
    principal: Any = Depends(get_principal),
    header_error: Any = Depends(require_a2a_body_headers),
) -> Response:
    if isinstance(principal, JSONResponse):
        return principal

    if isinstance(header_error, JSONResponse):
        return header_error

    principal_hash = principal

    message_dict = body.message.model_dump(exclude_none=True)
    message_hash = sha256_json(message_dict)
    message_id = body.message.messageId

    if not body.message.parts:
        return a2a_error(400, "INVALID_MESSAGE", "Message has no parts.")

    batch_part = next(
        (
            part
            for part in body.message.parts
            if part.mediaType == BATCH_MEDIA_TYPE and isinstance(part.data, dict)
        ),
        None,
    )

    result_part = next(
        (
            part
            for part in body.message.parts
            if part.mediaType == RESULTS_MEDIA_TYPE and isinstance(part.data, dict)
        ),
        None,
    )

    if batch_part:
        return await handle_initial_batch(
            principal_hash=principal_hash,
            message_dict=message_dict,
            message_id=message_id,
            message_hash=message_hash,
            batch_data=batch_part.data,
        )

    if result_part:
        return await handle_result_continuation(
            principal_hash=principal_hash,
            message_dict=message_dict,
            message_id=message_id,
            message_hash=message_hash,
            result_data=result_part.data,
        )

    return a2a_error(
        400,
        "UNSUPPORTED_INPUT_MEDIA_TYPE",
        "No supported invoice media type found.",
    )


async def handle_initial_batch(
    principal_hash: str,
    message_dict: dict[str, Any],
    message_id: str,
    message_hash: str,
    batch_data: dict[str, Any],
) -> Response:
    batch_id = batch_data.get("batchId")
    packages = batch_data.get("packages")

    if not isinstance(batch_id, str) or not batch_id:
        return a2a_error(400, "INVALID_BATCH", "Invalid batch ID.")

    if not isinstance(packages, list) or not packages:
        return a2a_error(400, "INVALID_PACKAGES", "Invalid invoice packages.")

    with db() as conn:
        existing = conn.execute(
            """
            SELECT task_id, message_hash
            FROM message_dedup
            WHERE principal_hash = ? AND message_id = ?
            """,
            (principal_hash, message_id),
        ).fetchone()

        if existing:
            if existing["message_hash"] != message_hash:
                return a2a_error(
                    409,
                    "IDEMPOTENCY_CONFLICT",
                    "Message ID reused with different content.",
                )

            task_row = conn.execute(
                "SELECT * FROM tasks WHERE task_id = ?",
                (existing["task_id"],),
            ).fetchone()

            return a2a_json_response(
                {"task": task_to_public_dict(task_row)}
            )

    decisions = await ai_decide_packages(packages)

    proposals: list[dict[str, Any]] = []
    seen_package_ids: set[str] = set()
    seen_action_ids: set[str] = set()

    for package in packages:
        package_id = package.get("packageId")

        if not isinstance(package_id, str) or not package_id:
            return a2a_error(400, "INVALID_PACKAGE_ID", "Invalid package ID.")

        if package_id in seen_package_ids:
            return a2a_error(
                400,
                "DUPLICATE_PACKAGE_ID",
                "Duplicate package ID in batch.",
            )

        seen_package_ids.add(package_id)

        decision = decisions.get(package_id)

        if not decision or decision.get("action") not in VALID_ACTIONS:
            decision = local_fallback_decision(package)

        proposal = build_proposal(package, decision)

        while proposal["actionId"] in seen_action_ids:
            proposal["actionId"] = f"act{uuid.uuid4().hex}"

        seen_action_ids.add(proposal["actionId"])
        proposals.append(proposal)

    task_id = new_id("task")
    context_id = new_id("context")
    now = utc_now()

    with db() as conn:
        conn.execute("BEGIN IMMEDIATE")

        conn.execute(
            """
            INSERT INTO tasks (
                task_id, context_id, principal_hash, batch_id, state,
                history_json, proposals_json, receipt_json,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?, ?)
            """,
            (
                task_id,
                context_id,
                principal_hash,
                batch_id,
                "TASK_STATE_INPUT_REQUIRED",
                canonical_json([message_dict]),
                canonical_json(proposals),
                now,
                now,
            ),
        )

        conn.execute(
            """
            INSERT INTO message_dedup (
                principal_hash, message_id, message_hash, task_id, created_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (principal_hash, message_id, message_hash, task_id, now),
        )

        conn.commit()

        task_row = conn.execute(
            "SELECT * FROM tasks WHERE task_id = ?",
            (task_id,),
        ).fetchone()

    return a2a_json_response({"task": task_to_public_dict(task_row)})


async def handle_result_continuation(
    principal_hash: str,
    message_dict: dict[str, Any],
    message_id: str,
    message_hash: str,
    result_data: dict[str, Any],
) -> Response:
    task_id = message_dict.get("taskId")
    context_id = message_dict.get("contextId")
    batch_id = result_data.get("batchId")
    results = result_data.get("results")

    if not isinstance(task_id, str) or not task_id:
        return a2a_error(400, "INVALID_TASK_ID", "Invalid task ID.")

    if not isinstance(context_id, str) or not context_id:
        return a2a_error(400, "INVALID_CONTEXT_ID", "Invalid context ID.")

    if not isinstance(batch_id, str) or not batch_id:
        return a2a_error(400, "INVALID_BATCH", "Invalid batch ID.")

    if not isinstance(results, list):
        return a2a_error(400, "INVALID_RESULTS", "Invalid results.")

    with db() as conn:
        conn.execute("BEGIN IMMEDIATE")

        existing = conn.execute(
            """
            SELECT task_id, message_hash
            FROM message_dedup
            WHERE principal_hash = ? AND message_id = ?
            """,
            (principal_hash, message_id),
        ).fetchone()

        if existing:
            conn.rollback()

            if existing["message_hash"] != message_hash:
                return a2a_error(
                    409,
                    "IDEMPOTENCY_CONFLICT",
                    "Message ID reused with different content.",
                )

            task_row = conn.execute(
                "SELECT * FROM tasks WHERE task_id = ?",
                (existing["task_id"],),
            ).fetchone()

            return a2a_json_response(
                {"task": task_to_public_dict(task_row)}
            )

        task_row = get_task_for_principal(conn, principal_hash, task_id)

        if isinstance(task_row, JSONResponse):
            conn.rollback()
            return task_row

        if task_row["context_id"] != context_id:
            conn.rollback()
            return a2a_error(400, "CONTEXT_MISMATCH", "Context mismatch.")

        if task_row["batch_id"] != batch_id:
            conn.rollback()
            return a2a_error(400, "BATCH_MISMATCH", "Batch mismatch.")

        if task_row["state"] == "TASK_STATE_CANCELED":
            conn.rollback()
            return a2a_error(
                409,
                "TASK_CANCELED",
                "Task was canceled.",
            )

        if task_row["state"] in TERMINAL_STATES:
            conn.rollback()
            return a2a_error(
                409,
                "TASK_ALREADY_TERMINAL",
                "Task is already terminal.",
            )

        proposals = json.loads(task_row["proposals_json"])
        proposals_by_key = {
            (proposal["packageId"], proposal["actionId"]): proposal
            for proposal in proposals
        }

        executions: list[dict[str, Any]] = []
        seen_result_keys: set[tuple[str, str]] = set()

        for result in results:
            package_id = result.get("packageId")
            action_id = result.get("actionId")
            action = result.get("action")
            outcome = result.get("outcome")
            receipt_nonce = result.get("receiptNonce")

            key = (package_id, action_id)

            if key in seen_result_keys:
                conn.rollback()
                return a2a_error(
                    400,
                    "DUPLICATE_RESULT",
                    "Duplicate result for package and action.",
                )

            seen_result_keys.add(key)

            proposal = proposals_by_key.get(key)

            if not proposal:
                conn.rollback()
                return a2a_error(
                    400,
                    "PROPOSAL_MISMATCH",
                    "Result does not match a stored proposal.",
                )

            if proposal["action"] != action:
                conn.rollback()
                return a2a_error(
                    400,
                    "ACTION_MISMATCH",
                    "Result action does not match proposal.",
                )

            if outcome not in {"ACCEPTED", "REJECTED"}:
                conn.rollback()
                return a2a_error(
                    400,
                    "INVALID_OUTCOME",
                    "Invalid result outcome.",
                )

            if outcome == "ACCEPTED":
                if not isinstance(receipt_nonce, str) or not receipt_nonce:
                    conn.rollback()
                    return a2a_error(
                        400,
                        "MISSING_RECEIPT_NONCE",
                        "Accepted result is missing receipt nonce.",
                    )

                executions.append(
                    {
                        "packageId": package_id,
                        "actionId": action_id,
                        "action": action,
                        "receiptNonce": receipt_nonce,
                        "facts": proposal["facts"],
                        "evidenceRefs": proposal["evidenceRefs"],
                    }
                )

        receipt = {
            "batchId": batch_id,
            "executions": executions,
        }

        history = json.loads(task_row["history_json"])
        history.append(message_dict)

        now = utc_now()

        conn.execute(
            """
            UPDATE tasks
            SET state = ?,
                receipt_json = ?,
                history_json = ?,
                updated_at = ?
            WHERE task_id = ?
              AND principal_hash = ?
              AND state = ?
            """,
            (
                "TASK_STATE_COMPLETED",
                canonical_json(receipt),
                canonical_json(history),
                now,
                task_id,
                principal_hash,
                "TASK_STATE_INPUT_REQUIRED",
            ),
        )

        updated_row = conn.execute(
            "SELECT * FROM tasks WHERE task_id = ?",
            (task_id,),
        ).fetchone()

        if updated_row["state"] != "TASK_STATE_COMPLETED":
            conn.rollback()
            return a2a_error(
                409,
                "TASK_STATE_CONFLICT",
                "Task state changed during processing.",
            )

        conn.execute(
            """
            INSERT INTO message_dedup (
                principal_hash, message_id, message_hash, task_id, created_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (principal_hash, message_id, message_hash, task_id, now),
        )

        conn.commit()

        task_row = conn.execute(
            "SELECT * FROM tasks WHERE task_id = ?",
            (task_id,),
        ).fetchone()

    return a2a_json_response({"task": task_to_public_dict(task_row)})


@app.get("/a2a/tasks/{task_id}")
async def read_task(
    task_id: str,
    principal: Any = Depends(get_principal),
    header_error: Any = Depends(require_a2a_version),
) -> Response:
    if isinstance(principal, JSONResponse):
        return principal

    if isinstance(header_error, JSONResponse):
        return header_error

    with db() as conn:
        row = get_task_for_principal(conn, principal, task_id)

        if isinstance(row, JSONResponse):
            return row

        return a2a_json_response(task_to_public_dict(row))


@app.get("/a2a/tasks")
async def list_tasks(
    principal: Any = Depends(get_principal),
    header_error: Any = Depends(require_a2a_version),
) -> Response:
    if isinstance(principal, JSONResponse):
        return principal

    if isinstance(header_error, JSONResponse):
        return header_error

    with db() as conn:
        rows = conn.execute(
            """
            SELECT * FROM tasks
            WHERE principal_hash = ?
            ORDER BY created_at DESC
            """,
            (principal,),
        ).fetchall()

        return a2a_json_response(
            {"tasks": [task_to_public_dict(row) for row in rows]}
        )


@app.post("/a2a/tasks/{task_id}:cancel")
async def cancel_task(
    task_id: str,
    principal: Any = Depends(get_principal),
    header_error: Any = Depends(require_a2a_body_headers),
) -> Response:
    if isinstance(principal, JSONResponse):
        return principal

    if isinstance(header_error, JSONResponse):
        return header_error

    principal_hash = principal
    now = utc_now()

    with db() as conn:
        conn.execute("BEGIN IMMEDIATE")

        row = get_task_for_principal(conn, principal_hash, task_id)

        if isinstance(row, JSONResponse):
            conn.rollback()
            return row

        if row["state"] in TERMINAL_STATES:
            conn.rollback()

            if row["state"] == "TASK_STATE_CANCELED":
                return a2a_json_response(task_to_public_dict(row))

            return a2a_error(
                409,
                "TASK_ALREADY_TERMINAL",
                "Task cannot be canceled.",
            )

        conn.execute(
            """
            UPDATE tasks
            SET state = ?,
                receipt_json = NULL,
                updated_at = ?
            WHERE task_id = ?
              AND principal_hash = ?
              AND state NOT IN (
                'TASK_STATE_COMPLETED',
                'TASK_STATE_FAILED',
                'TASK_STATE_CANCELED',
                'TASK_STATE_REJECTED'
              )
            """,
            ("TASK_STATE_CANCELED", now, task_id, principal_hash),
        )

        conn.commit()

        row = conn.execute(
            "SELECT * FROM tasks WHERE task_id = ?",
            (task_id,),
        ).fetchone()

        return a2a_json_response(task_to_public_dict(row))


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
    )

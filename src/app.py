"""IBHelm Chat Service - AI chat with Python sandbox and database access."""

import asyncio
import hashlib
import json
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from . import settings, storage
from .auth import get_current_user
from .database import get_pool, close_pool
from .llm import get_model_configs, invalidate_model_cache, resolve_model
from . import tasks

logger = logging.getLogger("ibhelm.chat")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.validate()
    logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL))
    logger.info("Chat Service starting on port %s", settings.PORT)

    agent_task = None
    if settings.AGENT_ENABLED:
        from .agent import orchestrator_loop
        pool = await get_pool()
        agent_task = asyncio.create_task(orchestrator_loop(pool))
        logger.info("Project activity agent started")

    yield

    if agent_task:
        agent_task.cancel()
        try:
            await agent_task
        except asyncio.CancelledError:
            pass
    await close_pool()
    await storage.close()
    logger.info("Chat Service stopped")


app = FastAPI(title="IBHelm Chat Service", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Health + Models
# =============================================================================

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/models")
async def list_models(user: dict = Depends(get_current_user)):
    """Return available models from app_settings.body->'chat_models'."""
    pool = await get_pool()
    invalidate_model_cache()
    configs = await get_model_configs(pool)
    return [m for m in configs if not m.get("hidden")]


# =============================================================================
# Sessions
# =============================================================================

class SessionCreate(BaseModel):
    title: Optional[str] = None

class SessionUpdate(BaseModel):
    title: str


@app.get("/sessions")
async def list_sessions(user: dict = Depends(get_current_user), q: Optional[str] = None):
    pool = await get_pool()
    user_id = user["sub"]
    if q and q.strip():
        rows = await pool.fetch(
            """SELECT DISTINCT s.id, s.title, s.created_at, s.updated_at
               FROM chat_sessions s
               LEFT JOIN chat_messages m ON m.session_id = s.id
               WHERE s.user_id = $1
                 AND (s.title ILIKE $2 OR m.content ILIKE $2)
               ORDER BY s.updated_at DESC""",
            user_id, f"%{q.strip()}%"
        )
    else:
        rows = await pool.fetch(
            """SELECT id, title, created_at, updated_at
               FROM chat_sessions WHERE user_id = $1
               ORDER BY updated_at DESC""",
            user_id
        )
    return [
        {"id": str(r["id"]), "title": r["title"],
         "created_at": r["created_at"].isoformat(), "updated_at": r["updated_at"].isoformat()}
        for r in rows
    ]


@app.post("/sessions")
async def create_session(body: SessionCreate, user: dict = Depends(get_current_user)):
    pool = await get_pool()
    user_id = user["sub"]
    row = await pool.fetchrow(
        """INSERT INTO chat_sessions (user_id, title)
           VALUES ($1, $2) RETURNING id, title, created_at, updated_at""",
        user_id, body.title
    )
    return {
        "id": str(row["id"]), "title": row["title"],
        "created_at": row["created_at"].isoformat(), "updated_at": row["updated_at"].isoformat()
    }


@app.patch("/sessions/{session_id}")
async def update_session(session_id: str, body: SessionUpdate, user: dict = Depends(get_current_user)):
    pool = await get_pool()
    user_id = user["sub"]
    row = await pool.fetchrow(
        """UPDATE chat_sessions SET title = $1, updated_at = NOW()
           WHERE id = $2 AND user_id = $3 RETURNING id""",
        body.title, session_id, user_id
    )
    if not row:
        raise HTTPException(404, "Session not found")
    return {"ok": True}


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str, user: dict = Depends(get_current_user)):
    pool = await get_pool()
    user_id = user["sub"]
    row = await pool.fetchrow(
        "DELETE FROM chat_sessions WHERE id = $1 AND user_id = $2 RETURNING id",
        session_id, user_id
    )
    if not row:
        raise HTTPException(404, "Session not found")
    return {"ok": True}


# =============================================================================
# Messages
# =============================================================================

@app.get("/sessions/{session_id}/messages")
async def get_messages(session_id: str, user: dict = Depends(get_current_user)):
    pool = await get_pool()
    user_id = user["sub"]
    session = await pool.fetchrow(
        "SELECT id FROM chat_sessions WHERE id = $1 AND user_id = $2",
        session_id, user_id
    )
    if not session:
        raise HTTPException(404, "Session not found")

    rows = await pool.fetch(
        """SELECT id, role, content, blocks, metadata, status, created_at
           FROM chat_messages WHERE session_id = $1
           ORDER BY created_at ASC""",
        session_id
    )

    file_rows = await pool.fetch("""
        SELECT cf.id, cf.message_id, cf.filename, cf.content_hash, cf.bucket,
               cf.origin, cf.size_bytes, cf.mime_type
        FROM chat_files cf
        JOIN chat_messages m ON cf.message_id = m.id
        WHERE m.session_id = $1
    """, session_id)
    files_by_msg: dict[str, list] = {}
    for fr in file_rows:
        mid = str(fr["message_id"])
        files_by_msg.setdefault(mid, []).append({
            "id": str(fr["id"]), "filename": fr["filename"],
            "content_hash": fr["content_hash"], "bucket": fr["bucket"],
            "origin": fr["origin"], "size_bytes": fr["size_bytes"],
            "mime_type": fr["mime_type"],
        })

    return [
        {"id": str(r["id"]), "role": r["role"], "content": r["content"],
         "blocks": json.loads(r["blocks"]) if r["blocks"] else None,
         "metadata": json.loads(r["metadata"]) if r["metadata"] else None,
         "status": r["status"],
         "created_at": r["created_at"].isoformat(),
         "files": files_by_msg.get(str(r["id"]), [])}
        for r in rows
    ]


class MessageUpdate(BaseModel):
    content: str


@app.delete("/sessions/{session_id}/messages/from/{message_id}")
async def delete_messages_from(session_id: str, message_id: str, user: dict = Depends(get_current_user)):
    """Delete a message and all subsequent messages in the session."""
    pool = await get_pool()
    user_id = user["sub"]
    session = await pool.fetchrow(
        "SELECT id FROM chat_sessions WHERE id = $1 AND user_id = $2",
        session_id, user_id
    )
    if not session:
        raise HTTPException(404, "Session not found")

    msg = await pool.fetchrow(
        "SELECT created_at FROM chat_messages WHERE id = $1 AND session_id = $2",
        message_id, session_id
    )
    if not msg:
        raise HTTPException(404, "Message not found")

    await pool.execute(
        "DELETE FROM chat_messages WHERE session_id = $1 AND created_at >= $2",
        session_id, msg["created_at"]
    )
    return {"ok": True}


@app.patch("/sessions/{session_id}/messages/{message_id}")
async def update_message(session_id: str, message_id: str, body: MessageUpdate, user: dict = Depends(get_current_user)):
    """Update a user message's content."""
    pool = await get_pool()
    user_id = user["sub"]
    session = await pool.fetchrow(
        "SELECT id FROM chat_sessions WHERE id = $1 AND user_id = $2",
        session_id, user_id
    )
    if not session:
        raise HTTPException(404, "Session not found")

    row = await pool.fetchrow(
        "UPDATE chat_messages SET content = $1 WHERE id = $2 AND session_id = $3 AND role = 'user' RETURNING id",
        body.content, message_id, session_id
    )
    if not row:
        raise HTTPException(404, "User message not found")
    return {"ok": True}


_INCOMPLETE_NOTE = "[This response was interrupted — it may have been canceled by the user or ended due to an error. The content above may be incomplete.]"


async def _load_history(pool, session_id: str) -> list[dict]:
    """Load conversation history for LLM context. Includes incomplete messages with annotation."""
    rows = await pool.fetch(
        """SELECT m.id, m.role, m.content, m.blocks, m.status, m.created_at
           FROM chat_messages m
           WHERE m.session_id = $1
           ORDER BY m.created_at ASC""",
        session_id
    )
    file_rows = await pool.fetch("""
        SELECT cf.message_id, cf.id, cf.filename, cf.content_hash, cf.size_bytes, cf.mime_type
        FROM chat_files cf
        JOIN chat_messages m ON cf.message_id = m.id
        WHERE m.session_id = $1
    """, session_id)
    files_by_msg: dict[str, list] = {}
    for fr in file_rows:
        mid = str(fr["message_id"])
        files_by_msg.setdefault(mid, []).append({
            "id": str(fr["id"]), "filename": fr["filename"],
            "content_hash": fr["content_hash"], "size_bytes": fr["size_bytes"],
            "mime_type": fr["mime_type"],
        })
    history = []
    for r in rows:
        blocks = json.loads(r["blocks"]) if r["blocks"] else None
        content = r["content"]
        if r["role"] == "assistant" and r["status"] in ("error", "generating", "canceled"):
            if blocks:
                blocks.append({"type": "text", "text": _INCOMPLETE_NOTE})
            content = ((content or "") + "\n\n" + _INCOMPLETE_NOTE).strip()
        history.append({
            "role": r["role"], "content": content, "blocks": blocks,
            "created_at": r["created_at"].isoformat(),
            "files": files_by_msg.get(str(r["id"]), []),
        })
    return history


async def _cancel_active(session_id: str):
    """Cancel any active generation for this session."""
    gen = tasks.get_active(session_id)
    if gen and gen.task and not gen.task.done():
        gen.task.cancel()
        await asyncio.sleep(0.1)


async def _start_generation(
    pool, session_id: str, user_email: str | None, user_id: str,
    user_content: str, session_title: str | None, model_id: str | None,
) -> tuple[str, tasks.GenerationTask]:
    """Shared generation setup. Call AFTER user message is ready in DB.

    1. Cancel any active generation
    2. Clean stale 'generating' messages
    3. Load history (ends with user message)
    4. Resolve model early
    5. Insert assistant message with model in metadata
    6. Start generation task
    """
    await _cancel_active(session_id)

    await pool.execute(
        "UPDATE chat_messages SET status = 'error' WHERE session_id = $1 AND status = 'generating'",
        session_id,
    )

    history = await _load_history(pool, session_id)

    model_config = await resolve_model(model_id, pool)

    asst_msg_row = await pool.fetchrow(
        """INSERT INTO chat_messages (session_id, role, status, metadata)
           VALUES ($1, 'assistant', 'generating', $2) RETURNING id""",
        session_id, json.dumps({"model": model_config["id"]}),
    )
    asst_msg_id = str(asst_msg_row["id"])

    gen = tasks.GenerationTask(session_id=session_id, assistant_msg_id=asst_msg_id)
    gen.task = asyncio.create_task(tasks.run_generation(
        gen, history, pool, user_email, model_config,
        user_content, session_title, user_id=user_id,
    ))
    tasks.register(session_id, gen)

    return asst_msg_id, gen


def _sse_response(*initial_events: dict, gen: tasks.GenerationTask) -> StreamingResponse:
    """Create an SSE StreamingResponse from initial events + generation tail."""
    async def stream():
        for evt in initial_events:
            yield f"data: {json.dumps(evt, ensure_ascii=False)}\n\n"
        async for event in gen.tail():
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
    return StreamingResponse(
        stream(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/sessions/{session_id}/messages")
async def send_message(
    session_id: str,
    user: dict = Depends(get_current_user),
    content: str = Form(...),
    model: Optional[str] = Form(None),
    files: list[UploadFile] = File(default=[]),
):
    """Send a message (with optional files) and stream the AI response via SSE."""
    pool = await get_pool()
    user_id = user["sub"]
    user_email = user.get("email")

    session = await pool.fetchrow(
        "SELECT id, title FROM chat_sessions WHERE id = $1 AND user_id = $2",
        session_id, user_id
    )
    if not session:
        raise HTTPException(404, "Session not found")

    user_msg_row = await pool.fetchrow(
        "INSERT INTO chat_messages (session_id, role, content) VALUES ($1, 'user', $2) RETURNING id",
        session_id, content
    )
    user_msg_id = str(user_msg_row["id"])

    uploaded_files = []
    for f in files:
        data = await f.read()
        file_hash = hashlib.sha256(data).hexdigest()
        filename = f.filename or "unnamed"
        mime = f.content_type or "application/octet-stream"

        existing = await pool.fetchrow(
            "SELECT content_hash FROM file_contents WHERE content_hash = $1", file_hash
        )
        if existing:
            bucket, origin = "files", "reference"
        else:
            bucket, origin = settings.CHAT_FILES_BUCKET, "upload"
            ok = await storage.upload_file(bucket, file_hash, data, mime)
            if not ok:
                logger.warning("Failed to upload file %s", filename)
                continue

        row = await pool.fetchrow("""
            INSERT INTO chat_files (message_id, filename, content_hash, bucket, origin, size_bytes, mime_type)
            VALUES ($1, $2, $3, $4, $5, $6, $7) RETURNING id
        """, user_msg_id, filename, file_hash, bucket, origin, len(data), mime)
        uploaded_files.append({
            "id": str(row["id"]), "filename": filename, "content_hash": file_hash,
            "bucket": bucket, "origin": origin, "size_bytes": len(data), "mime_type": mime,
        })

    asst_msg_id, gen = await _start_generation(
        pool, session_id, user_email, str(user_id),
        content, session["title"], model,
    )

    initial = [{"type": "user_message_id", "id": user_msg_id}]
    if uploaded_files:
        initial.append({"type": "files_uploaded", "files": uploaded_files})
    initial.append({"type": "assistant_message_id", "id": asst_msg_id})
    return _sse_response(*initial, gen=gen)


class RegenerateRequest(BaseModel):
    message_id: str
    model: Optional[str] = None


@app.post("/sessions/{session_id}/regenerate")
async def regenerate(session_id: str, body: RegenerateRequest, user: dict = Depends(get_current_user)):
    """Regenerate an assistant response. Keeps the user message + files intact."""
    pool = await get_pool()
    user_id = user["sub"]
    user_email = user.get("email")

    session = await pool.fetchrow(
        "SELECT id, title FROM chat_sessions WHERE id = $1 AND user_id = $2",
        session_id, user_id
    )
    if not session:
        raise HTTPException(404, "Session not found")

    msg = await pool.fetchrow(
        "SELECT id, role, content, created_at FROM chat_messages WHERE id = $1 AND session_id = $2",
        body.message_id, session_id
    )
    if not msg:
        raise HTTPException(404, "Message not found")

    if msg["role"] == "assistant":
        user_msg = await pool.fetchrow(
            """SELECT id, content, created_at FROM chat_messages
               WHERE session_id = $1 AND role = 'user' AND created_at < $2
               ORDER BY created_at DESC LIMIT 1""",
            session_id, msg["created_at"]
        )
        if not user_msg:
            raise HTTPException(400, "No user message found before this assistant message")
    else:
        user_msg = msg

    await pool.execute(
        "DELETE FROM chat_messages WHERE session_id = $1 AND created_at > $2",
        session_id, user_msg["created_at"]
    )

    asst_msg_id, gen = await _start_generation(
        pool, session_id, user_email, str(user_id),
        user_msg["content"], session["title"], body.model,
    )
    return _sse_response({"type": "assistant_message_id", "id": asst_msg_id}, gen=gen)


@app.get("/sessions/{session_id}/stream")
async def stream_session(session_id: str, user: dict = Depends(get_current_user)):
    """Reconnect to an active generation's SSE stream."""
    pool = await get_pool()
    user_id = user["sub"]
    session = await pool.fetchrow(
        "SELECT id FROM chat_sessions WHERE id = $1 AND user_id = $2",
        session_id, user_id
    )
    if not session:
        raise HTTPException(404, "Session not found")

    gen = tasks.get_active(session_id)
    if not gen:
        return {"status": "no_active_generation"}

    async def sse_stream():
        async for event in gen.tail():
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        sse_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


@app.post("/sessions/{session_id}/cancel")
async def cancel_generation(session_id: str, user: dict = Depends(get_current_user)):
    """Cancel an active generation for a session."""
    pool = await get_pool()
    user_id = user["sub"]
    session = await pool.fetchrow(
        "SELECT id FROM chat_sessions WHERE id = $1 AND user_id = $2",
        session_id, user_id
    )
    if not session:
        raise HTTPException(404, "Session not found")

    gen = tasks.get_active(session_id)
    if not gen:
        return {"ok": True, "was_active": False}

    if gen.task and not gen.task.done():
        gen.task.cancel()
    return {"ok": True, "was_active": True}


# =============================================================================
# Files
# =============================================================================


@app.post("/sessions/{session_id}/files")
async def upload_file(
    session_id: str,
    file: UploadFile = File(...),
    message_id: str = Form(...),
    content_hash: str = Form(...),
    user: dict = Depends(get_current_user),
):
    """Upload a file to a chat message. Deduplicates against existing file_contents."""
    pool = await get_pool()
    user_id = user["sub"]
    session = await pool.fetchrow(
        "SELECT id FROM chat_sessions WHERE id = $1 AND user_id = $2",
        session_id, user_id,
    )
    if not session:
        raise HTTPException(404, "Session not found")

    data = await file.read()
    computed_hash = hashlib.sha256(data).hexdigest()
    if computed_hash != content_hash:
        raise HTTPException(400, "Content hash mismatch")

    filename = file.filename or "unnamed"
    mime = file.content_type or "application/octet-stream"
    size_bytes = len(data)

    # Check if file already exists in file_contents (NAS/TTE processed)
    existing = await pool.fetchrow(
        "SELECT content_hash FROM file_contents WHERE content_hash = $1", content_hash
    )

    if existing:
        bucket = "files"
        origin = "reference"
    else:
        bucket = settings.CHAT_FILES_BUCKET
        origin = "upload"
        ok = await storage.upload_file(bucket, content_hash, data, mime)
        if not ok:
            raise HTTPException(502, "Failed to upload file to storage")

    row = await pool.fetchrow("""
        INSERT INTO chat_files (message_id, filename, content_hash, bucket, origin, size_bytes, mime_type)
        VALUES ($1, $2, $3, $4, $5, $6, $7)
        RETURNING id
    """, message_id, filename, content_hash, bucket, origin, size_bytes, mime)

    return {
        "id": str(row["id"]),
        "filename": filename,
        "content_hash": content_hash,
        "bucket": bucket,
        "origin": origin,
        "size_bytes": size_bytes,
        "mime_type": mime,
    }


@app.get("/sessions/{session_id}/files")
async def list_files(session_id: str, user: dict = Depends(get_current_user)):
    """List all files in a chat session."""
    pool = await get_pool()
    user_id = user["sub"]
    session = await pool.fetchrow(
        "SELECT id FROM chat_sessions WHERE id = $1 AND user_id = $2",
        session_id, user_id,
    )
    if not session:
        raise HTTPException(404, "Session not found")

    rows = await pool.fetch("""
        SELECT cf.id, cf.message_id, cf.filename, cf.content_hash, cf.bucket,
               cf.origin, cf.size_bytes, cf.mime_type, cf.created_at
        FROM chat_files cf
        JOIN chat_messages m ON cf.message_id = m.id
        WHERE m.session_id = $1
        ORDER BY cf.created_at
    """, session_id)

    return [
        {"id": str(r["id"]), "message_id": str(r["message_id"]),
         "filename": r["filename"], "content_hash": r["content_hash"],
         "bucket": r["bucket"], "origin": r["origin"],
         "size_bytes": r["size_bytes"], "mime_type": r["mime_type"],
         "created_at": r["created_at"].isoformat()}
        for r in rows
    ]


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.HOST, port=settings.PORT)

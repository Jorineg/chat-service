"""IBHelm Chat Service - AI chat with Python sandbox and database access."""

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from . import settings
from .auth import get_current_user
from .database import get_pool, close_pool
from .llm import get_model_configs, invalidate_model_cache
from . import tasks

logger = logging.getLogger("ibhelm.chat")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.validate()
    logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL))
    logger.info("Chat Service starting on port %s", settings.PORT)
    yield
    await close_pool()
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
    return [
        {"id": str(r["id"]), "role": r["role"], "content": r["content"],
         "blocks": json.loads(r["blocks"]) if r["blocks"] else None,
         "metadata": json.loads(r["metadata"]) if r["metadata"] else None,
         "status": r["status"],
         "created_at": r["created_at"].isoformat()}
        for r in rows
    ]


class MessageCreate(BaseModel):
    content: str
    model: Optional[str] = None

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


@app.post("/sessions/{session_id}/messages")
async def send_message(session_id: str, body: MessageCreate, user: dict = Depends(get_current_user)):
    """Send a message and stream the AI response via SSE."""
    pool = await get_pool()
    user_id = user["sub"]
    user_email = user.get("email")

    session = await pool.fetchrow(
        "SELECT id, title FROM chat_sessions WHERE id = $1 AND user_id = $2",
        session_id, user_id
    )
    if not session:
        raise HTTPException(404, "Session not found")

    if tasks.has_active(session_id):
        raise HTTPException(409, "Generation already in progress for this session")

    # Insert user message
    user_msg_row = await pool.fetchrow(
        "INSERT INTO chat_messages (session_id, role, content) VALUES ($1, 'user', $2) RETURNING id",
        session_id, body.content
    )
    user_msg_id = str(user_msg_row["id"])

    # Insert placeholder assistant message
    asst_msg_row = await pool.fetchrow(
        "INSERT INTO chat_messages (session_id, role, status) VALUES ($1, 'assistant', 'generating') RETURNING id",
        session_id
    )
    asst_msg_id = str(asst_msg_row["id"])

    # Load history (excluding the generating placeholder)
    history_rows = await pool.fetch(
        """SELECT role, content, blocks, created_at FROM chat_messages
           WHERE session_id = $1 AND status = 'complete'
           ORDER BY created_at ASC""",
        session_id
    )
    history = [
        {"role": r["role"], "content": r["content"],
         "blocks": json.loads(r["blocks"]) if r["blocks"] else None,
         "created_at": r["created_at"].isoformat()}
        for r in history_rows
    ]

    # Create and register the generation task
    gen = tasks.GenerationTask(session_id=session_id, assistant_msg_id=asst_msg_id)
    gen.task = asyncio.create_task(tasks.run_generation(
        gen, history, pool, user_email, body.model,
        body.content, session["title"],
    ))
    tasks.register(session_id, gen)

    # SSE stream tails the task's events
    async def sse_stream():
        yield f"data: {json.dumps({'type': 'user_message_id', 'id': user_msg_id})}\n\n"
        yield f"data: {json.dumps({'type': 'assistant_message_id', 'id': asst_msg_id})}\n\n"

        async for event in gen.tail():
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        sse_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


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


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.HOST, port=settings.PORT)

"""Generation task registry — decouples LLM generation from HTTP connection lifecycle.

Each active generation is an asyncio.Task that:
- Runs the tool loop independently of any SSE connection
- Pushes events to a list (supports multiple readers / reconnection)
- Writes to DB incrementally after each tool result
- Cleans up on completion or error
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field

import asyncpg

logger = logging.getLogger("ibhelm.chat.tasks")


@dataclass
class GenerationTask:
    session_id: str
    assistant_msg_id: str
    task: asyncio.Task | None = None
    events: list[dict] = field(default_factory=list)
    _notify: asyncio.Event = field(default_factory=asyncio.Event)
    finished: bool = False

    def push(self, event: dict):
        self.events.append(event)
        self._notify.set()

    def finish(self):
        self.finished = True
        self._notify.set()

    async def tail(self, start: int = 0):
        """Async generator yielding events from position `start`. Completes when task finishes."""
        pos = start
        while True:
            while pos < len(self.events):
                yield self.events[pos]
                pos += 1
            if self.finished:
                return
            self._notify.clear()
            try:
                await asyncio.wait_for(self._notify.wait(), timeout=1.0)
            except asyncio.TimeoutError:
                pass


# Session-level registry (one active generation per session)
_active: dict[str, GenerationTask] = {}


def get_active(session_id: str) -> GenerationTask | None:
    return _active.get(session_id)


def has_active(session_id: str) -> bool:
    return session_id in _active


def register(session_id: str, gen: GenerationTask):
    _active[session_id] = gen


def unregister(session_id: str):
    _active.pop(session_id, None)


async def run_generation(
    gen: GenerationTask,
    history: list[dict],
    pool: asyncpg.Pool,
    user_email: str | None,
    model_id: str | None,
    user_content: str,
    session_title: str | None,
):
    """Run LLM generation as an independent task. Pushes events + writes to DB incrementally."""
    from .llm import stream_chat_response, generate_title

    all_blocks: list[dict] = []
    full_text = ""
    final_metadata: dict | None = None

    # Fire title generation in parallel — pushes event as soon as ready
    if not session_title:
        async def _gen_title():
            try:
                title = await generate_title(user_content, pool)
                await pool.execute(
                    "UPDATE chat_sessions SET title = $1 WHERE id = $2",
                    title, gen.session_id,
                )
                gen.push({"type": "title", "title": title})
            except Exception as e:
                logger.warning("Title generation failed: %s", e)
        asyncio.create_task(_gen_title())

    try:
        async for event in stream_chat_response(history, pool, user_email, model_id):
            gen.push(event)
            etype = event["type"]

            if etype == "tool_result":
                tc_record = {"type": "tool_call", "id": event["id"], "code": ""}
                # Find code from previous tool_call event
                for prev in reversed(gen.events):
                    if prev.get("type") == "tool_call" and prev.get("id") == event["id"]:
                        tc_record["code"] = prev.get("code", "")
                        break
                if event.get("result") is not None:
                    tc_record["result"] = event["result"]
                if event.get("error") is not None:
                    tc_record["error"] = event["error"]
                all_blocks.append(tc_record)

                # Incremental DB write after each tool result
                await pool.execute(
                    "UPDATE chat_messages SET blocks = $1 WHERE id = $2",
                    json.dumps(all_blocks, ensure_ascii=False),
                    gen.assistant_msg_id,
                )

            elif etype == "text":
                full_text += event["content"]
                if not all_blocks or all_blocks[-1]["type"] != "text":
                    all_blocks.append({"type": "text", "text": event["content"]})
                else:
                    all_blocks[-1]["text"] += event["content"]

            elif etype == "thinking":
                if not all_blocks or all_blocks[-1]["type"] != "thinking":
                    all_blocks.append({"type": "thinking", "text": event["content"]})
                else:
                    all_blocks[-1]["text"] += event["content"]

            elif etype == "done":
                final_metadata = event.get("metadata")
                final_blocks = event.get("blocks") or all_blocks or None
                await pool.execute(
                    """UPDATE chat_messages
                       SET content = $1, blocks = $2, metadata = $3, status = 'complete'
                       WHERE id = $4""",
                    full_text,
                    json.dumps(final_blocks, ensure_ascii=False) if final_blocks else None,
                    json.dumps(final_metadata, ensure_ascii=False) if final_metadata else None,
                    gen.assistant_msg_id,
                )
                await pool.execute(
                    "UPDATE chat_sessions SET updated_at = NOW() WHERE id = $1",
                    gen.session_id,
                )

            elif etype == "error":
                # Save partial work on error
                await pool.execute(
                    """UPDATE chat_messages
                       SET content = $1, blocks = $2, status = 'error'
                       WHERE id = $3""",
                    full_text or None,
                    json.dumps(all_blocks, ensure_ascii=False) if all_blocks else None,
                    gen.assistant_msg_id,
                )

        # Handle max tool iterations (stream ends without done event)
        row = await pool.fetchrow(
            "SELECT status FROM chat_messages WHERE id = $1", gen.assistant_msg_id
        )
        if row and row["status"] == "generating":
            await pool.execute(
                """UPDATE chat_messages
                   SET content = $1, blocks = $2, status = 'error'
                   WHERE id = $3""",
                full_text or None,
                json.dumps(all_blocks, ensure_ascii=False) if all_blocks else None,
                gen.assistant_msg_id,
            )

    except Exception as e:
        logger.error("Generation task failed: %s", e, exc_info=True)
        gen.push({"type": "error", "message": str(e)})
        try:
            await pool.execute(
                """UPDATE chat_messages
                   SET content = $1, blocks = $2, status = 'error'
                   WHERE id = $3""",
                full_text or None,
                json.dumps(all_blocks, ensure_ascii=False) if all_blocks else None,
                gen.assistant_msg_id,
            )
        except Exception:
            pass
    finally:
        gen.finish()
        unregister(gen.session_id)
        logger.info("Generation task finished for session %s", gen.session_id)

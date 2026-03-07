"""Generation task registry — decouples LLM generation from HTTP connection lifecycle.

Each active generation is an asyncio.Task that:
- Runs the tool loop independently of any SSE connection
- Pushes events to a list (supports multiple readers / reconnection)
- Writes to DB incrementally after each tool result
- Manages sandbox container lifecycle (per response)
- Cleans up on completion or error
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field

import asyncpg

from .sandbox import SandboxSession, get_session_files

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
    model_config: dict,
    user_content: str,
    session_title: str | None,
    user_id: str | None = None,
    system_prompt: str | None = None,
):
    """Run LLM generation as an independent task. Pushes events + writes to DB incrementally."""
    from .llm import stream_chat_response, generate_title

    all_blocks: list[dict] = []
    full_text = ""
    active_model: str = model_config["id"]
    sandbox: SandboxSession | None = None

    async def _finalize(status: str, error: str | None = None, usage: dict | None = None):
        """Single exit point: push 'done' event + write same state to DB."""
        metadata = {"model": active_model, **(usage or {})}
        done_event: dict = {
            "type": "done", "status": status,
            "content": full_text, "blocks": all_blocks or None,
            "metadata": metadata,
        }
        if error:
            done_event["error"] = error
        gen.push(done_event)
        try:
            await pool.execute(
                """UPDATE chat_messages
                   SET content = $1, blocks = $2, metadata = $3, status = $4
                   WHERE id = $5""",
                full_text or None,
                json.dumps(all_blocks, ensure_ascii=False) if all_blocks else None,
                json.dumps(metadata, ensure_ascii=False),
                status,
                gen.assistant_msg_id,
            )
            if status == "complete":
                await pool.execute(
                    "UPDATE chat_sessions SET updated_at = NOW() WHERE id = $1",
                    gen.session_id,
                )
        except Exception as e:
            logger.error("Failed to write final state to DB: %s", e)

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
        sandbox = SandboxSession(pool, user_email, gen.session_id, user_id=user_id)
        conversation_files = await get_session_files(pool, gen.session_id)
        await sandbox.start(conversation_files)

        async for event in stream_chat_response(
            history, pool, user_email, model_config, sandbox, gen.assistant_msg_id,
            system_prompt=system_prompt,
        ):
            etype = event["type"]

            # done/error are handled exclusively by _finalize — don't push raw
            if etype == "done":
                usage = event.get("metadata") or {}
                usage.pop("model", None)
                await _finalize("complete", usage=usage)
                continue
            if etype == "error":
                await _finalize("error", error=event.get("message"))
                continue

            gen.push(event)

            if etype == "tool_result":
                tc_record = {"type": "tool_call", "id": event["id"], "code": ""}
                for prev in reversed(gen.events):
                    if prev.get("type") == "tool_call" and prev.get("id") == event["id"]:
                        tc_record["code"] = prev.get("code", "")
                        break
                if event.get("result") is not None:
                    tc_record["result"] = event["result"]
                if event.get("error") is not None:
                    tc_record["error"] = event["error"]
                all_blocks.append(tc_record)

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

        # Safety net: stream ended without done/error
        row = await pool.fetchrow(
            "SELECT status FROM chat_messages WHERE id = $1", gen.assistant_msg_id
        )
        if row and row["status"] == "generating":
            await _finalize("error", error="Generation ended unexpectedly")

    except asyncio.CancelledError:
        logger.info("Generation cancelled for session %s", gen.session_id)
        await _finalize("canceled")
    except Exception as e:
        logger.error("Generation task failed: %s", e, exc_info=True)
        await _finalize("error", error=str(e))
    finally:
        if sandbox:
            await sandbox.cleanup()
        gen.finish()
        unregister(gen.session_id)
        logger.info("Generation task finished for session %s", gen.session_id)

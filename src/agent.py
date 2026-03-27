"""Project Activity Agent — background processor for Tier 3/2/1 generation.

Runs as an asyncio task inside chat-service. Periodically:
1. Computes diffs for Tier 4 events with old_content
2. For each project with unprocessed events, runs an LLM agent turn
3. Processes bootstrap requests (full Tier 1+2 from scratch)
"""

import asyncio
import difflib
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone

import asyncpg

from . import settings
from . import template_resolver as tr
from .llm import stream_chat_response, resolve_model, get_app_setting
from .sandbox import SandboxSession

logger = logging.getLogger("ibhelm.chat.agent")

AGENT_USER_ID = "00000000-0000-0000-0000-000000000001"

_SOURCE_CONTENT_QUERIES = {
    "teamwork.tasks": "SELECT description FROM teamwork.tasks WHERE id = $1::int",
    "craft_documents": "SELECT markdown_content FROM craft_documents WHERE id = $1",
}


@dataclass
class AgentResult:
    session_id: str
    content: str
    usage: dict




# ---------------------------------------------------------------------------
# Diff Processor
# ---------------------------------------------------------------------------

async def process_pending_diffs(pool: asyncpg.Pool):
    """Compute unified diffs for events with old_content.

    Groups events by (source_table, source_id) and processes in chronological
    order.  For consecutive events on the same source, event N+1's old_content
    is exactly the state *after* event N's change, so we use it as new_content
    for event N instead of fetching the (potentially much newer) current state.
    Only the last event per source needs a live DB fetch.
    """
    events = await pool.fetch(
        "SELECT id, source_table, source_id, old_content "
        "FROM project_event_log "
        "WHERE old_content IS NOT NULL AND NOT processed_by_diff "
        "ORDER BY id"
    )
    if not events:
        return

    logger.info("Processing %d pending diffs", len(events))

    groups: dict[tuple[str, str], list] = {}
    for e in events:
        groups.setdefault((e["source_table"], e["source_id"]), []).append(e)

    for (source_table, source_id), chain in groups.items():
        for i, event in enumerate(chain):
            try:
                if i + 1 < len(chain):
                    new_content = chain[i + 1]["old_content"]
                else:
                    new_content = await _fetch_current_content(pool, source_table, source_id)
                diff = _compute_diff(event["old_content"], new_content)
                await pool.execute(
                    "UPDATE project_event_log "
                    "SET content_diff = $1, old_content = NULL, processed_by_diff = TRUE "
                    "WHERE id = $2",
                    diff, event["id"],
                )
            except Exception as e:
                logger.warning("Diff failed for event %d: %s", event["id"], e)
                await pool.execute(
                    "UPDATE project_event_log "
                    "SET content_diff = '(diff error)', old_content = NULL, processed_by_diff = TRUE "
                    "WHERE id = $1",
                    event["id"],
                )


async def _fetch_current_content(
    pool: asyncpg.Pool, source_table: str, source_id: str
) -> str | None:
    query = _SOURCE_CONTENT_QUERIES.get(source_table)
    if not query:
        return None
    return await pool.fetchval(query, source_id)


def _compute_diff(old_text: str | None, new_text: str | None) -> str:
    result = "".join(difflib.unified_diff(
        (old_text or "").splitlines(keepends=True),
        (new_text or "").splitlines(keepends=True),
        n=3,
    ))
    return result or "(no changes)"


# ---------------------------------------------------------------------------
# Agent Runner
# ---------------------------------------------------------------------------

async def _resolve_agent_model(pool: asyncpg.Pool) -> dict:
    agent_model_id = await get_app_setting(pool, "agent_model_id")
    return await resolve_model(agent_model_id, pool)


async def run_agent_turn(
    system_prompt: str,
    user_message: str,
    pool: asyncpg.Pool,
    model_config: dict | None = None,
    context_metadata: dict | None = None,
    nudge_check: Callable | None = None,
    title_prefix: str = "PAA",
) -> AgentResult:
    """Run agent to completion with optional nudge loop.

    nudge_check: async callable(pool) -> str|None. Called after each turn.
    If it returns a string, that string is sent as a follow-up user message
    and the agent continues with full conversation context. If None, done.
    """
    if not model_config:
        model_config = await _resolve_agent_model(pool)

    title = f"{title_prefix} {datetime.now(timezone.utc).strftime('%d.%m.%y %H:%M')}"
    session_row = await pool.fetchrow(
        "INSERT INTO chat_sessions (user_id, title, system_prompt) VALUES ($1, $2, $3) RETURNING id",
        AGENT_USER_ID, title, system_prompt,
    )
    session_id = str(session_row["id"])

    await pool.fetchrow(
        "INSERT INTO chat_messages (session_id, role, content) VALUES ($1, 'user', $2) RETURNING id",
        session_id, user_message,
    )

    asst_row = await pool.fetchrow(
        "INSERT INTO chat_messages (session_id, role, status, metadata) "
        "VALUES ($1, 'assistant', 'generating', $2) RETURNING id",
        session_id, json.dumps({"model": model_config["id"], **(context_metadata or {})}),
    )
    asst_msg_id = str(asst_row["id"])

    sandbox = SandboxSession(pool, None, session_id, model_id=model_config["id"],
                             agent_context=title_prefix.lower())
    await sandbox.start()
    try:
        messages = [{"role": "user", "content": user_message}]
        result_text, all_blocks, usage = "", [], {}

        async for event in stream_chat_response(
            messages, pool, model_config=model_config,
            sandbox=sandbox, assistant_msg_id=asst_msg_id,
            system_prompt=system_prompt,
        ):
            etype = event["type"]
            if etype == "text":
                result_text += event.get("content", "")
            elif etype == "done":
                result_text = event.get("content", result_text)
                all_blocks = event.get("blocks") or []
                usage = event.get("metadata") or {}
            elif etype == "error":
                await pool.execute(
                    "UPDATE chat_messages SET status='error', content=$1 WHERE id=$2",
                    event.get("message"), asst_msg_id,
                )
                raise RuntimeError(event.get("message"))

        # Save first turn
        await pool.execute(
            "UPDATE chat_messages SET content=$1, blocks=$2, metadata=$3, status='complete' WHERE id=$4",
            result_text or None,
            json.dumps(all_blocks) if all_blocks else None,
            json.dumps(usage),
            asst_msg_id,
        )

        # Nudge loop: keep checking and nudging until task is complete (max 3 nudges)
        if nudge_check:
            for nudge_attempt in range(1, 4):
                nudge_msg = await nudge_check(pool)
                if not nudge_msg:
                    break

                logger.info("Nudging agent (attempt %d/3) in session %s", nudge_attempt, session_id)
                messages.append({"role": "assistant", "content": result_text, "blocks": all_blocks})
                messages.append({"role": "user", "content": nudge_msg})

                await pool.execute(
                    "INSERT INTO chat_messages (session_id, role, content) VALUES ($1, 'user', $2)",
                    session_id, nudge_msg,
                )
                nudge_asst_row = await pool.fetchrow(
                    "INSERT INTO chat_messages (session_id, role, status, metadata) "
                    "VALUES ($1, 'assistant', 'generating', $2) RETURNING id",
                    session_id, json.dumps({"model": model_config["id"]}),
                )
                nudge_asst_id = str(nudge_asst_row["id"])

                result_text, all_blocks, usage = "", [], {}
                async for event in stream_chat_response(
                    messages, pool, model_config=model_config,
                    sandbox=sandbox, assistant_msg_id=nudge_asst_id,
                    system_prompt=system_prompt,
                ):
                    etype = event["type"]
                    if etype == "text":
                        result_text += event.get("content", "")
                    elif etype == "done":
                        result_text = event.get("content", result_text)
                        all_blocks = event.get("blocks") or []
                        usage = event.get("metadata") or {}
                    elif etype == "error":
                        await pool.execute(
                            "UPDATE chat_messages SET status='error', content=$1 WHERE id=$2",
                            event.get("message"), nudge_asst_id,
                        )
                        break

                await pool.execute(
                    "UPDATE chat_messages SET content=$1, blocks=$2, metadata=$3, status='complete' WHERE id=$4",
                    result_text or None,
                    json.dumps(all_blocks) if all_blocks else None,
                    json.dumps(usage),
                    nudge_asst_id,
                )

        return AgentResult(session_id=session_id, content=result_text, usage=usage)
    finally:
        await sandbox.cleanup()


# ---------------------------------------------------------------------------
# Orchestrator Loop
# ---------------------------------------------------------------------------

async def orchestrator_loop(pool: asyncpg.Pool):
    """Background loop: process diffs, run agent for events, handle bootstrap requests."""
    logger.info("Project activity agent started (interval=%ds)", settings.AGENT_CHECK_INTERVAL_S)
    while True:
        try:
            await process_pending_diffs(pool)
            await _process_events(pool)
            await _process_bootstrap_requests(pool)
        except asyncio.CancelledError:
            logger.info("Project activity agent cancelled")
            return
        except Exception as e:
            logger.error("Orchestrator loop error: %s", e, exc_info=True)

        await asyncio.sleep(settings.AGENT_CHECK_INTERVAL_S)


async def _process_events(pool: asyncpg.Pool):
    """Find projects with unprocessed events and run the agent for each."""
    projects = await pool.fetch("""
        SELECT DISTINCT pel.tw_project_id
        FROM project_event_log pel
        JOIN teamwork.projects tp ON tp.id = pel.tw_project_id
        WHERE NOT pel.processed_by_agent
          AND (pel.processed_by_diff OR pel.old_content IS NULL)
          AND tp.status = 'active'
    """)
    if not projects:
        return

    logger.info("Agent processing %d project(s) with new events", len(projects))
    for row in projects:
        project_id = row["tw_project_id"]
        try:
            await _run_event_agent(pool, project_id)
        except Exception as e:
            logger.error("Agent failed for project %s: %s", project_id, e, exc_info=True)


async def _run_event_agent(pool: asyncpg.Pool, project_id: int):
    """Build prompt with Tier 4 events + current Tier 1/2, run agent, mark events processed."""
    event_ids = [r["id"] for r in await pool.fetch(
        "SELECT id FROM project_event_log "
        "WHERE tw_project_id = $1 AND NOT processed_by_agent "
        "AND (processed_by_diff OR old_content IS NULL) ORDER BY occurred_at",
        project_id,
    )]
    if not event_ids:
        return

    vars_ = {"project_id": str(project_id), "event_ids": ",".join(str(i) for i in event_ids)}
    prompt, user_message = await asyncio.gather(
        tr.resolve(pool, "agent.event_prompt"),
        tr.resolve(pool, "agent.event_user_message", vars_),
    )

    logger.info("Running event agent for project %s — %d events", project_id, len(event_ids))
    result = await run_agent_turn(
        prompt, user_message, pool,
        context_metadata={"tw_project_id": project_id, "action": "events"},
    )

    await pool.execute(
        "UPDATE project_event_log SET processed_by_agent = TRUE WHERE id = ANY($1::bigint[])",
        event_ids,
    )
    logger.info("Agent completed for project %s — session %s, marked %d events processed",
                project_id, result.session_id, len(event_ids))


async def _process_bootstrap_requests(pool: asyncpg.Pool):
    """Process pending bootstrap requests."""
    requests = await pool.fetch("""
        SELECT id, tw_project_id
        FROM project_agent_requests
        WHERE status = 'pending' AND action = 'bootstrap'
        ORDER BY created_at
    """)
    for req in requests:
        request_id = req["id"]
        project_id = req["tw_project_id"]
        await pool.execute(
            "UPDATE project_agent_requests SET status = 'processing' WHERE id = $1",
            request_id,
        )
        try:
            await _run_bootstrap_agent(pool, project_id, request_id)
        except Exception as e:
            logger.error("Bootstrap failed for project %s: %s", project_id, e, exc_info=True)
            await pool.execute(
                "UPDATE project_agent_requests SET status = 'error', error_message = $1, processed_at = NOW() WHERE id = $2",
                str(e)[:500], request_id,
            )


async def _run_bootstrap_agent(pool: asyncpg.Pool, project_id: int, request_id: str):
    """Run the bootstrap agent to generate Tier 1+2 from scratch. Nudges if results not saved."""
    vars_ = {"project_id": str(project_id)}
    prompt, user_message = await asyncio.gather(
        tr.resolve(pool, "agent.bootstrap_prompt", vars_),
        tr.resolve(pool, "agent.bootstrap_user_message", vars_),
    )

    async def _bootstrap_nudge(p: asyncpg.Pool) -> str | None:
        row = await p.fetchrow(
            "SELECT profile_markdown IS NOT NULL as has_t1, status_markdown IS NOT NULL as has_t2 "
            "FROM project_extensions WHERE tw_project_id = $1", project_id
        )
        if not row:
            return None
        missing = []
        if not row["has_t1"]:
            missing.append("Tier 1 (call update_project_profile)")
        if not row["has_t2"]:
            missing.append("Tier 2 (call update_project_status)")
        if not missing:
            return None
        logger.info("Nudging bootstrap for project %s — missing: %s", project_id, ", ".join(missing))
        return (
            f"[system] You have not saved your results yet. Still missing: {', '.join(missing)}.\n\n"
            f"Continue your research if needed, then call the write functions to save "
            f"Tier 1 profile and Tier 2 status for project {project_id}."
        )

    logger.info("Running bootstrap agent for project %s", project_id)
    result = await run_agent_turn(
        prompt, user_message, pool,
        context_metadata={"tw_project_id": project_id, "action": "bootstrap"},
        nudge_check=_bootstrap_nudge,
        title_prefix="BST",
    )

    await pool.execute(
        "UPDATE project_agent_requests SET status = 'done', result_session_id = $1, processed_at = NOW() WHERE id = $2",
        result.session_id, request_id,
    )
    logger.info("Bootstrap completed for project %s — session %s", project_id, result.session_id)

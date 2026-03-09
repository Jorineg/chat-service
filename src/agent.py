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
from pathlib import Path

import asyncpg

from . import settings
from .llm import stream_chat_response, resolve_model
from .sandbox import SandboxSession

_DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"
_SCHEMA_DOC = (_DOCS_DIR / "schema_doc_output.md").read_text().strip() if (_DOCS_DIR / "schema_doc_output.md").exists() else ""

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
# Shared context block — included in both event and bootstrap prompts
# ---------------------------------------------------------------------------

_TIER_CONTEXT = """\
You are an automated background agent running without a human operator. \
No one reads your text output — only your tool calls matter. \
Do not narrate, explain, or ask questions. Just execute the task using `run_python` tool calls.

## Project Activity System — Overview

IBHelm manages 15+ active building engineering projects (TGA/HVAC) across clients. Project knowledge is scattered across Teamwork tasks, Missive emails, Craft docs, \
and NAS files. This system creates a unified, evolving project narrative through four tiers:

### Tier 4: Mechanical Event Log (automatic, no AI)
Raw facts captured by DB triggers: task created/changed, email linked, Craft doc edited, file added. \
One row per event with JSONB details. You receive these as input.

### Tier 3: Activity Narrative (AI-generated, what you produce)
Semantic summaries of what happened. Each entry has a `category` and a concise `summary` in German.

**Categories:** `decision`, `blocker`, `resolution`, `progress`, `milestone`, `risk`, `scope_change`, `communication`

**Guidelines:**
- 1-3 sentences per entry, in German
- Group related events into single entries when they tell one story
- You may produce zero, one, or multiple entries per run — whatever fits the events
- Skip truly irrelevant changes: typo fixes, wording tweaks, routine metadata updates without semantic meaning. \
It is perfectly fine to produce no Tier 3 entries if the events contain nothing noteworthy.
- Include `source_event_ids` linking back to Tier 4 events
- Include `kgr_codes` (e.g. "KGR 434") when events relate to specific Kostengruppen
- Include `involved_persons` when specific people are mentioned

### Tier 2: Current Status Snapshot (markdown, what you maintain)
A markdown document per project capturing the **current state only**. Resolved items belong in Tier 3, not here.

**Structure:** Section by Gewerk (KGR cost group) when applicable. Simple projects with one Gewerk are short; \
complex projects scale proportionally.
**Length guidance:** ~50-150 tokens per active Gewerk section, ~100-200 tokens for general sections.

**Example:**
```
## Aktueller Stand
Projekt in Entwurfsplanung. Nächste Baubesprechung am 13.03.

## KGR 434 — Kälteanlagen
Variante B für Serverraum bestätigt. 4 Klimaschränke.
Warten auf finale USV-Kapazitätsdaten von Firma X für Dimensionierung.

## KGR 440 — Elektrische Anlagen
Elektroleistungszusammenstellung in Arbeit (Fällig: 13.03).

## Nächste Schritte
- Finale Entscheidung Aufstellvariante BR/SR (Fällig: 13.03)
```

### Tier 1: Project Profile (markdown, rarely changes)
A standardized project overview: scope, client, stakeholders, locations, systems, constraints. \
Only update on major scope changes.
**Length guidance:** ~500-1500 tokens depending on project complexity.

**Example:**
```
## Projekt
Planung der Klimatisierung und IT-Infrastruktur für zwei Serverräume
bei Firma X am Standort Y in Z.

## Auftraggeber & Beteiligte
- Auftraggeber: Firma X
- Ansprechpartner: Hr. Müller (IT), Hr. Schmidt (Elektro)

## Gewerke & Systeme
- KGR 434 Kälteanlagen: Klimaschränke, Rückkühler, Kühlwassersystem
- KGR 440 Elektro: USV-Versorgung, Bestandsverkabelung

## Umfang & Randbedingungen
Entwurfsplanung bis LV-Erstellung. Bestandsgebäude mit eingeschränkter Tragfähigkeit.
```

## Database Schema

Use this reference for all `db()` queries.

""" + _SCHEMA_DOC + """

## Tool: run_python

You have one tool: `run_python`. It executes Python code in an isolated container.

**Pre-loaded functions (no import needed):**
- `db(sql)` — execute read-only SQL (SELECT/WITH). Returns list[dict].
- `fmt(rows)` — format rows as compact table for inspection.
- `file_text(id_or_path)` — extract text from a file (PDF, docx, pptx, xlsx, csv, txt). Accepts file UUID or /work/ path.
- `file_image(id_or_path, page=None, max_dim=None)` — queue an image for you to see. For PDFs pass page number. max_dim resizes longest edge. Accepts file UUID or /work/ path.
- `describe_image(id_or_path, question=None, page=None)` — send image to a vision model, get text description back. Pass a specific question for targeted analysis. Accepts file UUID or /work/ path.
- `download_file(content_hash)` — download a NAS file into /work/ by content_hash. Returns local path.
- `download_craft_file(storage_path)` — download a Craft doc media file (image, PDF, etc.) into /work/. The storage_path is the part after `craft-files/` in the URL. Returns local path.
- `add_activity_entry(project_id, logged_at, category, summary, source_event_ids=[], kgr_codes=[], involved_persons=[])` — insert a Tier 3 entry. Returns the new entry's UUID on success.
- `update_project_status(project_id, markdown)` — replace Tier 2 status. Returns confirmation with char count. Rejected if new text is less than half the length of current (to prevent accidental truncation).
- `update_project_profile(project_id, markdown)` — replace Tier 1 profile. Same length protection as status.

Call these functions from within `run_python` code blocks. Example:
```python
add_activity_entry(
    project_id=711736,
    logged_at="2026-03-06T19:05:00+00:00",
    category="progress",
    summary="Kälteplanung für Serverraum gestartet.",
    source_event_ids=[1, 2],
    kgr_codes=["KGR 434"],
)
```

## Inspecting files for better context

When events reference Craft documents, email attachments, or NAS files that contain images, PDFs, or other media, \
you may download and inspect them to understand what actually happened — especially when the text metadata alone \
is insufficient. This often produces much better, more specific summaries.

- **Craft doc media**: URLs in Craft markdown like `.../craft-files/DOC_ID/BLOCK_ID_filename.pdf` — use \
`download_craft_file("DOC_ID/BLOCK_ID_filename.pdf")` to pull into /work/.
- **NAS files**: Use `download_file(content_hash)` with the hash from `file_contents` table.
- **Email attachments**: Query `email_attachment_files` → `file_contents` to get the content_hash, then `download_file()`.

Once a file is in /work/, use `file_image(path)` to view images, `file_text(path)` to extract text from PDFs/docs, \
or `describe_image(path, question)` for vision analysis.

## Important rules

- Write ALL summaries and markdown in German (the team's working language).
- Do NOT fabricate information. Only summarize what the data tells you.
- If you update Tier 2 or Tier 1, include the FULL updated markdown (it replaces the existing one).
- Variables persist across tool calls within this response.
"""


# ---------------------------------------------------------------------------
# Event Processing Prompt
# ---------------------------------------------------------------------------

AGENT_EVENT_PROMPT = _TIER_CONTEXT + """\
## Your specific task: Process Tier 4 Events

1. Read the Tier 4 events provided in the user message
2. Use `db(sql)` to research context when it would improve your understanding — read email bodies, \
Craft doc content, related tasks, previous email threads, etc. Better context produces better summaries.
3. Create Tier 3 entries via `add_activity_entry()` for each meaningful activity — or none if the events are trivial
4. Update Tier 2 if the events indicate meaningful status changes via `update_project_status()`
5. Update Tier 1 only if there's a major scope change via `update_project_profile()`
"""


# ---------------------------------------------------------------------------
# Bootstrap Prompt
# ---------------------------------------------------------------------------

AGENT_BOOTSTRAP_PROMPT = _TIER_CONTEXT + """\
## Your specific task: Bootstrap Tier 1 and Tier 2 from scratch

Generate a comprehensive Tier 1 (profile) and Tier 2 (status) for this project by reading ALL available data:

1. Use `db(sql)` to read:
   - Project info: `SELECT * FROM teamwork.projects WHERE id = {project_id}`
   - Craft docs: `SELECT cd.title, cd.markdown_content FROM craft_documents cd JOIN project_craft_documents pcd ON cd.id = pcd.craft_document_id WHERE pcd.tw_project_id = {project_id}`
   - Current tasks with structure: tasklists, statuses, assignees, tags (especially KGR tags)
   - Recent emails: subjects, senders (via project_conversations + missive tables)
   - Project extensions: locations, cost groups, client, contractors
   - Files: recent project files

2. From all this data, generate:
   - **Tier 1 (Profile):** Standardized overview — scope, client, stakeholders, locations, systems (Gewerke/KGR), constraints. Draw especially from Craft docs (Machbarkeit/feasibility studies) and Anforderungen tasklist.
   - **Tier 2 (Status):** Current state snapshot — what's active, what's blocked, next steps. Sectioned by Gewerk (KGR). Reflect only current state.

3. Write results via `update_project_profile(project_id, markdown)` and `update_project_status(project_id, markdown)`.

Read the Craft docs thoroughly — they contain the richest project context (scope, requirements, technical details).
"""


# ---------------------------------------------------------------------------
# Diff Processor
# ---------------------------------------------------------------------------

async def process_pending_diffs(pool: asyncpg.Pool):
    """Compute unified diffs for events with old_content."""
    events = await pool.fetch(
        "SELECT id, source_table, source_id, old_content "
        "FROM project_event_log "
        "WHERE old_content IS NOT NULL AND NOT processed_by_diff"
    )
    if not events:
        return

    logger.info("Processing %d pending diffs", len(events))
    for event in events:
        try:
            current = await _fetch_current_content(
                pool, event["source_table"], event["source_id"]
            )
            diff = _compute_diff(event["old_content"], current)
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
    return await resolve_model(settings.AGENT_MODEL, pool)


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

    sandbox = SandboxSession(pool, None, session_id)
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
    events = await pool.fetch("""
        SELECT id, occurred_at, source_table, source_id, event_type, details, content_diff
        FROM project_event_log
        WHERE tw_project_id = $1 AND NOT processed_by_agent
          AND (processed_by_diff OR old_content IS NULL)
        ORDER BY occurred_at
    """, project_id)
    if not events:
        return

    event_ids = [e["id"] for e in events]
    project_name = await pool.fetchval(
        "SELECT name FROM teamwork.projects WHERE id = $1", project_id
    )
    current_t1 = await pool.fetchval(
        "SELECT profile_markdown FROM project_extensions WHERE tw_project_id = $1", project_id
    )
    current_t2 = await pool.fetchval(
        "SELECT status_markdown FROM project_extensions WHERE tw_project_id = $1", project_id
    )
    recent_t3 = await pool.fetch("""
        SELECT logged_at, category, summary
        FROM project_activity_log
        WHERE tw_project_id = $1
        ORDER BY logged_at DESC LIMIT 30
    """, project_id)

    user_message = _build_event_prompt(project_id, project_name, events, current_t1, current_t2, recent_t3)

    logger.info("Running event agent for project %s (%s) — %d events", project_id, project_name, len(events))
    result = await run_agent_turn(
        AGENT_EVENT_PROMPT, user_message, pool,
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
    project_name = await pool.fetchval(
        "SELECT name FROM teamwork.projects WHERE id = $1", project_id
    )

    user_message = f"[system] # Bootstrap project: {project_name or 'Unknown'} (ID: {project_id})\n\nGenerate Tier 1 (profile) and Tier 2 (status) from scratch for this project."

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

    logger.info("Running bootstrap agent for project %s (%s)", project_id, project_name)
    result = await run_agent_turn(
        AGENT_BOOTSTRAP_PROMPT, user_message, pool,
        context_metadata={"tw_project_id": project_id, "action": "bootstrap"},
        nudge_check=_bootstrap_nudge,
        title_prefix="BST",
    )

    await pool.execute(
        "UPDATE project_agent_requests SET status = 'done', result_session_id = $1, processed_at = NOW() WHERE id = $2",
        result.session_id, request_id,
    )
    logger.info("Bootstrap completed for project %s — session %s", project_id, result.session_id)


# ---------------------------------------------------------------------------
# Prompt Builders
# ---------------------------------------------------------------------------

def _build_event_prompt(
    project_id: int,
    project_name: str | None,
    events: list,
    current_t1: str | None,
    current_t2: str | None,
    recent_t3: list,
) -> str:
    parts = [f"[system] # Project: {project_name or 'Unknown'} (ID: {project_id})", ""]

    if current_t1:
        parts.extend(["## Current Tier 1 (Profile)", current_t1, ""])
    else:
        parts.extend(["## Tier 1 (Profile): Not yet generated", ""])

    if current_t2:
        parts.extend(["## Current Tier 2 (Status)", current_t2, ""])
    else:
        parts.extend(["## Tier 2 (Status): Not yet generated", ""])

    if recent_t3:
        parts.append(f"## Recent Tier 3 Entries (last {len(recent_t3)}, newest first)")
        parts.append("")
        for t3 in recent_t3:
            parts.append(
                f"- [{t3['logged_at'].strftime('%Y-%m-%d %H:%M')}] **{t3['category']}**: {t3['summary']}"
            )
        parts.append("")

    parts.append(f"## Tier 4 Events to Process ({len(events)} events)")
    parts.append("")

    for e in events:
        details = e["details"] if isinstance(e["details"], dict) else json.loads(e["details"])
        line = (
            f"- **Event {e['id']}** [{e['occurred_at'].strftime('%Y-%m-%d %H:%M')}] "
            f"`{e['source_table']}` {e['event_type']}: "
            f"{json.dumps(details, ensure_ascii=False, default=str)}"
        )
        if e["content_diff"]:
            line += f"\n  Diff:\n  ```\n  {e['content_diff'][:2000]}\n  ```"
        parts.append(line)

    parts.extend(["", "Process these events: create Tier 3 entries, update Tier 2 if needed."])

    return "\n".join(parts)

"""LLM integration - multi-provider streaming with tool loop and prompt caching."""

import json
import logging
import re
from typing import AsyncGenerator

import asyncpg

from . import settings
from .sandbox import execute_python, create_sandbox_env
from .providers import get_provider

logger = logging.getLogger("ibhelm.chat.llm")

_LEAKED_TOKENS_RE = re.compile(r'\s*<\|tool_calls_section_begin\|>.*', re.DOTALL)
_MARKDOWN_CODE_RE = re.compile(r'```(?:python|py)?\s*\n(.*?)```', re.DOTALL)


def _strip_leaked_tokens(blocks: list[dict]):
    """Remove leaked model tokens (e.g. <|tool_calls_section_begin|>) from the last thinking block."""
    if not blocks:
        return
    last = blocks[-1]
    if last.get("type") == "thinking" and '<|tool_calls_section_begin|>' in last.get("text", ""):
        last["text"] = _LEAKED_TOKENS_RE.sub("", last["text"]).rstrip()
        if not last["text"]:
            blocks.pop()


def _extract_code_blocks(text: str) -> list[str]:
    """Extract Python code from markdown fenced code blocks."""
    return [m.strip() for m in _MARKDOWN_CODE_RE.findall(text) if m.strip()]

# ---------------------------------------------------------------------------
# Tool definition (provider-agnostic content, formatted per-provider)
# ---------------------------------------------------------------------------

TOOL_NAME = "run_python"
TOOL_DESCRIPTION = (
    "Execute Python code with database access.\n\n"
    "Available: db(sql) → list[dict], fmt(rows) → str (TOON format), "
    "math, json, re, datetime, timedelta, date, Counter, defaultdict.\n\n"
    "Variables persist across tool calls within this response."
)
TOOL_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "code": {"type": "string", "description": "Python code to execute"}
    },
    "required": ["code"]
}

# ---------------------------------------------------------------------------
# System prompt - STATIC part (cached by Anthropic)
# ---------------------------------------------------------------------------

STATIC_SYSTEM_PROMPT = """\
You are an AI assistant for IBHelm, a data management system for an engineering office \
(Ingenieurbüro) specializing in technische Gebäudeausrüstung (TGA / HVAC / building services engineering). \
You have read-only access to the company's central database through Python code execution.

The database aggregates data from three source systems:
- **Teamwork**: Project management (tasks, projects, timelogs, tags)
- **Missive**: Email communication (conversations, messages, contacts, attachments)
- **Craft**: Documentation (documents with markdown content)
Plus local files synced from the office NAS.

## Behavior
- Respond in the same language the user writes in.
- Each user message ends with a system-injected timestamp (e.g. `[2026-03-04 14:30 UTC]`). This is NOT from the user — it provides temporal context. Do not mention it.
- Be precise and helpful. Present query results clearly using Markdown (tables, lists, bold).
- Always verify by querying rather than making assumptions. Don't guess IDs or dates.
- When referencing specific items (tasks, emails, projects, documents, people), always include clickable links.
- Be specific - reference actual task names, dates, assignees, and project names.
- Don't make up information. If you don't know or can't find something, say so.
- Keep responses focused and actionable. Avoid unnecessary pleasantries.
- Variables persist across tool calls within one response - reuse them for multi-step analysis.

## Tool: run_python
You have one tool: `run_python`. Write Python code using the available functions and modules.

### Functions
- `db(sql)` - Execute read-only SQL (SELECT/WITH only). Returns list of dicts. Keys are column names.
- `fmt(rows, max_rows=50, max_cell=80)` - Format rows as compact TOON table. Set max_rows=None or max_cell=None to disable truncation.

### Modules (all pre-loaded — NEVER use import/from statements, they will error)
math, json, re, datetime, timedelta, date, Counter, defaultdict

### Example
```python
tasks = db(\"\"\"
    SELECT t.id, t.name, t.due_date, t.status,
           string_agg(u.first_name || ' ' || u.last_name, ', ') as assignees
    FROM teamwork.tasks t
    LEFT JOIN teamwork.task_assignees ta ON t.id = ta.task_id
    LEFT JOIN teamwork.users u ON ta.user_id = u.id
    WHERE t.project_id = 682843 AND t.status != 'completed'
      AND t.due_date < NOW()
    GROUP BY t.id, t.name, t.due_date, t.status
    ORDER BY t.due_date
\"\"\")
print(fmt(tasks))
```

## Database Schema

### teamwork (project management)
- **projects**: id, name, description, status, company_id, start_date, end_date, created_at, updated_at
- **tasks**: id, project_id, tasklist_id, name, description, status ('new'/'in progress'/'completed'), priority ('low'/'medium'/'high'), progress (0-100), parent_task, start_date, due_date, estimate_minutes, accumulated_estimated_minutes, created_at
- **tasklists**: id, name, description, project_id, status
- **users**: id, first_name, last_name, email, title, company_id, is_admin
- **companies**: id, name, email_one, phone, website, address_one, city, zip
- **tags**: id, name, color, project_id
- **timelogs**: id, task_id, project_id, user_id, minutes, description, time_logged, is_billable
- Junction: **task_tags**(task_id, tag_id), **task_assignees**(task_id, user_id), **user_teams**(user_id, team_id)

### missive (email)
- **conversations**: id, subject, latest_message_subject, messages_count, last_activity_at, web_url, app_url
- **messages**: id, conversation_id, subject, preview, body, body_plain_text, from_contact_id, delivered_at, created_at
- **contacts**: id, name, email
- **attachments**: id, message_id, filename, extension, size, media_type
- **shared_labels**: id, name
- Junction: **conversation_labels**(label_id, conversation_id), **message_recipients**(message_id, recipient_type, contact_id)

### public (core)
- **files**: id, full_path, content_hash, project_id, document_type_id, fs_mtime, deleted_at
- **file_contents**: content_hash, size_bytes, mime_type, extracted_text, thumbnail_path
- **craft_documents**: id, title, markdown_content, folder_path, craft_last_modified_at, daily_note_date
- **unified_persons**: id, display_name, primary_email, is_internal, is_company
- **locations**: id, parent_id, name, type (building/level/room), path, depth
- **cost_groups**: id, parent_id, code, name, path (DIN 276 Kostengruppen, 3-digit codes 100-999)
- **project_extensions**: tw_project_id, default_location_id, default_cost_group_id, nas_folder_path, client_person_id
- **task_types**: id, name, slug, color, icon
- **task_extensions**: tw_task_id, task_type_id

### Key Views (query like tables)
- **project_overview**: id, name, status, company_name, client_name, file_count, task_count, conversation_count
- **unified_items_secure**: Unified view across tasks, emails, files, and Craft docs
  - type: 'task' | 'email' | 'file' | 'craft'
  - Columns: id, type, name, status, project, project_status, customer, due_date, created_at, updated_at, creator, search_text, teamwork_url, missive_url, craft_url, assigned_to (jsonb), tags (jsonb), location, location_path, cost_group, cost_group_code, priority, progress, tasklist, body, preview, conversation_subject, recipients (jsonb), attachments (jsonb), attachment_count
  - Emails are filtered by current user's visibility (RLS)
  - Use ILIKE on search_text for full-text search
  - For newest emails: `WHERE type = 'email' ORDER BY created_at DESC`
  - For tasks: `WHERE type = 'task'`
- **file_details**: id, full_path, content_hash, document_type, thumbnail_path, project (jsonb), size_bytes
- **unified_person_details**: id, display_name, primary_email, tw_user_email, m_contact_email, is_internal, is_company
- **location_hierarchy**: id, name, type, depth, path, parent_name, building_name, child_count

### Key Relationships
- task → project: tasks.project_id → projects.id
- task → assignees: task_assignees (task_id, user_id)
- task → tags: task_tags (task_id, tag_id)
- conversation → labels: conversation_labels (label_id, conversation_id)
- message → sender: messages.from_contact_id → contacts.id
- message → recipients: message_recipients (message_id, recipient_type [to/cc/bcc], contact_id)
- file → project: files.project_id → projects.id
- object → location: object_locations (tw_task_id | m_conversation_id | file_id, location_id)
- object → cost_group: object_cost_groups (tw_task_id | m_conversation_id | file_id, cost_group_id)
- project → conversations: project_conversations (m_conversation_id, tw_project_id)
- project → craft docs: project_craft_documents (craft_document_id, tw_project_id)

## Generating Links
When you mention specific tasks, emails, projects, or documents, always include clickable Markdown links.

- **Teamwork task/Anforderung/Hinweis**: `[Task Name](https://ibhelm.teamwork.com/#/tasks/{task_id})`
- **Teamwork project**: `[Project Name](https://ibhelm.teamwork.com/app/projects/{project_id})`
- **Missive conversation**: `[Subject](https://mail.missiveapp.com/#inbox/conversations/{conversation_id})` — or use the `web_url` column from `missive.conversations`
- **Craft document**: `[Title](craftdocs://open?spaceId=fa51f40a-da64-2cc0-6a32-d489be2d5528&blockId={document_id})` — opens directly in Craft app
- The `unified_items_secure` view has `teamwork_url`, `missive_url`, and `craft_url` columns for convenience.

## Row Level Security (RLS)
The database enforces row-level security. Email-related queries through `unified_items_secure` \
are automatically filtered by the current user's visibility. You will only see emails the user \
has access to. This is normal — do not treat missing results as an error.

## Limitations
- The database is READ-ONLY. No INSERT, UPDATE, or DELETE.
- If the user wants to edit something, they must do it in the source tool (Teamwork, Missive, Craft). Changes sync to the database within a few minutes.
- You cannot send notifications, schedule tasks, or access external websites.
- Long print output is automatically truncated. If that happens, use targeted slicing on your variables to inspect specific parts.
- The `import` statement is disabled in the sandbox. All modules and functions are pre-loaded. Never write `import` or `from ... import`."""

# ---------------------------------------------------------------------------
# Model config resolution (cached from DB)
# ---------------------------------------------------------------------------

_model_configs: list[dict] | None = None


async def get_model_configs(pool: asyncpg.Pool) -> list[dict]:
    """Load and cache model configs from app_settings."""
    global _model_configs
    if _model_configs is not None:
        return _model_configs
    row = await pool.fetchrow("SELECT body->>'chat_models' as models FROM app_settings LIMIT 1")
    if row and row["models"]:
        _model_configs = json.loads(row["models"])
    else:
        _model_configs = [{"id": settings.CLAUDE_MODEL, "provider": "anthropic",
                           "name": "Claude Sonnet 4", "default": True}]
    return _model_configs


async def resolve_model(model_id: str | None, pool: asyncpg.Pool) -> dict:
    """Resolve a model ID to its full config dict."""
    configs = await get_model_configs(pool)
    if model_id:
        match = next((m for m in configs if m["id"] == model_id), None)
        if match:
            return match
    default = next((m for m in configs if m.get("default")), configs[0])
    return default


def invalidate_model_cache():
    """Call when app_settings might have changed."""
    global _model_configs
    _model_configs = None

# ---------------------------------------------------------------------------
# Dynamic context builder
# ---------------------------------------------------------------------------


async def _build_dynamic_context(pool: asyncpg.Pool, user_email: str | None) -> str:
    parts = [f"## Current Context\n- **User**: {user_email or 'unknown'}"]
    try:
        rows = await pool.fetch(
            "SELECT id, name FROM teamwork.projects WHERE status = 'active' ORDER BY name"
        )
        if rows:
            parts.append("\n## Active Projects")
            parts.append("Users often use abbreviations. Match against this list:")
            for r in rows:
                parts.append(f"- {r['name']} (id: {r['id']})")
    except Exception as e:
        logger.warning("Failed to fetch projects for context: %s", e)
    return "\n".join(parts)

# ---------------------------------------------------------------------------
# Streaming chat response with shared tool loop
# ---------------------------------------------------------------------------


async def stream_chat_response(
    messages: list[dict],
    pool: asyncpg.Pool,
    user_email: str | None = None,
    model_override: str | None = None,
) -> AsyncGenerator[dict, None]:
    """Stream a chat response with tool calling loop.

    Yields SSE events:
      {"type": "text", "content": "..."}
      {"type": "thinking", "content": "..."}
      {"type": "tool_call", "id": "...", "code": "..."}
      {"type": "tool_result", "id": "...", "result"|"error": "..."}
      {"type": "done", "content": "...", "blocks": [...], "metadata": {...}}
      {"type": "error", "message": "..."}
    """
    model_config = await resolve_model(model_override, pool)
    provider = get_provider(model_config)
    active_model = model_config["id"]

    dynamic_context = await _build_dynamic_context(pool, user_email)
    static = STATIC_SYSTEM_PROMPT
    if model_config.get("system_prompt_addition"):
        static += "\n\n" + model_config["system_prompt_addition"]
    system = provider.build_system_prompt(static, dynamic_context)
    tools = [provider.build_tool_definition(TOOL_NAME, TOOL_DESCRIPTION, TOOL_INPUT_SCHEMA)]
    api_messages = provider.build_api_messages(messages)

    thinking_config = None
    if settings.ENABLE_THINKING and model_config.get("provider") == "anthropic":
        thinking_config = {"budget_tokens": settings.THINKING_BUDGET}

    max_tokens = 16384 if thinking_config else 8192
    sandbox_env = create_sandbox_env()
    all_blocks: list[dict] = []
    full_text = ""

    total_usage = {
        "input_tokens": 0, "output_tokens": 0,
        "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0,
    }

    def _append_block(btype: str, text: str):
        if all_blocks and all_blocks[-1]["type"] == btype:
            all_blocks[-1]["text"] += text
        else:
            all_blocks.append({"type": btype, "text": text})

    for iteration in range(settings.MAX_TOOL_ITERATIONS):
        turn_tool_calls = []
        turn_text = ""

        async for event in provider.stream_turn(
            api_messages, system, tools, active_model, max_tokens, thinking_config
        ):
            etype = event["type"]

            if etype == "text":
                full_text += event["content"]
                turn_text += event["content"]
                _append_block("text", event["content"])
                yield {"type": "text", "content": event["content"]}

            elif etype == "thinking":
                _append_block("thinking", event["content"])
                yield {"type": "thinking", "content": event["content"]}

            elif etype == "turn_end":
                if event.get("error"):
                    yield {"type": "error", "message": event["error"]}
                    return

                turn_tool_calls = event["tool_calls"]
                usage = event["usage"]
                for k in total_usage:
                    total_usage[k] += usage.get(k, 0)

                if event["stop_reason"] != "tool_use":
                    # Check for code blocks in text that should be executed
                    if model_config.get("auto_execute_code_blocks") and turn_text:
                        code_blocks = _extract_code_blocks(turn_text)
                        if code_blocks:
                            logger.info("Auto-executing %d code block(s) from text", len(code_blocks))
                            turn_tool_calls = [
                                {"id": f"auto_{iteration}_{i}", "name": TOOL_NAME, "input": {"code": code}}
                                for i, code in enumerate(code_blocks)
                            ]
                            break  # fall through to tool execution below

                    # Nudge: if no text was produced but tools ran, force a final text response
                    has_tool_results = any(b.get("type") == "tool_call" for b in all_blocks)
                    if not full_text and has_tool_results and iteration < settings.MAX_TOOL_ITERATIONS - 1:
                        logger.info("Nudging model for final text response (no text after tool calls)")
                        provider.append_assistant_turn(api_messages, turn_text, [])
                        continue

                    total_usage["model"] = active_model
                    yield {
                        "type": "done",
                        "content": full_text,
                        "blocks": all_blocks if all_blocks else None,
                        "metadata": total_usage,
                    }
                    return

                # Strip leaked tool-call tokens from the last thinking block
                _strip_leaked_tokens(all_blocks)

        if not turn_tool_calls:
            total_usage["model"] = active_model
            yield {"type": "done", "content": full_text,
                   "blocks": all_blocks if all_blocks else None, "metadata": total_usage}
            return

        provider.append_assistant_turn(api_messages, turn_text, turn_tool_calls)

        # Execute each tool call
        tool_results = []
        for tc in turn_tool_calls:
            code = tc["input"].get("code", "")
            tc_id = tc["id"]

            yield {"type": "tool_call", "id": tc_id, "code": code}

            result = await execute_python(code, pool, user_email, sandbox_env)
            tc_record: dict = {"type": "tool_call", "id": tc_id, "code": code}

            if result.get("error"):
                output = result["error"]
                if result.get("output"):
                    output = result["output"] + "\n" + output
                tc_record["error"] = output
                yield {"type": "tool_result", "id": tc_id, "error": output}
            else:
                parts = []
                if result.get("output"):
                    parts.append(result["output"])
                if result.get("result") is not None:
                    parts.append(json.dumps(result["result"], ensure_ascii=False, default=str))
                output = "\n".join(parts) if parts else "(no output)"
                tc_record["result"] = output
                yield {"type": "tool_result", "id": tc_id, "result": output}

            tool_results.append({"tool_use_id": tc_id, "content": output})
            all_blocks.append(tc_record)

        provider.append_tool_results(api_messages, tool_results)

    # Max iterations reached — do one final call without tools to force a text summary
    logger.info("Max tool iterations reached, forcing final response without tools")
    try:
        async for event in provider.stream_turn(
            api_messages, system, [], active_model, max_tokens, None
        ):
            if event["type"] == "text":
                full_text += event["content"]
                _append_block("text", event["content"])
                yield {"type": "text", "content": event["content"]}
            elif event["type"] == "turn_end":
                usage = event.get("usage", {})
                for k in total_usage:
                    total_usage[k] += usage.get(k, 0)
    except Exception as e:
        logger.warning("Final nudge failed: %s", e)

    total_usage["model"] = active_model
    yield {
        "type": "done",
        "content": full_text,
        "blocks": all_blocks if all_blocks else None,
        "metadata": total_usage,
    }

# ---------------------------------------------------------------------------
# Title generation
# ---------------------------------------------------------------------------


async def generate_title(content: str, pool: asyncpg.Pool) -> str:
    """Generate a short title for a chat session from the first message."""
    prompt = f"Generate a short title (2-5 words) for this chat message. Reply with ONLY the title — no alternatives, no explanations, no quotes, no period.\n\n{content[:500]}"

    try:
        title_model_id = settings.TITLE_MODEL
        if title_model_id:
            model_config = await resolve_model(title_model_id, pool)
        else:
            model_config = await resolve_model(None, pool)

        provider = get_provider(model_config)
        max_tokens = 1000 if model_config.get("provider") != "anthropic" else 30
        raw = await provider.generate_simple(model_config["id"], prompt, max_tokens)
        title = raw.strip('"').strip("'")[:100]
        if not title:
            raise ValueError("Empty response from model")
        return title
    except Exception as e:
        logger.warning("Title generation failed: %s", e)
        return content[:60].strip()


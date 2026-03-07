"""LLM integration - multi-provider streaming with tool loop and prompt caching."""

import json
import logging
import re
from pathlib import Path
from typing import AsyncGenerator

import asyncpg

from . import settings
from .sandbox import SandboxSession, resolve_images
from . import storage as st
from .providers import get_provider

logger = logging.getLogger("ibhelm.chat.llm")

# Auto-generated schema doc and sandbox requirements (loaded once at import)
_DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"
_SCHEMA_DOC = (_DOCS_DIR / "schema_doc_output.md").read_text().strip() if (_DOCS_DIR / "schema_doc_output.md").exists() else ""
_SANDBOX_REQUIREMENTS = (Path(__file__).resolve().parent.parent / "sandbox" / "requirements.txt").read_text().strip() if (Path(__file__).resolve().parent.parent / "sandbox" / "requirements.txt").exists() else ""

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
    "Execute Python in an isolated container with full library access.\n\n"
    "Pre-loaded functions (no import needed):\n"
    "- db(sql) -> list[dict]: Execute read-only SQL (SELECT/WITH)\n"
    "- fmt(rows, max_rows=50, max_cell=80) -> str: Format as compact TOON table\n"
    "- file_info(id) -> dict: File metadata {filename, size, mime_type, nas_path, project_name, extracted_text}\n"
    "- file_text(id_or_path) -> str: Extract text (PDF, docx, pptx, xlsx, csv, txt). Accepts file UUID or /work/ path.\n"
    "- file_image(id_or_path, page=None, max_dim=None): Queue image for you to see (vision models only). For PDFs: page number. max_dim resizes.\n"
    "- describe_image(id_or_path, question=None, page=None): Send image to a vision model, get text description back. Use when you cannot view images directly. Pass a specific question for targeted analysis.\n"
    "- download_file(content_hash) -> str: Download NAS file into /work/ by content_hash from DB\n"
    "- download_craft_file(storage_path) -> str: Download Craft doc media (image/PDF/file) into /work/ by its storage path from craft-files bucket\n"
    "- download_url(file_id) -> str: Get URL for user to click\n\n"
    "Full Python: import anything, subprocess, os, open() all work.\n"
    "/work/ is your workspace with conversation files pre-populated.\n"
    "New files created in /work/ are automatically saved.\n"
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

## Previous conversations
- while every new conversation starts with a fresh context, you have access to the previous conversations and can use them to understand the current context and the user's intent.
- use this tool wisely if you think the user might be refrencing something from previous conversations or it could be helpful to look them up
- use db function: search_chat_history for this purpose

## Tool: run_python
You have one tool: `run_python`. Execute Python code in an isolated container with full library access.

### Pre-loaded Functions (always available, no import needed)
- `db(sql)` - Execute read-only SQL (SELECT/WITH only). Returns list[dict].
- `fmt(rows, max_rows=50, max_cell=80)` - Format rows as compact TOON table.
- `file_info(id)` - Get metadata for a conversation file: {filename, size_bytes, mime_type, nas_path, project_name, extracted_text}.
- `file_text(id_or_path)` - Extract text from a file (PDF, docx, pptx, xlsx, csv, plain text). Returns string. Accepts file UUID or /work/ path.
- `file_image(id_or_path, page=None, max_dim=None)` - Queue an image for you to see (vision models only). For PDFs pass page number (1-indexed). max_dim resizes longest edge to save tokens. Accepts file UUID or local path.
- `describe_image(id_or_path, question=None, page=None)` - Send image to a vision model and get a text description back. Use this if you cannot view images directly. Pass a specific question for targeted analysis (e.g. "what text is on this document?"), or None for a general description. Accepts file UUID or local /work/ path.
- `download_file(content_hash)` - Download a NAS file into /work/ by its content_hash (from file_contents table). Returns local path.
- `download_craft_file(storage_path)` - Download a Craft document media file (image, PDF, etc.) into /work/ by its storage path from the craft-files bucket. The storage path is the part after `craft-files/` in the URL. Returns local path.
- `download_url(file_id)` - Get a download URL for a file the user can click.

### Environment
- Full Python with all standard libraries. `import` works normally.
- Available packages: """ + _SANDBOX_REQUIREMENTS.replace('\n', ', ') + """.
- `subprocess`, `os`, `pathlib`, `open()` all work.
- /opt/sandbox/docs/ contains reference documentation. Read files there when you need details about the dashboard UI, system architecture, or deployment.
- /work/ is your workspace. Files the user attached to the conversation are pre-populated there.
- Any new files you save to /work/ are automatically uploaded and available to the user after execution.
- Variables persist across tool calls within this response.

### File IDs
- Files attached to messages have UUIDs (use with file_info, file_text, file_image, download_url).
- NAS files found via SQL have a `content_hash` in the `file_contents` table (use with download_file to pull into /work/).
- The `files` table has: id, full_path, content_hash, project_id. Join with `file_contents` for size, mime_type, extracted_text.

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
""" + _SCHEMA_DOC + """

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
- The sandbox has no internet access. All external data must come through db() or pre-populated files."""

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


async def build_dynamic_context(pool: asyncpg.Pool, user_email: str | None) -> str:
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


async def build_session_prompt(pool: asyncpg.Pool, user_email: str | None) -> str:
    """Build the full system prompt to snapshot on a new chat session."""
    dynamic = await build_dynamic_context(pool, user_email)
    return STATIC_SYSTEM_PROMPT + "\n\n" + dynamic

# ---------------------------------------------------------------------------
# Streaming chat response with shared tool loop
# ---------------------------------------------------------------------------


async def stream_chat_response(
    messages: list[dict],
    pool: asyncpg.Pool,
    user_email: str | None = None,
    model_config: dict | None = None,
    sandbox: SandboxSession | None = None,
    assistant_msg_id: str | None = None,
    system_prompt: str | None = None,
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
    if not model_config:
        model_config = await resolve_model(None, pool)
    provider = get_provider(model_config)
    active_model = model_config["id"]

    yield {"type": "model_resolved", "model": active_model}

    from datetime import datetime, timezone
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    if not system_prompt:
        dynamic_context = await build_dynamic_context(pool, user_email)
        system_prompt = STATIC_SYSTEM_PROMPT + "\n\n" + dynamic_context
    if model_config.get("system_prompt_addition"):
        system_prompt += "\n\n" + model_config["system_prompt_addition"]
    system = provider.build_system_prompt(system_prompt, f"Current time: {ts}")
    tools = [provider.build_tool_definition(TOOL_NAME, TOOL_DESCRIPTION, TOOL_INPUT_SCHEMA)]
    api_messages = provider.build_api_messages(messages)

    thinking_config = None
    if settings.ENABLE_THINKING and model_config.get("provider") == "anthropic":
        thinking_config = {"budget_tokens": settings.THINKING_BUDGET}

    max_tokens = 16384 if thinking_config else 8192
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

        # Execute each tool call via sandbox container
        tool_results = []
        for tc in turn_tool_calls:
            code = tc["input"].get("code", "")
            tc_id = tc["id"]

            yield {"type": "tool_call", "id": tc_id, "code": code}

            if sandbox:
                result = await sandbox.execute(code)
            else:
                result = {"stdout": None, "result": None, "error": "Sandbox not available",
                          "pending_images": [], "new_files": []}

            tc_record: dict = {"type": "tool_call", "id": tc_id, "code": code}

            if result.get("error"):
                output = result["error"]
                if result.get("stdout"):
                    output = result["stdout"] + "\n" + output
                tc_record["error"] = output
                yield {"type": "tool_result", "id": tc_id, "error": output}
            else:
                parts = []
                if result.get("stdout"):
                    parts.append(result["stdout"])
                if result.get("result") is not None:
                    parts.append(json.dumps(result["result"], ensure_ascii=False, default=str))
                output = "\n".join(parts) if parts else "(no output)"
                tc_record["result"] = output
                yield {"type": "tool_result", "id": tc_id, "result": output}

            # Upload new files from sandbox and append info to output
            if result.get("new_files") and sandbox and assistant_msg_id:
                created = await sandbox.upload_new_files(result["new_files"], assistant_msg_id)
                if created:
                    yield {"type": "files_created", "files": created}
                    file_lines = ["\n\n[Files saved]"]
                    for cf in created:
                        url = st.public_url("chat-files", cf["content_hash"])
                        file_lines.append(f"- {cf['filename']}: id={cf['id']}, url={url}")
                    output += "\n".join(file_lines)
                    tc_record["result"] = output

            tool_content = output
            if result.get("pending_images") and sandbox:
                if model_config.get("supports_vision"):
                    images = await resolve_images(sandbox, result["pending_images"])
                    if images:
                        tool_content = [{"type": "text", "text": output}] + images
                        yield {"type": "pending_images", "images": result["pending_images"]}
                else:
                    vision_hint = (
                        "\n\nERROR: This model cannot view images directly. "
                        "Use describe_image(id_or_path, question=None) instead — "
                        "it sends the image to a vision model and returns a text description. "
                        "Pass a specific question for targeted analysis, or None for a general description."
                    )
                    output += vision_hint
                    tc_record["result"] = output
                    tool_content = output

            tool_results.append({"tool_use_id": tc_id, "content": tool_content})
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


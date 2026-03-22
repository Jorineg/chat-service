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
from . import template_resolver as tr
from .providers import get_provider

logger = logging.getLogger("ibhelm.chat.llm")

# Sandbox requirements (loaded once at import)
_SANDBOX_REQUIREMENTS = (Path(__file__).resolve().parent.parent / "sandbox" / "requirements.txt").read_text().strip() if (Path(__file__).resolve().parent.parent / "sandbox" / "requirements.txt").exists() else ""

_LEAKED_TOKENS_RE = re.compile(r'\s*<\|tool_calls_section_begin\|>.*', re.DOTALL)
_MARKDOWN_CODE_RE = re.compile(r'```(?:python|py)?\s*\n(.*?)```', re.DOTALL)
_USAGE_KEYS = (
    "input_tokens",
    "output_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
)


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


def empty_usage() -> dict:
    """Create an empty usage object with stable keys."""
    return {k: 0 for k in _USAGE_KEYS}


def normalize_usage(usage: dict | None) -> dict:
    """Normalize provider usage payload to the four tracked token counters."""
    normalized = empty_usage()
    if not usage:
        return normalized
    for key in _USAGE_KEYS:
        normalized[key] = int(usage.get(key, 0) or 0)
    return normalized


def add_usage(total: dict, usage: dict | None) -> None:
    """Accumulate normalized usage into a mutable totals dict."""
    normalized = normalize_usage(usage)
    for key in _USAGE_KEYS:
        total[key] += normalized[key]


def calculate_usage_cost_usd(model_config: dict | None, usage: dict | None) -> float:
    """Calculate USD cost for a usage payload from model pricing fields."""
    if not model_config or not usage:
        return 0.0
    normalized = normalize_usage(usage)
    return (
        normalized["input_tokens"] * float(model_config.get("input_price") or 0)
        + normalized["output_tokens"] * float(model_config.get("output_price") or 0)
        + normalized["cache_read_input_tokens"] * float(model_config.get("cache_read_price") or 0)
        + normalized["cache_creation_input_tokens"] * float(model_config.get("cache_write_price") or 0)
    ) / 1_000_000

# ---------------------------------------------------------------------------
# Tool definition (provider-agnostic content, formatted per-provider)
# ---------------------------------------------------------------------------

TOOL_NAME = "run_python"
TOOL_DESCRIPTION = (
    "Execute Python in an isolated container. "
    "Pre-loaded: db(sql), fmt(rows), file_text(id), file_image(id), describe_image(id), "
    "download_file(hash), download_craft_file(path), download_url(id), web_search(query), fetch_url(url), "
    "add_activity_entry(...), update_activity_entry(...), update_project_status(id, md), update_project_profile(id, md). "
    "Full Python with all standard libraries. /work/ is your workspace. Always print() results."
)
TOOL_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "code": {"type": "string", "description": "Python code to execute"}
    },
    "required": ["code"]
}

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

async def build_session_prompt(pool: asyncpg.Pool, user_email: str | None) -> str:
    """Build the full system prompt from the chat.system_prompt template."""
    return await tr.resolve(pool, "chat.system_prompt", {
        "user_email": user_email or "unknown",
        "sandbox_requirements": _SANDBOX_REQUIREMENTS.replace('\n', ', '),
    })

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
        system_prompt = await _build_system_prompt(pool, user_email)
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

    total_usage = empty_usage()
    subcalls: list[dict] = []
    subcall_index = 0

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
                usage = normalize_usage(event["usage"])
                add_usage(total_usage, usage)
                subcall_index += 1
                subcalls.append({
                    "kind": "llm_turn",
                    "index": subcall_index,
                    "stop_reason": event["stop_reason"],
                    **usage,
                })

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

                    metadata = {**total_usage, "model": active_model}
                    if subcalls:
                        metadata["subcalls"] = subcalls
                    yield {
                        "type": "done",
                        "content": full_text,
                        "blocks": all_blocks if all_blocks else None,
                        "metadata": metadata,
                    }
                    return

                # Strip leaked tool-call tokens from the last thinking block
                _strip_leaked_tokens(all_blocks)

        if not turn_tool_calls:
            metadata = {**total_usage, "model": active_model}
            if subcalls:
                metadata["subcalls"] = subcalls
            yield {"type": "done", "content": full_text,
                   "blocks": all_blocks if all_blocks else None, "metadata": metadata}
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
                output = "\n".join(parts) if parts else \
                    "(no output — your code ran but printed nothing. Use print() to see db() results or function return values.)"
                tc_record["result"] = output
                yield {"type": "tool_result", "id": tc_id, "result": output}

            for tool_cost in result.get("tool_costs") or []:
                cost_usd = float(tool_cost.get("cost_usd") or 0)
                if cost_usd <= 0:
                    continue
                subcall_index += 1
                subcalls.append({
                    "kind": "tool",
                    "index": subcall_index,
                    "tool_name": tool_cost.get("tool_name") or TOOL_NAME,
                    "cost_usd": cost_usd,
                })

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
                usage = normalize_usage(event.get("usage"))
                add_usage(total_usage, usage)
                subcall_index += 1
                subcalls.append({
                    "kind": "llm_turn",
                    "index": subcall_index,
                    "stop_reason": event["stop_reason"],
                    **usage,
                })
    except Exception as e:
        logger.warning("Final nudge failed: %s", e)

    metadata = {**total_usage, "model": active_model}
    if subcalls:
        metadata["subcalls"] = subcalls
    yield {
        "type": "done",
        "content": full_text,
        "blocks": all_blocks if all_blocks else None,
        "metadata": metadata,
    }

# ---------------------------------------------------------------------------
# Title generation
# ---------------------------------------------------------------------------


async def generate_title(content: str, pool: asyncpg.Pool) -> str:
    """Generate a short title for a chat session from the first message."""
    prompt = await tr.resolve(pool, "chat.title_generation", {"message_content": content[:500]})

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


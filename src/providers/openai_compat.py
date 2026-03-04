"""OpenAI-compatible provider (Nebius, etc.)."""

import json
import logging
import re
import uuid
from typing import Any, AsyncGenerator

from openai import AsyncOpenAI, APIError

from .base import LLMProvider, inject_timestamp, truncate_tool_output

logger = logging.getLogger("ibhelm.chat.provider.openai_compat")

TOOL_NAME = "run_python"

# Regex to extract tool calls leaked into reasoning_content by models like Kimi-K2.5.
# Format: <|tool_call_argument_begin|> {"code": "..."} <|tool_call_end|>
_LEAKED_TOOL_CALL_RE = re.compile(
    r'<\|tool_call_argument_begin\|>\s*(\{.*?\})\s*<\|tool_call_end\|>',
    re.DOTALL,
)


def _parse_leaked_tool_calls(reasoning: str) -> list[dict]:
    """Extract tool calls that leaked into reasoning content as raw model tokens."""
    if '<|tool_calls_section_begin|>' not in reasoning:
        return []
    results = []
    for match in _LEAKED_TOOL_CALL_RE.finditer(reasoning):
        try:
            args = json.loads(match.group(1))
            results.append({
                "id": f"leaked_{uuid.uuid4().hex[:12]}",
                "name": TOOL_NAME,
                "input": args,
            })
        except json.JSONDecodeError:
            continue
    if results:
        logger.info("Recovered %d tool call(s) from reasoning content", len(results))
    return results


class OpenAICompatProvider(LLMProvider):
    def __init__(self, api_key: str, base_url: str):
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    def build_system_prompt(self, static: str, dynamic: str) -> str:
        return f"{static}\n\n{dynamic}"

    def build_tool_definition(self, name: str, description: str, input_schema: dict) -> dict:
        return {
            "type": "function",
            "function": {"name": name, "description": description, "parameters": input_schema},
        }

    def build_api_messages(self, db_messages: list[dict]) -> list[dict]:
        api_msgs = []
        for msg in db_messages:
            role = msg["role"]
            content = msg.get("content") or ""
            blocks = msg.get("blocks")

            if role == "user":
                api_msgs.append({"role": "user", "content": inject_timestamp(content, msg.get("created_at"))})
                continue

            if not blocks:
                if content:
                    api_msgs.append({"role": "assistant", "content": content})
                continue

            if not blocks[0].get("type"):
                self._build_legacy_messages(api_msgs, blocks, content)
                continue

            self._build_blocks_messages(api_msgs, blocks, content)
        return api_msgs

    def _build_blocks_messages(self, api_msgs: list, blocks: list, final_content: str):
        """Convert neutral blocks to OpenAI assistant + tool messages."""
        i = 0
        while i < len(blocks):
            text_parts = []
            tool_calls_in_turn = []

            while i < len(blocks):
                b = blocks[i]
                if b["type"] == "text":
                    text_parts.append(b["text"])
                    i += 1
                elif b["type"] == "tool_call":
                    tool_calls_in_turn.append(b)
                    i += 1
                    while i < len(blocks) and blocks[i]["type"] == "tool_call":
                        tool_calls_in_turn.append(blocks[i])
                        i += 1
                    break
                else:
                    i += 1

            if text_parts or tool_calls_in_turn:
                msg: dict = {"role": "assistant"}
                msg["content"] = "\n".join(text_parts) if text_parts else None
                if tool_calls_in_turn:
                    msg["tool_calls"] = [
                        {
                            "id": tc["id"], "type": "function",
                            "function": {
                                "name": TOOL_NAME,
                                "arguments": json.dumps({"code": tc["code"]}),
                            },
                        }
                        for tc in tool_calls_in_turn
                    ]
                api_msgs.append(msg)

            for tc in tool_calls_in_turn:
                result_text = self._extract_text_result(tc)
                api_msgs.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": truncate_tool_output(result_text),
                })

        if final_content and (not blocks or blocks[-1]["type"] != "text"):
            api_msgs.append({"role": "assistant", "content": final_content})

    def _build_legacy_messages(self, api_msgs: list, blocks: list, final_content: str):
        """Handle legacy format: [{id, code, result, error}, ...]."""
        for tc in blocks:
            api_msgs.append({
                "role": "assistant", "content": None,
                "tool_calls": [{
                    "id": tc["id"], "type": "function",
                    "function": {"name": TOOL_NAME, "arguments": json.dumps({"code": tc["code"]})},
                }],
            })
            result_text = tc.get("result") or tc.get("error") or ""
            api_msgs.append({"role": "tool", "tool_call_id": tc["id"],
                             "content": truncate_tool_output(result_text)})
        if final_content:
            api_msgs.append({"role": "assistant", "content": final_content})

    @staticmethod
    def _extract_text_result(tc: dict) -> str:
        raw = tc.get("result") or tc.get("error") or ""
        if isinstance(raw, str):
            return raw
        if isinstance(raw, list):
            return "\n".join(p.get("text", "") for p in raw if p.get("type") == "text")
        return str(raw)

    def append_assistant_turn(self, messages: list[dict], text: str, tool_calls: list[dict]):
        msg: dict = {"role": "assistant", "content": text or None}
        if tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc["id"], "type": "function",
                    "function": {"name": tc["name"], "arguments": json.dumps(tc["input"])},
                }
                for tc in tool_calls
            ]
        messages.append(msg)

    def append_tool_results(self, messages: list[dict], results: list[dict]):
        for r in results:
            messages.append({
                "role": "tool",
                "tool_call_id": r["tool_use_id"],
                "content": truncate_tool_output(r["content"]),
            })

    async def stream_turn(
        self,
        messages: list[dict],
        system: Any,
        tools: list,
        model_id: str,
        max_tokens: int,
        thinking_config: dict | None = None,
    ) -> AsyncGenerator[dict, None]:
        all_messages = [{"role": "system", "content": system}] + messages

        kwargs: dict = {
            "model": model_id,
            "messages": all_messages,
            "tools": tools,
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        try:
            stream = await self.client.chat.completions.create(**kwargs)
        except APIError as e:
            logger.error("OpenAI-compat API error: %s", e)
            yield {"type": "turn_end", "stop_reason": "error", "error": str(e),
                   "tool_calls": [], "usage": _empty_usage()}
            return

        tool_calls_map: dict[int, dict] = {}
        finish_reason = None
        usage = None
        reasoning_buffer = ""

        try:
            async for chunk in stream:
                if chunk.usage:
                    usage = {
                        "input_tokens": chunk.usage.prompt_tokens or 0,
                        "output_tokens": chunk.usage.completion_tokens or 0,
                        "cache_read_input_tokens": 0,
                        "cache_creation_input_tokens": 0,
                    }

                if not chunk.choices:
                    continue

                choice = chunk.choices[0]
                delta = choice.delta

                if delta.content:
                    yield {"type": "text", "content": delta.content}

                if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                    reasoning_buffer += delta.reasoning_content
                    yield {"type": "thinking", "content": delta.reasoning_content}

                if delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        idx = tc_delta.index
                        if idx not in tool_calls_map:
                            tool_calls_map[idx] = {"id": "", "name": "", "arguments": ""}
                        entry = tool_calls_map[idx]
                        if tc_delta.id:
                            entry["id"] = tc_delta.id
                        if tc_delta.function:
                            if tc_delta.function.name:
                                entry["name"] = tc_delta.function.name
                            if tc_delta.function.arguments:
                                entry["arguments"] += tc_delta.function.arguments

                if choice.finish_reason:
                    finish_reason = choice.finish_reason
        except APIError as e:
            logger.error("OpenAI-compat streaming error: %s", e)
            yield {"type": "turn_end", "stop_reason": "error", "error": str(e),
                   "tool_calls": [], "usage": usage or _empty_usage()}
            return

        tool_calls = []
        for idx in sorted(tool_calls_map.keys()):
            tc = tool_calls_map[idx]
            try:
                parsed_input = json.loads(tc["arguments"]) if tc["arguments"] else {}
            except json.JSONDecodeError:
                parsed_input = {"code": tc["arguments"]}
            tool_calls.append({"id": tc["id"], "name": tc["name"], "input": parsed_input})

        # Fallback: recover tool calls leaked into reasoning content
        if not tool_calls and finish_reason != "tool_calls" and reasoning_buffer:
            leaked = _parse_leaked_tool_calls(reasoning_buffer)
            if leaked:
                tool_calls = leaked
                finish_reason = "tool_calls"

        stop = "tool_use" if finish_reason == "tool_calls" else "end_turn"
        yield {"type": "turn_end", "stop_reason": stop,
               "tool_calls": tool_calls, "usage": usage or _empty_usage()}

    async def generate_simple(self, model_id: str, prompt: str, max_tokens: int) -> str:
        resp = await self.client.chat.completions.create(
            model=model_id, max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return (resp.choices[0].message.content or "").strip()


def _empty_usage() -> dict:
    return {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}

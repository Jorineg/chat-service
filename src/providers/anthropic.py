"""Anthropic Claude provider."""

import logging
from typing import Any, AsyncGenerator

import anthropic

from .base import LLMProvider, inject_timestamp, inject_file_context, truncate_tool_output

logger = logging.getLogger("ibhelm.chat.provider.anthropic")

TOOL_NAME = "run_python"


class AnthropicProvider(LLMProvider):
    def __init__(self, api_key: str):
        self.client = anthropic.AsyncAnthropic(api_key=api_key)

    def build_system_prompt(self, static: str, dynamic: str) -> list[dict]:
        return [
            {"type": "text", "text": static, "cache_control": {"type": "ephemeral", "ttl": "1h"}},
            {"type": "text", "text": dynamic},
        ]

    def build_tool_definition(self, name: str, description: str, input_schema: dict) -> dict:
        return {"name": name, "description": description, "input_schema": input_schema}

    def build_api_messages(self, db_messages: list[dict]) -> list[dict]:
        api_msgs = []
        for msg in db_messages:
            role = msg["role"]
            content = msg.get("content") or ""
            blocks = msg.get("blocks")

            if role == "user":
                text = inject_file_context(content, msg.get("files") or [])
                api_msgs.append({"role": "user", "content": inject_timestamp(text, msg.get("created_at"))})
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
        """Convert neutral blocks to Anthropic assistant turns + tool results."""
        i = 0
        while i < len(blocks):
            assistant_content = []
            while i < len(blocks):
                b = blocks[i]
                if b["type"] == "text":
                    assistant_content.append({"type": "text", "text": b["text"]})
                    i += 1
                elif b["type"] == "tool_call":
                    assistant_content.append({
                        "type": "tool_use", "id": b["id"],
                        "name": TOOL_NAME, "input": {"code": b["code"]}
                    })
                    i += 1
                    while i < len(blocks) and blocks[i]["type"] == "tool_call":
                        tb = blocks[i]
                        assistant_content.append({
                            "type": "tool_use", "id": tb["id"],
                            "name": TOOL_NAME, "input": {"code": tb["code"]}
                        })
                        i += 1
                    break
                else:
                    i += 1

            if assistant_content:
                api_msgs.append({"role": "assistant", "content": assistant_content})

            tool_uses = [c for c in assistant_content if c["type"] == "tool_use"]
            if tool_uses:
                results = []
                for tu in tool_uses:
                    tc = next((b for b in blocks if b.get("id") == tu["id"]), {})
                    result_text = self._extract_text_result(tc)
                    results.append({
                        "type": "tool_result",
                        "tool_use_id": tu["id"],
                        "content": truncate_tool_output(result_text),
                    })
                api_msgs.append({"role": "user", "content": results})

        if final_content and (not blocks or blocks[-1]["type"] != "text"):
            api_msgs.append({"role": "assistant", "content": final_content})

    def _build_legacy_messages(self, api_msgs: list, blocks: list, final_content: str):
        """Handle legacy format: [{id, code, result, error}, ...]."""
        for tc in blocks:
            api_msgs.append({"role": "assistant", "content": [
                {"type": "tool_use", "id": tc["id"], "name": TOOL_NAME,
                 "input": {"code": tc["code"]}}
            ]})
            result_text = tc.get("result") or tc.get("error") or ""
            api_msgs.append({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": tc["id"],
                 "content": truncate_tool_output(result_text)}
            ]})
        if final_content:
            api_msgs.append({"role": "assistant", "content": final_content})

    @staticmethod
    def _extract_text_result(tc: dict) -> str:
        """Extract text from a tool call result (string or content parts array)."""
        raw = tc.get("result") or tc.get("error") or ""
        if isinstance(raw, str):
            return raw
        if isinstance(raw, list):
            return "\n".join(p.get("text", "") for p in raw if p.get("type") == "text")
        return str(raw)

    def append_assistant_turn(self, messages: list[dict], text: str, tool_calls: list[dict]):
        content: list[dict] = []
        if text:
            content.append({"type": "text", "text": text})
        for tc in tool_calls:
            content.append({
                "type": "tool_use", "id": tc["id"],
                "name": tc["name"], "input": tc["input"],
            })
        messages.append({"role": "assistant", "content": content})

    def append_tool_results(self, messages: list[dict], results: list[dict]):
        tool_results = []
        for r in results:
            content = r["content"]
            if isinstance(content, list):
                parts = []
                for block in content:
                    if block["type"] == "text":
                        parts.append({"type": "text", "text": truncate_tool_output(block["text"])})
                    elif block["type"] == "image":
                        parts.append({
                            "type": "image",
                            "source": {"type": "base64", "media_type": block["media_type"],
                                       "data": block["base64"]},
                        })
                tool_results.append({"type": "tool_result", "tool_use_id": r["tool_use_id"],
                                     "content": parts})
            else:
                tool_results.append({"type": "tool_result", "tool_use_id": r["tool_use_id"],
                                     "content": truncate_tool_output(content)})
        messages.append({"role": "user", "content": tool_results})

    async def stream_turn(
        self,
        messages: list[dict],
        system: Any,
        tools: list,
        model_id: str,
        max_tokens: int,
        thinking_config: dict | None = None,
    ) -> AsyncGenerator[dict, None]:
        kwargs: dict = {
            "model": model_id,
            "max_tokens": max_tokens,
            "system": system,
            "tools": tools,
            "messages": messages,
        }
        if thinking_config:
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": thinking_config["budget_tokens"]}

        try:
            async with self.client.messages.stream(**kwargs) as stream:
                async for event in stream:
                    if event.type == "content_block_delta":
                        if hasattr(event.delta, 'text'):
                            yield {"type": "text", "content": event.delta.text}
                        elif hasattr(event.delta, 'thinking'):
                            yield {"type": "thinking", "content": event.delta.thinking}

                response = await stream.get_final_message()
        except anthropic.APIError as e:
            logger.error("Anthropic API error: %s", e)
            yield {"type": "turn_end", "stop_reason": "error", "error": str(e),
                   "tool_calls": [], "usage": _empty_usage()}
            return

        tool_calls = [
            {"id": b.id, "name": b.name, "input": b.input}
            for b in response.content if b.type == "tool_use"
        ]

        usage = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "cache_read_input_tokens": getattr(response.usage, 'cache_read_input_tokens', 0) or 0,
            "cache_creation_input_tokens": getattr(response.usage, 'cache_creation_input_tokens', 0) or 0,
        }

        stop = "tool_use" if response.stop_reason == "tool_use" else "end_turn"
        yield {"type": "turn_end", "stop_reason": stop, "tool_calls": tool_calls, "usage": usage}

    async def generate_simple(self, model_id: str, prompt: str, max_tokens: int) -> str:
        resp = await self.client.messages.create(
            model=model_id, max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip()


def _empty_usage() -> dict:
    return {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}

"""Abstract base for LLM providers.

Each provider converts between the neutral DB format and its wire format,
and handles streaming API calls.

Neutral block format (stored in chat_messages.blocks JSONB):
  {type: "text", text: "..."}
  {type: "tool_call", id: "...", code: "...", result: "..." | [{type:"text",...},{type:"image",...}], error: "..."}
  {type: "thinking", text: "..."}

Normalized stream events yielded by stream_turn():
  {type: "text", content: "delta"}
  {type: "thinking", content: "delta"}
  {type: "turn_end", stop_reason: "end_turn"|"tool_use", tool_calls: [{id,name,input}], usage: {...}}
"""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, AsyncGenerator


OUTPUT_TRUNCATION_THRESHOLD = 12_000
OUTPUT_HEAD_CHARS = 4_000
OUTPUT_TAIL_CHARS = 2_000


def truncate_tool_output(output: str) -> str:
    """Truncate long tool output, keeping head + tail with an explanation."""
    if len(output) <= OUTPUT_TRUNCATION_THRESHOLD:
        return output
    total = len(output)
    return (
        f"{output[:OUTPUT_HEAD_CHARS]}\n\n"
        f"--- OUTPUT TRUNCATED ({total:,} chars total, showing first {OUTPUT_HEAD_CHARS:,} + last {OUTPUT_TAIL_CHARS:,}) ---\n"
        f"Your variables are still in scope. Use targeted print() with slicing to inspect specific parts.\n\n"
        f"{output[-OUTPUT_TAIL_CHARS:]}"
    )


def inject_timestamp(content: str, created_at: str | None = None) -> str:
    """Append a system timestamp to a user message for temporal context."""
    if created_at:
        try:
            dt = datetime.fromisoformat(created_at)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            ts = dt.strftime("[%Y-%m-%d %H:%M UTC]")
        except (ValueError, TypeError):
            ts = datetime.now(timezone.utc).strftime("[%Y-%m-%d %H:%M UTC]")
    else:
        ts = datetime.now(timezone.utc).strftime("[%Y-%m-%d %H:%M UTC]")
    return f"{content}\n{ts}"


class LLMProvider(ABC):
    """Abstract LLM provider. Subclasses handle Anthropic, OpenAI-compatible APIs, etc."""

    @abstractmethod
    def build_system_prompt(self, static: str, dynamic: str) -> Any:
        """Build provider-specific system prompt representation."""

    @abstractmethod
    def build_api_messages(self, db_messages: list[dict]) -> list[dict]:
        """Convert neutral DB messages to provider-specific API message format."""

    @abstractmethod
    def build_tool_definition(self, name: str, description: str, input_schema: dict) -> Any:
        """Build provider-specific tool definition."""

    @abstractmethod
    def append_assistant_turn(self, messages: list[dict], text: str, tool_calls: list[dict]) -> None:
        """Append an assistant turn (with optional tool calls) to the API messages list."""

    @abstractmethod
    def append_tool_results(self, messages: list[dict], results: list[dict]) -> None:
        """Append tool results to the API messages list.

        results: [{tool_use_id: str, content: str}]
        """

    @abstractmethod
    async def stream_turn(
        self,
        messages: list[dict],
        system: Any,
        tools: list,
        model_id: str,
        max_tokens: int,
        thinking_config: dict | None = None,
    ) -> AsyncGenerator[dict, None]:
        """Stream one API turn. Yields normalized events:
          {type: "text", content: "delta"}
          {type: "thinking", content: "delta"}
          {type: "turn_end", stop_reason: "end_turn"|"tool_use",
           tool_calls: [{id, name, input}], usage: {input_tokens, output_tokens, ...}}
        """
        yield  # type: ignore

    @abstractmethod
    async def generate_simple(self, model_id: str, prompt: str, max_tokens: int) -> str:
        """Simple non-streaming completion for utility tasks (title generation, etc.)."""

"""LLM provider abstraction for multi-provider chat support."""

from .base import LLMProvider
from .anthropic import AnthropicProvider
from .openai_compat import OpenAICompatProvider

__all__ = ["LLMProvider", "AnthropicProvider", "OpenAICompatProvider", "get_provider"]


def get_provider(model_config: dict) -> LLMProvider:
    """Instantiate the correct provider based on model config."""
    from .. import settings

    provider_type = model_config.get("provider", "anthropic")
    if provider_type == "anthropic":
        return AnthropicProvider(api_key=settings.ANTHROPIC_API_KEY)
    if provider_type == "nebius":
        return OpenAICompatProvider(
            api_key=settings.NEBIUS_API_KEY,
            base_url=model_config.get("base_url", "https://api.tokenfactory.nebius.com/v1/"),
        )
    raise ValueError(f"Unknown provider: {provider_type}")

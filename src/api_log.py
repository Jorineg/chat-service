"""Rotating file logger for raw LLM API requests and responses."""

import json
import logging
import os
import time
from logging.handlers import RotatingFileHandler

LOG_DIR = os.getenv("API_LOG_DIR", "/var/log/chat-service")
LOG_FILE = os.path.join(LOG_DIR, "api_calls.jsonl")
RAW_LOG_FILE = os.path.join(LOG_DIR, "raw_chunks.jsonl")
MAX_BYTES = 50 * 1024 * 1024  # 50 MB per file
BACKUP_COUNT = 5
RAW_LOG_ENABLED = os.getenv("RAW_LOG_ENABLED", "").lower() in ("1", "true", "yes")

os.makedirs(LOG_DIR, exist_ok=True)

_logger = logging.getLogger("ibhelm.chat.api_log")
_logger.setLevel(logging.DEBUG)
_logger.propagate = False

_handler = RotatingFileHandler(LOG_FILE, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT)
_handler.setFormatter(logging.Formatter("%(message)s"))
_logger.addHandler(_handler)

_raw_logger: logging.Logger | None = None
if RAW_LOG_ENABLED:
    _raw_logger = logging.getLogger("ibhelm.chat.raw_log")
    _raw_logger.setLevel(logging.DEBUG)
    _raw_logger.propagate = False
    _raw_handler = RotatingFileHandler(RAW_LOG_FILE, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT)
    _raw_handler.setFormatter(logging.Formatter("%(message)s"))
    _raw_logger.addHandler(_raw_handler)


def _safe_json(obj) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        return repr(obj)


def log_raw(tag: str, data):
    """Log raw network-level data (SSE chunks, request bodies) to raw_chunks.jsonl.
    Only active when RAW_LOG_ENABLED=1 env var is set."""
    if _raw_logger:
        _raw_logger.debug(_safe_json({"ts": time.time(), "tag": tag, "data": data}))


def log_request(provider: str, model: str, messages: list, system=None, tools: list | None = None, **extra):
    record = {"dir": "req", "provider": provider, "model": model,
              "system": system, "messages": messages, "tools": tools}
    record.update(extra)
    _logger.debug(_safe_json(record))


def log_response(provider: str, model: str, *, content: str | None = None,
                 tool_calls: list | None = None, usage: dict | None = None,
                 error: str | None = None, reasoning: str | None = None,
                 **extra):
    record = {"dir": "resp", "provider": provider, "model": model,
              "content": content, "tool_calls": tool_calls, "usage": usage,
              "error": error}
    if reasoning:
        record["reasoning"] = reasoning
    record.update(extra)
    _logger.debug(_safe_json(record))

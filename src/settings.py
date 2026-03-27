"""Configuration for Chat Service."""
import os
from dotenv import load_dotenv

load_dotenv()

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8200"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

DATABASE_URL = os.getenv("DATABASE_URL")
SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_INTERNAL_URL = os.getenv("SUPABASE_INTERNAL_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
CHAT_FILES_BUCKET = os.getenv("CHAT_FILES_BUCKET", "chat-files")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
NEBIUS_API_KEY = os.getenv("NEBIUS_API_KEY")
LINKUP_API_KEY = os.getenv("LINKUP_API_KEY")

CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514")

MAX_TOOL_ITERATIONS = int(os.getenv("MAX_TOOL_ITERATIONS", "15"))
ENABLE_THINKING = os.getenv("ENABLE_THINKING", "false").lower() == "true"
THINKING_BUDGET = int(os.getenv("THINKING_BUDGET", "10000"))

AGENT_CHECK_INTERVAL_S = int(os.getenv("AGENT_CHECK_INTERVAL_S", "3600"))
AGENT_ENABLED = os.getenv("AGENT_ENABLED", "true").lower() == "true"

BETTERSTACK_SOURCE_TOKEN = os.getenv("BETTERSTACK_SOURCE_TOKEN")
BETTERSTACK_INGEST_HOST = os.getenv("BETTERSTACK_INGEST_HOST")


def validate():
    errors = []
    if not DATABASE_URL:
        errors.append("DATABASE_URL is required")
    if not SUPABASE_JWT_SECRET:
        errors.append("SUPABASE_JWT_SECRET is required")
    if not ANTHROPIC_API_KEY and not NEBIUS_API_KEY:
        errors.append("At least one of ANTHROPIC_API_KEY or NEBIUS_API_KEY is required")
    if not SUPABASE_URL:
        errors.append("SUPABASE_URL is required")
    if not SUPABASE_SERVICE_KEY:
        errors.append("SUPABASE_SERVICE_KEY is required")
    if errors:
        raise ValueError("Config errors:\n  " + "\n  ".join(errors))

"""Template resolver — thin wrapper over the DB function resolve_prompt_template().

All resolution logic (includes, variables, SQL directives) lives in plpgsql.
This module sets the appropriate DB role before calling so that RLS applies:
  - system mode (default): mcp_readonly — sees only public/project data
  - user mode: authenticated + user context — sees user-specific data via RLS
"""

import json
import logging

import asyncpg

logger = logging.getLogger("ibhelm.chat.resolver")

_ALLOWED_ROLES = frozenset({'mcp_readonly', 'authenticated'})


async def resolve(
    pool: asyncpg.Pool,
    template_id: str,
    runtime_vars: dict[str, str] | None = None,
    *,
    role: str = 'mcp_readonly',
    user_email: str | None = None,
    user_id: str | None = None,
) -> str:
    """Resolve a stored template by ID with all directives expanded."""
    if role not in _ALLOWED_ROLES:
        raise ValueError(f"Invalid role: {role}")
    vars_json = json.dumps(runtime_vars) if runtime_vars else '{}'
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(f"SET LOCAL ROLE {role}")
            if user_email:
                await conn.execute("SELECT set_config('app.user_email', $1, true)", user_email)
            if user_id:
                await conn.execute("SELECT set_config('app.user_id', $1, true)", user_id)
            result = await conn.fetchval(
                "SELECT resolve_prompt_template($1, $2::jsonb)", template_id, vars_json
            )
    return result or ''


async def resolve_raw(
    pool: asyncpg.Pool,
    content: str,
    runtime_vars: dict[str, str] | None = None,
    *,
    role: str = 'mcp_readonly',
    user_email: str | None = None,
    user_id: str | None = None,
) -> str:
    """Resolve directives in arbitrary text (not from a stored template)."""
    if role not in _ALLOWED_ROLES:
        raise ValueError(f"Invalid role: {role}")
    vars_json = json.dumps(runtime_vars) if runtime_vars else '{}'
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(f"SET LOCAL ROLE {role}")
            if user_email:
                await conn.execute("SELECT set_config('app.user_email', $1, true)", user_email)
            if user_id:
                await conn.execute("SELECT set_config('app.user_id', $1, true)", user_id)
            result = await conn.fetchval(
                "SELECT resolve_prompt_template_raw($1, $2::jsonb)", content, vars_json
            )
    return result or ''

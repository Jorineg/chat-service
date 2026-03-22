"""Template resolver for prompt_templates.

Resolves three directive types in order:
1. {{include:template_id}} — recursive composition (cycle detection, max depth 10)
2. {runtime_var} — replaced from caller-provided dict
3. {{sql:query |||prefix|||fallback}} — execute read-only SQL, insert result
"""

import asyncio
import json
import logging
import re
import time
from datetime import date, datetime, time as time_type, timedelta

import asyncpg

logger = logging.getLogger("ibhelm.chat.resolver")

_INCLUDE_RE = re.compile(r'\{\{include:([^}]+)\}\}')
_SQL_RE = re.compile(r'\{\{sql:(.*?)\}\}', re.DOTALL)

# ---------------------------------------------------------------------------
# Cache: all templates loaded in bulk, invalidated via LISTEN/NOTIFY
# ---------------------------------------------------------------------------

_cache: dict[str, str] = {}
_cache_ts: float = 0.0
_CACHE_TTL = 300.0  # fallback TTL if LISTEN fails
_listener_task: asyncio.Task | None = None


async def _load_cache(pool: asyncpg.Pool):
    global _cache, _cache_ts
    rows = await pool.fetch("SELECT id, content FROM prompt_templates")
    _cache = {r["id"]: r["content"] for r in rows}
    _cache_ts = time.time()
    logger.info("Template cache loaded: %d templates", len(_cache))


async def _ensure_cache(pool: asyncpg.Pool):
    if not _cache or time.time() - _cache_ts > _CACHE_TTL:
        await _load_cache(pool)


async def invalidate_cache():
    global _cache, _cache_ts
    _cache.clear()
    _cache_ts = 0.0


async def start_listener(pool: asyncpg.Pool):
    """Start LISTEN/NOTIFY listener for cache invalidation."""
    global _listener_task
    if _listener_task and not _listener_task.done():
        return

    async def _listen():
        while True:
            try:
                conn = await pool.acquire()
                try:
                    await conn.add_listener('prompt_templates_changed', _on_notify)
                    logger.info("LISTEN prompt_templates_changed active")
                    while not conn.is_closed():
                        await asyncio.sleep(60)
                finally:
                    await pool.release(conn)
            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.warning("LISTEN reconnect in 5s: %s", e)
                await asyncio.sleep(5)

    _listener_task = asyncio.create_task(_listen())


def _on_notify(conn, pid, channel, payload):
    logger.info("Template changed: %s — invalidating cache", payload)
    _cache.clear()
    global _cache_ts
    _cache_ts = 0.0


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------

async def resolve(
    pool: asyncpg.Pool,
    template_id: str,
    runtime_vars: dict[str, str] | None = None,
) -> str:
    """Resolve a template by ID with all directives expanded."""
    await _ensure_cache(pool)
    content = _cache.get(template_id)
    if content is None:
        await _load_cache(pool)
        content = _cache.get(template_id)
    if content is None:
        return f'{{{{error: template "{template_id}" not found}}}}'

    content = _resolve_includes(content, set())
    if runtime_vars:
        content = _resolve_vars(content, runtime_vars)
    content = await _resolve_sql(pool, content)
    return content


async def resolve_raw(
    pool: asyncpg.Pool,
    content: str,
    runtime_vars: dict[str, str] | None = None,
) -> str:
    """Resolve directives in arbitrary text (not from a stored template)."""
    await _ensure_cache(pool)
    content = _resolve_includes(content, set())
    if runtime_vars:
        content = _resolve_vars(content, runtime_vars)
    content = await _resolve_sql(pool, content)
    return content


# ---------------------------------------------------------------------------
# Step 1: {{include:template_id}}
# ---------------------------------------------------------------------------

def _resolve_includes(content: str, visited: set[str], depth: int = 0) -> str:
    if depth > 10:
        return content

    def _replacer(m: re.Match) -> str:
        tid = m.group(1).strip()
        if tid in visited:
            return f'{{{{error: circular reference "{tid}"}}}}'
        included = _cache.get(tid)
        if included is None:
            return f'{{{{error: template "{tid}" not found}}}}'
        return _resolve_includes(included, visited | {tid}, depth + 1)

    return _INCLUDE_RE.sub(_replacer, content)


# ---------------------------------------------------------------------------
# Step 2: {runtime_var}
# ---------------------------------------------------------------------------

def _resolve_vars(content: str, variables: dict[str, str]) -> str:
    for key, val in variables.items():
        content = content.replace(f'{{{key}}}', str(val))
    return content


# ---------------------------------------------------------------------------
# Step 3: {{sql:query |||prefix|||fallback}}
# ---------------------------------------------------------------------------

async def _resolve_sql(pool: asyncpg.Pool, content: str) -> str:
    matches = list(_SQL_RE.finditer(content))
    if not matches:
        return content

    # Process from last to first to preserve positions
    for m in reversed(matches):
        block = m.group(1).strip()
        parts = block.split('|||')
        query = parts[0].strip()
        prefix = parts[1].strip() if len(parts) >= 2 else None
        fallback = parts[2].strip() if len(parts) >= 3 else None

        result = await _exec_sql(pool, query)

        if result is not None and result != '':
            if prefix:
                result = prefix + '\n' + result
        else:
            result = fallback or ''

        content = content[:m.start()] + result + content[m.end():]

    return content


async def _exec_sql(pool: asyncpg.Pool, query: str) -> str | None:
    """Execute a read-only SQL query, return result as text."""
    q = query.strip().rstrip(';')
    q_upper = q.upper().lstrip()
    if not (q_upper.startswith('SELECT') or q_upper.startswith('WITH')):
        return '{{sql_error: only SELECT/WITH queries allowed}}'

    try:
        rows = await pool.fetch(q)
    except Exception as e:
        logger.warning("SQL directive failed: %s — %s", e, q[:200])
        return f'{{{{sql_error: {e}}}}}'

    if not rows:
        return None

    cols = list(rows[0].keys())

    # Single scalar
    if len(rows) == 1 and len(cols) == 1:
        return _format_value(rows[0][cols[0]])

    # Multiple rows — TOON-style table (no limits for prompt context)
    return _format_table(rows, cols)


def _format_value(v) -> str | None:
    if v is None:
        return None
    if isinstance(v, datetime):
        return v.strftime('%Y-%m-%d %H:%M')
    if isinstance(v, date):
        return v.isoformat()
    if isinstance(v, time_type):
        return v.strftime('%H:%M')
    if isinstance(v, timedelta):
        total = int(v.total_seconds())
        h, rem = divmod(total, 3600)
        m, s = divmod(rem, 60)
        return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"
    return str(v)


def _format_table(rows: list, cols: list[str]) -> str:
    """Format rows as TOON table (no limits — full content for prompts)."""
    header = f"rows[{len(rows)}]{{{','.join(cols)}}}:"
    lines = [header]
    for row in rows:
        cells = []
        for c in cols:
            v = row[c]
            if v is None:
                cells.append('∅')
            elif isinstance(v, bool):
                cells.append('T' if v else 'F')
            elif isinstance(v, (datetime, date)):
                cells.append(_format_value(v))
            elif isinstance(v, str):
                v = v.replace('\n', '↵').replace('\t', '→').replace('\r', '')
                if ',' in v or '"' in v:
                    v = '"' + v.replace('"', '""') + '"'
                cells.append(v)
            else:
                cells.append(str(v))
        lines.append('  ' + ','.join(cells))
    return '\n'.join(lines)

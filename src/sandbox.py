"""
Python execution sandbox for the chat AI agent.

Single tool: run_python. The AI writes Python code using helper functions:
  - db(sql)           → Execute read-only SQL, returns list of dicts
  - fmt(rows)         → Format rows as compact text table
  - Standard: math, json, re, datetime, timedelta, date, Counter, defaultdict
"""

import asyncio
import logging
import math
import json as json_module
import re as re_module
import sys
import threading
from collections import Counter, defaultdict
from datetime import datetime, timedelta, date
from io import StringIO

import asyncpg

logger = logging.getLogger("ibhelm.chat.sandbox")

MAX_QUERIES = 10
QUERY_TIMEOUT_S = 10
EXEC_TIMEOUT_S = 15
MAX_OUTPUT_CHARS = 50_000

SAFE_BUILTINS = {
    'abs': abs, 'all': all, 'any': any, 'bin': bin, 'bool': bool,
    'chr': chr, 'dict': dict, 'divmod': divmod, 'enumerate': enumerate,
    'filter': filter, 'float': float, 'format': format, 'frozenset': frozenset,
    'hash': hash, 'hex': hex, 'int': int, 'isinstance': isinstance,
    'issubclass': issubclass, 'iter': iter, 'len': len, 'list': list,
    'map': map, 'max': max, 'min': min, 'next': next, 'oct': oct,
    'ord': ord, 'pow': pow, 'print': print, 'range': range, 'repr': repr,
    'reversed': reversed, 'round': round, 'set': set, 'slice': slice,
    'sorted': sorted, 'str': str, 'sum': sum, 'tuple': tuple, 'type': type,
    'zip': zip, 'True': True, 'False': False, 'None': None,
    '__import__': lambda *a, **kw: (_ for _ in ()).throw(
        ImportError("Import not available. All modules (math, json, re, datetime, timedelta, date, Counter, defaultdict) are pre-loaded in scope.")),
}

BLOCKED_KEYWORDS = {'import os', 'import sys', 'import subprocess', '__import__',
                    'open(', 'exec(', 'eval(', 'compile('}


def _validate_sql(sql: str) -> str | None:
    """Returns error message if SQL is not a safe read-only query."""
    stripped = sql.strip().rstrip(';').strip()
    # Strip leading SQL comments (-- and /* */ style)
    while stripped.startswith('--'):
        stripped = stripped.split('\n', 1)[-1].strip() if '\n' in stripped else ''
    while stripped.startswith('/*'):
        end = stripped.find('*/')
        stripped = stripped[end + 2:].strip() if end >= 0 else ''
    upper = stripped.upper()
    if not (upper.startswith('SELECT') or upper.startswith('WITH')):
        return "Only SELECT/WITH queries allowed"
    for kw in ('INSERT', 'UPDATE', 'DELETE', 'DROP', 'ALTER', 'CREATE', 'TRUNCATE', 'GRANT', 'REVOKE'):
        parts = upper.split()
        if kw in parts and kw not in ('SELECT', 'WITH'):
            if kw == 'CREATE' and 'CREATE' not in upper.split('SELECT')[0]:
                continue
            return f"Write operation '{kw}' not allowed"
    return None


def _serialize_row(row: dict) -> dict:
    """Make row values JSON-serializable."""
    out = {}
    for k, v in row.items():
        if hasattr(v, 'isoformat'):
            out[k] = v.isoformat()
        elif isinstance(v, (bytes, bytearray)):
            out[k] = f"<{len(v)} bytes>"
        else:
            out[k] = v
    return out


def _fmt_table(rows: list[dict], max_rows: int | None = 50, max_cell: int | None = 80) -> str:
    """Format rows as compact TOON (Token-Oriented Object Notation) table.

    Format: rows[count]{col1,col2,...}:
              val1,val2,...
    Set max_rows=None or max_cell=None to disable respective truncation.
    """
    if not rows:
        return "rows[0]{}: (empty)"
    fields = list(rows[0].keys())
    display = rows[:max_rows] if max_rows else rows
    header = f"rows[{len(rows)}]{{{','.join(fields)}}}:"
    lines = [header]
    for row in display:
        cells = []
        for f in fields:
            v = row.get(f)
            if v is None:
                cells.append("∅")
            elif isinstance(v, bool):
                cells.append("T" if v else "F")
            elif isinstance(v, str):
                v = v.replace('\n', '↵').replace('\t', '→').replace('\r', '')
                if max_cell and len(v) > max_cell:
                    v = v[:max_cell - 3] + '...'
                if ',' in v or '"' in v:
                    v = '"' + v.replace('"', '""') + '"'
                cells.append(v)
            else:
                s = str(v)
                if max_cell and len(s) > max_cell:
                    s = s[:max_cell - 3] + '...'
                cells.append(s)
        lines.append("  " + ",".join(cells))
    if max_rows and len(rows) > max_rows:
        lines.append(f"  ...({len(rows) - max_rows} more)")
    return '\n'.join(lines)


def create_sandbox_env() -> dict:
    """Create a fresh sandbox environment with safe builtins and standard modules."""
    return {
        '__builtins__': SAFE_BUILTINS,
        'math': math,
        'json': json_module,
        're': re_module,
        'datetime': datetime,
        'timedelta': timedelta,
        'date': date,
        'Counter': Counter,
        'defaultdict': defaultdict,
    }


async def execute_python(
    code: str,
    pool: asyncpg.Pool,
    user_email: str | None = None,
    persistent_env: dict | None = None,
) -> dict:
    """
    Execute Python code in sandbox with database access.

    If persistent_env is provided, user variables persist across calls.
    Returns: {"result": ..., "output": ..., "error": ...}
    """
    for blocked in BLOCKED_KEYWORDS:
        if blocked in code:
            return {"error": f"Blocked: '{blocked}' not allowed in sandbox"}

    query_count = 0

    async def _exec_query(sql: str) -> list[dict]:
        nonlocal query_count
        query_count += 1
        if query_count > MAX_QUERIES:
            raise RuntimeError(f"Too many queries (max {MAX_QUERIES})")
        err = _validate_sql(sql)
        if err:
            raise ValueError(err)
        async with pool.acquire() as conn:
            await conn.execute(f"SET statement_timeout = '{QUERY_TIMEOUT_S}s'")
            if user_email:
                await conn.execute("SELECT set_config('app.user_email', $1, true)", user_email)
            rows = await conn.fetch(sql)
            return [_serialize_row(dict(r)) for r in rows]

    query_cache: dict[str, list[dict]] = {}

    async def _cached_query(sql: str) -> list[dict]:
        key = ' '.join(sql.split())
        if key not in query_cache:
            query_cache[key] = await _exec_query(key)
        return query_cache[key]

    # Pre-parse and execute all db() calls from the code
    import re
    patterns = [
        r'db\s*\(\s*"""([\s\S]+?)"""\s*\)',
        r"db\s*\(\s*'''([\s\S]+?)'''\s*\)",
        r'db\s*\(\s*"(?!"")([^"]+)"\s*\)',
        r"db\s*\(\s*'(?!'')([^']+)'\s*\)",
    ]
    found_queries = []
    for pat in patterns:
        found_queries.extend(re.findall(pat, code))

    for sql in found_queries:
        try:
            await _cached_query(sql)
        except Exception as e:
            return {"error": f"Query failed: {str(e)}\nSQL: {sql[:200]}"}

    _loop = asyncio.get_running_loop()

    def db_sync(sql: str) -> list[dict]:
        key = ' '.join(sql.split())
        if key in query_cache:
            return query_cache[key]
        future = asyncio.run_coroutine_threadsafe(_cached_query(sql), _loop)
        return future.result(timeout=QUERY_TIMEOUT_S)

    env = persistent_env if persistent_env is not None else create_sandbox_env()
    env['db'] = db_sync
    env['fmt'] = _fmt_table

    try:
        def _run():
            captured = StringIO()
            old_stdout = sys.stdout
            sys.stdout = captured
            try:
                exec(compile(code, '<code>', 'exec'), env)

                # Try to get the value of the last expression
                last_val = None
                stripped_code = code.strip()

                # First: try evaluating the entire code as a single expression
                try:
                    last_val = eval(compile(stripped_code, '<code>', 'eval'), env)
                except (SyntaxError, TypeError):
                    # Not a single expression — try the last line
                    lines = stripped_code.split('\n')
                    if lines:
                        last = lines[-1].strip()
                        skip = ('#', 'import', 'from', 'def', 'class', 'if', 'for',
                                'while', 'try', 'with', 'return', 'raise', 'assert',
                                'pass', 'break', 'continue', 'print(', 'print (')
                        if last and not any(last.startswith(p) for p in skip):
                            if '=' not in last or last.count('=') == last.count('=='):
                                try:
                                    last_val = eval(last, env)
                                except Exception:
                                    pass
                return last_val, captured.getvalue()
            except Exception as e:
                return e, captured.getvalue()
            finally:
                sys.stdout = old_stdout

        result_or_exc, output = await asyncio.wait_for(
            asyncio.to_thread(_run),
            timeout=EXEC_TIMEOUT_S
        )

        if isinstance(result_or_exc, Exception):
            e = result_or_exc
            return {"error": f"{type(e).__name__}: {e}", "output": output or None}

        result = result_or_exc
        if len(output) > MAX_OUTPUT_CHARS:
            output = output[:MAX_OUTPUT_CHARS] + f"\n... (truncated, {len(output)} total chars)"

        if result is not None:
            try:
                if hasattr(result, 'isoformat'):
                    result = result.isoformat()
                elif isinstance(result, (set, frozenset)):
                    result = list(result)
                json_module.dumps(result)
            except (TypeError, ValueError):
                result = str(result)

        return {"result": result, "output": output or None}

    except asyncio.TimeoutError:
        return {"error": f"Execution timed out after {EXEC_TIMEOUT_S}s"}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}

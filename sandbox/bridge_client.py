"""Bridge client for sandbox ↔ chat service communication.

Uses a shared socket connection (owned by executor.py) to proxy requests
to the chat service. All calls are synchronous and blocking — safe because
sandbox code execution is single-threaded.
"""

import json
import math
import re
import struct

_HEADER_FMT = "!I"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)
_MAX_MSG_SIZE = 100 * 1024 * 1024


def _send(sock, msg: dict):
    data = json.dumps(msg, ensure_ascii=False, default=str).encode("utf-8")
    sock.sendall(struct.pack(_HEADER_FMT, len(data)) + data)


def _recv(sock) -> dict:
    buf = bytearray()
    while len(buf) < _HEADER_SIZE:
        chunk = sock.recv(_HEADER_SIZE - len(buf))
        if not chunk:
            raise ConnectionError("Bridge socket closed")
        buf.extend(chunk)
    length = struct.unpack(_HEADER_FMT, bytes(buf))[0]
    if length > _MAX_MSG_SIZE:
        raise RuntimeError(f"Response too large: {length} bytes")
    data = bytearray()
    while len(data) < length:
        chunk = sock.recv(min(length - len(data), 65536))
        if not chunk:
            raise ConnectionError("Bridge socket closed")
        data.extend(chunk)
    return json.loads(data.decode("utf-8"))


def _normalize_placeholders(sql: str, params: tuple) -> tuple[str, tuple]:
    """Convert psycopg2-style %s placeholders to asyncpg-style $N."""
    if '%s' not in sql:
        return sql, params
    counter = 0
    def _replace(m):
        nonlocal counter
        counter += 1
        return f'${counter}'
    converted = re.sub(r'(?<!%)%s', _replace, sql).replace('%%', '%')
    return converted, params


class BridgeClient:
    """Proxy to chat service for DB/file operations. Uses shared socket."""

    def __init__(self, sock):
        self._sock = sock
        self._tool_costs: list[dict] = []

    def _request(self, msg_type: str, payload: dict | None = None) -> dict:
        msg = {"type": msg_type}
        if payload:
            msg.update(payload)
        _send(self._sock, msg)
        resp = _recv(self._sock)
        cost = resp.get("cost_usd")
        if isinstance(cost, (int, float)) and cost > 0:
            self._tool_costs.append({
                "tool_name": resp.get("tool_name") or msg_type,
                "cost_usd": float(cost),
            })
        return resp

    def consume_tool_costs(self) -> list[dict]:
        costs = self._tool_costs
        self._tool_costs = []
        return costs

    def db(self, sql=None, *params, **kwargs) -> 'DbResult':
        sql = sql or kwargs.get("query") or kwargs.get("sql")
        if params:
            sql, params = _normalize_placeholders(sql, params)
        payload = {"sql": sql}
        if params:
            payload["params"] = list(params)
        resp = self._request("db_query", payload)
        if resp.get("error"):
            raise RuntimeError(resp["error"])
        return DbResult(resp["rows"])

    def load_skill(self, skill_id: str = "", *, id: str = "") -> '_Loaded':
        """Load a skill/doc from the template system. Always prints the content."""
        skill_id = skill_id or id
        if not skill_id:
            raise ValueError("skill_id is required")
        result = self.db("SELECT load_skill($1)", skill_id)
        text = result[0][list(result[0].keys())[0]] if result else ""
        if not text or text.startswith("Error:"):
            raise ValueError(text or f"Skill '{skill_id}' not found")
        result._printed = True
        print(text)
        return _Loaded(skill_id)

    def file_info(self, file_id=None, **kwargs) -> dict:
        file_id = file_id or kwargs.get("id") or kwargs.get("id_or_path")
        resp = self._request("file_info", {"id": file_id})
        if resp.get("error"):
            raise RuntimeError(resp["error"])
        return resp["info"]

    def download_file(self, content_hash=None, **kwargs) -> str:
        """Download a NAS file into /work/. Returns local path."""
        content_hash = content_hash or kwargs.get("hash")
        resp = self._request("download_file", {"content_hash": content_hash})
        if resp.get("error"):
            raise RuntimeError(resp["error"])
        return resp["local_path"]

    def download_craft_file(self, storage_path=None, **kwargs) -> str:
        """Download a Craft doc media file into /work/. Returns local path."""
        storage_path = storage_path or kwargs.get("path") or kwargs.get("key")
        resp = self._request("download_craft_file", {"storage_path": storage_path})
        if resp.get("error"):
            raise RuntimeError(resp["error"])
        return resp["local_path"]

    def download_url(self, file_id=None, **kwargs) -> str:
        file_id = file_id or kwargs.get("id") or kwargs.get("id_or_path")
        resp = self._request("download_url", {"id": file_id})
        if resp.get("error"):
            raise RuntimeError(resp["error"])
        return resp["url"]

    def add_activity_entry(self, project_id, logged_at=None, category=None,
                           summary=None, source_event_ids=None, kgr_codes=None,
                           involved_persons=None, **kwargs) -> str:
        payload = {
            "project_id": project_id,
            "category": category,
            "summary": summary,
        }
        if logged_at:
            payload["logged_at"] = logged_at
        if source_event_ids:
            payload["source_event_ids"] = source_event_ids
        if kgr_codes:
            payload["kgr_codes"] = kgr_codes
        if involved_persons:
            payload["involved_persons"] = involved_persons
        resp = self._request("add_activity_entry", payload)
        if resp.get("error"):
            raise RuntimeError(resp["error"])
        return resp["id"]

    def update_activity_entry(self, entry_id, summary=None, category=None,
                              kgr_codes=None, involved_persons=None,
                              append_source_event_ids=None, **kwargs) -> str:
        payload = {"entry_id": entry_id}
        if summary is not None:
            payload["summary"] = summary
        if category is not None:
            payload["category"] = category
        if kgr_codes is not None:
            payload["kgr_codes"] = kgr_codes
        if involved_persons is not None:
            payload["involved_persons"] = involved_persons
        if append_source_event_ids:
            payload["append_source_event_ids"] = append_source_event_ids
        resp = self._request("update_activity_entry", payload)
        if resp.get("error"):
            raise RuntimeError(resp["error"])
        return resp.get("message", "ok")

    def update_project_status(self, project_id, markdown) -> str:
        resp = self._request("update_project_status", {"project_id": project_id, "markdown": markdown})
        if resp.get("error"):
            raise RuntimeError(resp["error"])
        return resp.get("message", "ok")

    def update_project_profile(self, project_id, markdown) -> str:
        resp = self._request("update_project_profile", {"project_id": project_id, "markdown": markdown})
        if resp.get("error"):
            raise RuntimeError(resp["error"])
        return resp.get("message", "ok")

    def create_prompt(self, id, title, category, content, summary=None,
                      hidden=False, tags=None, prompt_role=None,
                      db_functions=None, py_functions=None, system=False) -> str:
        """Create a prompt template. system=True requires admin (sets owner_id=NULL, is_system=TRUE)."""
        payload = {"id": id, "title": title, "category": category, "content": content, "system": system}
        for k, v in [("summary", summary), ("hidden", hidden), ("tags", tags),
                      ("prompt_role", prompt_role), ("db_functions", db_functions),
                      ("py_functions", py_functions)]:
            if v is not None and v is not False:
                payload[k] = v
        resp = self._request("create_prompt", payload)
        if resp.get("error"):
            raise RuntimeError(resp["error"])
        return resp.get("message", "ok")

    def update_prompt(self, id, title=None, content=None, category=None,
                      summary=None, hidden=None, tags=None, prompt_role=None,
                      db_functions=None, py_functions=None) -> str:
        """Update a prompt template. Only own rows or system rows (admin)."""
        payload = {"id": id}
        for k, v in [("title", title), ("content", content), ("category", category),
                      ("summary", summary), ("hidden", hidden), ("tags", tags),
                      ("prompt_role", prompt_role), ("db_functions", db_functions),
                      ("py_functions", py_functions)]:
            if v is not None:
                payload[k] = v
        resp = self._request("update_prompt", payload)
        if resp.get("error"):
            raise RuntimeError(resp["error"])
        return resp.get("message", "ok")

    def delete_prompt(self, id) -> str:
        """Delete a prompt template. Cannot delete is_system rows."""
        resp = self._request("delete_prompt", {"id": id})
        if resp.get("error"):
            raise RuntimeError(resp["error"])
        return resp.get("message", "ok")

    def web_search(self, query=None, depth="standard", **kwargs) -> list[dict]:
        """Search the web. Returns list of {name, url, content}."""
        query = query or kwargs.get("q")
        if not query:
            raise ValueError("web_search requires a query string")
        resp = self._request("web_search", {"query": query, "depth": depth})
        if resp.get("error"):
            raise RuntimeError(resp["error"])
        return resp["results"]

    def fetch_url(self, url=None, **kwargs) -> str:
        """Fetch a webpage and return its markdown content."""
        url = url or kwargs.get("href")
        if not url:
            raise ValueError("fetch_url requires a URL")
        resp = self._request("fetch_url", {"url": url})
        if resp.get("error"):
            raise RuntimeError(resp["error"])
        return resp["markdown"]

    def update_settings(self, scope=None, patch=None, **kwargs):
        """Update admin or user settings via deep merge. Empty patch {} = read current."""
        scope = scope or kwargs.get("scope")
        if scope not in ("admin", "user"):
            raise ValueError("scope must be 'admin' or 'user'")
        if patch is None:
            patch = kwargs.get("patch", {})
        resp = self._request("update_settings", {"scope": scope, "patch": patch})
        if resp.get("error"):
            raise RuntimeError(resp["error"])
        if "settings" in resp:
            return resp["settings"]
        return resp.get("message", "ok")

    def describe_image(self, ref=None, question=None, page=None, **kwargs) -> str:
        ref = ref or kwargs.get("id_or_path") or kwargs.get("id") or kwargs.get("path")
        question = question or kwargs.get("q") or kwargs.get("prompt")
        payload = {"ref": ref}
        if question:
            payload["question"] = question
        if page is not None:
            payload["page"] = page
        resp = self._request("describe_image", payload)
        if resp.get("error"):
            raise RuntimeError(resp["error"])
        return resp["description"]


class _Loaded:
    """Sentinel returned by load_skill() — prints nothing to avoid double output."""
    def __init__(self, skill_id: str):
        self._id = skill_id
    def __repr__(self):
        return ""
    def __str__(self):
        return ""
    def __bool__(self):
        return True


_PAGE_SIZE = 30
_PAGE_BUDGET = 6000


def _format_cell(v, cell_limit: int | None = None) -> str:
    """Format a single cell value for TOON output."""
    if v is None:
        return "∅"
    if isinstance(v, bool):
        return "T" if v else "F"
    if isinstance(v, str):
        s = v.replace('\n', '↵').replace('\t', '→').replace('\r', '')
        if cell_limit and len(s) > cell_limit:
            s = _smart_cut_flat(s, cell_limit)
        if ',' in s or '"' in s:
            s = '"' + s.replace('"', '""') + '"'
        return s
    s = str(v)
    if cell_limit and len(s) > cell_limit:
        s = _smart_cut_flat(s, cell_limit)
    return s


def _smart_cut_flat(text: str, limit: int) -> str:
    """Cut already-flattened text (↵ instead of \\n) at a natural boundary."""
    for sep in ['↵↵', '↵', '. ', '! ', '? ', ' ']:
        cut = text.rfind(sep, 0, limit)
        if cut > limit * 0.5:
            remaining = len(text) - cut
            return text[:cut] + f'…[+{remaining}]'
    return text[:limit] + f'…[+{len(text) - limit}]'


def _estimate_row_chars(row: dict) -> int:
    total = 0
    for v in row.values():
        if v is None:
            total += 1
        elif isinstance(v, str):
            total += len(v)
        else:
            total += len(str(v))
    return total


def _compute_cell_limit(rows: list[dict], budget: int) -> int | None:
    """Adaptive cell limit: None if total fits budget, else proportional."""
    total = sum(_estimate_row_chars(r) for r in rows)
    if total <= budget:
        return None
    ratio = budget / total
    return max(60, int(300 * ratio))


def _render_table(rows: list[dict], fields: list[str], total_rows: int,
                  cell_limit: int | None = None, offset: int = 0,
                  total_chars: int | None = None,
                  var_hint: str | None = None) -> str:
    """Render rows as TOON table with optional footer."""
    header = f"rows[{total_rows}]{{{','.join(fields)}}}:"
    lines = [header]
    if offset > 0:
        lines.append(f"  … (skipped {offset})")
    for row in rows:
        cells = [_format_cell(row.get(f), cell_limit) for f in fields]
        lines.append("  " + ",".join(cells))
    footer_parts = []
    remaining = total_rows - offset - len(rows)
    if remaining > 0:
        shown = f"{len(rows)} of {total_rows}"
        char_info = f" | ~{_human_size(total_chars)}" if total_chars else ""
        footer_parts.append(f"  … ({shown} rows{char_info})")
        hints = []
        if var_hint:
            hints.append(f"print({var_hint}.more)")
            hints.append(f"print({var_hint}.raw)")
            hints.append(f"print({var_hint}.raw[N])")
        footer_parts.append(f"  → {' | '.join(hints)}" if hints else "")
    elif cell_limit is not None:
        hint = f"print({var_hint}.raw[N]) for full row" if var_hint else "use .raw[N] for full row"
        footer_parts.append(f"  [{hint}]")
    return '\n'.join(lines + [p for p in footer_parts if p])


def _render_keyvalue(row: dict, fields: list[str]) -> str:
    """Render a single row as key: value pairs (no cell truncation)."""
    lines = []
    for f in fields:
        v = row.get(f)
        if v is None:
            lines.append(f"{f}: ∅")
        elif isinstance(v, bool):
            lines.append(f"{f}: {'T' if v else 'F'}")
        elif isinstance(v, str):
            lines.append(f"{f}: {v}")
        else:
            lines.append(f"{f}: {v}")
    return '\n'.join(lines)


def _human_size(n: int | None) -> str:
    if n is None:
        return "?"
    if n < 1000:
        return f"{n}"
    if n < 10_000:
        return f"{n / 1000:.1f}K"
    return f"{n // 1000}K"


class RawView:
    """Untruncated view of DbResult — no page limits, no cell limits."""

    def __init__(self, rows: list[dict]):
        self._rows = rows

    def __repr__(self):
        if not self._rows:
            return "rows[0]{}: (empty)"
        fields = list(self._rows[0].keys())
        if len(self._rows) == 1 and len(fields) == 1:
            return str(self._rows[0][fields[0]])
        if len(self._rows) == 1:
            return _render_keyvalue(self._rows[0], fields)
        return _render_table(self._rows, fields, len(self._rows), cell_limit=None)

    def __getitem__(self, key):
        if isinstance(key, int):
            row = self._rows[key]
            fields = list(row.keys())
            return _RawRow(row, fields)
        if isinstance(key, slice):
            sliced = self._rows[key]
            return RawView(sliced)
        raise TypeError(f"indices must be integers or slices, not {type(key).__name__}")

    def __len__(self):
        return len(self._rows)

    def __iter__(self):
        return iter(self._rows)

    def __bool__(self):
        return bool(self._rows)


class _RawRow:
    """Single row printed as key-value, no truncation."""

    def __init__(self, row: dict, fields: list[str]):
        self._row = row
        self._fields = fields

    def __repr__(self):
        return _render_keyvalue(self._row, self._fields)

    def __getitem__(self, key):
        return self._row[key]

    def keys(self):
        return self._row.keys()

    def values(self):
        return self._row.values()

    def items(self):
        return self._row.items()


class DbResult:
    """Smart DB result wrapper. print() auto-formats with pagination and adaptive truncation.

    Shapes:
      1×1 (scalar)    → prints bare value
      1×N (one row)   → prints key: value pairs
      N×M (table)     → prints paginated table with adaptive cell limits

    Pagination:  print(r.more) for next page
    Full access: print(r.raw), print(r.raw[N]), print(r.raw[N:M])
    Data access: r.rows (list[dict]), r.df (pandas DataFrame)
    List compat: r[i], len(r), for row in r, bool(r)
    """

    _unprinted: list['DbResult'] = []

    @classmethod
    def _reset_tracking(cls):
        cls._unprinted.clear()

    def __init__(self, rows: list[dict], page_size: int = _PAGE_SIZE,
                 page_budget: int = _PAGE_BUDGET):
        self._rows = rows
        self._page_size = page_size
        self._page_budget = page_budget
        self._page = 0
        self._total_chars: int | None = None
        self._var_name: str | None = None
        self._printed = False
        DbResult._unprinted.append(self)

    def _get_total_chars(self) -> int:
        if self._total_chars is None:
            self._total_chars = sum(_estimate_row_chars(r) for r in self._rows)
        return self._total_chars

    def __repr__(self):
        self._printed = True
        if not self._rows:
            return "rows[0]{}: (empty)"
        fields = list(self._rows[0].keys())

        # 1×1 scalar: bare value
        if len(self._rows) == 1 and len(fields) == 1:
            v = self._rows[0][fields[0]]
            return str(v) if v is not None else "∅"

        # 1×N single row: key-value, no truncation
        if len(self._rows) == 1:
            return _render_keyvalue(self._rows[0], fields)

        # N×M table: paginated with adaptive cell limits
        n = len(self._rows)
        start = self._page * self._page_size
        if start >= n:
            return "(no more rows)"
        end = min(start + self._page_size, n)
        page_rows = self._rows[start:end]
        cell_limit = _compute_cell_limit(page_rows, self._page_budget)
        total_chars = self._get_total_chars() if n > self._page_size else None
        return _render_table(
            page_rows, fields, n,
            cell_limit=cell_limit, offset=start,
            total_chars=total_chars, var_hint=self._var_name,
        )

    @property
    def more(self):
        """Advance to next page and return self for printing."""
        self._page += 1
        return self

    @property
    def raw(self) -> RawView:
        """All rows, no truncation. Supports slicing: .raw[N], .raw[N:M]."""
        return RawView(self._rows)

    @property
    def rows(self) -> list[dict]:
        """Raw list[dict] for computation."""
        return self._rows

    @property
    def df(self):
        """Pandas DataFrame (lazy import)."""
        import pandas as pd
        return pd.DataFrame(self._rows)

    def __getitem__(self, key):
        return self._rows[key]

    def __len__(self):
        return len(self._rows)

    def __iter__(self):
        return iter(self._rows)

    def __bool__(self):
        return bool(self._rows)

    def __contains__(self, item):
        return item in self._rows

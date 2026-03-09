"""Bridge client for sandbox ↔ chat service communication.

Uses a shared socket connection (owned by executor.py) to proxy requests
to the chat service. All calls are synchronous and blocking — safe because
sandbox code execution is single-threaded.
"""

import json
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

    def db(self, sql=None, **kwargs) -> list[dict]:
        sql = sql or kwargs.get("query") or kwargs.get("sql")
        resp = self._request("db_query", {"sql": sql})
        if resp.get("error"):
            raise RuntimeError(resp["error"])
        return resp["rows"]

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


def fmt(rows: list[dict], max_rows: int | None = 50, max_cell: int | None = 80) -> str:
    """Format rows as compact TOON table. Pure local, no bridge needed."""
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

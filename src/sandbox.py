"""
Sandbox orchestrator — manages gVisor containers for Python code execution.

Replaces in-process exec() with an isolated Docker container per response.
The container has full Python access but no network, no credentials, and
runs under gVisor (runsc) for kernel-level isolation.

Protocol: single Unix socket, bidirectional length-prefixed JSON.
  Orchestrator → Sandbox: execute, shutdown
  Sandbox → Orchestrator: db_query, file_info, download_file, download_url, result
"""

import asyncio
import hashlib
import json
import logging
import os
import socket
import struct
import tempfile
import uuid
from pathlib import Path

import asyncpg

from . import settings, storage

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

logger = logging.getLogger("ibhelm.chat.sandbox")

SANDBOX_IMAGE = os.getenv("SANDBOX_IMAGE", "ibhelm-sandbox:latest")
SANDBOX_RUNTIME = os.getenv("SANDBOX_RUNTIME", "")  # "runsc" for gVisor, "" for default
SANDBOX_MEM_LIMIT = os.getenv("SANDBOX_MEM_LIMIT", "2g")
SANDBOX_PIDS_LIMIT = int(os.getenv("SANDBOX_PIDS_LIMIT", "200"))
EXEC_TIMEOUT_S = int(os.getenv("SANDBOX_EXEC_TIMEOUT", "60"))
QUERY_TIMEOUT_S = 10

_HEADER_FMT = "!I"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)


def _send_msg(sock: socket.socket, msg: dict):
    data = json.dumps(msg, ensure_ascii=False, default=str).encode("utf-8")
    sock.sendall(struct.pack(_HEADER_FMT, len(data)) + data)


def _recv_msg(sock: socket.socket) -> dict:
    header = _recv_exact(sock, _HEADER_SIZE)
    length = struct.unpack(_HEADER_FMT, header)[0]
    data = _recv_exact(sock, length)
    return json.loads(data.decode("utf-8"))


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(n - len(buf), 65536))
        if not chunk:
            raise ConnectionError("Socket closed")
        buf.extend(chunk)
    return bytes(buf)


class SandboxSession:
    """Manages one sandbox container for the duration of an assistant response."""

    def __init__(self, pool: asyncpg.Pool, user_email: str | None, session_id: str, user_id: str | None = None):
        self.pool = pool
        self.user_email = user_email
        self.user_id = user_id
        self.session_id = session_id
        self._container_id: str | None = None
        self._shared_dir: str | None = None
        self._server_sock: socket.socket | None = None
        self._conn: socket.socket | None = None
        self._started = False

    async def start(self, conversation_files: list[dict] | None = None):
        """Create shared volume, start container, establish socket connection."""
        self._shared_dir = tempfile.mkdtemp(prefix="sandbox_")
        work_dir = os.path.join(self._shared_dir, "work")
        os.makedirs(work_dir, exist_ok=True)

        if conversation_files:
            await self._populate_files(work_dir, conversation_files)

        sock_path = os.path.join(self._shared_dir, "sandbox.sock")
        self._server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server_sock.bind(sock_path)
        self._server_sock.listen(1)
        self._server_sock.settimeout(30)
        os.chmod(sock_path, 0o777)

        await self._start_container()

        loop = asyncio.get_running_loop()
        self._conn, _ = await asyncio.wait_for(
            loop.run_in_executor(None, self._server_sock.accept),
            timeout=30,
        )
        self._conn.settimeout(EXEC_TIMEOUT_S + 10)
        self._started = True
        logger.info("Sandbox container started for session %s", self.session_id)

    async def _start_container(self):
        """Start the sandbox Docker container."""
        import docker
        client = docker.from_env()

        container_name = f"sandbox-{uuid.uuid4().hex[:12]}"
        kwargs = {
            "image": SANDBOX_IMAGE,
            "name": container_name,
            "detach": True,
            "network_mode": "none",
            "read_only": True,
            "mem_limit": SANDBOX_MEM_LIMIT,
            "pids_limit": SANDBOX_PIDS_LIMIT,
            "volumes": {
                os.path.join(self._shared_dir, "work"): {"bind": "/work", "mode": "rw"},
                self._shared_dir: {"bind": "/shared", "mode": "rw"},
            },
            "tmpfs": {"/tmp": "size=512m"},
            "labels": {"ibhelm.sandbox": "true", "ibhelm.session": self.session_id},
        }
        if SANDBOX_RUNTIME:
            kwargs["runtime"] = SANDBOX_RUNTIME

        loop = asyncio.get_running_loop()
        container = await loop.run_in_executor(None, lambda: client.containers.run(**kwargs))
        self._container_id = container.id
        logger.info("Started sandbox container %s (%s)", container_name, container.short_id)

    async def _populate_files(self, work_dir: str, files: list[dict]):
        """Download conversation files into the work directory."""
        for f in files:
            try:
                data = await storage.download_file(f["bucket"], f["content_hash"])
                if data:
                    filepath = os.path.join(work_dir, f["filename"])
                    # Avoid name collisions
                    if os.path.exists(filepath):
                        name, ext = os.path.splitext(f["filename"])
                        filepath = os.path.join(work_dir, f"{name}_{f['id'][:8]}{ext}")
                    with open(filepath, "wb") as fh:
                        fh.write(data)
            except Exception as e:
                logger.warning("Failed to populate file %s: %s", f.get("filename"), e)

    async def execute(self, code: str) -> dict:
        """Send code to sandbox, handle bridge requests, return result.

        Returns: {stdout, result, error, pending_images, new_files, tool_costs}
        """
        if not self._started:
            raise RuntimeError("Sandbox not started")

        loop = asyncio.get_running_loop()

        def _blocking_execute():
            _send_msg(self._conn, {"type": "execute", "code": code})
            while True:
                msg = _recv_msg(self._conn)
                if msg["type"] == "result":
                    return msg
                response = self._handle_bridge_request_sync(msg, loop)
                _send_msg(self._conn, response)

        try:
            return await asyncio.wait_for(
                loop.run_in_executor(None, _blocking_execute),
                timeout=EXEC_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            return {"stdout": None, "result": None, "error": f"Execution timed out after {EXEC_TIMEOUT_S}s",
                    "pending_images": [], "new_files": [], "tool_costs": []}

    def _handle_bridge_request_sync(self, msg: dict, loop: asyncio.AbstractEventLoop) -> dict:
        """Handle a proxy request from the sandbox (runs in executor thread)."""
        msg_type = msg["type"]

        if msg_type == "db_query":
            future = asyncio.run_coroutine_threadsafe(self._exec_query(msg["sql"]), loop)
            try:
                rows = future.result(timeout=QUERY_TIMEOUT_S)
                return {"type": "db_result", "rows": rows}
            except Exception as e:
                return {"type": "db_result", "error": str(e)}

        if msg_type == "file_info":
            future = asyncio.run_coroutine_threadsafe(self._get_file_info(msg["id"]), loop)
            try:
                info = future.result(timeout=10)
                return {"type": "file_info_result", "info": info}
            except Exception as e:
                return {"type": "file_info_result", "error": str(e)}

        if msg_type == "download_file":
            future = asyncio.run_coroutine_threadsafe(
                self._download_nas_file(msg["content_hash"]), loop
            )
            try:
                local_path = future.result(timeout=30)
                return {"type": "download_file_result", "local_path": local_path}
            except Exception as e:
                return {"type": "download_file_result", "error": str(e)}

        if msg_type == "download_craft_file":
            future = asyncio.run_coroutine_threadsafe(
                self._download_craft_file(msg["storage_path"]), loop
            )
            try:
                local_path = future.result(timeout=30)
                return {"type": "download_craft_file_result", "local_path": local_path}
            except Exception as e:
                return {"type": "download_craft_file_result", "error": str(e)}

        if msg_type == "download_url":
            future = asyncio.run_coroutine_threadsafe(self._get_download_url(msg["id"]), loop)
            try:
                url = future.result(timeout=10)
                return {"type": "download_url_result", "url": url}
            except Exception as e:
                return {"type": "download_url_result", "error": str(e)}

        if msg_type == "describe_image":
            future = asyncio.run_coroutine_threadsafe(
                self._describe_image(msg["ref"], msg.get("question"), msg.get("page")), loop
            )
            try:
                desc = future.result(timeout=60)
                return {
                    "type": "describe_image_result",
                    "description": desc["description"],
                    "cost_usd": desc["cost_usd"],
                    "tool_name": "describe_image",
                }
            except Exception as e:
                return {"type": "describe_image_result", "error": str(e)}

        if msg_type == "add_activity_entry":
            future = asyncio.run_coroutine_threadsafe(
                self._add_activity_entry(msg), loop
            )
            try:
                result = future.result(timeout=10)
                return {"type": "add_activity_entry_result", "id": result}
            except Exception as e:
                return {"type": "add_activity_entry_result", "error": str(e)}

        if msg_type == "update_project_status":
            future = asyncio.run_coroutine_threadsafe(
                self._update_project_status(msg["project_id"], msg["markdown"]), loop
            )
            try:
                message = future.result(timeout=10)
                return {"type": "update_project_status_result", "ok": True, "message": message}
            except Exception as e:
                return {"type": "update_project_status_result", "error": str(e)}

        if msg_type == "update_project_profile":
            future = asyncio.run_coroutine_threadsafe(
                self._update_project_profile(msg["project_id"], msg["markdown"]), loop
            )
            try:
                message = future.result(timeout=10)
                return {"type": "update_project_profile_result", "ok": True, "message": message}
            except Exception as e:
                return {"type": "update_project_profile_result", "error": str(e)}

        return {"type": "error", "error": f"Unknown request type: {msg_type}"}

    async def _exec_query(self, sql: str) -> list[dict]:
        """Execute a read-only SQL query with RLS."""
        stripped = sql.strip().rstrip(';').strip()
        upper = stripped.upper()
        while upper.startswith('--'):
            stripped = stripped.split('\n', 1)[-1].strip() if '\n' in stripped else ''
            upper = stripped.upper()
        if not (upper.startswith('SELECT') or upper.startswith('WITH')):
            raise ValueError("Only SELECT/WITH queries allowed")

        async with self.pool.acquire() as conn:
            await conn.execute(f"SET statement_timeout = '{QUERY_TIMEOUT_S}s'")
            async with conn.transaction():
                await conn.execute("SET LOCAL ROLE mcp_readonly")
                if self.user_email:
                    await conn.execute("SELECT set_config('app.user_email', $1, true)", self.user_email)
                if self.user_id:
                    await conn.execute("SELECT set_config('app.user_id', $1, true)", self.user_id)
                rows = await conn.fetch(sql)
            return [_serialize_row(dict(r)) for r in rows]

    _VALID_CATEGORIES = frozenset([
        'decision', 'blocker', 'resolution', 'progress',
        'milestone', 'risk', 'scope_change', 'communication',
    ])

    async def _add_activity_entry(self, msg: dict) -> str:
        """Insert a Tier 3 activity log entry. Returns the new UUID."""
        from datetime import datetime, timezone as tz
        category = msg["category"]
        if category not in self._VALID_CATEGORIES:
            raise ValueError(f"Invalid category: {category}")
        logged_at = msg.get("logged_at")
        if isinstance(logged_at, str):
            logged_at = datetime.fromisoformat(logged_at)
        if not logged_at:
            logged_at = datetime.now(tz.utc)
        row = await self.pool.fetchrow("""
            INSERT INTO project_activity_log
                (tw_project_id, logged_at, category, summary, source_event_ids, kgr_codes, involved_persons)
            VALUES ($1, $2, $3, $4, $5, $6, $7) RETURNING id
        """,
            msg["project_id"],
            logged_at,
            category,
            msg["summary"],
            msg.get("source_event_ids") or [],
            msg.get("kgr_codes") or [],
            msg.get("involved_persons") or [],
        )
        return str(row["id"])

    async def _update_project_status(self, project_id: int, markdown: str) -> str:
        """Update Tier 2 status markdown with length protection."""
        current = await self.pool.fetchval(
            "SELECT status_markdown FROM project_extensions WHERE tw_project_id = $1", project_id
        )
        new_len = len(markdown)
        if current and new_len < len(current) // 2:
            raise ValueError(
                f"Rejected: new status ({new_len} chars) is less than half of current ({len(current)} chars). "
                f"You must provide the FULL updated status markdown, not a partial update or diff."
            )
        await self.pool.execute("""
            UPDATE project_extensions
            SET status_markdown = $1, status_generated_at = NOW()
            WHERE tw_project_id = $2
        """, markdown, project_id)
        return f"Status updated ({new_len} chars)"

    async def _update_project_profile(self, project_id: int, markdown: str) -> str:
        """Update Tier 1 profile markdown with length protection."""
        current = await self.pool.fetchval(
            "SELECT profile_markdown FROM project_extensions WHERE tw_project_id = $1", project_id
        )
        new_len = len(markdown)
        if current and new_len < len(current) // 2:
            raise ValueError(
                f"Rejected: new profile ({new_len} chars) is less than half of current ({len(current)} chars). "
                f"You must provide the FULL updated profile markdown, not a partial update or diff."
            )
        await self.pool.execute("""
            UPDATE project_extensions
            SET profile_markdown = $1, profile_generated_at = NOW()
            WHERE tw_project_id = $2
        """, markdown, project_id)
        return f"Profile updated ({new_len} chars)"

    async def _get_file_info(self, file_id: str) -> dict:
        """Get metadata for a file. Checks chat_files first, then falls back to files table."""
        row = await self.pool.fetchrow("""
            SELECT cf.id, cf.filename, cf.content_hash, cf.bucket, cf.origin,
                   cf.size_bytes, cf.mime_type,
                   fc.extracted_text, fc.mime_type as fc_mime_type,
                   fc.size_bytes as fc_size_bytes,
                   f.full_path, p.name as project_name
            FROM chat_files cf
            LEFT JOIN file_contents fc ON cf.content_hash = fc.content_hash
            LEFT JOIN files f ON f.content_hash = cf.content_hash AND f.deleted_at IS NULL
            LEFT JOIN teamwork.projects p ON f.project_id = p.id
            WHERE cf.id = $1
            LIMIT 1
        """, file_id)
        if row:
            return {
                "id": str(row["id"]),
                "filename": row["filename"],
                "content_hash": row["content_hash"],
                "bucket": row["bucket"],
                "size_bytes": row["size_bytes"] or row["fc_size_bytes"],
                "mime_type": row["mime_type"] or row["fc_mime_type"],
                "extracted_text": row["extracted_text"],
                "nas_path": row["full_path"],
                "project_name": row["project_name"],
            }

        # Fallback: check the files table (NAS files found via SQL)
        row = await self.pool.fetchrow("""
            SELECT f.id, f.full_path, f.content_hash,
                   fc.size_bytes, fc.mime_type, fc.extracted_text,
                   p.name as project_name
            FROM files f
            JOIN file_contents fc ON f.content_hash = fc.content_hash
            LEFT JOIN teamwork.projects p ON f.project_id = p.id
            WHERE f.id = $1 AND f.deleted_at IS NULL
            LIMIT 1
        """, file_id)
        if row:
            filename = Path(row["full_path"]).name if row["full_path"] else file_id
            return {
                "id": str(row["id"]),
                "filename": filename,
                "content_hash": row["content_hash"],
                "bucket": "files",
                "size_bytes": row["size_bytes"],
                "mime_type": row["mime_type"],
                "extracted_text": row["extracted_text"],
                "nas_path": row["full_path"],
                "project_name": row["project_name"],
            }

        raise ValueError(f"File not found: {file_id}")

    async def _download_nas_file(self, content_hash: str) -> str:
        """Download a file from the files bucket into /work/."""
        row = await self.pool.fetchrow("""
            SELECT fc.mime_type, f.full_path
            FROM file_contents fc
            LEFT JOIN files f ON f.content_hash = fc.content_hash AND f.deleted_at IS NULL
            WHERE fc.content_hash = $1
        """, content_hash)
        if not row:
            raise ValueError(f"Content hash not found: {content_hash}")

        data = await storage.download_file("files", content_hash)
        if not data:
            raise RuntimeError(f"Failed to download from storage: {content_hash}")

        # Derive filename from NAS path or use hash
        if row["full_path"]:
            filename = Path(row["full_path"]).name
        else:
            ext = _mime_to_ext(row["mime_type"] or "")
            filename = f"{content_hash[:16]}{ext}"

        work_dir = os.path.join(self._shared_dir, "work")
        local_path = os.path.join(work_dir, filename)
        if os.path.exists(local_path):
            name, ext = os.path.splitext(filename)
            local_path = os.path.join(work_dir, f"{name}_{content_hash[:8]}{ext}")

        with open(local_path, "wb") as f:
            f.write(data)

        # Return path as seen from inside the container
        return f"/work/{Path(local_path).name}"

    async def _download_craft_file(self, storage_path: str) -> str:
        """Download a file from the craft-files bucket into /work/."""
        MAX = 100 * 1024 * 1024
        data = await storage.download_file("craft-files", storage_path)
        if not data:
            raise RuntimeError(f"Failed to download craft file: {storage_path}")
        if len(data) > MAX:
            raise RuntimeError(f"File too large: {len(data)} bytes (limit {MAX})")

        filename = Path(storage_path).name
        # Strip block_id prefix if present (format: {block_id}_{filename})
        if "_" in filename and len(filename.split("_", 1)[0]) > 30:
            filename = filename.split("_", 1)[1]

        work_dir = os.path.join(self._shared_dir, "work")
        local_path = os.path.join(work_dir, filename)
        if os.path.exists(local_path):
            name, ext = os.path.splitext(filename)
            local_path = os.path.join(work_dir, f"{name}_{storage_path.split('/')[0][:8]}{ext}")

        with open(local_path, "wb") as f:
            f.write(data)

        return f"/work/{Path(local_path).name}"

    async def _describe_image(self, ref: str, question: str | None, page: int | None) -> dict:
        """Load an image and send it to a vision model for description."""
        image_data, media_type = await self._load_image(ref, page)
        if not image_data:
            raise ValueError(f"Could not load image: {ref}")
        return await _call_vision_model(image_data, media_type, question, self.pool)

    async def _load_image(self, ref: str, page: int | None) -> tuple[bytes | None, str]:
        """Load image bytes from a file UUID or local /work/ path."""
        image_data = None
        media_type = "image/png"

        if "/" in ref:
            host_path = ref.replace("/work/", os.path.join(self._shared_dir, "work/"), 1)
            if os.path.exists(host_path):
                with open(host_path, "rb") as f:
                    image_data = f.read()
                ext = Path(host_path).suffix.lower()
                media_type = {
                    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                    ".gif": "image/gif", ".webp": "image/webp",
                    ".heic": "image/heic", ".heif": "image/heif",
                }.get(ext, "image/png")
        else:
            row = await self.pool.fetchrow(
                "SELECT bucket, content_hash, mime_type FROM chat_files WHERE id = $1", ref
            )
            if row:
                file_data = await storage.download_file(row["bucket"], row["content_hash"])
                if file_data:
                    image_data = file_data
                    media_type = row["mime_type"] or "image/png"

        if not image_data:
            return None, media_type

        if page is not None and (media_type == "application/pdf" or str(ref).endswith(".pdf")):
            import fitz
            doc = fitz.open(stream=image_data, filetype="pdf")
            if 0 < page <= len(doc):
                pix = doc[page - 1].get_pixmap(dpi=150)
                image_data = pix.tobytes("png")
                media_type = "image/png"
            doc.close()

        needs_convert = media_type in ("image/heic", "image/heif")
        if media_type.startswith("image/"):
            from PIL import Image
            from io import BytesIO
            img = Image.open(BytesIO(image_data))
            if max(img.size) > 1024 or needs_convert:
                if max(img.size) > 1024:
                    img.thumbnail((1024, 1024))
                buf = BytesIO()
                img.save(buf, format="JPEG", quality=85)
                image_data = buf.getvalue()
                media_type = "image/jpeg"

        return image_data, media_type

    async def _get_download_url(self, file_id: str) -> str:
        """Generate a download URL for a file (chat_files or files table)."""
        row = await self.pool.fetchrow(
            "SELECT bucket, content_hash FROM chat_files WHERE id = $1", file_id
        )
        if row:
            bucket, content_hash = row["bucket"], row["content_hash"]
        else:
            row = await self.pool.fetchrow(
                "SELECT content_hash FROM files WHERE id = $1 AND deleted_at IS NULL", file_id
            )
            if not row:
                raise ValueError(f"File not found: {file_id}")
            bucket, content_hash = "files", row["content_hash"]

        if bucket == "chat-files":
            return storage.public_url("chat-files", content_hash)
        url = await storage.create_signed_url("files", content_hash)
        if not url:
            raise RuntimeError("Failed to generate signed URL")
        return url

    async def cleanup(self):
        """Kill the container and remove shared directory."""
        if self._conn:
            try:
                _send_msg(self._conn, {"type": "shutdown"})
            except Exception:
                pass
            try:
                self._conn.close()
            except Exception:
                pass

        if self._server_sock:
            try:
                self._server_sock.close()
            except Exception:
                pass

        if self._container_id:
            try:
                import docker
                client = docker.from_env()
                container = client.containers.get(self._container_id)
                try:
                    container.stop(timeout=2)
                except Exception:
                    pass
                container.remove(force=True)
                logger.info("Removed sandbox container %s", self._container_id[:12])
            except Exception as e:
                logger.warning("Failed to remove sandbox container: %s", e)

        if self._shared_dir:
            try:
                import shutil
                shutil.rmtree(self._shared_dir, ignore_errors=True)
            except Exception:
                pass

    async def upload_new_files(self, new_files: list[dict], message_id: str) -> list[dict]:
        """Upload newly created files from sandbox to S3, create chat_files entries."""
        created = []
        for nf in new_files:
            content_hash = nf["content_hash"]
            filename = nf["filename"]
            size_bytes = nf["size_bytes"]
            local_path = nf["path"]

            # The local_path is relative to shared_dir/work, read from host
            host_path = local_path.replace("/work/", os.path.join(self._shared_dir, "work/"), 1)
            if not os.path.exists(host_path):
                host_path = local_path  # fallback
            try:
                with open(host_path, "rb") as f:
                    data = f.read()
            except FileNotFoundError:
                logger.warning("New file not found at %s", host_path)
                continue

            mime = _guess_mime(filename)
            ok = await storage.upload_file("chat-files", content_hash, data, mime)
            if not ok:
                logger.warning("Failed to upload generated file %s", filename)
                continue

            file_id = str(uuid.uuid4())
            await self.pool.execute("""
                INSERT INTO chat_files (id, message_id, filename, content_hash, bucket, origin, size_bytes, mime_type)
                VALUES ($1, $2, $3, $4, 'chat-files', 'generated', $5, $6)
            """, file_id, message_id, filename, content_hash, size_bytes, mime)

            created.append({"id": file_id, "filename": filename, "content_hash": content_hash,
                            "bucket": "chat-files", "origin": "generated",
                            "size_bytes": size_bytes, "mime_type": mime})
        return created


async def _call_vision_model(image_data: bytes, media_type: str, question: str | None,
                             pool: asyncpg.Pool) -> dict:
    """Call the vision fallback model to describe an image."""
    import base64
    from .llm import calculate_usage_cost_usd, normalize_usage, resolve_model

    vision_model_id = settings.VISION_FALLBACK_MODEL
    if not vision_model_id:
        raise RuntimeError("No VISION_FALLBACK_MODEL configured")

    model_config = await resolve_model(vision_model_id, pool)
    if not model_config.get("supports_vision"):
        raise RuntimeError(f"Vision fallback model {vision_model_id} does not support vision")

    b64 = base64.b64encode(image_data).decode("ascii")
    prompt = question or "Describe this image in detail. Include all visible text, objects, people, colors, and layout."

    provider_type = model_config.get("provider", "anthropic")
    if provider_type == "anthropic":
        import anthropic
        client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
        resp = await client.messages.create(
            model=model_config["id"], max_tokens=2000,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}},
                {"type": "text", "text": prompt},
            ]}],
        )
        usage = normalize_usage({
            "input_tokens": getattr(resp.usage, "input_tokens", 0),
            "output_tokens": getattr(resp.usage, "output_tokens", 0),
            "cache_read_input_tokens": getattr(resp.usage, "cache_read_input_tokens", 0),
            "cache_creation_input_tokens": getattr(resp.usage, "cache_creation_input_tokens", 0),
        })
        return {
            "description": resp.content[0].text.strip(),
            "cost_usd": calculate_usage_cost_usd(model_config, usage),
        }
    else:
        from openai import AsyncOpenAI
        base_url = model_config.get("base_url", "https://api.tokenfactory.nebius.com/v1/")
        client = AsyncOpenAI(api_key=settings.NEBIUS_API_KEY, base_url=base_url)
        resp = await client.chat.completions.create(
            model=model_config["id"], max_tokens=2000,
            messages=[{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{b64}"}},
                {"type": "text", "text": prompt},
            ]}],
        )
        usage = normalize_usage({
            "input_tokens": getattr(resp.usage, "prompt_tokens", 0) if resp.usage else 0,
            "output_tokens": getattr(resp.usage, "completion_tokens", 0) if resp.usage else 0,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        })
        return {
            "description": (resp.choices[0].message.content or "").strip(),
            "cost_usd": calculate_usage_cost_usd(model_config, usage),
        }


def _serialize_row(row: dict) -> dict:
    out = {}
    for k, v in row.items():
        if hasattr(v, 'isoformat'):
            out[k] = v.isoformat()
        elif isinstance(v, (bytes, bytearray)):
            out[k] = f"<{len(v)} bytes>"
        else:
            out[k] = v
    return out


def _mime_to_ext(mime: str) -> str:
    mapping = {
        "application/pdf": ".pdf", "image/png": ".png", "image/jpeg": ".jpg",
        "text/plain": ".txt", "text/csv": ".csv",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    }
    return mapping.get(mime, "")


def _guess_mime(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    mapping = {
        ".pdf": "application/pdf", ".png": "image/png", ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg", ".gif": "image/gif", ".webp": "image/webp",
        ".svg": "image/svg+xml", ".csv": "text/csv", ".txt": "text/plain",
        ".json": "application/json", ".html": "text/html",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    }
    return mapping.get(ext, "application/octet-stream")


async def resolve_images(
    sandbox: "SandboxSession",
    image_requests: list[dict],
) -> list[dict]:
    """Load images requested by file_image() and encode as base64.

    Returns: [{base64: str, media_type: str}]
    """
    import base64
    results = []
    for req in image_requests:
        try:
            ref = req["ref"]
            page = req.get("page")
            max_dim = req.get("max_dim")

            image_data = None
            media_type = "image/png"

            if "/" in ref:
                host_path = ref.replace("/work/", os.path.join(sandbox._shared_dir, "work/"), 1)
                if os.path.exists(host_path):
                    with open(host_path, "rb") as f:
                        image_data = f.read()
                    ext = Path(host_path).suffix.lower()
                    media_type = {".png": "image/png", ".jpg": "image/jpeg",
                                  ".jpeg": "image/jpeg", ".gif": "image/gif",
                                  ".webp": "image/webp",
                                  ".heic": "image/heic", ".heif": "image/heif",
                                  }.get(ext, "image/png")
            else:
                row = await sandbox.pool.fetchrow(
                    "SELECT bucket, content_hash, mime_type FROM chat_files WHERE id = $1", ref
                )
                if row:
                    file_data = await storage.download_file(row["bucket"], row["content_hash"])
                    if file_data:
                        image_data = file_data
                        media_type = row["mime_type"] or "image/png"

            if not image_data:
                continue

            if page is not None and (media_type == "application/pdf" or str(ref).endswith(".pdf")):
                import fitz
                doc = fitz.open(stream=image_data, filetype="pdf")
                if 0 < page <= len(doc):
                    pix = doc[page - 1].get_pixmap(dpi=150)
                    image_data = pix.tobytes("png")
                    media_type = "image/png"
                doc.close()

            needs_convert = media_type in ("image/heic", "image/heif")
            needs_resize = max_dim and media_type.startswith("image/")
            if needs_convert or needs_resize:
                from PIL import Image
                from io import BytesIO
                img = Image.open(BytesIO(image_data))
                resized = max_dim and max(img.size) > max_dim
                if resized:
                    img.thumbnail((max_dim, max_dim))
                if needs_convert or resized:
                    buf = BytesIO()
                    img.save(buf, format="JPEG", quality=85)
                    image_data = buf.getvalue()
                    media_type = "image/jpeg"

            b64 = base64.b64encode(image_data).decode("ascii")
            results.append({"type": "image", "base64": b64, "media_type": media_type})
        except Exception as e:
            logger.warning("Failed to resolve image %s: %s", req.get("ref"), e)
    return results


async def get_session_files(pool: asyncpg.Pool, session_id: str) -> list[dict]:
    """Load all files for a chat session (for pre-populating sandbox /work/)."""
    rows = await pool.fetch("""
        SELECT cf.id, cf.filename, cf.content_hash, cf.bucket, cf.mime_type
        FROM chat_files cf
        JOIN chat_messages m ON cf.message_id = m.id
        WHERE m.session_id = $1
        ORDER BY cf.created_at
    """, session_id)
    return [dict(r) for r in rows]

"""Supabase Storage client for file uploads/downloads.

Uses SUPABASE_INTERNAL_URL (Docker network) for API calls and
SUPABASE_URL (external hostname) for browser-facing URLs.
"""

import logging
from urllib.parse import quote

import httpx

from . import settings

logger = logging.getLogger("ibhelm.chat.storage")

_client: httpx.AsyncClient | None = None


def _internal_url() -> str:
    """URL for server-to-server storage API calls (inside Docker network)."""
    return settings.SUPABASE_INTERNAL_URL or settings.SUPABASE_URL


def _external_url() -> str:
    """URL for browser-facing links."""
    return settings.SUPABASE_URL


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=60.0)
    return _client


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {settings.SUPABASE_SERVICE_KEY}",
        "apikey": settings.SUPABASE_SERVICE_KEY,
    }


async def upload_file(bucket: str, path: str, data: bytes, content_type: str = "application/octet-stream") -> bool:
    """Upload a file to Supabase Storage. Returns True on success."""
    client = _get_client()
    url = f"{_internal_url()}/storage/v1/object/{bucket}/{quote(path, safe='/')}"
    resp = await client.post(url, content=data, headers={**_headers(), "Content-Type": content_type})
    if resp.status_code == 400 and "already exists" in resp.text.lower():
        resp = await client.put(url, content=data, headers={**_headers(), "Content-Type": content_type})
    if resp.status_code not in (200, 201):
        logger.error("Storage upload failed: %s %s", resp.status_code, resp.text[:200])
        return False
    return True


async def download_file(bucket: str, path: str) -> bytes | None:
    """Download a file from Supabase Storage. Returns bytes or None on error."""
    client = _get_client()
    url = f"{_internal_url()}/storage/v1/object/{bucket}/{quote(path, safe='/')}"
    resp = await client.get(url, headers=_headers())
    if resp.status_code != 200:
        logger.error("Storage download failed: %s %s", resp.status_code, resp.text[:200])
        return None
    return resp.content


async def create_signed_url(bucket: str, path: str, expires_in: int = 3600) -> str | None:
    """Create a signed URL for a private bucket. Returns URL or None."""
    client = _get_client()
    url = f"{_internal_url()}/storage/v1/object/sign/{bucket}/{quote(path, safe='/')}"
    resp = await client.post(url, json={"expiresIn": expires_in}, headers=_headers())
    if resp.status_code != 200:
        logger.error("Signed URL failed: %s %s", resp.status_code, resp.text[:200])
        return None
    data = resp.json()
    return f"{_external_url()}/storage/v1{data['signedURL']}"


def public_url(bucket: str, path: str) -> str:
    """Construct a public URL (for public buckets). Uses external hostname for browser access."""
    return f"{_external_url()}/storage/v1/object/public/{bucket}/{quote(path, safe='/')}"


async def close():
    global _client
    if _client:
        await _client.aclose()
        _client = None

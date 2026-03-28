"""
Web tools for Tinyclaw — search and extract web content.

Uses Firecrawl as the scraping backend (requires FIRECRAWL_API_KEY).
Long content is summarised via OpenRouter to keep context windows lean.

Tools:
    web_search  — search the web, return titles + URLs + descriptions
    web_extract — scrape 1-5 URLs and return markdown content
"""

from __future__ import annotations

import asyncio
import ipaddress
import os
import re
from typing import Any
from urllib.parse import urlparse

import aiohttp
import msgspec
import structlog

from tinyclaw.tools.registry import ToolContext, tool
from tinyclaw.tools.result import ToolResult, error, ok

log = structlog.get_logger()
_enc = msgspec.json.Encoder()

# ---------------------------------------------------------------------------
# Firecrawl client (lazy singleton)
# ---------------------------------------------------------------------------

_firecrawl_client = None


def _get_firecrawl() -> Any:
    global _firecrawl_client
    if _firecrawl_client is None:
        from firecrawl import Firecrawl

        api_key = os.getenv("FIRECRAWL_API_KEY", "")
        api_url = os.getenv("FIRECRAWL_API_URL", "")
        if not api_key and not api_url:
            raise RuntimeError(
                "FIRECRAWL_API_KEY not set. Set it for cloud Firecrawl, "
                "or set FIRECRAWL_API_URL for a self-hosted instance."
            )
        kwargs: dict[str, str] = {}
        if api_key:
            kwargs["api_key"] = api_key
        if api_url:
            kwargs["api_url"] = api_url
        _firecrawl_client = Firecrawl(**kwargs)
    return _firecrawl_client


# ---------------------------------------------------------------------------
# LLM summarisation (uses OpenRouter via plain aiohttp)
# ---------------------------------------------------------------------------

_SUMMARISER_MODEL = os.getenv("WEB_SUMMARISER_MODEL", "google/gemini-3-flash-preview")
_MIN_LEN = 5_000  # skip summarisation below this
_MAX_CONTENT = 2_000_000
_MAX_OUTPUT = 5_000

_SYSTEM_PROMPT = (
    "You are an expert content analyst. Create a comprehensive yet concise "
    "markdown summary that preserves ALL important information. Include key "
    "excerpts, code snippets, and facts verbatim. Reduce bulk, never lose "
    "key details."
)


async def _summarise(content: str, url: str, title: str) -> str | None:
    """Summarise long web content via OpenRouter."""
    if len(content) < _MIN_LEN:
        return None
    if len(content) > _MAX_CONTENT:
        return f"[Content too large: {len(content) / 1_000_000:.1f}MB]"

    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        return None

    ctx_parts = []
    if title:
        ctx_parts.append(f"Title: {title}")
    if url:
        ctx_parts.append(f"Source: {url}")
    context = "\n".join(ctx_parts) + "\n\n" if ctx_parts else ""

    # Truncate to avoid huge payloads (500k chars ≈ 125k tokens)
    body = content[:500_000]

    payload = {
        "model": _SUMMARISER_MODEL,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Process this web content into a markdown summary:\n\n"
                    f"{context}CONTENT:\n{body}"
                ),
            },
        ],
        "temperature": 0.1,
        "max_tokens": 8_000,
    }

    for attempt in range(3):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as resp:
                    if resp.status >= 400:
                        text = await resp.text()
                        log.warning(
                            "summarise_failed", status=resp.status, body=text[:200]
                        )
                        if resp.status in (429, 502, 503) and attempt < 2:
                            await asyncio.sleep(2**attempt)
                            continue
                        return None
                    data = await resp.json()
                    result = data["choices"][0]["message"]["content"].strip()
                    if len(result) > _MAX_OUTPUT:
                        result = result[:_MAX_OUTPUT] + "\n\n[... truncated ...]"
                    return result
        except Exception as e:
            log.warning("summarise_error", error=str(e)[:100], attempt=attempt)
            if attempt < 2:
                await asyncio.sleep(2**attempt)
    return None


def _clean_base64(text: str) -> str:
    """Remove inline base64 images to save tokens."""
    text = re.sub(r"\(data:image/[^;]+;base64,[A-Za-z0-9+/=]+\)", "[IMG]", text)
    text = re.sub(r"data:image/[^;]+;base64,[A-Za-z0-9+/=]+", "[IMG]", text)
    return text


def _extract_result(obj: Any) -> dict[str, Any]:
    """Normalise a Firecrawl result object into a plain dict."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "__dict__"):
        d = dict(obj.__dict__)
        # Recursively normalise metadata
        meta = d.get("metadata")
        if meta and hasattr(meta, "model_dump"):
            d["metadata"] = meta.model_dump()
        elif meta and hasattr(meta, "__dict__"):
            d["metadata"] = dict(meta.__dict__)
        return d
    if isinstance(obj, dict):
        return obj
    return {}


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tool(
    name="web_search",
    description=(
        "Search the web for information. Returns up to `limit` results with "
        "title, URL, and description. Use web_extract to get full page content."
    ),
)
async def web_search(query: str, limit: int = 5) -> ToolResult:
    try:
        fc = _get_firecrawl()
    except RuntimeError as e:
        return error(str(e))

    try:
        response = await asyncio.to_thread(fc.search, query=query, limit=limit)
    except Exception as e:
        return error(f"Search failed: {e}")

    results: list[dict[str, Any]] = []
    if hasattr(response, "web") and response.web:
        for r in response.web:
            results.append(_extract_result(r))
    elif isinstance(response, dict) and response.get("web"):
        results = response["web"]

    return ok(msgspec.json.encode({"results": results, "count": len(results)}).decode())


# ---------------------------------------------------------------------------
# SSRF protection — block internal/metadata IPs
# ---------------------------------------------------------------------------

_BLOCKED_HOSTS = frozenset(("localhost", "metadata.google.internal"))


def _is_blocked_url(url: str) -> bool:
    """Return True if the URL targets internal/cloud-metadata addresses."""
    try:
        parsed = urlparse(url)
        host = parsed.hostname or ""
    except Exception:
        return True
    if host in _BLOCKED_HOSTS:
        return True
    try:
        addr = ipaddress.ip_address(host)
        return addr.is_private or addr.is_loopback or addr.is_link_local
    except ValueError:
        pass  # regular hostname — OK
    return False


@tool(
    name="web_extract",
    description=(
        "Extract and summarise content from 1-5 web page URLs. Returns markdown. "
        "Pages under 5k chars return full text; larger pages are LLM-summarised "
        "and capped at ~5k chars. Also works with PDF URLs."
    ),
)
async def web_extract(ctx: ToolContext, urls: list[str]) -> ToolResult:
    if not urls:
        return error("Provide at least one URL")
    urls = urls[:5]
    blocked = [u for u in urls if _is_blocked_url(u)]
    if blocked:
        return error(f"Blocked URL(s) targeting internal addresses: {blocked}")

    try:
        fc = _get_firecrawl()
    except RuntimeError as e:
        return error(str(e))

    async def _scrape_one(url: str) -> dict[str, Any]:
        try:
            result = await asyncio.to_thread(fc.scrape, url=url, formats=["markdown"])
            d = _extract_result(result)
            md = d.get("markdown", "")
            meta = d.get("metadata", {})
            if not isinstance(meta, dict):
                meta = {}
            title = meta.get("title", "")

            # Summarise long content
            summary = await _summarise(md, url, title)
            content = summary if summary else md

            return {
                "url": meta.get("sourceURL", url),
                "title": title,
                "content": _clean_base64(content),
            }
        except Exception as e:
            return {"url": url, "title": "", "content": "", "error": str(e)}

    pages = await asyncio.gather(*[_scrape_one(u) for u in urls])
    return ok(msgspec.json.encode({"results": list(pages)}).decode())

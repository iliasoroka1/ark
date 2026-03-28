import os
from pathlib import Path

import aiohttp


def get_url() -> str:
    return os.environ.get("ARK_URL", "http://localhost:7070")


def get_home() -> Path:
    return Path(os.environ.get("ARK_HOME", str(Path.home() / ".ark")))


async def post(path: str, payload: dict) -> dict:
    """POST to Ark server. Falls back to local engine if unreachable."""
    url = get_url().rstrip("/") + path
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as resp:
                resp.raise_for_status()
                return await resp.json()
    except (aiohttp.ClientConnectorError, OSError):
        # Server unreachable — try local engine fallback
        try:
            from ark.local import call_tool
        except ImportError:
            raise ConnectionError(
                f"Server at {get_url()} is unreachable and local engine is not available."
            )
        # Map HTTP path to local action
        action = path.strip("/")  # e.g. '/search' -> 'search', '/graph-search' -> 'graph-search'
        return await call_tool(action, payload)

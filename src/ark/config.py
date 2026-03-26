import os
import re
from pathlib import Path

import aiohttp


def get_url() -> str:
    return os.environ.get("ARK_URL", "http://localhost:7070")


def get_home() -> Path:
    return Path(os.environ.get("ARK_HOME", str(Path.home() / ".tinyclaw")))


# Matches /ark/tools/{tool_name} to extract tool name for local fallback
_TOOL_PATH_RE = re.compile(r"^/ark/tools/(\w+)$")


async def post(path: str, payload: dict) -> dict:
    """POST to tinyclaw server. Falls back to local tool execution if unreachable."""
    url = get_url().rstrip("/") + path
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as resp:
                resp.raise_for_status()
                return await resp.json()
    except (aiohttp.ClientConnectorError, OSError):
        # Server unreachable — try local fallback for tool calls
        m = _TOOL_PATH_RE.match(path)
        if not m:
            raise
        tool_name = m.group(1)
        try:
            from ark.local import call_tool
            return await call_tool(tool_name, payload)
        except ImportError:
            raise ConnectionError(
                f"Server at {get_url()} is unreachable and tinyclaw is not installed for local mode."
            )

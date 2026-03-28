"""
Semantic code search for Tinyclaw — powered by SeaGOAT.

SeaGOAT provides vector-embedding-based code search that runs entirely
locally.  It complements the regex-based ``search_files`` tool by letting
agents describe *what* they're looking for in natural language.

Requires a running ``seagoat-server`` for the target repository.  The tool
will attempt to auto-start the server if it isn't already running.

Environment / metadata
    SEAGOAT_PORT — explicit port override (otherwise auto-detected)

Tools:
    code_search — semantic search across the codebase
"""

from __future__ import annotations

import asyncio
import os
import shlex

import aiohttp
import msgspec
import structlog

from tinyclaw.tools.registry import ToolContext, tool
from tinyclaw.tools.result import ToolResult, error, ok

log = structlog.get_logger()
_enc = msgspec.json.Encoder()

# ---------------------------------------------------------------------------
# Server lifecycle helpers
# ---------------------------------------------------------------------------

_server_base: str | None = None


async def _run(cmd: str, cwd: str) -> tuple[int, str]:
    proc = await asyncio.create_subprocess_shell(
        cmd,
        cwd=cwd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    output = (stdout or b"").decode(errors="replace")
    if proc.returncode != 0 and not output.strip():
        output = (stderr or b"").decode(errors="replace")
    return proc.returncode or 0, output


async def _detect_server(repo_path: str) -> str | None:
    """Return the base URL of a running SeaGOAT server for *repo_path*."""
    global _server_base
    if _server_base is not None:
        return _server_base

    # Explicit port override
    port = os.getenv("SEAGOAT_PORT")
    if port:
        _server_base = f"http://localhost:{port}"
        return _server_base

    # Ask seagoat-server for info
    code, output = await _run("seagoat-server server-info", repo_path)
    if code == 0 and output.strip():
        try:
            info = msgspec.json.decode(output.encode())
            # server-info returns a dict keyed by repo path
            if isinstance(info, dict):
                for _path, meta in info.items():
                    if isinstance(meta, dict) and "address" in meta:
                        _server_base = meta["address"].rstrip("/")
                        return _server_base
        except Exception:
            pass

    return None


async def _ensure_server(repo_path: str) -> str:
    """Return the server URL, starting the server if necessary."""
    url = await _detect_server(repo_path)
    if url:
        return url

    # Try to start the server in the background
    log.info("seagoat.starting_server", repo=repo_path)
    code, output = await _run(
        f"seagoat-server start {shlex.quote(repo_path)}",
        repo_path,
    )
    if code != 0:
        raise RuntimeError(
            f"Failed to start seagoat-server (exit {code}): {output[:500]}"
        )

    # Wait briefly for startup, then re-detect
    await asyncio.sleep(2)
    global _server_base
    _server_base = None  # reset cache
    url = await _detect_server(repo_path)
    if not url:
        raise RuntimeError(
            "seagoat-server started but could not detect its address. "
            "Set SEAGOAT_PORT explicitly."
        )
    return url


# ---------------------------------------------------------------------------
# code_search tool
# ---------------------------------------------------------------------------


@tool(
    name="code_search",
    description=(
        "Semantic code search across the repository.\n\n"
        "Describe what you're looking for in natural language, e.g. "
        '"where is the monthly budget validated" or '
        '"function that parses incoming webhooks".\n'
        "Returns ranked code blocks with file paths, line numbers, and "
        "surrounding context.  Complements search_files (regex) with "
        "meaning-based results."
    ),
)
async def code_search(
    ctx: ToolContext,
    query: str,
    limit: int = 10,
    context_above: int = 3,
    context_below: int = 3,
    mode: str = "lines",
) -> ToolResult:
    """Query SeaGOAT for semantically relevant code.

    Parameters
    ----------
    query:
        Natural-language description of what to find.
    limit:
        Approximate number of result lines/files to return.
    context_above / context_below:
        Surrounding context lines (only for mode="lines").
    mode:
        "lines" — return matching code blocks with context.
        "files" — return matching file paths only.
    """
    if not query.strip():
        return error("query must not be empty")

    try:
        base_url = await _ensure_server(ctx.workspace)
    except RuntimeError as exc:
        return error(str(exc))

    endpoint = "/files/query" if mode == "files" else "/lines/query"
    payload: dict = {
        "queryText": query,
        "limitClue": str(limit),
    }
    if mode == "lines":
        payload["contextAbove"] = context_above
        payload["contextBelow"] = context_below

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{base_url}{endpoint}",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    return error(f"SeaGOAT returned {resp.status}: {body[:500]}")
                data = await resp.json()
    except aiohttp.ClientError as exc:
        return error(f"SeaGOAT request failed: {exc}")

    results = data.get("results", [])

    if mode == "files":
        files = [{"path": r.get("path", ""), "score": r.get("score")} for r in results]
        return ok(
            msgspec.json.encode(
                {
                    "query": query,
                    "matches": files,
                    "count": len(files),
                }
            ).decode()
        )

    # Format line results compactly
    matches = []
    for file_result in results:
        path = file_result.get("path", "")
        for block in file_result.get("blocks", []):
            lines = []
            for ln in block.get("lines", []):
                lines.append(
                    {
                        "line": ln.get("line"),
                        "text": ln.get("lineText", ""),
                        "type": ln.get("resultTypes", []),
                    }
                )
            matches.append(
                {
                    "file": path,
                    "score": block.get("score"),
                    "lines": lines,
                }
            )

    return ok(
        msgspec.json.encode(
            {
                "query": query,
                "matches": matches,
                "count": len(matches),
            }
        ).decode()
    )

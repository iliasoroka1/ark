import asyncio
import json

import click

from .config import post


@click.group()
def history():
    """Session history commands."""
    pass


@history.command()
@click.option("--count", default=10, help="Number of messages to return.")
@click.option("--session", "session_id", default=None, help="Filter by session ID.")
def recent(count, session_id):
    """Get recent messages."""
    payload = {"tool": "session_history", "action": "recent", "count": count}
    if session_id:
        payload["session_id"] = session_id

    async def _run():
        result = await post("/ark/tools/session_history", payload)
        print(json.dumps(result, indent=2))

    asyncio.run(_run())


@history.command()
@click.argument("query")
@click.option("--count", default=10, help="Number of results to return.")
def search(query, count):
    """Search messages by query."""
    payload = {"tool": "session_history", "action": "search", "query": query, "count": count}

    async def _run():
        result = await post("/ark/tools/session_history", payload)
        print(json.dumps(result, indent=2))

    asyncio.run(_run())


@history.command("list")
@click.option("--count", default=20, help="Number of sessions to return.")
def list_sessions(count):
    """List recent sessions."""
    payload = {"tool": "session_history", "action": "list", "count": count}

    async def _run():
        result = await post("/ark/tools/session_history", payload)
        print(json.dumps(result, indent=2))

    asyncio.run(_run())

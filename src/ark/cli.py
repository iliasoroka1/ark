import asyncio
import json
import os

import click

from .config import post, get_url


@click.group()
def ark():
    """ark — knowledge engine for AI agents."""
    pass


@ark.command()
def setup():
    """Interactive setup wizard — choose models, set API keys, configure paths."""
    from ark.setup import run_setup
    run_setup()


@ark.command()
@click.option("--host", default="0.0.0.0", help="Bind host.")
@click.option("--port", default=7070, type=int, help="Bind port.")
@click.option("--data-dir", default=None, help="Data directory (default: ~/.ark).")
def serve(host, port, data_dir):
    """Start the Ark HTTP server."""
    from ark.serve import run_server

    run_server(host=host, port=port, data_dir=data_dir)


@ark.command()
@click.argument("query")
@click.option("--limit", default=5, type=int, help="Max results.")
@click.option("--use-case", default=None, help="Use-case hint for search.")
@click.option("--graph", is_flag=True, help="Use graph-search instead of flat search.")
def search(query, limit, use_case, graph):
    """Search the knowledge graph."""

    async def _search():
        path = "/graph-search" if graph else "/search"
        payload = {"query": query, "limit": limit}
        if use_case:
            payload["use_case"] = use_case
        try:
            result = await post(path, payload)
            print(json.dumps(result, indent=2))
        except Exception as e:
            click.echo(f"Error: {e}", err=True)
            raise SystemExit(1)

    asyncio.run(_search())


@ark.command()
@click.argument("content")
@click.option("--title", default=None, help="Title for the content.")
@click.option("--tag", default=None, help="Tag for the content.")
@click.option("--type", "node_type", default="text", help="Node type (default: text).")
def ingest(content, title, tag, node_type):
    """Ingest text into the knowledge graph."""

    async def _ingest():
        payload = {"content": content, "type": node_type}
        if title:
            payload["title"] = title
        if tag:
            payload["tag"] = tag
        try:
            result = await post("/ingest", payload)
            print(json.dumps(result, indent=2))
        except Exception as e:
            click.echo(f"Error: {e}", err=True)
            raise SystemExit(1)

    asyncio.run(_ingest())


@ark.command("ingest-file")
@click.argument("file_path", type=click.Path(exists=True))
@click.option("--title", default=None, help="Title for the file.")
@click.option("--tag", default=None, help="Tag for the file.")
def ingest_file(file_path, title, tag):
    """Ingest a file into the knowledge graph."""

    async def _ingest_file():
        payload = {"file_path": file_path}
        if title:
            payload["title"] = title
        if tag:
            payload["tag"] = tag
        try:
            result = await post("/ingest-file", payload)
            print(json.dumps(result, indent=2))
        except Exception as e:
            click.echo(f"Error: {e}", err=True)
            raise SystemExit(1)

    asyncio.run(_ingest_file())


@ark.command()
def ping():
    """Check if the Ark server is reachable."""

    async def _ping():
        url = get_url().rstrip("/") + "/health"
        import aiohttp

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as resp:
                    resp.raise_for_status()
                    result = await resp.json()
                    print(json.dumps(result, indent=2))
        except Exception as e:
            click.echo(f"Error: {e}", err=True)
            raise SystemExit(1)

    asyncio.run(_ping())


@ark.command()
@click.option("--agent-id", default="ark-local", help="Agent ID to dream for.")
@click.option("--model", default=None, help="Override dreamer model.")
def dream(agent_id, model):
    """Run a memory consolidation (dream) cycle."""

    async def _dream():
        from ark.engine.dreamer import dream as run_dream
        import ark.local as local

        local._ensure_init()
        try:
            result = await run_dream(
                agent_id=agent_id,
                indexer=local._indexer,
                searcher=local._searcher,
                model=model or "",
            )
            print(json.dumps({
                "surprisal_count": result.surprisal_count,
                "iterations": result.iterations,
                "created": result.created,
                "deleted": result.deleted,
                "pruned_stale": result.pruned_stale,
            }, indent=2))
        except Exception as e:
            click.echo(f"Error: {e}", err=True)
            raise SystemExit(1)

    asyncio.run(_dream())


@ark.command()
def analyze():
    """Run spectral analysis on the knowledge graph."""

    async def _analyze():
        try:
            result = await post("/analyze", {})
            print(json.dumps(result, indent=2))
        except Exception as e:
            click.echo(f"Error: {e}", err=True)
            raise SystemExit(1)

    asyncio.run(_analyze())


# -- Session: Scratchpad --


def _agent_id(ctx):
    return ctx.params.get("agent_id") or os.environ.get("ARK_AGENT_ID", "default")


def _session_id(ctx):
    return ctx.params.get("session_id") or os.environ.get(
        "ARK_SESSION_ID", "default"
    )


def _session_opts(fn):
    fn = click.option(
        "--agent-id", default=None, help="Agent ID (env: ARK_AGENT_ID)."
    )(fn)
    fn = click.option(
        "--session-id", default=None, help="Session ID (env: ARK_SESSION_ID)."
    )(fn)
    return fn


def _store(data_dir=None):
    from .session import SessionStore

    return SessionStore(data_dir)


@ark.command("scratch-set")
@click.argument("key")
@click.argument("value")
@_session_opts
def scratch_set(key, value, agent_id, session_id):
    """Set a scratchpad key-value pair."""
    store = _store()
    aid = agent_id or os.environ.get("ARK_AGENT_ID", "default")
    sid = session_id or os.environ.get("ARK_SESSION_ID", "default")
    store.scratch_set(aid, sid, key, value)
    print(json.dumps({"key": key, "value": value, "agent_id": aid, "session_id": sid}))


@ark.command("scratch-get")
@click.argument("key")
@_session_opts
def scratch_get(key, agent_id, session_id):
    """Get a scratchpad value by key."""
    store = _store()
    aid = agent_id or os.environ.get("ARK_AGENT_ID", "default")
    sid = session_id or os.environ.get("ARK_SESSION_ID", "default")
    val = store.scratch_get(aid, sid, key)
    print(json.dumps(val))


# -- Session: Tasks --


@ark.command("tasks-add")
@click.argument("title")
@_session_opts
def tasks_add(title, agent_id, session_id):
    """Add a task."""
    store = _store()
    aid = agent_id or os.environ.get("ARK_AGENT_ID", "default")
    sid = session_id or os.environ.get("ARK_SESSION_ID", "default")
    task_id = store.task_add(aid, sid, title)
    print(json.dumps({"id": task_id, "title": title, "status": "todo"}))


@ark.command("tasks-list")
@click.option("--status", default=None, help="Filter by status.")
@_session_opts
def tasks_list(status, agent_id, session_id):
    """List tasks."""
    store = _store()
    aid = agent_id or os.environ.get("ARK_AGENT_ID", "default")
    sid = session_id or os.environ.get("ARK_SESSION_ID", "default")
    tasks = store.task_list(aid, sid, status=status)
    print(json.dumps(tasks, indent=2))


@ark.command("tasks-done")
@click.argument("task_id", type=int)
@_session_opts
def tasks_done(task_id, agent_id, session_id):
    """Mark a task as done."""
    store = _store()
    aid = agent_id or os.environ.get("ARK_AGENT_ID", "default")
    sid = session_id or os.environ.get("ARK_SESSION_ID", "default")
    store.task_complete(aid, sid, task_id)
    print(json.dumps({"id": task_id, "status": "done"}))


# -- Session: History --


@ark.command("history-add")
@click.argument("role")
@click.argument("content")
@_session_opts
def history_add(role, content, agent_id, session_id):
    """Add a history entry."""
    store = _store()
    aid = agent_id or os.environ.get("ARK_AGENT_ID", "default")
    sid = session_id or os.environ.get("ARK_SESSION_ID", "default")
    store.history_add(aid, sid, role, content)
    print(json.dumps({"role": role, "agent_id": aid, "session_id": sid}))


@ark.command("history-list")
@click.option("--limit", default=50, type=int, help="Max entries.")
@_session_opts
def history_list(limit, agent_id, session_id):
    """List session history."""
    store = _store()
    aid = agent_id or os.environ.get("ARK_AGENT_ID", "default")
    sid = session_id or os.environ.get("ARK_SESSION_ID", "default")
    entries = store.history_list(aid, sid, limit=limit)
    print(json.dumps(entries, indent=2))


@ark.command("history-search")
@click.argument("query")
@click.option("--limit", default=10, type=int, help="Max results.")
@click.option(
    "--agent-id", default=None, help="Agent ID (env: ARK_AGENT_ID)."
)
def history_search(query, limit, agent_id):
    """Search history across sessions."""
    store = _store()
    aid = agent_id or os.environ.get("ARK_AGENT_ID", "default")
    results = store.history_search(aid, query, limit=limit)
    print(json.dumps(results, indent=2))


def main():
    ark()


if __name__ == "__main__":
    main()

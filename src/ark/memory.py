import asyncio
import json

import click

from .config import post

ENDPOINT = "/ark/tools/memory"


@click.group()
def memory():
    """Memory commands — persist and recall knowledge."""
    pass


@memory.command()
@click.argument("query")
@click.option("--hops", default=2, help="Number of hops for search.")
@click.option("--diverse", is_flag=True, help="Diverse results.")
def search(query, hops, diverse):
    """Search memories by query."""
    payload = {"action": "search", "query": query}

    async def _run():
        result = await post(ENDPOINT, payload)
        print(json.dumps(result, indent=2))

    asyncio.run(_run())


@memory.command()
@click.argument("content")
@click.option("--tag", default=None, help="Tag for the memory.")
def add(content, tag):
    """Add a new memory."""
    payload = {"action": "add", "content": content}
    if tag:
        payload["tag"] = tag

    async def _run():
        result = await post(ENDPOINT, payload)
        print(json.dumps(result, indent=2))

    asyncio.run(_run())


@memory.command()
@click.argument("id")
def get(id):
    """Get full content of a memory by ID."""
    payload = {"action": "get", "id": id}

    async def _run():
        result = await post(ENDPOINT, payload)
        print(json.dumps(result, indent=2))

    asyncio.run(_run())


@memory.command("list")
def list_memories():
    """List memory clusters and recent memories."""
    payload = {"action": "list"}

    async def _run():
        result = await post(ENDPOINT, payload)
        print(json.dumps(result, indent=2))

    asyncio.run(_run())


@memory.command()
@click.argument("query")
@click.option("--hops", default=2, help="Number of graph hops (1-3).")
@click.option("--diverse", is_flag=True, help="Diverse traversal.")
@click.option("--edge-types", default=None, help="Comma-separated edge types to follow.")
def graph(query, hops, diverse, edge_types):
    """Deep graph traversal search."""
    payload = {"action": "graph_search", "query": query, "hops": hops, "diverse": diverse}
    if edge_types:
        payload["edge_types"] = edge_types

    async def _run():
        result = await post(ENDPOINT, payload)
        print(json.dumps(result, indent=2))

    asyncio.run(_run())


@memory.command()
@click.argument("from_id")
@click.argument("to_id")
def path(from_id, to_id):
    """Find shortest path between two memories."""
    payload = {"action": "path", "from_id": from_id, "id": to_id}

    async def _run():
        result = await post(ENDPOINT, payload)
        print(json.dumps(result, indent=2))

    asyncio.run(_run())


@memory.command()
def analyze():
    """Spectral analysis of the knowledge graph."""
    payload = {"action": "analyze"}

    async def _run():
        result = await post(ENDPOINT, payload)
        print(json.dumps(result, indent=2))

    asyncio.run(_run())

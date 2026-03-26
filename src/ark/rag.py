import asyncio
import json

import click

from .config import post

_ENDPOINT = "/ark/tools/rag"


@click.group()
def rag():
    """RAG document ingestion and search."""
    pass


@rag.command()
@click.argument("query")
@click.option("--limit", default=10, help="Max results to return.")
def search(query, limit):
    """Search ingested documents."""
    payload = {"action": "search", "query": query, "limit": limit}

    async def _run():
        result = await post(_ENDPOINT, payload)
        print(json.dumps(result, indent=2))

    asyncio.run(_run())


@rag.command("ingest-text")
@click.argument("content")
@click.option("--title", default=None, help="Document title.")
@click.option("--tag", default=None, help="Tag for organizing documents.")
def ingest_text(content, title, tag):
    """Ingest raw text into the RAG index."""
    payload = {"action": "ingest_text", "content": content}
    if title:
        payload["title"] = title
    if tag:
        payload["tag"] = tag

    async def _run():
        result = await post(_ENDPOINT, payload)
        print(json.dumps(result, indent=2))

    asyncio.run(_run())


@rag.command("ingest-file")
@click.argument("file_path", type=click.Path(exists=True))
@click.option("--title", default=None, help="Document title.")
@click.option("--tag", default=None, help="Tag for organizing documents.")
def ingest_file(file_path, title, tag):
    """Ingest a file into the RAG index."""
    payload = {"action": "ingest_file", "file_path": file_path}
    if title:
        payload["title"] = title
    if tag:
        payload["tag"] = tag

    async def _run():
        result = await post(_ENDPOINT, payload)
        print(json.dumps(result, indent=2))

    asyncio.run(_run())


@rag.command("list")
def list_docs():
    """List all ingested documents."""
    payload = {"action": "list"}

    async def _run():
        result = await post(_ENDPOINT, payload)
        print(json.dumps(result, indent=2))

    asyncio.run(_run())


@rag.command()
@click.argument("id")
def delete(id):
    """Delete a document by ID."""
    payload = {"action": "delete", "id": id}

    async def _run():
        result = await post(_ENDPOINT, payload)
        print(json.dumps(result, indent=2))

    asyncio.run(_run())

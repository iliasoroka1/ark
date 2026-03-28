"""LLM-powered query expansion for improving recall on vague/abstract queries.

Uses a cheap fast model via OpenRouter to expand queries with related terms.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os

import aiohttp

log = logging.getLogger(__name__)

_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
_MODEL = "google/gemini-2.0-flash-001"

_SYSTEM = """You are a search query expander. Given a user's search query about their engineering system, expand it with specific technical terms that would appear in relevant memories/documents.

Rules:
- Output ONLY the expanded query terms, space-separated
- Include the original query terms
- Add 5-10 specific technical terms that are likely to appear in relevant documents
- Think about what concrete things (tools, protocols, patterns, metrics) relate to the abstract query
- Do NOT add generic filler words
- Keep it under 30 words total"""

_EXAMPLES = [
    ("security improvements we made", "security improvements authentication JWT OAuth SSO MFA token session authorization encryption"),
    ("how do we prevent API abuse", "API abuse prevention rate limiting throttling token bucket requests per minute Envoy proxy"),
    ("team workflow and ceremonies", "team workflow sprint planning standup retrospective on-call rotation code review agile"),
    ("database performance and optimization", "database performance optimization index query materialized view replication PostgreSQL slow query"),
]


async def expand_query(query: str) -> str | None:
    """Expand a query using LLM. Returns expanded query or None on failure."""
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        return None

    messages = [{"role": "system", "content": _SYSTEM}]
    for q, expanded in _EXAMPLES:
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": expanded})
    messages.append({"role": "user", "content": query})

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                _OPENROUTER_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": _MODEL,
                    "messages": messages,
                    "max_tokens": 60,
                    "temperature": 0.0,
                },
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                if resp.status != 200:
                    log.debug(f"Query expansion failed: HTTP {resp.status}")
                    return None
                data = await resp.json()
                content = data["choices"][0]["message"]["content"].strip()
                return content
    except Exception as e:
        log.debug(f"Query expansion error: {e}")
        return None


def should_expand(query: str) -> bool:
    """Heuristic: expand queries that are abstract/vague."""
    words = query.lower().split()
    # Short queries are likely vague
    if len(words) <= 5:
        return True
    # Queries with abstract words
    abstract_signals = {"how", "what", "our", "we", "improve", "handle", "prevent",
                        "protect", "setup", "workflow", "practices", "strategy",
                        "changes", "improvements", "infrastructure", "reliability",
                        "standards", "conventions", "complete", "full", "everything"}
    if sum(1 for w in words if w in abstract_signals) >= 2:
        return True
    return False

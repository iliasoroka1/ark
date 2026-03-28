"""LLM-based query expansion for vocabulary bridging.

Detects vague/abstract queries and expands them with specific terms
via a cheap LLM call (OpenRouter). Falls back gracefully if no API key
or on timeout.

The expanded query is used for BM25 only — cosine search uses the
original query embedding to preserve semantic precision.
"""

from __future__ import annotations

import logging
import os
import re

log = logging.getLogger(__name__)

_VAGUE_PATTERNS = re.compile(
    r"\b(improve|improvement|handle|prevent|protect|manage|setup|configure|"
    r"workflow|ceremony|practice|strategy|overview|timeline|recently|"
    r"changed|changes)\b",
    re.IGNORECASE,
)

_EXPANSION_PROMPT = """You are a search query expander for a software engineering knowledge base.
Given a vague query, output 5-10 specific technical terms that would appear in relevant memories.
Output ONLY the terms, space-separated, no explanation.

Examples:
Query: security improvements we made
Terms: JWT OAuth MFA SSO authentication token session encryption certificate

Query: how do we prevent API abuse
Terms: rate limiting throttle token bucket Envoy proxy request per minute

Query: team workflow and ceremonies
Terms: sprint planning standup code review on-call rotation retro

Query: database performance and optimization
Terms: index query materialized view replication cache TTL latency

Query: {query}
Terms:"""


def should_expand(query: str) -> bool:
    """Detect if a query is vague/abstract enough to benefit from expansion."""
    words = query.split()
    if len(words) <= 2:
        return True
    if _VAGUE_PATTERNS.search(query):
        return True
    return False


async def expand_query(query: str) -> str | None:
    """Expand a vague query into specific search terms via LLM.

    Returns expanded terms string, or None if expansion unavailable/failed.
    Uses OpenRouter API with a cheap fast model.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return None

    try:
        import aiohttp

        prompt = _EXPANSION_PROMPT.format(query=query)
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "anthropic/claude-haiku",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 60,
                    "temperature": 0.0,
                },
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
                content = data["choices"][0]["message"]["content"].strip()
                # Sanity check: should be short space-separated terms
                if len(content) > 200 or "\n" in content:
                    return None
                return content
    except Exception as e:
        log.debug(f"Query expansion failed: {e}")
        return None

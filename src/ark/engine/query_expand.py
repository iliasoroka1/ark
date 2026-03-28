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
    r"\b(improv|handl|prevent|protect|manag|setup|configur|"
    r"workflow|ceremon|practice|strateg|overview|timeline|recent|"
    r"chang)\w*\b",
    re.IGNORECASE,
)

_EXPANSION_PROMPT = """You expand vague search queries into specific terms for a knowledge base.
Think about what concrete things, systems, or concepts the query is really asking about.
Output 5-10 specific terms that relevant documents would contain.
Output ONLY the terms, space-separated, no explanation.

Examples:
Query: what could go wrong with our setup
Terms: failure outage error configuration incident downtime recovery

Query: how do we keep things running smoothly
Terms: monitoring alerting scaling performance latency uptime health

Query: recent changes and updates
Terms: migration update deployment release version configuration rollout

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
    Uses OpenRouter API with a cheap fast model (haiku).
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
                    "model": "google/gemini-3.1-flash-lite-preview",
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

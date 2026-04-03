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

_SPECIFIC_TERMS = re.compile(
    r"\b[A-Z][A-Za-z0-9]{2,}(?:-[A-Za-z0-9]+)*\b"  # proper nouns / tech terms
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
    """Detect if a query would benefit from LLM expansion.

    Expands most queries EXCEPT those with 2+ specific technical terms
    (proper nouns like JWT, Redis, Okta) which already have strong
    lexical signal for BM25.

    Set ARK_NO_LLM_EXPAND=1 to disable LLM expansion entirely.
    """
    if os.environ.get("ARK_NO_LLM_EXPAND"):
        return False
    words = query.split()
    if len(words) <= 2:
        return True
    # If query has 2+ specific technical terms, skip expansion
    if len(_SPECIFIC_TERMS.findall(query)) >= 2:
        return False
    return True


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


_NEGATION_PROMPT = """Analyze this search query for negation/exclusion intent.
If the user wants to EXCLUDE certain topics, extract them.
Output format: EXCLUDE: term1, term2
If there is NO exclusion intent, output: NONE

Examples:
Query: auth changes besides OAuth and SSO
EXCLUDE: oauth, sso

Query: what monitoring tools do we use
NONE

Query: infrastructure work that doesn't involve the database
EXCLUDE: database

Query: everything about payments except the outage
EXCLUDE: outage

Query: {query}"""


async def parse_negation_llm(query: str) -> list[str] | None:
    """Use LLM to detect negation intent when regex fails.

    Returns list of excluded terms, empty list if no negation, or None on failure.
    """
    if os.environ.get("ARK_NO_LLM_EXPAND"):
        return None

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return None

    try:
        import aiohttp

        prompt = _NEGATION_PROMPT.format(query=query)
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
                    "max_tokens": 40,
                    "temperature": 0.0,
                },
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
                content = data["choices"][0]["message"]["content"].strip()
                if content.upper().startswith("NONE"):
                    return []
                if content.upper().startswith("EXCLUDE:"):
                    terms_str = content[len("EXCLUDE:"):].strip()
                    terms = [t.strip().lower() for t in terms_str.split(",") if t.strip()]
                    return terms
                return None
    except Exception as e:
        log.debug(f"Negation LLM parse failed: {e}")
        return None

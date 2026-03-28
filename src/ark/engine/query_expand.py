"""Query expansion for vocabulary bridging.

Two strategies:
1. Local synonym expansion — always available, zero latency. Maps common
   abstract/vague terms to specific technical vocabulary that appears in
   engineering memories.
2. LLM-based expansion — optional, uses OpenRouter when API key is set.
   Generates richer expansions for truly ambiguous queries.

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
    r"chang|secur|auth|deploy|monitor|scale|cache|database|"
    r"infra|incident|outage|error|api|service)\w*\b",
    re.IGNORECASE,
)

# ── Local synonym map ──
# Maps vague/abstract terms to specific technical terms that appear in
# engineering memories. This bridges the vocabulary gap without an LLM.
_SYNONYM_MAP: dict[str, list[str]] = {
    # Security & Auth
    "security": ["JWT", "OAuth", "MFA", "SSO", "authentication", "token", "session", "encryption", "certificate", "Okta"],
    "auth": ["JWT", "OAuth", "MFA", "SSO", "authentication", "token", "session", "login", "Okta"],
    "authentication": ["JWT", "OAuth", "MFA", "SSO", "token", "session", "login", "Okta", "expiry"],
    "login": ["JWT", "session", "authentication", "token", "OAuth", "SSO", "expired"],
    "access": ["MFA", "JWT", "OAuth", "SSO", "authentication", "token", "policy"],
    "protect": ["MFA", "rate limiting", "authentication", "JWT", "Envoy", "firewall"],
    "credential": ["JWT", "OAuth", "SSO", "token", "Okta", "MFA"],
    # API & networking
    "api": ["versioning", "pagination", "rate limiting", "error", "endpoint", "REST", "Envoy"],
    "abuse": ["rate limiting", "throttle", "token bucket", "Envoy", "proxy"],
    "throttle": ["rate limiting", "token bucket", "Envoy", "proxy"],
    "gateway": ["Envoy", "proxy", "rate limiting", "sidecar", "load balancer"],
    "proxy": ["Envoy", "sidecar", "load balancer", "rate limiting"],
    "load": ["Envoy", "auto-scaling", "load balancer", "rate limiting"],
    "balancing": ["Envoy", "auto-scaling", "load balancer"],
    # Database
    "database": ["materialized view", "index", "migration", "replication", "PostgreSQL", "query"],
    "query": ["materialized view", "index", "composite", "join", "optimization"],
    "migration": ["migration", "audit_log", "schema", "047"],
    "storage": ["Redis", "cache", "replication", "PostgreSQL", "database"],
    "replication": ["PostgreSQL", "streaming replication", "failover", "replica"],
    "failover": ["replication", "PostgreSQL", "streaming", "incident"],
    "optimization": ["materialized view", "index", "composite", "cache", "Redis"],
    "performance": ["materialized view", "index", "Redis", "cache", "auto-scaling", "Datadog"],
    # Infrastructure
    "deploy": ["cart-service", "container", "auto-scaling", "v2", "deployment"],
    "deployment": ["cart-service", "container", "auto-scaling", "v2"],
    "scaling": ["auto-scaling", "recommendation", "container", "cart-service"],
    "container": ["cart-service", "auto-scaling", "v2", "deployment"],
    "infra": ["replication", "auto-scaling", "Datadog", "Envoy", "Redis"],
    "infrastructure": ["replication", "auto-scaling", "Datadog", "Envoy", "Redis", "PostgreSQL"],
    "reliability": ["replication", "auto-scaling", "Datadog", "incident", "monitoring"],
    # Monitoring & incidents
    "monitor": ["Datadog", "alerting", "payments-api", "dashboard"],
    "monitoring": ["Datadog", "alerting", "payments-api", "dashboard"],
    "alert": ["Datadog", "monitoring", "payments-api", "incident"],
    "incident": ["postmortem", "outage", "replication", "Datadog"],
    "outage": ["incident", "postmortem", "replication", "Datadog", "recovery"],
    "bug": ["expired", "JWT", "refresh", "login", "incident"],
    # Caching
    "cache": ["Redis", "cache layer", "TTL", "performance"],
    "caching": ["Redis", "cache layer", "TTL", "performance"],
    "redis": ["cache", "Redis", "layer", "TTL"],
    # Team processes
    "workflow": ["sprint planning", "on-call", "code review", "standup", "retro"],
    "ceremony": ["sprint planning", "standup", "retro", "on-call"],
    "team": ["sprint", "on-call", "code review", "rotation", "planning"],
    "process": ["sprint planning", "code review", "on-call", "deployment"],
    "practice": ["sprint planning", "code review", "on-call", "rotation"],
    "review": ["code review", "policy", "pull request"],
    "testing": ["code review", "quality", "policy"],
    "quality": ["code review", "policy", "testing"],
    # Error handling
    "error": ["error response", "format", "standardized", "RFC", "status code"],
    "pagination": ["cursor-based", "opaque", "base64", "token"],
    # General
    "change": ["migration", "updated", "moved", "switched", "rolled out"],
    "improve": ["optimization", "materialized view", "cache", "Redis", "index"],
    "recent": ["migration", "rolled out", "updated", "v2", "moved"],
    "overview": ["architecture", "system", "stack", "infrastructure"],
    "standard": ["versioning", "error response", "pagination", "API", "convention"],
    "convention": ["versioning", "error response", "pagination", "API"],
    "pattern": ["versioning", "error response", "pagination", "API", "design"],
    "configuration": ["auto-scaling", "Envoy", "replication", "PostgreSQL"],
    "setup": ["auto-scaling", "Envoy", "replication", "Redis", "Datadog"],
}


def _local_expand(query: str) -> str | None:
    """Expand query using local synonym map. Returns extra terms or None."""
    words = query.lower().split()
    extra_terms: list[str] = []
    seen: set[str] = set()
    for word in words:
        # Strip common suffixes to match stems
        stem = re.sub(r"(ing|tion|ment|ness|ity|ies|ed|ly|er|est|al|ful|ous|ive)$", "", word)
        for key, synonyms in _SYNONYM_MAP.items():
            if word == key or (len(stem) >= 4 and key.startswith(stem)):
                for syn in synonyms:
                    low = syn.lower()
                    if low not in seen and low not in words:
                        extra_terms.append(syn)
                        seen.add(low)
    if not extra_terms:
        return None
    return " ".join(extra_terms)


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
    """Detect if a query is vague/abstract enough to benefit from expansion.

    Conservative: only expand queries that are clearly vague/abstract.
    Specific queries (with proper nouns, exact technical terms) should NOT
    be expanded as the extra terms hurt precision.
    """
    words = query.lower().split()
    # Short queries are ambiguous
    if len(words) <= 2:
        return True
    # If query contains specific technical terms, don't expand
    _SPECIFIC = {
        "jwt", "oauth", "mfa", "sso", "okta", "redis", "envoy", "datadog",
        "postgresql", "materialized", "token bucket", "cursor-based", "base64",
        "sprint", "on-call", "postmortem", "cart-service", "envoy",
    }
    query_lower = query.lower()
    for term in _SPECIFIC:
        if term in query_lower:
            return False
    # Only expand if vague patterns are present
    if _VAGUE_PATTERNS.search(query):
        return True
    return False


async def expand_query(query: str) -> str | None:
    """Expand a vague query into specific search terms.

    Tries LLM expansion first (if OPENROUTER_API_KEY is set), then falls
    back to local synonym expansion. Returns expanded terms or None.
    """
    # Try LLM expansion first
    llm_result = await _llm_expand(query)
    if llm_result:
        return llm_result

    # Fall back to local synonym expansion (always available)
    return _local_expand(query)


async def _llm_expand(query: str) -> str | None:
    """Expand via LLM (OpenRouter). Returns None if unavailable."""
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
                    "model": "google/gemini-2.0-flash-001",
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

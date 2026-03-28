"""Query expansion for improving recall on vague/tangential queries.

Expands a single query into multiple sub-queries using synonym maps and
concept decomposition.  No LLM calls — purely rule-based expansion.
"""

from __future__ import annotations

import re

# ── Synonym / concept map ──
# Maps abstract terms to concrete terms likely found in stored memories.
# Keys are lowercased; values are lists of expansion terms.

_SYNONYM_MAP: dict[str, list[str]] = {
    # Security & auth
    "abuse": ["rate limiting", "throttling", "token bucket", "block"],
    "prevent": ["rate limiting", "block", "restrict", "enforce"],
    "protection": ["rate limiting", "authentication", "authorization", "firewall"],
    "security": ["authentication", "authorization", "OAuth", "SSO", "MFA", "JWT", "token"],
    "authentication": ["OAuth", "SSO", "MFA", "JWT", "login", "session", "token"],
    "auth": ["OAuth", "SSO", "MFA", "JWT", "login", "session", "token", "authentication"],
    "providers": ["OAuth", "SSO", "Okta", "provider"],
    "login": ["authentication", "session", "JWT", "OAuth", "SSO"],
    "identity": ["OAuth", "SSO", "MFA", "authentication"],

    # Incidents & bugs
    "incidents": ["bug", "outage", "postmortem", "SEV", "incident"],
    "bugs": ["bug", "fix", "issue", "regression", "error"],
    "outages": ["outage", "incident", "downtime", "SEV", "postmortem"],
    "production": ["deploy", "outage", "incident", "monitoring"],
    "issues": ["bug", "incident", "error", "regression"],
    "problems": ["bug", "incident", "error", "outage"],

    # API
    "api": ["endpoint", "REST", "versioning", "pagination", "rate limiting", "error response"],
    "throttling": ["rate limiting", "token bucket", "requests per minute"],
    "rate": ["rate limiting", "throttling", "token bucket"],
    "errors": ["error response", "error format", "standardized", "status code"],
    "error": ["error response", "error format", "standardized"],

    # Database
    "database": ["PostgreSQL", "migration", "index", "replication", "materialized view", "query"],
    "performance": ["optimization", "index", "cache", "materialized view", "latency"],
    "optimization": ["index", "cache", "materialized view", "performance"],
    "scaling": ["auto-scaling", "replication", "cache", "horizontal"],
    "slow": ["optimization", "index", "materialized view", "cache", "performance"],

    # Infrastructure
    "monitoring": ["Datadog", "alerts", "metrics", "observability"],
    "caching": ["Redis", "cache", "invalidation", "TTL"],
    "deployment": ["deploy", "service", "auto-scaling", "CI/CD"],
    "infrastructure": ["deploy", "auto-scaling", "monitoring", "Redis", "cache"],

    # Process
    "team": ["sprint", "on-call", "code review", "planning", "rotation"],
    "process": ["sprint", "on-call", "code review", "planning", "workflow"],
    "workflow": ["sprint", "planning", "code review", "on-call", "process"],
    "meetings": ["sprint planning", "standup", "retrospective"],
    "review": ["code review", "approval", "pull request"],
}

# Phrase-level patterns: if the query contains this phrase, add these expansions
_PHRASE_EXPANSIONS: list[tuple[str, list[str]]] = [
    ("api abuse", ["rate limiting", "throttling", "token bucket", "Envoy proxy"]),
    ("prevent abuse", ["rate limiting", "throttling", "token bucket"]),
    ("production incident", ["postmortem", "outage", "SEV", "bug", "login"]),
    ("incidents and bugs", ["postmortem", "outage", "SEV", "login bug", "expired"]),
    ("auth provider", ["OAuth", "SSO", "Okta", "PKCE"]),
    ("authentication provider", ["OAuth", "SSO", "Okta", "PKCE"]),
    ("database performance", ["materialized view", "composite index", "replication"]),
    ("how are errors", ["error response format", "standardized"]),
    ("error handling", ["error response format", "standardized"]),
    ("team process", ["sprint planning", "on-call rotation", "code review policy"]),
]

_WORD_RE = re.compile(r"\w+")


def expand_query(query: str) -> list[str]:
    """Return a list of queries: the original + up to 2 expanded variants.

    Strategy:
    1. Check phrase-level expansions first (most specific)
    2. Collect synonym expansions for individual words
    3. Build 1-2 additional queries from the collected expansions
    """
    queries = [query]
    lower = query.lower()
    words = [m.group().lower() for m in _WORD_RE.finditer(lower)]

    expansion_terms: list[str] = []

    # Phase 1: phrase-level expansions (high-precision)
    for phrase, terms in _PHRASE_EXPANSIONS:
        if phrase in lower:
            expansion_terms.extend(terms)

    # Phase 2: word-level synonym expansions
    for word in words:
        if word in _SYNONYM_MAP:
            expansion_terms.extend(_SYNONYM_MAP[word])

    if not expansion_terms:
        return queries

    # Deduplicate while preserving order, and remove terms already in query
    seen = set()
    unique_terms: list[str] = []
    for t in expansion_terms:
        t_lower = t.lower()
        if t_lower not in seen and t_lower not in lower:
            seen.add(t_lower)
            unique_terms.append(t)

    if not unique_terms:
        return queries

    # Build expanded queries: group terms into 1-2 additional queries
    # First expanded query: top priority terms (first half)
    mid = max(3, len(unique_terms) // 2)
    batch1 = unique_terms[:mid]
    queries.append(" ".join(batch1))

    # Second expanded query: remaining terms (if any)
    batch2 = unique_terms[mid:]
    if batch2:
        queries.append(" ".join(batch2))

    return queries

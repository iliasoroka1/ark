"""Dreamer — memory consolidation via autonomous LLM specialists.

Runs as a background task after sufficient observations accumulate.
Currently implements Phase 0 (surprisal) + Phase 1 (deduction).

Phase 0: Identify geometrically anomalous observations via surprisal scoring.
Phase 1: Self-directed deduction agent that searches for knowledge updates,
         logical implications, and contradictions — then acts on them.

Designed for ark's async architecture. The specialist gets a small set of
tools for reading/writing memory and runs a tool-calling loop against
OpenRouter (same pattern as query_expand.py).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import aiohttp
import msgspec
import tantivy

from ark.engine.index import (
    F_ATTRIBUTES,
    F_CHUNK_ATTRIBUTES,
    F_CHUNK_ID,
    F_ID,
    F_UPDATED_AT,
    Indexer,
)
from ark.engine.result import Error, Ok
from ark.engine.search import Searcher
from ark.engine.surprisal import SurprisalScore, compute_surprisal
from ark.engine.types import IndexDoc, SearchParams

log = logging.getLogger(__name__)

_DREAMER_MODEL = os.getenv(
    "DREAMER_MODEL", "google/gemini-3.1-flash-lite-preview"
)
_MAX_ITERATIONS = int(os.getenv("DREAMER_MAX_ITERATIONS", "12"))
_MIN_OBSERVATIONS = int(os.getenv("DREAMER_MIN_OBSERVATIONS", "20"))
_SURPRISAL_K = int(os.getenv("DREAMER_SURPRISAL_K", "5"))
_SURPRISAL_TOP_PERCENT = float(
    os.getenv("DREAMER_SURPRISAL_TOP_PERCENT", "0.15")
)
_STALE_PRUNE_MIN_AGE_DAYS = int(os.getenv("DREAMER_STALE_PRUNE_DAYS", "90"))


# ── Result types ──────────────────────────────────────────────────────


@dataclass(slots=True)
class DreamResult:
    """Metrics from a dream cycle."""

    surprisal_count: int = 0
    iterations: int = 0
    created: int = 0
    deleted: int = 0
    pruned_stale: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


# ── Tool definitions for the specialist (OpenAI-format) ──────────────

_TOOL_DEFS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_memory",
            "description": "Semantic search across all observations. Returns L0 summaries with IDs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_observation",
            "description": "Load the full text of an observation by ID (from search results).",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "Observation ID from search results",
                    },
                },
                "required": ["id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_observation",
            "description": (
                "Create a new deductive observation. Must include source_ids linking to "
                "premise observations. Level is one of: deductive, contradiction."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The observation text",
                    },
                    "level": {
                        "type": "string",
                        "enum": ["deductive", "contradiction"],
                        "description": "Type of derived observation",
                    },
                    "source_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "IDs of premise observations this was derived from",
                    },
                },
                "required": ["content", "level", "source_ids"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_observation",
            "description": "Delete an outdated or superseded observation by ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Observation ID to delete"},
                },
                "required": ["id"],
            },
        },
    },
]


# ── System prompt ─────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a memory consolidation specialist. Your job is to review an agent's \
stored observations and improve their quality through deductive reasoning.

You have tools to search, read, create, and delete observations. Work in two phases:

## PHASE 1: DISCOVERY
Explore what's in memory using search_memory. Understand the landscape before acting.

## PHASE 2: ACTION
Look for these three patterns (in priority order):

### A. Knowledge Updates (HIGHEST PRIORITY)
Same entity or fact with different values across time.
Example: "Project deadline is March 15" + "Deadline moved to April 1"
→ DELETE the outdated observation, CREATE a new one noting the change.

### B. Logical Implications
Facts that logically entail other useful facts not yet recorded.
Example: "Agent manages a team of 5 engineers"
→ CREATE: "Agent has management responsibilities" (level: deductive)
Only create implications that are genuinely useful for future context.

### C. Contradictions
Facts that cannot both be true simultaneously.
Example: "Prefers async communication" vs "Always wants sync standups"
→ CREATE a contradiction observation linking both source IDs.
Do NOT delete either side — flag the conflict for human review.

## RULES
- Always provide source_ids when creating observations (link to premises).
- Never create duplicates of existing observations.
- Never delete observations you just created.
- Stop when you've exhausted productive patterns — don't force actions.
- Quality over quantity. A dream cycle that finds nothing wrong is a good outcome.
"""


# ── OpenRouter LLM call ──────────────────────────────────────────────


async def _chat_completion(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    model: str,
) -> dict[str, Any] | None:
    """POST to OpenRouter chat/completions with tool-calling support.

    Returns the raw JSON response dict, or None on failure.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        log.warning("dreamer: OPENROUTER_API_KEY not set")
        return None

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": 4096,
        "temperature": 0.2,
    }
    if tools:
        payload["tools"] = tools

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=aiohttp.ClientTimeout(total=60),
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    log.warning("dreamer_llm_error: status=%s body=%s", resp.status, body[:200])
                    return None
                return await resp.json()
    except Exception as e:
        log.warning("dreamer_llm_exception: %s", e)
        return None


# ── L0 extraction helper ─────────────────────────────────────────────


def _extract_l0(content: str) -> str:
    """Extract a one-line summary (L0) from observation content."""
    first_line = content.split("\n", 1)[0].strip()
    if len(first_line) <= 120:
        return first_line
    return first_line[:117] + "..."


# ── Tool executor ─────────────────────────────────────────────────────


class _ToolExecutor:
    """Executes dreamer tools against the memory subsystem."""

    __slots__ = (
        "_indexer",
        "_searcher",
        "_corpus",
        "_agent_id",
        "_created",
        "_deleted",
    )

    def __init__(
        self, indexer: Indexer, searcher: Searcher, corpus: str, agent_id: str
    ) -> None:
        self._indexer = indexer
        self._searcher = searcher
        self._corpus = corpus
        self._agent_id = agent_id
        self._created = 0
        self._deleted = 0

    @property
    def created(self) -> int:
        return self._created

    @property
    def deleted(self) -> int:
        return self._deleted

    async def execute(self, name: str, args: dict[str, Any]) -> str:
        match name:
            case "search_memory":
                return await self._search(args.get("query", ""))
            case "get_observation":
                return await self._get(args.get("id", ""))
            case "create_observation":
                return await self._create(
                    args.get("content", ""),
                    args.get("level", "deductive"),
                    args.get("source_ids", []),
                )
            case "delete_observation":
                return self._delete(args.get("id", ""))
            case _:
                return f"Unknown tool: {name}"

    async def _search(self, query: str) -> str:
        if not query:
            return "Error: query is required"
        params = SearchParams(
            num_to_return=15,
            num_to_score=40,
            min_rrf_score=0.003,
            max_hits_per_doc=1,
        )
        match await self._searcher.search(query, corpus=self._corpus, params=params):
            case Ok(hits) if hits:
                results = []
                for h in hits:
                    attrs = h.attributes or {}
                    entry: dict[str, Any] = {
                        "id": h.doc_id,
                        "l0": attrs.get("l0", h.body[:200]),
                    }
                    if attrs.get("tag"):
                        entry["tag"] = attrs["tag"]
                    if attrs.get("observation_level"):
                        entry["level"] = attrs["observation_level"]
                    results.append(entry)
                return msgspec.json.encode(results).decode()
            case Ok(_):
                return "No observations found."
            case err:
                return f"Search error: {err}"

    async def _get(self, doc_id: str) -> str:
        if not doc_id:
            return "Error: id is required"

        self._indexer._index.reload()
        searcher = self._indexer._index.searcher()
        query = tantivy.Query.term_query(self._indexer.schema, F_ID, doc_id)
        hits = searcher.search(query, limit=50).hits

        if not hits:
            return f"Observation {doc_id!r} not found."

        chunks: list[tuple[str, str]] = []
        attrs: dict[str, Any] = {}
        for _score, addr in hits:
            doc = searcher.doc(addr)
            cids = doc.get_all(F_CHUNK_ID)
            if not cids:
                raw = doc.get_all("attributes")
                if raw and isinstance(raw[0], dict):
                    attrs = raw[0]
                continue
            ca = doc.get_all(F_CHUNK_ATTRIBUTES)
            if ca and isinstance(ca[0], dict):
                chunks.append((str(cids[0]), ca[0].get("body", "")))

        if not chunks:
            return f"Observation {doc_id!r} has no content."

        chunks.sort(key=lambda x: x[0])
        body = "\n\n".join(b for _, b in chunks)
        result: dict[str, Any] = {"id": doc_id, "content": body}
        if attrs.get("tag"):
            result["tag"] = attrs["tag"]
        if attrs.get("observation_level"):
            result["level"] = attrs["observation_level"]
        if attrs.get("source_ids"):
            result["source_ids"] = attrs["source_ids"]
        return msgspec.json.encode(result).decode()

    async def _create(self, content: str, level: str, source_ids: list[str]) -> str:
        if not content:
            return "Error: content is required"
        if not source_ids:
            return "Error: source_ids required — link to premise observations"

        doc_id = f"dream-{hashlib.sha256(content.encode()).hexdigest()[:16]}"

        if self._indexer.is_indexed(doc_id):
            return (
                f"Duplicate: observation with this content already exists (id={doc_id})"
            )

        l0 = _extract_l0(content)
        now = datetime.now(UTC).isoformat()

        doc = IndexDoc(
            id=doc_id,
            source_id=self._agent_id,
            corpus=self._corpus,
            body=content,
            attributes={
                "agent_id": self._agent_id,
                "l0": l0,
                "tag": "dreamer",
                "observation_level": level,
                "source_ids": source_ids,
                "derived_at": now,
            },
        )

        match await self._indexer.add(doc):
            case Ok(n) if n > 0:
                self._indexer.commit()
                self._created += 1
                return msgspec.json.encode(
                    {"status": "created", "id": doc_id, "l0": l0}
                ).decode()
            case result:
                return f"Failed to create observation: {result}"

    def _delete(self, doc_id: str) -> str:
        if not doc_id:
            return "Error: id is required"
        if not self._indexer.is_indexed(doc_id):
            return f"Observation {doc_id!r} not found."
        # Soft-delete: invalidate graph edges, keep node in tantivy for history.
        self._indexer.invalidate_observation(doc_id)
        self._indexer.delete(doc_id)
        self._indexer.commit()
        self._deleted += 1
        return msgspec.json.encode({"status": "deleted", "id": doc_id}).decode()


# ── Specialist loop ───────────────────────────────────────────────────


async def _run_specialist(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    executor: _ToolExecutor,
    max_iterations: int,
    model: str,
) -> tuple[int, int, int]:
    """Run the tool-calling loop. Returns (iterations, input_tokens, output_tokens)."""
    total_in = 0
    total_out = 0

    for iteration in range(max_iterations):
        data = await _chat_completion(messages, tools, model)
        if data is None:
            return iteration + 1, total_in, total_out

        # Track usage
        usage = data.get("usage", {})
        total_in += usage.get("prompt_tokens", 0)
        total_out += usage.get("completion_tokens", 0)

        choices = data.get("choices", [])
        if not choices:
            return iteration + 1, total_in, total_out

        message = choices[0].get("message", {})
        # Append assistant message to conversation
        messages.append(message)

        tool_calls = message.get("tool_calls")
        if tool_calls:
            for tc in tool_calls:
                func = tc.get("function", {})
                name = func.get("name", "")
                try:
                    args = json.loads(func.get("arguments", "{}"))
                except (json.JSONDecodeError, TypeError):
                    args = {}

                result = await executor.execute(name, args)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id", ""),
                    "content": result,
                })
        else:
            # No tool calls — specialist is done
            return iteration + 1, total_in, total_out

    return max_iterations, total_in, total_out


# ── Stale pruning ────────────────────────────────────────────────────


def _prune_stale(indexer: Indexer, corpus: str, agent_id: str) -> int:
    """Delete never-accessed explicit observations older than the threshold.

    Preserves dreamer-created observations (deductive, contradiction) and
    manually added memories (no observation_level attribute). Only prunes
    raw deriver extractions that were never useful enough to be retrieved.
    """
    if indexer.embed_cache is None:
        return 0

    stale_ids = indexer.embed_cache.find_stale(corpus)
    if not stale_ids:
        return 0

    indexer._index.reload()
    searcher = indexer._index.searcher()
    now = datetime.now(UTC)
    pruned = 0

    for doc_id in stale_ids:
        query = tantivy.Query.term_query(indexer.schema, F_ID, doc_id)
        hits = searcher.search(query, limit=5).hits
        if not hits:
            continue

        should_prune = False
        for _score, addr in hits:
            doc = searcher.doc(addr)
            cids = doc.get_all(F_CHUNK_ID)
            if cids:
                continue  # skip chunk docs, we want the parent

            attrs_raw = doc.get_all(F_ATTRIBUTES)
            if not attrs_raw or not isinstance(attrs_raw[0], dict):
                continue

            attrs = attrs_raw[0]
            level = attrs.get("observation_level", "")

            # Only prune raw deriver extractions ("explicit", "action")
            if level not in ("explicit", "action"):
                break

            updated_raw = doc.get_all(F_UPDATED_AT)
            if not updated_raw:
                break
            try:
                updated = datetime.fromisoformat(str(updated_raw[0]))
                age_days = (now - updated).total_seconds() / 86400.0
            except (ValueError, TypeError):
                break

            if age_days >= _STALE_PRUNE_MIN_AGE_DAYS:
                should_prune = True
            break

        if should_prune:
            indexer.delete(doc_id)
            indexer.embed_cache.delete(doc_id)
            pruned += 1

    if pruned:
        indexer.commit()
        log.info("dreamer_pruned_stale: agent_id=%s pruned=%d", agent_id, pruned)

    return pruned


# ── Public API ────────────────────────────────────────────────────────


async def dream(
    agent_id: str,
    indexer: Indexer,
    searcher: Searcher,
    model: str = "",
    hints: list[SurprisalScore] | None = None,
) -> DreamResult:
    """Run a full dream cycle for an agent.

    Args:
        agent_id: The agent whose memory to consolidate.
        indexer: Memory indexer (write side).
        searcher: Memory searcher (read side).
        model: Override model (defaults to DREAMER_MODEL env).
        hints: Pre-computed surprisal scores (skips Phase 0 if provided).

    Returns:
        DreamResult with metrics.
    """
    model = model or _DREAMER_MODEL
    corpus = f"agent:{agent_id}"
    result = DreamResult()

    # ── Phase 0: Surprisal sampling ──────────────────────────────
    if hints is None and indexer.embed_cache is not None:
        observations = indexer.embed_cache.get_corpus(corpus)
        if len(observations) >= _MIN_OBSERVATIONS:
            hints = compute_surprisal(
                observations,
                k=_SURPRISAL_K,
                top_percent=_SURPRISAL_TOP_PERCENT,
            )
            result.surprisal_count = len(hints)
            log.info(
                "dreamer_surprisal_done: agent_id=%s total=%d anomalies=%d",
                agent_id, len(observations), len(hints),
            )
        else:
            log.info(
                "dreamer_skip_surprisal: agent_id=%s count=%d",
                agent_id, len(observations),
            )

    # ── Phase 1: Deduction specialist ────────────────────────────
    executor = _ToolExecutor(indexer, searcher, corpus, agent_id)

    # Build the user prompt with optional hints
    user_parts = []
    if hints:
        user_parts.append(
            "The following observations scored as geometrically anomalous "
            "(they don't cluster well with other memories). They may indicate "
            "outdated facts, contradictions, or isolated knowledge worth investigating:\n"
        )
        for h in hints[:10]:
            user_parts.append(f"- [{h.doc_id}] (surprisal: {h.normalized:.2f})")
        user_parts.append(
            "\nStart by searching for context around these anomalies. "
            "But also explore freely — these are hints, not constraints."
        )
    else:
        user_parts.append(
            "Explore the observation space. Search for knowledge updates "
            "(same topic, different values), logical implications, and contradictions. "
            "Start with broad searches, then drill into specific topics."
        )

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(user_parts)},
    ]

    iterations, in_tokens, out_tokens = await _run_specialist(
        messages=messages,
        tools=_TOOL_DEFS,
        executor=executor,
        max_iterations=_MAX_ITERATIONS,
        model=model,
    )

    result.iterations = iterations
    result.created = executor.created
    result.deleted = executor.deleted
    result.input_tokens = in_tokens
    result.output_tokens = out_tokens

    # ── Phase 2: Prune stale observations ─────────────────────────
    result.pruned_stale = _prune_stale(indexer, corpus, agent_id)

    log.info(
        "dreamer_done: agent_id=%s iterations=%d created=%d deleted=%d "
        "pruned_stale=%d input_tokens=%d output_tokens=%d",
        agent_id, iterations, executor.created, executor.deleted,
        result.pruned_stale, in_tokens, out_tokens,
    )

    return result


async def maybe_dream(
    agent_id: str,
    indexer: Indexer,
    searcher: Searcher,
) -> DreamResult | None:
    """Conditionally run a dream cycle if enough observations have accumulated.

    Call this from the deriver after storing new observations. Returns None
    if conditions aren't met (not enough observations).
    """
    if indexer.embed_cache is None:
        return None

    corpus = f"agent:{agent_id}"
    count = indexer.embed_cache.count(corpus)

    if count < _MIN_OBSERVATIONS:
        return None

    # Simple gating: dream every _MIN_OBSERVATIONS new observations.
    state_key = f"_dream_last_count_{agent_id}"
    last_count = _dream_state.get(state_key, 0)

    if count - last_count < _MIN_OBSERVATIONS:
        return None

    log.info(
        "dreamer_triggered: agent_id=%s observations=%d since_last=%d",
        agent_id, count, count - last_count,
    )

    try:
        result = await dream(agent_id, indexer, searcher)
        _dream_state[state_key] = count
        return result
    except Exception:
        log.exception("dreamer_failed: agent_id=%s", agent_id)
        return None


# Module-level state for dream gating (reset on process restart)
_dream_state: dict[str, int] = {}

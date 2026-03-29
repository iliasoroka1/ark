# Ark

Knowledge engine for AI agents. Ingest, search, and reason over structured memory — from the terminal or over HTTP.

## Install

```bash
pip install ark-agent
```

## Quick start

```bash
# Start the server (default: localhost:7070)
ark serve

# Ingest knowledge
ark ingest "RRF merges BM25 + cosine signals" --title "Search Architecture" --tag technical

# Search
ark search "How does ranking work?"

# Graph-powered search (multi-hop traversal)
ark search "How does ranking work?" --graph

# Ingest a file
ark ingest-file ./notes.md --tag research

# Spectral analysis — find novel, foundational, and bridge memories
ark analyze
```

## HTTP API

All endpoints available at `http://localhost:7070` when the server is running.

```bash
# Ingest
curl -X POST localhost:7070/ingest \
  -H "Content-Type: application/json" \
  -d '{"content": "Ark uses BM25 + embeddings", "title": "Architecture", "tag": "technical"}'

# Search
curl -X POST localhost:7070/search \
  -H "Content-Type: application/json" \
  -d '{"query": "How does search work?", "limit": 5}'

# Graph search (multi-hop)
curl -X POST localhost:7070/graph-search \
  -H "Content-Type: application/json" \
  -d '{"query": "ranking", "hops": 2}'

# Health check
curl localhost:7070/health
```

## How search works

Ark runs a multi-signal hybrid search pipeline with automatic query expansion:

### Query-time pipeline

```
Query
  │
  ├─ Signal 1: Full-precision cosine similarity
  │    Embed query with pplx-embed-v1-0.6b (1024d) via OpenRouter
  │    Brute-force cosine against all corpus vectors in SQLite
  │
  ├─ Pseudo-Relevance Feedback (PRF)
  │    Extract top-3 cosine hits (sim ≥ 0.35)
  │    Tokenize their body text, extract top-12 frequent terms
  │    → feeds into Signal 1b and Signal 2b
  │
  ├─ Signal 1b: Enriched cosine
  │    Re-embed "query + PRF terms" as single string
  │    Merge with original cosine results (max sim per doc)
  │
  ├─ Signal 2: BM25 (original query, full weight)
  │    Tantivy with en_stem tokenizer (lowercase + Porter stemmer)
  │
  ├─ Signal 2b: BM25 (expanded query, half weight)
  │    Uses PRF terms or LLM-expanded terms for vocabulary bridging
  │
  ├─ Temporal candidate injection (when query has dates)
  │    Regex detects dates in query → resolve to period node IDs
  │    Look up hypergraph period nodes (month:YYYY-MM, quarter:YYYY-QN)
  │    Inject connected docs into candidate pool via occurred_in edges
  │
  ├─ Signal 3: Temporal proximity (when query has dates)
  │    Gaussian decay: score = exp(-distance²/2σ²), σ=15 days
  │    Weight 1.5x in RRF — only active for temporal queries
  │
  ├─ RRF Merge
  │    Score-weighted Reciprocal Rank Fusion (K=15)
  │    Embedding weight: 2.0x, BM25 weight: 1.5x
  │    Expanded BM25: 0.5x, Temporal: 1.5x (when active)
  │
  ├─ Temporal Decay (per-agent)
  │    decay = max(0.3, 1.0 - age_days/365 * 0.5)
  │    Access boost = min(1.3, 1.0 + access_count * 0.02)
  │    Scoped per agent_id — one agent's usage doesn't affect another's
  │
  └─ Graph Expansion
       2-hop traversal from top-5 hits
       Neighbors with cosine ≥ 0.55 get interpolated RRF scores
```

### Index-time pipeline

1. **Chunking** — TextChunker (256 tokens), MarkdownChunker, or SymbolChunker (code-aware regex AST)
2. **Embedding** — `embed_document()` with `search_document:` prefix (default: pplx-embed-v1-0.6b via OpenRouter; fallback: fastembed nomic-embed-text-v1.5)
3. **Dedup** — skip chunks with cosine ≥ 0.95 to existing corpus vectors
4. **Tantivy indexing** — `chunk_body` (en_stem analyzed for BM25), `chunk_tokens` (raw), metadata as JSON
5. **Embedding cache** — SQLite sidecar (`embeddings.db`) storing raw float32 vectors
6. **Graph edges** — auto-generated from source_id attributes (`derives_from`, `contradicts`)
7. **Temporal hypergraph** — dates extracted from doc text, linked to period nodes (`month:2026-01`, `quarter:2026-Q1`) via `occurred_in` edges

### Query expansion

- **PRF** (always active) — extracts terms from the top cosine-similar documents in the corpus. No API key needed. Bridges vocabulary gaps automatically.
- **LLM expansion** (optional) — OpenRouter API with Gemini Flash. Fires for vague/abstract queries when `OPENROUTER_API_KEY` is set. Generates 5-10 specific terms. Disable with `ARK_NO_LLM_EXPAND=1`.

### Key parameters

| Param | Value | Purpose |
|---|---|---|
| RRF K | 15 | RRF smoothing constant |
| Embedding weight | 2.0 | Cosine signal weight in RRF |
| BM25 weight | 1.5 | BM25 signal weight in RRF |
| Expanded BM25 weight | 0.5x | Weight multiplier for expanded BM25 |
| PRF docs | 3 | Top cosine docs for term extraction |
| PRF terms | 12 | Max terms extracted per query |
| PRF min similarity | 0.35 | Cosine floor for PRF candidates |
| Graph min similarity | 0.55 | Cosine floor for graph expansion |
| Graph hops | 2 | Traversal depth |
| Dedup threshold | 0.95 | Cosine threshold for chunk dedup |
| Temporal weight | 1.5 | Temporal signal weight in RRF (only when query has dates) |
| Temporal sigma | 15 days | Gaussian decay std dev for temporal proximity |
| Decay half-life | 365 days | Temporal decay rate |

### Benchmark

220 queries across 20 categories, 1233 documents (83 target + 1150 noise).
Target docs: 23 core engineering + 20 infra/ops + 20 data/backend + 20 business/ops.
Baseline: `pplx-embed-v1-0.6b` via OpenRouter + `ARK_NO_DECAY=1`.

| Category | Hit@3 (no LLM) | Hit@3 (+LLM) | Notes |
|---|---|---|---|
| Exact | 97% | 97% | Verbatim phrase matching |
| Paraphrase | 97% | 93% | Rephrased concepts |
| Precision | 95% | 95% | Distinguish near-identical docs |
| Conversational | 100% | 100% | Vague/informal queries |
| Needle | 100% | 100% | Specific detail extraction |
| Adversarial | 80% | 85% | Confusable docs + noise overlap |
| Synonym hell | 60% | 100% | Zero lexical overlap |
| Compositional | 50% | 70% | Combining 2+ memories |
| Cross-domain | 47% | 67% | Connecting different domains |
| Temporal | 47% | 60% | Date-aware search (hypergraph + Gaussian) |
| Lexical traps | 60% | 60% | Query words match noise more |
| Multi-hop | 60% | 60% | Chaining facts |
| Tangential | 40% | 50% | Abstract/indirect queries |
| Negation | 20% | 20% | Exclusion logic not supported |
| Biz exact | 0% | 100% | Business language — needs LLM bridge |
| Biz paraphrase | 0% | 100% | Technical rephrasing of business docs |
| Biz cross-domain | 60% | 100% | Connecting business + engineering |
| Biz temporal | 0% | 100% | Business docs with dates |
| Biz adversarial | 0% | 67% | Business vs noise overlap |
| **Overall** | **75.8%** | **82.8%** | **MRR: 0.679 / 0.716, 0% false positives** |

## Spectral analysis

`ark analyze` runs graph-level analysis:

- **RMT** (Random Matrix Theory) — detects novel/anomalous memories
- **PageRank** — identifies foundational memories
- **Betweenness centrality** — finds bridge memories between knowledge domains
- **Fiedler vector** — reveals knowledge boundaries

## Session state

Per-agent scratchpad, task tracking, and conversation history. Stored in SQLite (`~/.ark/sessions.db`).

```bash
# Scratchpad
ark scratch-set current_focus "search optimization"
ark scratch-get current_focus

# Tasks
ark tasks-add "Implement query expansion"
ark tasks-list
ark tasks-done 1

# History
ark history-add user "How does Ark handle embeddings?"
ark history-list --limit 10
ark history-search "embeddings"
```

Environment variables `ARK_AGENT_ID` and `ARK_SESSION_ID` scope state per agent/session.

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ARK_URL` | `http://localhost:7070` | Server address |
| `ARK_HOME` | `~/.ark` | Data directory |
| `ARK_AGENT_ID` | `default` | Agent ID for session commands |
| `ARK_SESSION_ID` | `default` | Session ID for session commands |
| `EMBEDDING_MODEL` | — | Custom embedding HTTP endpoint |
| `EMBEDDING_DIMS` | `768` | Embedding dimensions |
| `OPENROUTER_API_KEY` | — | Enables LLM query expansion and OpenRouter embedding |
| `OPENROUTER_EMBED_MODEL` | — | OpenRouter embedding model (e.g. `perplexity/pplx-embed-v1-0.6b`) |
| `ARK_NO_LLM_EXPAND` | — | Set to `1` to disable LLM query expansion (keeps embedding provider) |
| `ARK_NO_DECAY` | — | Set to `1` to disable temporal decay (use for deterministic benchmarks) |

## License

MIT

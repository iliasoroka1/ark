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
  │    Embed query with nomic-embed-text-v1.5 (768d)
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
  ├─ RRF Merge
  │    Score-weighted Reciprocal Rank Fusion (K=15)
  │    Embedding weight: 2.0x, BM25 weight: 1.5x
  │    Expanded BM25: 0.5x multiplier
  │
  ├─ Temporal Decay
  │    decay = max(0.3, 1.0 - age_days/365 * 0.5)
  │    Access boost = min(1.3, 1.0 + access_count * 0.02)
  │
  └─ Graph Expansion
       2-hop traversal from top-5 hits
       Neighbors with cosine ≥ 0.55 get interpolated RRF scores
```

### Index-time pipeline

1. **Chunking** — TextChunker (256 tokens), MarkdownChunker, or SymbolChunker (code-aware regex AST)
2. **Embedding** — `embed_document()` with `search_document:` prefix (nomic asymmetric retrieval)
3. **Dedup** — skip chunks with cosine ≥ 0.95 to existing corpus vectors
4. **Tantivy indexing** — `chunk_body` (en_stem analyzed for BM25), `chunk_tokens` (raw), metadata as JSON
5. **Embedding cache** — SQLite sidecar (`embeddings.db`) storing raw float32 vectors
6. **Graph edges** — auto-generated from source_id attributes (`derives_from`, `contradicts`)

### Query expansion

- **PRF** (always active) — extracts terms from the top cosine-similar documents in the corpus. No API key needed. Bridges vocabulary gaps automatically.
- **LLM expansion** (optional) — OpenRouter API with Gemini Flash. Fires for vague/abstract queries when `OPENROUTER_API_KEY` is set. Generates 5-10 specific terms.

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
| Decay half-life | 365 days | Temporal decay rate |

### Benchmark

130 queries across 15 categories, 1173 documents (23 engineering + 1150 noise).

| Category | Hit@3 | Notes |
|---|---|---|
| Exact | 100% | Verbatim phrase matching |
| Precision | 100% | Single-memory retrieval |
| Needle | 100% | Specific detail extraction |
| Conversational | 90% | Vague/informal queries — PRF biggest win |
| Paraphrase | 80% | Rephrased concepts |
| Adversarial | 70% | Terms overlap with noise corpus |
| Lexical traps | 70% | Query words match noise more than targets |
| Synonym hell | 60% | Zero lexical overlap with documents |
| Multi-hop | 60% | Requires chaining facts |
| Compositional | 50% | Requires combining 2+ memories |
| Tangential | 40% | Abstract/indirect queries |
| Negation | 40% | Exclusion logic not supported |
| Cross-domain | 30% | Requires inference chains |
| Temporal | 20% | Date reasoning not supported |
| **Overall** | **64.8%** | **MRR: 0.569, 0% false positives** |

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
| `OPENROUTER_API_KEY` | — | Enables LLM query expansion |

## License

MIT

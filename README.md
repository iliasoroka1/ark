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

Ark runs a hybrid search pipeline:

1. **BM25** (Tantivy, stemmed English) — keyword matching
2. **Embeddings** (cosine similarity, full-precision) — semantic matching
3. **RRF** (Reciprocal Rank Fusion, k=15) — merges both signals

Results include temporal decay (365-day half-life) and access boost (up to 1.3x on repeated reads).

**Graph search** adds multi-hop beam traversal with MMR (maximal marginal relevance). Edge types: `relates_to`, `same_tag`, `co_session`, `derives_from`, `contradicts`. Hop decay 0.8x per hop, beam width 8.

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

## Architecture

- **Indexing**: Tantivy (BM25) + in-memory embedding cache
- **Graph**: SQLite temporal edge store with label propagation clustering
- **Chunking**: Smart file-aware chunker (1024 tokens, 256 overlap)
- **Embeddings**: FastEmbed (local) or custom HTTP endpoint via `EMBEDDING_MODEL`
- **Server**: aiohttp on port 7070
- **Local fallback**: CLI works without the server — falls back to direct engine access

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ARK_URL` | `http://localhost:7070` | Server address |
| `ARK_HOME` | `~/.ark` | Data directory |
| `ARK_AGENT_ID` | `default` | Agent ID for session commands |
| `ARK_SESSION_ID` | `default` | Session ID for session commands |
| `EMBEDDING_MODEL` | — | Custom embedding HTTP endpoint |
| `EMBEDDING_DIMS` | `1024` | Embedding dimensions |

## License

MIT

# Ark Agent Research Guide

How to work on ark's search pipeline in parallel autoresearch mode.

## Architecture Quick Reference

```
Query → should_expand? → LLM expansion (gemini-3.1-flash-lite via OpenRouter)
  ↓
Signal 1: Cosine similarity (embedding cache, weight 2.0)
  ↓
PRF: extract terms from top cosine hits → enriched cosine (signal 1b)
  ↓
Signal 2: BM25 (tantivy, weight 1.5)
Signal 2b: BM25 on expanded query (half weight)
  ↓
RRF merge → temporal decay → graph expansion → top-K results
```

Key files:
- `src/ark/engine/search.py` — RRF merge, PRF, graph expansion, decay
- `src/ark/engine/query_expand.py` — LLM expansion (should_expand + expand_query)
- `src/ark/engine/embed.py` — embedding providers (OpenRouter, FastEmbed, Catsu)
- `src/ark/engine/index.py` — write side (chunking, embedding, tantivy)
- `src/ark/engine/embedding_cache.py` — SQLite vector store
- `src/ark/local.py` — local engine init, ARK_HOME, embedding provider selection

## Running Benchmarks

```bash
# Baseline: pplx-embed, no LLM expansion
OPENROUTER_EMBED_MODEL=perplexity/pplx-embed-v1-0.6b \
OPENROUTER_API_KEY=<key> \
ARK_HOME=/tmp/ark-pplx-test \
ARK_NO_DECAY=1 \
ARK_NO_LLM_EXPAND=1 \
uv run python test_recall_brutal.py

# Best config: pplx-embed + LLM expansion
OPENROUTER_EMBED_MODEL=perplexity/pplx-embed-v1-0.6b \
OPENROUTER_API_KEY=<key> \
ARK_HOME=/tmp/ark-pplx-test \
ARK_NO_DECAY=1 \
uv run python test_recall_brutal.py

# Verbose (see which queries miss)
OPENROUTER_EMBED_MODEL=perplexity/pplx-embed-v1-0.6b \
OPENROUTER_API_KEY=<key> \
ARK_HOME=/tmp/ark-pplx-test \
ARK_NO_DECAY=1 \
uv run python test_recall_brutal.py -v
```

Always use `ARK_NO_DECAY=1` for benchmarks. Without it, results drift ~2-5pp between runs due to `datetime.now()` in temporal decay.

**IMPORTANT: kill any running `ark serve` before benchmarking.** The CLI falls back to local engine when no server is reachable — if a server is running at `localhost:7070`, the benchmark hits it instead of your `ARK_HOME` index. Run `pkill -f "ark serve"` first.

**IMPORTANT: parallel benchmarks need separate `ARK_HOME`.** Two processes can't share the same tantivy index (writer lock conflict). Copy the index: `cp -r /tmp/ark-pplx-test /tmp/ark-pplx-nollm`.

## Working in Parallel (Multiple Agents)

### Rule 1: Use separate worktrees for code changes
```bash
git worktree add /tmp/ark-<experiment-name> improve/<branch-name>
cd /tmp/ark-<experiment-name>
uv sync
```

### Rule 2: Use separate ARK_HOME for index experiments
```bash
# Each agent gets its own index directory
ARK_HOME=/tmp/ark-<agent-name>-test uv run python reembed.py
ARK_HOME=/tmp/ark-<agent-name>-test uv run python test_recall_brutal.py
```

Never share an index between concurrent benchmark runs (writer lock conflict). The canonical pplx baseline index lives at `/tmp/ark-pplx-test` (pplx-embed-v1-0.6b, 1024D, 1173 docs). Copy it for each parallel run.

### Rule 3: Fix test cwd when using worktrees
Test files hardcode `cwd="/Users/iliasoroka/ark"`. After copying to worktree:
```bash
sed -i '' 's|cwd="/Users/iliasoroka/ark"|cwd="/tmp/ark-<worktree>"|g' test_recall*.py
```

### Rule 4: Coordinate via rz
```bash
rz send <agent-name> "what I changed, what I found, what to try next"
```

## Re-indexing

When changing embedding model or re-ingesting:
```bash
ARK_HOME=/tmp/ark-test uv run python reembed.py
```

This ingests 1173 docs: 23 engineering + 150 noise + 500 AG News + 500 tech news. Takes ~2min locally (fastembed) or ~5min via API (pplx-embed).

## Embedding Models

| Model | Type | Dims | Hit@3 (no LLM) | Hit@3 (+LLM) | How to use |
|-------|------|------|----------------|--------------|------------|
| pplx-embed-v1-0.6b | API | 1024 | **74.3%** | **85.7%** | `OPENROUTER_EMBED_MODEL=perplexity/pplx-embed-v1-0.6b` |
| BGE-large-en-v1.5 | local | 1024 | 71.2% | — | `FASTEMBED_MODEL=BAAI/bge-large-en-v1.5` |
| BGE-base-en-v1.5 | local | 768 | 69.6% | — | `FASTEMBED_MODEL=BAAI/bge-base-en-v1.5` |
| Nomic v1.5 | local | 768 | 64.8% | — | default fastembed fallback |

pplx-embed is the current baseline. LLM expansion via gemini-3.1-flash-lite adds ~14pp on top.

## Current Benchmark Results (220 queries, 20 categories)

Model: pplx-embed-v1-0.6b. Corpus: 83 target docs + 1150 noise = 1233 total.
Target docs: 23 core engineering + 20 infra/ops + 20 data/backend + 20 business/ops.
Features: temporal hypergraph (period nodes + Gaussian proximity RRF signal).
Env: `ARK_NO_DECAY=1`, no server running, isolated `ARK_HOME`.

| Category | Queries | Hit@3 (no LLM) | Hit@3 (+LLM) | Notes |
|----------|---------|----------------|--------------|-------|
| Exact | 30 | 97% | 97% | verbatim phrases |
| Paraphrase | 30 | 97% | 93% | rephrased concepts |
| Precision | 20 | 95% | 95% | distinguish near-identical docs |
| Conversational | 10 | 100% | 100% | vague human-like queries |
| Needle | 5 | 100% | 100% | detail buried in long memory |
| Adversarial | 20 | 80% | 85% | confusable docs + noise overlap |
| Biz exact | 5 | 0% | 100% | business language — needs LLM bridge |
| Biz paraphrase | 5 | 0% | 100% | technical rephrasing of business docs |
| Biz cross-domain | 5 | 60% | 100% | connecting business + engineering |
| Biz temporal | 2 | 0% | 100% | business docs with dates |
| Biz adversarial | 3 | 0% | 67% | business vs noise overlap |
| Synonym hell | 10 | 60% | 100% | zero lexical overlap |
| Compositional | 10 | 50% | 70% | combining 2+ memories |
| Cross-domain | 15 | 47% | 67% | connecting different domains |
| Temporal | 15 | 47% | 60% | date-aware search (hypergraph + Gaussian) |
| Multi-hop | 5 | 60% | 60% | chaining facts |
| Lexical traps | 10 | 60% | 60% | query words match noise more |
| Tangential | 10 | 40% | 50% | indirect/abstract queries |
| Negation | 5 | 20% | 20% | exclusion logic not supported |
| Negative | 5 | 0% FP | 0% FP | correctly returns nothing |
| **OVERALL** | **215** | **75.8%** | **82.8%** | MRR: 0.679 / 0.716 |

**Key findings:**
- Business docs (non-technical language) score 0% without LLM expansion, 100% with — this is the strongest evidence that LLM expansion genuinely bridges vocabulary gaps.
- Temporal hypergraph improved temporal from 50% → 60% (+LLM) by injecting date-matched docs into candidate pool.
- Original categories (tangential, lexical traps) dropped vs earlier benchmarks — the 83-doc corpus is genuinely harder.

## What Doesn't Work (Lessons Learned)

1. **Synonym maps** — hardcoding domain-specific term mappings is cheating. Doesn't generalize.
2. **Global embedding weight changes** — boosting `_EMBED_WEIGHT` helps tangential but kills exact recall.
3. **Dense graph edges** — connecting all docs with cosine > 0.6 floods results with semi-related docs.
4. **Multi-seed cosine expansion** — using top hits as secondary queries fails when seeds are noise.
5. **Overlap-based adaptive weights** — comparing BM25/cosine top-5 overlap is too noisy and non-deterministic.

## What Works

1. **PRF (pseudo-relevance feedback)** — extract terms from top cosine hits, use for BM25 expansion. Domain-agnostic, no API needed.
2. **Enriched cosine** — embed `query + expansion_terms` as single vector for signal 1b.
3. **LLM expansion with aggressive should_expand** — expand most queries, skip only those with 2+ proper nouns.
4. **Better embedding models** — pplx-embed-v1-0.6b >> BGE >> nomic for retrieval.
5. **ARK_NO_DECAY** — essential for deterministic benchmarking.

## Benchmark Validity / Overfitting Concerns

Be skeptical of the headline numbers. Known issues:

1. **No train/test split** — all pipeline parameters (PRF top-k, RRF weights, `should_expand` heuristic, LLM prompt examples) were tuned by running against the same 130 queries. The benchmark is the dev set.

2. **Oracle query crafting** — queries were written by the same person who wrote the documents. Real users don't know the exact phrasing in the corpus. The queries are unnaturally well-matched.

3. **Only 23 target documents** — with 23 targets, random retrieval hits ~13% of queries by chance at Hit@3. Real production corpora have hundreds of relevant documents, all confusable with each other. **Fix in progress**: expanding to 60+ target docs across infra, data, and backend domains (branches `improve/infra-docs`, `improve/data-docs`).

4. **Fixed noise corpus** — the 1150 noise docs never change between runs. The system has effectively seen this noise during tuning. Production noise is domain-matched and unpredictable.

**Rough honest estimate**: subtract 10-15pp from headline numbers for a realistic out-of-distribution estimate. The relative rankings (pplx > BGE > nomic, LLM expansion helps vague queries) are real. The absolute numbers are optimistic.

## Remaining Challenges

With best config (pplx-embed + LLM expansion):

- **Temporal** (50%) — needs date-aware search, not similarity matching
- **Negation** (60%) — needs query parsing for exclusion logic
- **Adversarial** (70%) — noise docs with same vocabulary as our docs

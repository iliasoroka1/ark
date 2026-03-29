---
name: ark-memory
description: Search, add, dream, and manage agent memories via the ark CLI. Use when the user asks about agent memory, wants to store knowledge, search for information, or run memory consolidation.
argument-hint: "[action] [args...] e.g. search 'query', ingest 'fact', dream, analyze"
allowed-tools: Bash(ark *), Read, Grep
---

# Ark Memory

Interact with the ark knowledge engine via the `ark` CLI.

## Core commands

| Command | Usage | Description |
|---------|-------|-------------|
| search | `ark search "QUERY"` | Hybrid search (cosine + BM25 + temporal + graph) |
| search --graph | `ark search "QUERY" --graph` | Deep graph traversal with multi-hop |
| ingest | `ark ingest "CONTENT" --tag TAG` | Store a new memory |
| ingest-file | `ark ingest-file ./path.md --tag TAG` | Ingest a file with smart chunking |
| dream | `ark dream` | Memory consolidation (find updates, contradictions, implications) |
| analyze | `ark analyze` | Spectral analysis — RMT anomalies, PageRank, bridges |
| setup | `ark setup` | Interactive setup wizard |

## Session commands

| Command | Usage | Description |
|---------|-------|-------------|
| scratch-set | `ark scratch-set KEY VALUE` | Set a scratchpad value |
| scratch-get | `ark scratch-get KEY` | Get a scratchpad value |
| tasks-add | `ark tasks-add "TITLE"` | Add a task |
| tasks-list | `ark tasks-list` | List tasks |
| tasks-done | `ark tasks-done ID` | Mark task done |
| history-add | `ark history-add ROLE "CONTENT"` | Add history entry |
| history-search | `ark history-search "QUERY"` | Search history |

## Instructions

1. Parse the user's intent from `$ARGUMENTS`
2. Run the appropriate `ark` command
3. Present results clearly — summarize L0 abstracts, highlight connections
4. If the user wants to explore further, suggest follow-up commands

## How search works

Multi-signal pipeline: cosine (pplx-embed 1024d) + PRF + BM25 (tantivy en_stem) + temporal (hypergraph period nodes + Gaussian proximity) + LLM expansion + RRF merge + graph expansion (2-hop).

## Dreamer

`ark dream` runs an LLM specialist that finds knowledge updates, logical implications, and contradictions. Creates `derives_from` and `contradicts` graph edges for richer traversal.

## Environment

Config auto-loads from `~/.ark/config.json` (set via `ark setup`).
Key env vars: `OPENROUTER_API_KEY`, `ARK_HOME`, `ARK_NO_DECAY=1`, `ARK_NO_DREAM=1`, `ARK_NO_LLM_EXPAND=1`.

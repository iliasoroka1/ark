---
name: ark-memory
description: Search, add, and manage agent memories via the ark CLI. Use when the user asks about agent memory, wants to store knowledge, or explore the knowledge graph.
argument-hint: "[action] [args...] e.g. search 'query', add 'fact', list, analyze"
allowed-tools: Bash(ark *), Read, Grep
---

# Ark Memory

Interact with the tinyclaw agent memory system via the `ark` CLI.

## Available commands

| Command | Usage | Description |
|---------|-------|-------------|
| search | `ark memory search "QUERY"` | Find relevant memories |
| add | `ark memory add "CONTENT" --tag TAG` | Store a new memory |
| get | `ark memory get ID` | Get full content by ID |
| list | `ark memory list` | List memory clusters |
| graph | `ark memory graph "QUERY" --hops 2 --diverse` | Deep graph traversal |
| path | `ark memory path FROM_ID TO_ID` | Shortest path between memories |
| analyze | `ark memory analyze` | Spectral analysis of knowledge graph |

## Instructions

1. Parse the user's intent from `$ARGUMENTS`
2. Run the appropriate `ark memory` command
3. Present the results clearly — summarize L0 abstracts, highlight connections
4. If the user wants to explore further, suggest follow-up commands (e.g. `get` after `search`)

## Notes

- Search returns L0 summaries. Use `get` to load full content (L2)
- Graph search with `--diverse` returns varied results across the knowledge graph
- The `analyze` command shows novel memories (RMT), important ones (PageRank), and bridges (betweenness)
- Works in both API mode (tinyclaw server running) and local mode (reads ~/.tinyclaw/memory directly)

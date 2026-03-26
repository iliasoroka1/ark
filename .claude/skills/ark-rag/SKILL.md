---
name: ark-rag
description: Search and manage RAG documents via the ark CLI. Use when the user asks about ingested documents, wants to search knowledge base, or ingest new content.
argument-hint: "[action] [args...] e.g. search 'query', ingest-text 'content', list"
allowed-tools: Bash(ark *), Read, Grep
---

# Ark RAG

Interact with the tinyclaw RAG (Retrieval-Augmented Generation) document index.

## Available commands

| Command | Usage | Description |
|---------|-------|-------------|
| search | `ark rag search "QUERY" --limit 10` | Search ingested documents |
| ingest-text | `ark rag ingest-text "CONTENT" --title "Title" --tag TAG` | Ingest raw text |
| ingest-file | `ark rag ingest-file /path/to/file --title "Title"` | Ingest a file |
| list | `ark rag list` | List all ingested documents |
| delete | `ark rag delete ID` | Delete a document by ID |

## Instructions

1. Parse the user's intent from `$ARGUMENTS`
2. Run the appropriate `ark rag` command
3. Present search results with relevance context
4. For ingestion, confirm what was indexed and how many chunks were created

---
name: ark-history
description: View and search agent session history via the ark CLI. Use when the user asks about past conversations, recent agent activity, or wants to find something from a previous session.
argument-hint: "[action] [args...] e.g. recent --count 5, search 'query', list"
allowed-tools: Bash(ark *), Read, Grep
---

# Ark History

View and search tinyclaw agent session history.

## Available commands

| Command | Usage | Description |
|---------|-------|-------------|
| recent | `ark history recent --count 10 --session ID` | Get recent messages |
| search | `ark history search "QUERY" --count 10` | Search messages by query |
| list | `ark history list --count 20` | List recent sessions |

## Instructions

1. Parse the user's intent from `$ARGUMENTS`
2. Run the appropriate `ark history` command
3. Present results chronologically, highlight key exchanges

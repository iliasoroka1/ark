"""Re-embed all memories with nomic-embed-text-v1.5.

Clears old embeddings.db (384d bge-small) and tantivy index,
then re-ingests all memories with the new 768d model.
"""
import asyncio
import json
import os
import shutil


# All engineering memories
ENGINEERING = [
    'The authentication system uses JWT tokens with 24-hour expiry.',
    'Login bug was expired JWT tokens not caught in refresh flow.',
    'Rate limiting uses token bucket algorithm, 100 req/min per user.',
    'Session management refactored: switched from server-side sessions in PostgreSQL to stateless JWTs with 15-min access token + 30-day refresh token.',
    'MFA enforcement rolled out company-wide 2026-01-10.',
    'Migrated to OAuth 2.0 with PKCE flow for all public clients, replacing legacy implicit grant.',
    'SSO integration with Okta completed for internal tools.',
    'Deployed cart-service v2.3.1 to production on k8s cluster east-1, 3 replicas behind Envoy proxy.',
    'Set up Datadog monitors for payments-api: p99 latency > 400ms triggers PagerDuty alert to #payments-oncall.',
    'Auto-scaling config for recommendation-engine: min 3 pods, max 20, target CPU 65%.',
    'Redis cache layer added in front of catalog-service. TTL 5min for product listings, 1hr for category tree.',
    'Rewrote the product search query to use materialized view instead of 6-table join. Query time dropped from 800ms to 12ms.',
    'Migration 047_add_audit_log_table ran on prod 2026-02-18. Adds audit_log table with user_id, action, resource, timestamp columns.',
    'Added composite index on orders table: (customer_id, created_at DESC). Speeds up order history lookup 40x.',
    'Set up PostgreSQL streaming replication: primary on db-prod-01, replica on db-prod-02. Failover tested 2026-02-25.',
    'API versioning strategy: URL path prefix /api/v{N}. Current stable is v3, v2 deprecated, v1 sunset 2026-06-01.',
    'Error response format standardized across all services: {"error": {"code": "RESOURCE_NOT_FOUND", "message": "..."}}.',
    'Pagination standard for all list endpoints: cursor-based using opaque base64 token. Max page size 100, default 25.',
    'Rate limiting implemented via Envoy proxy sidecar. Per-service config in envoy-ratelimit ConfigMap.',
    'Sprint planning moved to Tuesdays 10am CET. Capacity: 60 story points per 2-week sprint.',
    'On-call rotation: weekly, handoff Monday 9am. Primary + secondary. Escalation path: primary -> secondary -> engineering manager.',
    'Code review policy updated 2026-03-01: PRs require 2 approvals for services in critical path (payments, auth, orders).',
    'Incident postmortem for 2026-03-12 payments outage (SEV-1, 47 min downtime): root cause was connection pool exhaustion in payments-db after config change removed max_connections limit.',
]


async def main():
    ark_home = os.path.expanduser("~/.ark")
    memory_dir = os.path.join(ark_home, "memory")

    # Clear old data — tantivy index, embedding cache, and graph
    print("Clearing old index, embeddings, and graph...")
    if os.path.exists(memory_dir):
        shutil.rmtree(memory_dir)
    os.makedirs(memory_dir, exist_ok=True)
    for db_file in ["embeddings.db", "graph.db"]:
        p = os.path.join(ark_home, db_file)
        if os.path.exists(p):
            os.remove(p)
            print(f"  Removed {db_file}")

    # Force re-init with new model
    import ark.local as local
    local._initialized = False
    local._indexer = None
    local._searcher = None
    local._graph_store = None

    from ark.local import call_tool

    # Ingest engineering memories
    print(f"Ingesting {len(ENGINEERING)} engineering memories...")
    for i, text in enumerate(ENGINEERING):
        await call_tool('ingest', {'content': text})
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(ENGINEERING)}]")

    # Ingest noise from seed_noise.py
    print("Ingesting 150 noise memories...")
    from seed_noise import NOISE
    for i, text in enumerate(NOISE):
        await call_tool('ingest', {'content': text})
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(NOISE)}]")

    # Ingest AG News noise
    print("Ingesting 500 AG News memories...")
    with open('ag_news_noise.json') as f:
        ag_news = json.load(f)
    for i, text in enumerate(ag_news):
        await call_tool('ingest', {'content': text})
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(ag_news)}]")

    # Ingest tech news noise
    print("Ingesting 500 tech news memories...")
    with open('tech_noise.json') as f:
        tech_news = json.load(f)
    for i, text in enumerate(tech_news):
        await call_tool('ingest', {'content': text})
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(tech_news)}]")

    total = len(ENGINEERING) + len(NOISE) + len(ag_news) + len(tech_news)
    print(f"\nDone: {total} memories ingested with nomic-embed-text-v1.5 (768d)")

    # Verify
    r = await call_tool('search', {'query': 'JWT tokens'})
    items = r.get('result', [])
    if isinstance(items, list) and items:
        print(f"Verification: search for 'JWT tokens' returned {len(items)} results")
        print(f"  Top: {items[0].get('l0', items[0].get('id', '?'))}")
    else:
        print("WARNING: verification search returned no results!")


if __name__ == '__main__':
    asyncio.run(main())

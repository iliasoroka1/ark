"""Benchmark raptis-mem-cli against the same brutal test set as ark.

Seeds the same 1173 memories, runs the same 70 queries, measures same metrics.
"""
import asyncio
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile

RAPTIS_BIN = "/Users/iliasoroka/meepo/target/release/raptis-mem"
os.environ["OPENROUTER_API_KEY"] = "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"

# Use a temp index dir to avoid polluting workspace
INDEX_DIR = "/tmp/raptis-bench-index"

K = 3

# Import test queries and ground truth from brutal benchmark
sys.path.insert(0, "/Users/iliasoroka/ark")
from test_recall_brutal import (
    EXACT, PARAPHRASE, TANGENTIAL, ADVERSARIAL, NEGATIVE, MULTIHOP,
    LEXICAL_TRAPS, COMPOSITIONAL, _ID_FINGERPRINTS, TestQuery
)
from seed_noise import NOISE

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


def raptis(cmd, *args, input_text=None):
    full = [RAPTIS_BIN, "--index", INDEX_DIR] + [cmd] + list(args)
    result = subprocess.run(full, capture_output=True, text=True, input=input_text, timeout=120)
    return result.stdout, result.stderr, result.returncode


def seed_all():
    print("Clearing old index...")
    if os.path.exists(INDEX_DIR):
        shutil.rmtree(INDEX_DIR)

    print("Initializing raptis index...")
    raptis("init")

    all_memories = ENGINEERING + NOISE
    # Skip AG News + tech noise for API-based benchmark (too slow)
    # Uncomment to include:
    # with open("/Users/iliasoroka/ark/ag_news_noise.json") as f:
    #     all_memories += json.load(f)
    # with open("/Users/iliasoroka/ark/tech_noise.json") as f:
    #     all_memories += json.load(f)

    total = len(all_memories)
    print(f"Seeding {total} memories...")
    for i, text in enumerate(all_memories):
        stdout, stderr, rc = raptis("add", text)
        if rc != 0 and i < 3:
            print(f"  Error on {i}: {stderr[:100]}")
        if (i + 1) % 100 == 0 or i == total - 1:
            print(f"  [{i+1}/{total}]")

    print(f"Done seeding {total} memories")


def run_search(query):
    stdout, stderr, rc = raptis("search", query, "--limit", "5")
    if rc != 0:
        return []
    # Parse raptis text output: "--- Result N (doc: ID, rrf: SCORE) ---\nBODY\n"
    import re
    results = []
    blocks = re.split(r'--- Result \d+', stdout)
    for block in blocks[1:]:  # skip empty first split
        match = re.search(r'\(doc: ([^,]+), rrf: ([^)]+)\) ---\n(.*?)(?=\n--- Result|\Z)', block, re.DOTALL)
        if match:
            doc_id = match.group(1).strip()
            rrf = float(match.group(2).strip())
            body = match.group(3).strip()
            results.append({"id": doc_id, "body": body, "rrf": rrf})
    return results


def is_relevant(item, tq):
    item_id = item.get("id", item.get("doc_id", ""))
    if item_id in tq.expected_ids:
        return True
    content = ""
    for key in ("body", "content", "l0", "text", "chunk_body"):
        content += " " + str(item.get(key, ""))
    content = content.lower()
    for eid in tq.expected_ids:
        fp = _ID_FINGERPRINTS.get(eid)
        if fp and fp in content:
            return True
    return False


def is_engineering(item):
    content = ""
    for key in ("body", "content", "l0", "text", "chunk_body"):
        content += " " + str(item.get(key, ""))
    content = content.lower()
    return any(fp in content for fp in _ID_FINGERPRINTS.values())


def dcg(rels, k):
    return sum(r / math.log2(i + 2) for i, r in enumerate(rels[:k]))


def ndcg(rels, k, n_rel):
    ideal = [1] * min(n_rel, k) + [0] * max(0, k - n_rel)
    idcg = dcg(ideal, k)
    return dcg(rels, k) / idcg if idcg > 0 else 0.0


def evaluate(queries, verbose=False):
    precisions, hit_rates, mrrs, ndcgs, fp_rates = [], [], [], [], []
    for tq in queries:
        results = run_search(tq.query)
        top_k = results[:K]
        if tq.is_negative:
            eng = sum(1 for r in top_k if is_engineering(r))
            fp_rates.append(eng / K if top_k else 0.0)
            if verbose:
                s = "CLEAN" if eng == 0 else f"LEAK({eng})"
                print(f"  [{s}] q=\"{tq.query[:50]}\"")
            continue
        rels = [1 if is_relevant(r, tq) else 0 for r in top_k]
        n_hits = sum(rels)
        n_rel = max(1, len(tq.expected_ids))
        precisions.append(n_hits / K)
        hit_rates.append(1.0 if n_hits > 0 else 0.0)
        rr = 0.0
        for i, r in enumerate(rels):
            if r:
                rr = 1.0 / (i + 1)
                break
        mrrs.append(rr)
        ndcgs.append(ndcg(rels, K, n_rel))
        if verbose:
            s = "HIT" if n_hits > 0 else "MISS"
            print(f"  [{s}] q=\"{tq.query[:50]}\" hits={n_hits}/{K} rr={rr:.2f}")
            if n_hits == 0 and top_k:
                for r in top_k[:2]:
                    body = ""
                    for key in ("body", "content", "l0", "text"):
                        if r.get(key):
                            body = str(r[key])[:70]
                            break
                    print(f"         -> {body}")
    m = {}
    if precisions:
        n = len(precisions)
        m.update({"precision@3": sum(precisions)/n, "hit_rate@3": sum(hit_rates)/n,
                  "mrr": sum(mrrs)/n, "ndcg@3": sum(ndcgs)/n, "n": n})
    if fp_rates:
        m.update({"false_positive_rate": sum(fp_rates)/len(fp_rates), "n_negative": len(fp_rates)})
    return m


def main():
    verbose = "-v" in sys.argv
    skip_seed = "--skip-seed" in sys.argv

    if not skip_seed:
        seed_all()

    cats = [
        ("Exact", EXACT), ("Paraphrase", PARAPHRASE), ("Tangential", TANGENTIAL),
        ("Adversarial", ADVERSARIAL), ("Negative", NEGATIVE), ("Multi-hop", MULTIHOP),
        ("Lexical traps", LEXICAL_TRAPS), ("Compositional", COMPOSITIONAL),
    ]

    total = sum(len(q) for _, q in cats)
    print("=" * 80)
    print(f"  RAPTIS-MEM benchmark — top-{K}, {total} queries")
    print(f"  Model: bge-m3 1024d via OpenRouter | RRF K=60")
    print("=" * 80)

    rows = []
    for name, queries in cats:
        print(f"\n{name} ({len(queries)} queries):")
        m = evaluate(queries, verbose)
        rows.append((name, m))

    pos = [(n, m) for n, m in rows if "hit_rate@3" in m]
    total_n = sum(m["n"] for _, m in pos)
    overall = {
        "precision@3": sum(m["precision@3"]*m["n"] for _,m in pos)/total_n,
        "hit_rate@3": sum(m["hit_rate@3"]*m["n"] for _,m in pos)/total_n,
        "mrr": sum(m["mrr"]*m["n"] for _,m in pos)/total_n,
        "ndcg@3": sum(m["ndcg@3"]*m["n"] for _,m in pos)/total_n,
        "n": total_n,
    }
    neg = [(n,m) for n,m in rows if "false_positive_rate" in m]
    if neg:
        overall["false_positive_rate"] = neg[0][1]["false_positive_rate"]

    print("\n" + "=" * 80)
    print(f"{'Category':<18} {'P@3':>7} {'Hit@3':>7} {'MRR':>7} {'NDCG@3':>7} {'FP':>6} {'N':>4}")
    print("-" * 80)
    for name, m in rows:
        p = f"{m['precision@3']:.1%}" if "precision@3" in m else "—"
        h = f"{m['hit_rate@3']:.1%}" if "hit_rate@3" in m else "—"
        mr = f"{m['mrr']:.3f}" if "mrr" in m else "—"
        nd = f"{m['ndcg@3']:.3f}" if "ndcg@3" in m else "—"
        fp = f"{m['false_positive_rate']:.0%}" if "false_positive_rate" in m else "—"
        n = m.get("n", m.get("n_negative", 0))
        print(f"{name:<18} {p:>7} {h:>7} {mr:>7} {nd:>7} {fp:>6} {n:>4}")
    print("-" * 80)
    fp = f"{overall.get('false_positive_rate',0):.0%}" if "false_positive_rate" in overall else "—"
    print(f"{'OVERALL':<18} {overall['precision@3']:.1%} {overall['hit_rate@3']:.1%} {overall['mrr']:.3f} {overall['ndcg@3']:.3f} {fp:>6} {overall['n']:>4}")
    print("=" * 80)

    print("\n# JSON:")
    out = {n: {k: round(v,4) if isinstance(v,float) else v for k,v in m.items()} for n,m in rows+[("overall",overall)]}
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

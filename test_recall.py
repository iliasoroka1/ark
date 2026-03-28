"""Recall quality evaluation for ark memory search.

Metrics:
  - Hit Rate@K: fraction of queries where at least 1 relevant result appears in top-K
  - Precision@K: fraction of top-K results that are relevant (averaged over queries)
  - MRR: mean reciprocal rank of first relevant result
  - NDCG@K: normalized discounted cumulative gain

A result is "relevant" if it matches ANY expected_id (ground truth doc IDs).
"""

import json
import subprocess
import sys
from dataclasses import dataclass, field
import math

K = 3  # evaluation cutoff


@dataclass
class TestQuery:
    query: str
    category: str
    expected_ids: list[str]  # ground-truth doc IDs that are relevant
    description: str = ""    # human-readable description of what we expect


# ── Ground-truth document IDs (from ark search) ──

# Auth memories
AUTH_JWT_24H = "e5cbc506b705bf20"       # JWT tokens with 24-hour expiry
AUTH_LOGIN_BUG = "5ebf4f751b53d0d7"     # login bug expired JWT refresh
AUTH_SESSION = "e17c981595bb266a"        # session mgmt refactored, stateless JWTs
AUTH_MFA = "ff2f93a81a68fc8c"           # MFA enforcement rolled out
AUTH_OAUTH = "6520003b5e121630"         # migrated to OAuth 2
AUTH_SSO = "9a0fa4d17f0fd207"           # SSO integration with Okta

# Infra memories
INFRA_RATELIMIT = "eb9a2f5d7a388697"    # rate limiting token bucket 100 req/min
INFRA_DATADOG = "d13e7d62d65240ac"      # Datadog monitors payments-api
INFRA_AUTOSCALE = "6fea38e8d09556fb"    # auto-scaling recommendation-engine
INFRA_CART = "08a91aeffee0a5e1"         # deployed cart-service v2
INFRA_REDIS = "b690f07b60f0257b"        # Redis cache catalog-service

# DB memories
DB_MATVIEW = "d5b1d205a0ff47c1"         # materialized view instead of 6-table join
DB_MIGRATION = "79fa0b520862f106"       # migration 047 audit log
DB_INDEX = "e2d12ca9459525ef"           # composite index orders table
DB_REPLICATION = "c5409c65f311628a"     # PostgreSQL streaming replication

# API memories
API_VERSIONING = "913d26b10564c09a"     # API versioning /api/v{N}
API_ERRORS = "dee0d1d453d7be80"         # error response format standardized
API_PAGINATION = "2442efb11a5228cd"     # cursor-based pagination
API_ENVOY = "beacc4ddd0cb061e"          # rate limiting via Envoy proxy

# Process memories
PROC_SPRINT = "d3fae1b223578479"        # sprint planning Tuesdays
PROC_ONCALL = "14cddd8bb324a254"        # on-call rotation weekly
PROC_CODEREVIEW = "c4658bd5ad47ab50"    # code review policy 2 approvals
PROC_INCIDENTS = "592ec5175080e20f"     # incident postmortem payments outage SEV-1


# ── Test queries covering all categories ──

EXACT_RECALL = [
    TestQuery("JWT tokens with 24-hour expiry", "exact",
              [AUTH_JWT_24H], "exact phrase from auth JWT memory"),
    TestQuery("token bucket algorithm 100 req/min", "exact",
              [INFRA_RATELIMIT], "exact phrase from rate limiting memory"),
    TestQuery("expired JWT tokens not caught in refresh flow", "exact",
              [AUTH_LOGIN_BUG], "exact phrase from login bug memory"),
    TestQuery("materialized view instead of 6-table join", "exact",
              [DB_MATVIEW], "exact phrase from DB query optimization"),
    TestQuery("cursor-based using opaque base64 token", "exact",
              [API_PAGINATION], "exact phrase from pagination memory"),
    TestQuery("sprint planning moved to Tuesdays", "exact",
              [PROC_SPRINT], "exact phrase from process memory"),
    TestQuery("SSO integration with Okta", "exact",
              [AUTH_SSO], "exact phrase from SSO memory"),
    TestQuery("composite index on orders table", "exact",
              [DB_INDEX], "exact phrase from DB index memory"),
    TestQuery("Datadog monitors for payments-api", "exact",
              [INFRA_DATADOG], "exact phrase from monitoring memory"),
    TestQuery("MFA enforcement rolled out", "exact",
              [AUTH_MFA], "exact phrase from MFA memory"),
]

PARAPHRASE_RECALL = [
    TestQuery("authentication tokens expire after a day", "paraphrase",
              [AUTH_JWT_24H], "paraphrase of JWT 24h expiry"),
    TestQuery("API throttling with bucket-based algorithm", "paraphrase",
              [INFRA_RATELIMIT], "paraphrase of rate limiting"),
    TestQuery("bug where stale auth tokens were not refreshed", "paraphrase",
              [AUTH_LOGIN_BUG], "paraphrase of login bug"),
    TestQuery("optimized slow product query with a precomputed view", "paraphrase",
              [DB_MATVIEW], "paraphrase of materialized view optimization"),
    TestQuery("paginating API responses with an opaque cursor", "paraphrase",
              [API_PAGINATION], "paraphrase of cursor pagination"),
    TestQuery("when does the team do sprint planning", "paraphrase",
              [PROC_SPRINT], "paraphrase of sprint planning schedule"),
    TestQuery("single sign-on setup for company tools", "paraphrase",
              [AUTH_SSO], "paraphrase of SSO/Okta integration"),
    TestQuery("database index for looking up orders by customer", "paraphrase",
              [DB_INDEX], "paraphrase of composite index"),
    TestQuery("monitoring and alerting for the payments service", "paraphrase",
              [INFRA_DATADOG], "paraphrase of Datadog monitoring"),
    TestQuery("mandatory two-factor authentication rollout", "paraphrase",
              [AUTH_MFA], "paraphrase of MFA enforcement"),
]

TANGENTIAL_RECALL = [
    TestQuery("security improvements we made", "tangential",
              [AUTH_JWT_24H, AUTH_MFA, AUTH_SSO, AUTH_OAUTH, AUTH_SESSION],
              "broad security query — should surface auth-related memories"),
    TestQuery("how do we prevent API abuse", "tangential",
              [INFRA_RATELIMIT, API_ENVOY],
              "abuse prevention — should find rate limiting"),
    TestQuery("what changed in our database recently", "tangential",
              [DB_MATVIEW, DB_MIGRATION, DB_INDEX, DB_REPLICATION],
              "broad DB query — should surface any db memory"),
    TestQuery("production incidents and bugs", "tangential",
              [AUTH_LOGIN_BUG],
              "incident query — should find login bug"),
    TestQuery("how does our caching work", "tangential",
              [INFRA_REDIS],
              "caching query — should find Redis cache memory"),
    TestQuery("deployment and scaling setup", "tangential",
              [INFRA_CART, INFRA_AUTOSCALE],
              "deployment query — should find cart/autoscale"),
    TestQuery("team workflow and ceremonies", "tangential",
              [PROC_SPRINT, PROC_ONCALL, PROC_CODEREVIEW],
              "process query — should find any team process memory"),
    TestQuery("how are errors handled in our APIs", "tangential",
              [API_ERRORS],
              "error handling query — should find error format memory"),
    TestQuery("what authentication providers do we use", "tangential",
              [AUTH_OAUTH, AUTH_SSO],
              "auth provider query — should find OAuth/SSO"),
    TestQuery("database performance and optimization", "tangential",
              [DB_MATVIEW, DB_INDEX, DB_REPLICATION],
              "DB perf query — should find optimization memories"),
]

ALL_QUERIES = EXACT_RECALL + PARAPHRASE_RECALL + TANGENTIAL_RECALL


def run_search(query: str) -> list[dict]:
    """Run ark search and return parsed results."""
    result = subprocess.run(
        ["uv", "run", "ark", "search", query],
        capture_output=True, text=True, cwd="/Users/iliasoroka/ark"
    )
    if result.returncode != 0:
        print(f"  ERROR: '{query}': {result.stderr.strip()}")
        return []
    try:
        data = json.loads(result.stdout)
        return extract_results(data)
    except json.JSONDecodeError:
        print(f"  ERROR parsing: '{query}': {result.stdout[:200]}")
        return []


def extract_results(data: dict) -> list[dict]:
    """Handle both response formats: {results: [...]} and {ok: true, result: [...]}."""
    if "results" in data:
        return data["results"]
    result = data.get("result", [])
    if isinstance(result, list):
        return result
    return []


# Map ground-truth IDs to content fingerprints for dedup-safe matching.
# When duplicate ingestion creates new IDs for the same content, we match
# on a unique substring from the original memory instead of exact ID.
_ID_FINGERPRINTS: dict[str, str] = {
    "e5cbc506b705bf20": "jwt tokens with 24-hour expiry",
    "5ebf4f751b53d0d7": "login bug was expired jwt",
    "e17c981595bb266a": "switched from server-side sessions",
    "ff2f93a81a68fc8c": "mfa enforcement rolled out",
    "6520003b5e121630": "migrated to oauth 2",
    "9a0fa4d17f0fd207": "sso integration with okta",
    "eb9a2f5d7a388697": "token bucket algorithm, 100 req/min",
    "d13e7d62d65240ac": "datadog monitors for payments-api",
    "6fea38e8d09556fb": "auto-scaling config for recommendation",
    "08a91aeffee0a5e1": "cart-service v2",
    "b690f07b60f0257b": "redis cache layer added",
    "d5b1d205a0ff47c1": "materialized view instead of 6-table",
    "79fa0b520862f106": "migration 047_add_audit_log",
    "e2d12ca9459525ef": "composite index on orders table",
    "c5409c65f311628a": "postgresql streaming replication",
    "913d26b10564c09a": "api versioning strategy",
    "dee0d1d453d7be80": "error response format standardized",
    "2442efb11a5228cd": "cursor-based using opaque base64",
    "beacc4ddd0cb061e": "envoy proxy sidecar",
    "d3fae1b223578479": "sprint planning moved to tuesdays",
    "14cddd8bb324a254": "on-call rotation: weekly",
    "c4658bd5ad47ab50": "code review policy updated",
    "592ec5175080e20f": "incident postmortem for 2026-03-12",
}


def is_relevant(item: dict, tq: TestQuery) -> bool:
    """Check if a result matches expected content, tolerating duplicate IDs."""
    item_id = item.get("id", "")
    if item_id in tq.expected_ids:
        return True
    # Fallback: match by content fingerprint for any expected ID
    content = (item.get("l0", "") + " " + item.get("content", "")).lower()
    for eid in tq.expected_ids:
        fp = _ID_FINGERPRINTS.get(eid)
        if fp and fp in content:
            return True
    return False


def dcg(relevances: list[int], k: int) -> float:
    """Discounted cumulative gain."""
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances[:k]))


def ndcg(relevances: list[int], k: int, n_relevant: int) -> float:
    """Normalized DCG — ideal is all relevant docs at top."""
    ideal = [1] * min(n_relevant, k) + [0] * max(0, k - n_relevant)
    ideal_dcg = dcg(ideal, k)
    if ideal_dcg == 0:
        return 0.0
    return dcg(relevances, k) / ideal_dcg


def evaluate(queries: list[TestQuery], verbose: bool = False) -> dict:
    """Evaluate queries, return metrics."""
    precisions = []
    hit_rates = []
    reciprocal_ranks = []
    ndcgs = []

    for tq in queries:
        results = run_search(tq.query)
        top_k = results[:K]
        relevances = [1 if is_relevant(r, tq) else 0 for r in top_k]
        n_hits = sum(relevances)
        n_relevant = len(tq.expected_ids)

        precisions.append(n_hits / K)
        hit_rates.append(1.0 if n_hits > 0 else 0.0)

        rr = 0.0
        for i, rel in enumerate(relevances):
            if rel:
                rr = 1.0 / (i + 1)
                break
        reciprocal_ranks.append(rr)
        ndcgs.append(ndcg(relevances, K, n_relevant))

        if verbose:
            status = "HIT" if n_hits > 0 else "MISS"
            print(f"  [{status}] q=\"{tq.query[:50]}\" hits={n_hits}/{K} rr={rr:.2f}")
            if n_hits == 0:
                returned_ids = [r.get("id", "?")[:8] for r in top_k]
                expected_ids = [eid[:8] for eid in tq.expected_ids]
                print(f"         expected: {expected_ids}  got: {returned_ids}")

    n = len(queries)
    return {
        "precision@3": sum(precisions) / n,
        "hit_rate@3": sum(hit_rates) / n,
        "mrr": sum(reciprocal_ranks) / n,
        "ndcg@3": sum(ndcgs) / n,
        "n": n,
    }


def main():
    verbose = "-v" in sys.argv or "--verbose" in sys.argv
    categories = [
        ("Exact recall", EXACT_RECALL),
        ("Paraphrase recall", PARAPHRASE_RECALL),
        ("Tangential recall", TANGENTIAL_RECALL),
    ]

    print("=" * 72)
    print(f"  ark recall baseline — top-{K}, {len(ALL_QUERIES)} queries, ground-truth IDs")
    print("=" * 72)

    all_p, all_h, all_m, all_n, total = [], [], [], [], 0
    rows = []

    for name, queries in categories:
        print(f"\n{name} ({len(queries)} queries):")
        m = evaluate(queries, verbose=verbose)
        rows.append((name, m))
        all_p.append(m["precision@3"] * m["n"])
        all_h.append(m["hit_rate@3"] * m["n"])
        all_m.append(m["mrr"] * m["n"])
        all_n.append(m["ndcg@3"] * m["n"])
        total += m["n"]

    overall = {
        "precision@3": sum(all_p) / total,
        "hit_rate@3": sum(all_h) / total,
        "mrr": sum(all_m) / total,
        "ndcg@3": sum(all_n) / total,
        "n": total,
    }

    print("\n" + "=" * 72)
    print(f"{'Category':<22} {'P@3':>8} {'Hit@3':>8} {'MRR':>8} {'NDCG@3':>8} {'N':>5}")
    print("-" * 72)
    for name, m in rows:
        print(f"{name:<22} {m['precision@3']:>7.1%} {m['hit_rate@3']:>7.1%} {m['mrr']:>8.3f} {m['ndcg@3']:>8.3f} {m['n']:>5}")
    print("-" * 72)
    print(f"{'OVERALL':<22} {overall['precision@3']:>7.1%} {overall['hit_rate@3']:>7.1%} {overall['mrr']:>8.3f} {overall['ndcg@3']:>8.3f} {overall['n']:>5}")
    print("=" * 72)

    # Machine-readable output for comparison
    print("\n# JSON baseline (copy for diff):")
    print(json.dumps({name: {k: round(v, 4) for k, v in m.items()} for name, m in rows + [("overall", overall)]}, indent=2))


if __name__ == "__main__":
    main()

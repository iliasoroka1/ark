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
AUTH_JWT_24H = "7d0bca77d6aae0a2"       # JWT tokens with 24-hour expiry
AUTH_LOGIN_BUG = "fa23c59e0072d1e1"     # login bug expired JWT refresh
AUTH_SESSION = "3ce793d76cd38eaa"        # session mgmt refactored, stateless JWTs
AUTH_MFA = "c8c3424bc410efd6"           # MFA enforcement rolled out
AUTH_OAUTH = "f9f921a659be8aaa"         # migrated to OAuth 2.0 PKCE
AUTH_SSO = "01e5c6fa83305bc8"           # SSO integration with Okta

# Infra memories
INFRA_RATELIMIT = "0cc63c7e0822d695"    # rate limiting token bucket 100 req/min
INFRA_DATADOG = "eef3c42aca4bd10d"      # Datadog monitors payments-api
INFRA_AUTOSCALE = "c03dea8ef4182cc6"    # auto-scaling recommendation-engine
INFRA_CART = "961d45b3a241e921"         # deployed cart-service v2.3.1
INFRA_REDIS = "df9bdfd4eac84e23"        # Redis cache catalog-service

# DB memories
DB_MATVIEW = "e72027dc1661d649"         # materialized view instead of 6-table join
DB_MIGRATION = "78e9fee166a3640d"       # migration 047 audit log
DB_INDEX = "72cbb27dba5288af"           # composite index orders table
DB_REPLICATION = "3cdebbffc0b53404"     # PostgreSQL streaming replication

# API memories
API_VERSIONING = "3cae47bea69627cf"     # API versioning /api/v{N}
API_ERRORS = "3f4050699ddb7055"         # error response format standardized
API_PAGINATION = "f9f6fa734426d4c1"     # cursor-based pagination
API_ENVOY = "38db5b3c761350e1"          # rate limiting via Envoy proxy

# Process memories
PROC_SPRINT = "8715f400cd17137e"        # sprint planning Tuesdays
PROC_ONCALL = "a7bfacf4ce1a4232"        # on-call rotation weekly
PROC_CODEREVIEW = "6a1aacd3f96e7cf3"    # code review policy 2 approvals
PROC_INCIDENTS = "2c7c5fc8a3fdb4b7"     # incident postmortem payments outage SEV-1


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
        return data.get("results", data.get("result", []))
    except json.JSONDecodeError:
        print(f"  ERROR parsing: '{query}': {result.stdout[:200]}")
        return []


def is_relevant(item: dict, tq: TestQuery) -> bool:
    """Check if a result matches any expected ground-truth ID."""
    return item.get("id", "") in tq.expected_ids


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

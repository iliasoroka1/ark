"""Hard recall benchmark for ark memory search.

Tests against a corpus of ~173 memories: 23 engineering + 150 noise (finance,
science, cooking, sports, history, geography, art, software-adjacent).

Categories:
  1. Exact recall (10) — verbatim phrases
  2. Paraphrase recall (10) — rephrased concepts
  3. Tangential recall (10) — indirect/abstract queries
  4. Adversarial distractors (10) — software terms that should match OUR memories, not generic ones
  5. Negative queries (5) — should return NO relevant engineering memories
  6. Multi-hop (5) — require chaining facts

Metrics: hit_rate@K, precision@K, MRR, NDCG@K, false_positive_rate (for negatives)
"""

import json
import math
import subprocess
import sys
from dataclasses import dataclass

K = 3


@dataclass
class TestQuery:
    query: str
    category: str
    expected_ids: list[str]
    description: str = ""
    is_negative: bool = False  # True = no results expected


# ── Content fingerprints for dedup-safe matching ──
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

# Ground-truth IDs
AUTH_JWT_24H = "e5cbc506b705bf20"
AUTH_LOGIN_BUG = "5ebf4f751b53d0d7"
AUTH_SESSION = "e17c981595bb266a"
AUTH_MFA = "ff2f93a81a68fc8c"
AUTH_OAUTH = "6520003b5e121630"
AUTH_SSO = "9a0fa4d17f0fd207"
INFRA_RATELIMIT = "eb9a2f5d7a388697"
INFRA_DATADOG = "d13e7d62d65240ac"
INFRA_AUTOSCALE = "6fea38e8d09556fb"
INFRA_CART = "08a91aeffee0a5e1"
INFRA_REDIS = "b690f07b60f0257b"
DB_MATVIEW = "d5b1d205a0ff47c1"
DB_MIGRATION = "79fa0b520862f106"
DB_INDEX = "e2d12ca9459525ef"
DB_REPLICATION = "c5409c65f311628a"
API_VERSIONING = "913d26b10564c09a"
API_ERRORS = "dee0d1d453d7be80"
API_PAGINATION = "2442efb11a5228cd"
API_ENVOY = "beacc4ddd0cb061e"
PROC_SPRINT = "d3fae1b223578479"
PROC_ONCALL = "14cddd8bb324a254"
PROC_CODEREVIEW = "c4658bd5ad47ab50"
PROC_INCIDENTS = "592ec5175080e20f"

# ── Test queries ──

EXACT = [
    TestQuery("JWT tokens with 24-hour expiry", "exact", [AUTH_JWT_24H]),
    TestQuery("token bucket algorithm 100 req/min", "exact", [INFRA_RATELIMIT]),
    TestQuery("expired JWT tokens not caught in refresh flow", "exact", [AUTH_LOGIN_BUG]),
    TestQuery("materialized view instead of 6-table join", "exact", [DB_MATVIEW]),
    TestQuery("cursor-based using opaque base64 token", "exact", [API_PAGINATION]),
    TestQuery("sprint planning moved to Tuesdays", "exact", [PROC_SPRINT]),
    TestQuery("SSO integration with Okta", "exact", [AUTH_SSO]),
    TestQuery("composite index on orders table", "exact", [DB_INDEX]),
    TestQuery("Datadog monitors for payments-api", "exact", [INFRA_DATADOG]),
    TestQuery("MFA enforcement rolled out", "exact", [AUTH_MFA]),
]

PARAPHRASE = [
    TestQuery("authentication tokens expire after a day", "paraphrase", [AUTH_JWT_24H]),
    TestQuery("API throttling with bucket-based algorithm", "paraphrase", [INFRA_RATELIMIT]),
    TestQuery("bug where stale auth tokens were not refreshed", "paraphrase", [AUTH_LOGIN_BUG]),
    TestQuery("optimized slow product query with a precomputed view", "paraphrase", [DB_MATVIEW]),
    TestQuery("paginating API responses with an opaque cursor", "paraphrase", [API_PAGINATION]),
    TestQuery("when does the team do sprint planning", "paraphrase", [PROC_SPRINT]),
    TestQuery("single sign-on setup for company tools", "paraphrase", [AUTH_SSO]),
    TestQuery("database index for looking up orders by customer", "paraphrase", [DB_INDEX]),
    TestQuery("monitoring and alerting for the payments service", "paraphrase", [INFRA_DATADOG]),
    TestQuery("mandatory two-factor authentication rollout", "paraphrase", [AUTH_MFA]),
]

TANGENTIAL = [
    TestQuery("security improvements we made", "tangential",
              [AUTH_JWT_24H, AUTH_MFA, AUTH_SSO, AUTH_OAUTH, AUTH_SESSION]),
    TestQuery("how do we prevent API abuse", "tangential",
              [INFRA_RATELIMIT, API_ENVOY]),
    TestQuery("what changed in our database recently", "tangential",
              [DB_MATVIEW, DB_MIGRATION, DB_INDEX, DB_REPLICATION]),
    TestQuery("production incidents and bugs", "tangential",
              [AUTH_LOGIN_BUG, PROC_INCIDENTS]),
    TestQuery("how does our caching work", "tangential",
              [INFRA_REDIS]),
    TestQuery("deployment and scaling setup", "tangential",
              [INFRA_CART, INFRA_AUTOSCALE]),
    TestQuery("team workflow and ceremonies", "tangential",
              [PROC_SPRINT, PROC_ONCALL, PROC_CODEREVIEW]),
    TestQuery("how are errors handled in our APIs", "tangential",
              [API_ERRORS]),
    TestQuery("what authentication providers do we use", "tangential",
              [AUTH_OAUTH, AUTH_SSO]),
    TestQuery("database performance and optimization", "tangential",
              [DB_MATVIEW, DB_INDEX, DB_REPLICATION]),
]

# Adversarial: queries that use terms shared with noise memories but should
# match OUR engineering memories, not generic software facts
ADVERSARIAL = [
    TestQuery("our Redis caching setup", "adversarial",
              [INFRA_REDIS],
              "should find OUR Redis config, not generic 'Redis supports five data types'"),
    TestQuery("how we use OAuth", "adversarial",
              [AUTH_OAUTH],
              "should find OUR OAuth migration, not generic 'OAuth 2.0 authorization code flow'"),
    TestQuery("database indexing strategy", "adversarial",
              [DB_INDEX],
              "should find OUR composite index, not generic 'B-trees are standard for indexes'"),
    TestQuery("our monitoring and alerting", "adversarial",
              [INFRA_DATADOG],
              "should find OUR Datadog setup, not generic 'Prometheus collects metrics'"),
    TestQuery("container deployment", "adversarial",
              [INFRA_CART],
              "should find OUR cart-service deploy, not generic 'Kubernetes pods' or 'Docker containers'"),
    TestQuery("circuit breaker in our services", "adversarial",
              [PROC_INCIDENTS],
              "should find OUR incident, not generic 'circuit breaker pattern prevents failures'"),
    TestQuery("how we handle load balancing", "adversarial",
              [API_ENVOY, INFRA_RATELIMIT],
              "should find OUR Envoy/rate limiting, not generic 'load balancers distribute traffic'"),
    TestQuery("our message queue setup", "adversarial",
              [],
              "we don't have a message queue — should return nothing relevant"),
    TestQuery("our PostgreSQL configuration", "adversarial",
              [DB_REPLICATION, AUTH_SESSION],
              "should find OUR PG replication/sessions, not generic DB facts"),
    TestQuery("API design patterns we follow", "adversarial",
              [API_VERSIONING, API_ERRORS, API_PAGINATION],
              "should find OUR versioning/errors/pagination, not generic 'microservices communicate via APIs'"),
]

# Negative: queries that should NOT match any engineering memory
NEGATIVE = [
    TestQuery("recipe for sourdough bread", "negative", [], is_negative=True,
              description="cooking — no engineering match"),
    TestQuery("World Cup soccer results", "negative", [], is_negative=True,
              description="sports — no engineering match"),
    TestQuery("stock market performance in 2023", "negative", [], is_negative=True,
              description="finance — no engineering match"),
    TestQuery("how do volcanoes form", "negative", [], is_negative=True,
              description="geology — no engineering match"),
    TestQuery("Beethoven symphony compositions", "negative", [], is_negative=True,
              description="music — no engineering match"),
]

# Multi-hop: require connecting multiple memories
MULTIHOP = [
    TestQuery("why did the payments service go down", "multihop",
              [PROC_INCIDENTS, DB_REPLICATION],
              "chain: outage → connection pool → payments-db"),
    TestQuery("what security layers protect our APIs", "multihop",
              [AUTH_JWT_24H, INFRA_RATELIMIT, API_ENVOY, AUTH_MFA],
              "chain: JWT auth + rate limiting + Envoy proxy + MFA"),
    TestQuery("how did we improve database query speed", "multihop",
              [DB_MATVIEW, DB_INDEX],
              "chain: materialized view + composite index"),
    TestQuery("what happens when a user session expires", "multihop",
              [AUTH_SESSION, AUTH_LOGIN_BUG, AUTH_JWT_24H],
              "chain: JWT expiry → refresh flow → login bug"),
    TestQuery("our complete deployment and monitoring pipeline", "multihop",
              [INFRA_CART, INFRA_AUTOSCALE, INFRA_DATADOG],
              "chain: deploy → autoscale → monitoring"),
]

ALL_QUERIES = EXACT + PARAPHRASE + TANGENTIAL + ADVERSARIAL + NEGATIVE + MULTIHOP


def run_search(query: str) -> list[dict]:
    result = subprocess.run(
        ["uv", "run", "ark", "search", query],
        capture_output=True, text=True, cwd="/Users/iliasoroka/ark"
    )
    if result.returncode != 0:
        return []
    try:
        data = json.loads(result.stdout)
        if "results" in data:
            return data["results"]
        r = data.get("result", [])
        return r if isinstance(r, list) else []
    except json.JSONDecodeError:
        return []


def is_relevant(item: dict, tq: TestQuery) -> bool:
    item_id = item.get("id", "")
    if item_id in tq.expected_ids:
        return True
    content = (item.get("l0", "") + " " + item.get("content", "")).lower()
    for eid in tq.expected_ids:
        fp = _ID_FINGERPRINTS.get(eid)
        if fp and fp in content:
            return True
    return False


def is_engineering_memory(item: dict) -> bool:
    """Check if result is one of our 23 engineering memories (not noise)."""
    content = (item.get("l0", "") + " " + item.get("content", "")).lower()
    return any(fp in content for fp in _ID_FINGERPRINTS.values())


def dcg(rels: list[int], k: int) -> float:
    return sum(r / math.log2(i + 2) for i, r in enumerate(rels[:k]))


def ndcg(rels: list[int], k: int, n_rel: int) -> float:
    ideal = [1] * min(n_rel, k) + [0] * max(0, k - n_rel)
    idcg = dcg(ideal, k)
    return dcg(rels, k) / idcg if idcg > 0 else 0.0


def evaluate(queries: list[TestQuery], verbose: bool = False) -> dict:
    precisions, hit_rates, mrrs, ndcgs = [], [], [], []
    fp_rates = []  # for negative queries

    for tq in queries:
        results = run_search(tq.query)
        top_k = results[:K]

        if tq.is_negative:
            # For negatives: count how many engineering memories leak in
            eng_hits = sum(1 for r in top_k if is_engineering_memory(r))
            fp_rate = eng_hits / K if top_k else 0.0
            fp_rates.append(fp_rate)
            if verbose:
                status = "CLEAN" if eng_hits == 0 else f"LEAK({eng_hits})"
                print(f"  [{status}] q=\"{tq.query[:50]}\"")
                if eng_hits > 0:
                    for r in top_k:
                        if is_engineering_memory(r):
                            c = r.get("l0", r.get("content", ""))[:60]
                            print(f"         leaked: {c}")
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
            status = "HIT" if n_hits > 0 else "MISS"
            print(f"  [{status}] q=\"{tq.query[:50]}\" hits={n_hits}/{K} rr={rr:.2f}")
            if n_hits == 0:
                got = [r.get("id", "?")[:8] for r in top_k]
                exp = [e[:8] for e in tq.expected_ids[:4]]
                print(f"         expected: {exp}  got: {got}")
                # Show what was returned
                for r in top_k[:2]:
                    c = r.get("l0", r.get("content", ""))[:70]
                    print(f"         -> {c}")

    m = {}
    if precisions:
        n = len(precisions)
        m["precision@3"] = sum(precisions) / n
        m["hit_rate@3"] = sum(hit_rates) / n
        m["mrr"] = sum(mrrs) / n
        m["ndcg@3"] = sum(ndcgs) / n
        m["n"] = n
    if fp_rates:
        m["false_positive_rate"] = sum(fp_rates) / len(fp_rates)
        m["n_negative"] = len(fp_rates)
    return m


def main():
    verbose = "-v" in sys.argv or "--verbose" in sys.argv
    categories = [
        ("Exact", EXACT),
        ("Paraphrase", PARAPHRASE),
        ("Tangential", TANGENTIAL),
        ("Adversarial", ADVERSARIAL),
        ("Negative", NEGATIVE),
        ("Multi-hop", MULTIHOP),
    ]

    total_q = len(ALL_QUERIES)
    print("=" * 78)
    print(f"  ark HARD recall benchmark — top-{K}, {total_q} queries, ~173 docs (23 eng + 150 noise)")
    print("=" * 78)

    rows = []
    for name, queries in categories:
        print(f"\n{name} ({len(queries)} queries):")
        m = evaluate(queries, verbose=verbose)
        rows.append((name, m))

    # Aggregate (excluding negatives from main metrics)
    pos_rows = [(n, m) for n, m in rows if "hit_rate@3" in m]
    total_n = sum(m["n"] for _, m in pos_rows)
    overall = {
        "precision@3": sum(m["precision@3"] * m["n"] for _, m in pos_rows) / total_n,
        "hit_rate@3": sum(m["hit_rate@3"] * m["n"] for _, m in pos_rows) / total_n,
        "mrr": sum(m["mrr"] * m["n"] for _, m in pos_rows) / total_n,
        "ndcg@3": sum(m["ndcg@3"] * m["n"] for _, m in pos_rows) / total_n,
        "n": total_n,
    }
    neg_rows = [(n, m) for n, m in rows if "false_positive_rate" in m]
    if neg_rows:
        overall["false_positive_rate"] = neg_rows[0][1]["false_positive_rate"]

    print("\n" + "=" * 78)
    print(f"{'Category':<16} {'P@3':>7} {'Hit@3':>7} {'MRR':>7} {'NDCG@3':>7} {'FP Rate':>8} {'N':>4}")
    print("-" * 78)
    for name, m in rows:
        p = f"{m['precision@3']:.1%}" if "precision@3" in m else "—"
        h = f"{m['hit_rate@3']:.1%}" if "hit_rate@3" in m else "—"
        mr = f"{m['mrr']:.3f}" if "mrr" in m else "—"
        nd = f"{m['ndcg@3']:.3f}" if "ndcg@3" in m else "—"
        fp = f"{m['false_positive_rate']:.1%}" if "false_positive_rate" in m else "—"
        n = m.get("n", m.get("n_negative", 0))
        print(f"{name:<16} {p:>7} {h:>7} {mr:>7} {nd:>7} {fp:>8} {n:>4}")
    print("-" * 78)
    p = f"{overall['precision@3']:.1%}"
    h = f"{overall['hit_rate@3']:.1%}"
    mr = f"{overall['mrr']:.3f}"
    nd = f"{overall['ndcg@3']:.3f}"
    fp = f"{overall.get('false_positive_rate', 0):.1%}" if "false_positive_rate" in overall else "—"
    print(f"{'OVERALL':<16} {p:>7} {h:>7} {mr:>7} {nd:>7} {fp:>8} {overall['n']:>4}")
    print("=" * 78)

    print("\n# JSON (copy for diff):")
    out = {n: {k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()} for n, m in rows + [("overall", overall)]}
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

"""Run recall test using our modified search directly (no HTTP server needed).

Opens the tantivy index read-only and uses Searcher directly.
"""

import asyncio
import json
import math
import sys
import os
from collections import defaultdict
from dataclasses import dataclass

import tantivy

from ark.engine.index import build_schema
from ark.engine.search import Searcher
from ark.engine.embedding_cache import EmbeddingCache
from ark.engine.types import SearchParams

K = 3

@dataclass
class TestQuery:
    query: str
    category: str
    expected_ids: list[str]
    description: str = ""


# Ground-truth document IDs
AUTH_JWT_24H = "7d0bca77d6aae0a2"
AUTH_LOGIN_BUG = "fa23c59e0072d1e1"
AUTH_SESSION = "3ce793d76cd38eaa"
AUTH_MFA = "c8c3424bc410efd6"
AUTH_OAUTH = "f9f921a659be8aaa"
AUTH_SSO = "01e5c6fa83305bc8"
INFRA_RATELIMIT = "0cc63c7e0822d695"
INFRA_DATADOG = "eef3c42aca4bd10d"
INFRA_AUTOSCALE = "c03dea8ef4182cc6"
INFRA_CART = "961d45b3a241e921"
INFRA_REDIS = "df9bdfd4eac84e23"
DB_MATVIEW = "e72027dc1661d649"
DB_MIGRATION = "78e9fee166a3640d"
DB_INDEX = "72cbb27dba5288af"
DB_REPLICATION = "3cdebbffc0b53404"
API_VERSIONING = "3cae47bea69627cf"
API_ERRORS = "3f4050699ddb7055"
API_PAGINATION = "f9f6fa734426d4c1"
API_ENVOY = "38db5b3c761350e1"
PROC_SPRINT = "8715f400cd17137e"
PROC_ONCALL = "a7bfacf4ce1a4232"
PROC_CODEREVIEW = "6a1aacd3f96e7cf3"
PROC_INCIDENTS = "2c7c5fc8a3fdb4b7"


EXACT_RECALL = [
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

PARAPHRASE_RECALL = [
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

TANGENTIAL_RECALL = [
    TestQuery("security improvements we made", "tangential",
              [AUTH_JWT_24H, AUTH_MFA, AUTH_SSO, AUTH_OAUTH, AUTH_SESSION]),
    TestQuery("how do we prevent API abuse", "tangential",
              [INFRA_RATELIMIT, API_ENVOY]),
    TestQuery("what changed in our database recently", "tangential",
              [DB_MATVIEW, DB_MIGRATION, DB_INDEX, DB_REPLICATION]),
    TestQuery("production incidents and bugs", "tangential",
              [AUTH_LOGIN_BUG]),
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

ALL_QUERIES = EXACT_RECALL + PARAPHRASE_RECALL + TANGENTIAL_RECALL


def make_searcher():
    """Create a read-only Searcher against the existing index."""
    data_dir = os.path.expanduser("~/.ark")
    schema = build_schema()
    index = tantivy.Index(schema, path=data_dir, reuse=True)

    # Embedding provider
    try:
        from ark.engine.embed import FastEmbedProvider
        embedding = FastEmbedProvider()
    except ImportError:
        model = os.environ.get("EMBEDDING_MODEL", "")
        dims = int(os.environ.get("EMBEDDING_DIMS", "1024") or "1024")
        from ark.engine.embed import CatsuEmbedding
        embedding = CatsuEmbedding(model=model, dims=dims)

    from ark.engine.circuit_breaker import CircuitBreakerEmbedding
    cb_embedding = CircuitBreakerEmbedding(embedding)

    embed_cache = EmbeddingCache(os.path.join(data_dir, "embeddings.db"))

    return Searcher(
        schema=schema,
        index=index,
        embedding=cb_embedding,
        embed_cache=embed_cache,
    )


async def run_search(searcher: Searcher, query: str) -> list[dict]:
    params = SearchParams(
        num_to_return=5,
        num_to_score=15,
        min_rrf_score=0.005,
        max_hits_per_doc=1,
    )
    from ark.engine.result import Ok
    match await searcher.search(query, corpus="agent:ark-serve", params=params):
        case Ok(hits):
            return [{"id": h.doc_id, "content": h.body, "score": h.scores.rrf} for h in hits]
        case _:
            return []


def dcg(relevances, k):
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances[:k]))

def ndcg(relevances, k, n_relevant):
    ideal = [1] * min(n_relevant, k) + [0] * max(0, k - n_relevant)
    ideal_dcg = dcg(ideal, k)
    return dcg(relevances, k) / ideal_dcg if ideal_dcg else 0.0


async def evaluate(searcher, queries, verbose=False):
    precisions, hit_rates, rrs, ndcgs = [], [], [], []

    for tq in queries:
        results = await run_search(searcher, tq.query)
        top_k = results[:K]
        relevances = [1 if r["id"] in tq.expected_ids else 0 for r in top_k]
        n_hits = sum(relevances)

        precisions.append(n_hits / K)
        hit_rates.append(1.0 if n_hits > 0 else 0.0)

        rr = 0.0
        for i, rel in enumerate(relevances):
            if rel:
                rr = 1.0 / (i + 1)
                break
        rrs.append(rr)
        ndcgs.append(ndcg(relevances, K, len(tq.expected_ids)))

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
        "mrr": sum(rrs) / n,
        "ndcg@3": sum(ndcgs) / n,
        "n": n,
    }


async def main():
    verbose = "-v" in sys.argv or "--verbose" in sys.argv
    searcher = make_searcher()

    categories = [
        ("Exact recall", EXACT_RECALL),
        ("Paraphrase recall", PARAPHRASE_RECALL),
        ("Tangential recall", TANGENTIAL_RECALL),
    ]

    print("=" * 72)
    print(f"  ark recall (query-expand) — top-{K}, {len(ALL_QUERIES)} queries")
    print("=" * 72)

    all_p, all_h, all_m, all_n, total = [], [], [], [], 0
    rows = []

    for name, queries in categories:
        print(f"\n{name} ({len(queries)} queries):")
        m = await evaluate(searcher, queries, verbose=verbose)
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

    print("\n# JSON results:")
    print(json.dumps({name: {k: round(v, 4) for k, v in m.items()} for name, m in rows + [("overall", overall)]}, indent=2))


if __name__ == "__main__":
    asyncio.run(main())

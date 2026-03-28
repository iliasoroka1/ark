"""Brutal recall benchmark for ark memory search.

~1173 docs: 23 engineering + 150 general noise + 500 AG News + 500 tech news.
70 queries across 7 categories including new harder ones.

Categories:
  1. Exact (10) — verbatim phrases
  2. Paraphrase (10) — rephrased concepts
  3. Tangential (10) — indirect/abstract queries
  4. Adversarial (10) — software terms vs generic tech noise
  5. Negative (5) — should return nothing relevant
  6. Multi-hop (5) — require chaining facts
  7. Lexical traps (10) — NEW: queries with words that match noise MORE than our memories
  8. Compositional (10) — NEW: require combining info from 2+ memories
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
    is_negative: bool = False


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

# Short aliases
JWT = "e5cbc506b705bf20"
BUG = "5ebf4f751b53d0d7"
SESSION = "e17c981595bb266a"
MFA = "ff2f93a81a68fc8c"
OAUTH = "6520003b5e121630"
SSO = "9a0fa4d17f0fd207"
RATELIMIT = "eb9a2f5d7a388697"
DATADOG = "d13e7d62d65240ac"
AUTOSCALE = "6fea38e8d09556fb"
CART = "08a91aeffee0a5e1"
REDIS = "b690f07b60f0257b"
MATVIEW = "d5b1d205a0ff47c1"
MIGRATION = "79fa0b520862f106"
INDEX = "e2d12ca9459525ef"
REPLICATION = "c5409c65f311628a"
VERSIONING = "913d26b10564c09a"
ERRORS = "dee0d1d453d7be80"
PAGINATION = "2442efb11a5228cd"
ENVOY = "beacc4ddd0cb061e"
SPRINT = "d3fae1b223578479"
ONCALL = "14cddd8bb324a254"
CODEREVIEW = "c4658bd5ad47ab50"
INCIDENT = "592ec5175080e20f"

# ── Queries ──

EXACT = [
    TestQuery("JWT tokens with 24-hour expiry", "exact", [JWT]),
    TestQuery("token bucket algorithm 100 req/min", "exact", [RATELIMIT]),
    TestQuery("expired JWT tokens not caught in refresh flow", "exact", [BUG]),
    TestQuery("materialized view instead of 6-table join", "exact", [MATVIEW]),
    TestQuery("cursor-based using opaque base64 token", "exact", [PAGINATION]),
    TestQuery("sprint planning moved to Tuesdays", "exact", [SPRINT]),
    TestQuery("SSO integration with Okta", "exact", [SSO]),
    TestQuery("composite index on orders table", "exact", [INDEX]),
    TestQuery("Datadog monitors for payments-api", "exact", [DATADOG]),
    TestQuery("MFA enforcement rolled out", "exact", [MFA]),
]

PARAPHRASE = [
    TestQuery("authentication tokens expire after a day", "paraphrase", [JWT]),
    TestQuery("API throttling with bucket-based algorithm", "paraphrase", [RATELIMIT]),
    TestQuery("bug where stale auth tokens were not refreshed", "paraphrase", [BUG]),
    TestQuery("optimized slow product query with a precomputed view", "paraphrase", [MATVIEW]),
    TestQuery("paginating API responses with an opaque cursor", "paraphrase", [PAGINATION]),
    TestQuery("when does the team do sprint planning", "paraphrase", [SPRINT]),
    TestQuery("single sign-on setup for company tools", "paraphrase", [SSO]),
    TestQuery("database index for looking up orders by customer", "paraphrase", [INDEX]),
    TestQuery("monitoring and alerting for the payments service", "paraphrase", [DATADOG]),
    TestQuery("mandatory two-factor authentication rollout", "paraphrase", [MFA]),
]

TANGENTIAL = [
    TestQuery("security improvements we made", "tangential", [JWT, MFA, SSO, OAUTH, SESSION]),
    TestQuery("how do we prevent API abuse", "tangential", [RATELIMIT, ENVOY]),
    TestQuery("what changed in our database recently", "tangential", [MATVIEW, MIGRATION, INDEX, REPLICATION]),
    TestQuery("production incidents and bugs", "tangential", [BUG, INCIDENT]),
    TestQuery("how does our caching work", "tangential", [REDIS]),
    TestQuery("deployment and scaling setup", "tangential", [CART, AUTOSCALE]),
    TestQuery("team workflow and ceremonies", "tangential", [SPRINT, ONCALL, CODEREVIEW]),
    TestQuery("how are errors handled in our APIs", "tangential", [ERRORS]),
    TestQuery("what authentication providers do we use", "tangential", [OAUTH, SSO]),
    TestQuery("database performance and optimization", "tangential", [MATVIEW, INDEX, REPLICATION]),
]

ADVERSARIAL = [
    TestQuery("our Redis caching setup", "adversarial", [REDIS]),
    TestQuery("how we use OAuth", "adversarial", [OAUTH]),
    TestQuery("database indexing strategy", "adversarial", [INDEX]),
    TestQuery("our monitoring and alerting", "adversarial", [DATADOG]),
    TestQuery("container deployment", "adversarial", [CART]),
    TestQuery("circuit breaker in our services", "adversarial", [INCIDENT]),
    TestQuery("how we handle load balancing", "adversarial", [ENVOY, RATELIMIT]),
    TestQuery("our message queue setup", "adversarial", []),
    TestQuery("our PostgreSQL configuration", "adversarial", [REPLICATION, SESSION]),
    TestQuery("API design patterns we follow", "adversarial", [VERSIONING, ERRORS, PAGINATION]),
]

NEGATIVE = [
    TestQuery("recipe for sourdough bread", "negative", [], is_negative=True),
    TestQuery("World Cup soccer results", "negative", [], is_negative=True),
    TestQuery("stock market performance in 2023", "negative", [], is_negative=True),
    TestQuery("how do volcanoes form", "negative", [], is_negative=True),
    TestQuery("Beethoven symphony compositions", "negative", [], is_negative=True),
]

MULTIHOP = [
    TestQuery("why did the payments service go down", "multihop", [INCIDENT, REPLICATION]),
    TestQuery("what security layers protect our APIs", "multihop", [JWT, RATELIMIT, ENVOY, MFA]),
    TestQuery("how did we improve database query speed", "multihop", [MATVIEW, INDEX]),
    TestQuery("what happens when a user session expires", "multihop", [SESSION, BUG, JWT]),
    TestQuery("our complete deployment and monitoring pipeline", "multihop", [CART, AUTOSCALE, DATADOG]),
]

# NEW: Lexical traps — queries where common words match noise harder than our memories
LEXICAL_TRAPS = [
    TestQuery("server configuration and scaling", "lexical_trap",
              [AUTOSCALE, CART, REPLICATION],
              "server/scaling appear in 100s of tech news articles"),
    TestQuery("software update and deployment process", "lexical_trap",
              [CART, CODEREVIEW],
              "'software update' matches tons of tech news"),
    TestQuery("network security and authentication", "lexical_trap",
              [JWT, MFA, SSO, OAUTH],
              "'network security' matches cybersecurity news"),
    TestQuery("data migration and storage", "lexical_trap",
              [MIGRATION, REPLICATION, REDIS],
              "'data migration' matches enterprise tech news"),
    TestQuery("performance monitoring dashboard", "lexical_trap",
              [DATADOG],
              "'performance monitoring' matches generic monitoring articles"),
    TestQuery("user access control policy", "lexical_trap",
              [MFA, CODEREVIEW, JWT],
              "'access control' matches security news"),
    TestQuery("service outage and recovery", "lexical_trap",
              [INCIDENT],
              "'outage' matches telecom/cloud outage news"),
    TestQuery("API gateway and proxy configuration", "lexical_trap",
              [ENVOY, RATELIMIT],
              "'API gateway' matches cloud service news"),
    TestQuery("database replication and failover", "lexical_trap",
              [REPLICATION],
              "'database replication' matches enterprise DB news"),
    TestQuery("automated testing and code quality", "lexical_trap",
              [CODEREVIEW],
              "'testing' and 'code quality' match dev tool news"),
]

# NEW: Compositional — require finding 2+ specific memories and recognizing they're related
COMPOSITIONAL = [
    TestQuery("auth system timeline: what changed and when", "compositional",
              [JWT, SESSION, MFA, OAUTH, SSO],
              "should find multiple auth memories spanning time"),
    TestQuery("everything about our payments infrastructure", "compositional",
              [DATADOG, INCIDENT, ENVOY],
              "payments-api monitoring + outage + proxy"),
    TestQuery("database schema changes we've made", "compositional",
              [MIGRATION, INDEX, MATVIEW],
              "migration + index + materialized view"),
    TestQuery("how we protect against overload", "compositional",
              [RATELIMIT, ENVOY, AUTOSCALE, REDIS],
              "rate limiting + proxy + scaling + cache"),
    TestQuery("our team's development practices", "compositional",
              [SPRINT, CODEREVIEW, ONCALL],
              "planning + review + on-call"),
    TestQuery("the full auth stack from login to session", "compositional",
              [JWT, SESSION, OAUTH, SSO, BUG],
              "JWT + sessions + OAuth + SSO + login bug"),
    TestQuery("infrastructure reliability measures", "compositional",
              [REPLICATION, AUTOSCALE, DATADOG, INCIDENT],
              "replication + scaling + monitoring + incident"),
    TestQuery("all API standards and conventions", "compositional",
              [VERSIONING, ERRORS, PAGINATION, ENVOY],
              "versioning + errors + pagination + proxy"),
    TestQuery("caching and query optimization strategy", "compositional",
              [REDIS, MATVIEW, INDEX],
              "Redis + materialized view + composite index"),
    TestQuery("incident response: what happened and how we prevent it", "compositional",
              [INCIDENT, REPLICATION, DATADOG, ONCALL],
              "postmortem + replication + monitoring + on-call"),
]

ALL_QUERIES = EXACT + PARAPHRASE + TANGENTIAL + ADVERSARIAL + NEGATIVE + MULTIHOP + LEXICAL_TRAPS + COMPOSITIONAL


def run_search(query: str) -> list[dict]:
    result = subprocess.run(
        ["uv", "run", "ark", "search", query],
        capture_output=True, text=True, cwd="/tmp/ark-charlie"
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


def is_engineering(item: dict) -> bool:
    content = (item.get("l0", "") + " " + item.get("content", "")).lower()
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
            if n_hits == 0:
                got = [r.get("id", "?")[:8] for r in top_k]
                exp = [e[:8] for e in tq.expected_ids[:4]]
                print(f"         expected: {exp}  got: {got}")
                for r in top_k[:2]:
                    c = r.get("l0", r.get("content", ""))[:70]
                    print(f"         -> {c}")
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
    cats = [
        ("Exact", EXACT), ("Paraphrase", PARAPHRASE), ("Tangential", TANGENTIAL),
        ("Adversarial", ADVERSARIAL), ("Negative", NEGATIVE), ("Multi-hop", MULTIHOP),
        ("Lexical traps", LEXICAL_TRAPS), ("Compositional", COMPOSITIONAL),
    ]
    total = len(ALL_QUERIES)
    print("=" * 80)
    print(f"  ark BRUTAL benchmark — top-{K}, {total} queries, ~1173 docs")
    print(f"  23 eng + 150 general + 500 AG News + 500 tech news")
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

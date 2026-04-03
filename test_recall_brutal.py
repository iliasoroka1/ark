"""Brutal recall benchmark for ark memory search.

~1233 docs: 83 engineering/business + 150 general noise + 500 AG News + 500 tech news.
  - 23 core engineering (auth, infra, DB, API, process)
  - 20 infra/ops (k8s, terraform, CI/CD, vault, istio, observability)
  - 20 data/backend (kafka, airflow, dbt, ML, snowflake, celery, data quality)
  - 20 business/ops (product, customer incidents, meetings, metrics, hiring, vendor, compliance)
200 queries across 15 categories + business.

Categories:
  1. Exact (10) — verbatim phrases
  2. Paraphrase (10) — rephrased concepts
  3. Tangential (10) — indirect/abstract queries
  4. Adversarial (10) — software terms vs generic tech noise
  5. Negative (5) — should return nothing relevant
  6. Multi-hop (5) — require chaining facts
  7. Lexical traps (10) — queries with words that match noise MORE than our memories
  8. Compositional (10) — require combining info from 2+ memories
  9. Temporal (10) — time-specific queries requiring date awareness
  10. Conversational (10) — messy, vague, human-like queries
  11. Synonym hell (10) — zero lexical overlap with source docs
  12. Precision (10) — must return ONE specific memory, not similar ones
  13. Negation (5) — exclude a specific topic
  14. Cross-domain (10) — connecting facts across different engineering domains
  15. Needle (5) — unique detail buried in a longer memory
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
    # infra/ops docs
    "e20fb0d54dd00b68": "kubernetes hpa for api-gateway min 2 max 10 cpu 70%",
    "da0ef2072d863866": "resource limits cpu 250m 1000m memory 256mi 1gi oomkilled",
    "37ccad1e28c5bfcd": "rbac eng-read-all clusterrole okta serviceaccount tokens",
    "370db464ee7fb506": "terraform state s3 tf-state-prod-us-east-1 dynamodb tf-locks",
    "0fc646377b06340a": "terraform modules vpc-module rds-module eks-module registry.internal",
    "ee9086aa47e88593": "terraform workspaces prod staging dev-engineer atlantis approval",
    "d5e2dd3a32e26bd3": "github actions ci lint unit integration docker ecr ubuntu-22.04",
    "514580a4390d606c": "github actions build cache docker layer ecr npm pip lock files",
    "949e9f75d0cc55e7": "deploy gates argo cd staging canary 10% 30 min manual approval",
    "4a6cd8de1781b393": "hashicorp vault v1.15.2 approle oidc dynamic secrets postgresql aws iam",
    "0403784b9c613cba": "sealed secrets controller v0.24.1 public key rotated 2026-01-15",
    "9e5090f3f46ecf06": "secrets rotation 90 days passwords 30 days api keys 7 days tokens",
    "68645b0bf4713c5e": "istio v1.20.3 mtls strict peerauthentication cert-manager tls",
    "723f12a447fa24d9": "istio circuit breaker payments-service 5xx ejection 30s retry 3",
    "65999b4a7d2a41c1": "opentelemetry collector v0.96.0 jaeger 5% prometheus loki operator",
    "0cd5afa2f85b40da": "grafana v10.2.3 slo-overview 47 alert rules provisioning",
    "7784a35655c2f9b3": "loki v2.9.4 promtail 30 days 2tb opa admission webhook log-format",
    "025e3843df69aa56": "kubernetes namespace prod staging dev network policies resourcequota 64 cpu 256gi",
    "f8f78d25f8ede6e1": "argo cd v2.9.5 automated staging manual prod applicationset crds",
    "8bcb0c5ad1115947": "node affinity gpu ml-inference spot taint noSchedule spot interruption",
    # data/backend docs
    "50f1e8613d70ebb9": "user-events kafka topic 12 partitions ingestion-team",
    "6a93ac4883bdcca2": "analytics-consumer order-completed kafka lag 10k",
    "74f7f6e517252c4a": "etl_user_activity airflow dag 02:00 utc depends_on_past",
    "cb2839d2f8135169": "dbt_analytics fct_orders incremental order_id unique key",
    "b7330951c5efcd66": "feast user_purchase_features redis online bigquery offline",
    "be461927d65e33df": "rec_model_v2 20% traffic launchdarkly shadow rec_shadow_log",
    "9e47d10b624e54c1": "mlflow rec-model/production run id a4f82c91",
    "df4c8084fd7510dd": "bigquery analytics.events partitioned event_date analytics-team-slot",
    "3a22c2a67d5dfc32": "snowflake analytics_role loader_role transformer_role acme_corp",
    "c1232523c63aabf9": "snowflake analytics_wh x-small auto_suspend 60 data-team budget",
    "504967e93b26b36d": "celery 3 queues default high bulk worker-01 redis-celery",
    "c1735b58a4eb924a": "email-notifications dlq email-dlq 3 retries exponential backoff",
    "8c64e6eec31b6724": "process_payment_webhook max_retries=5 payments-dlq sqs",
    "e988899f20df5a7f": "great expectations orders_suite 12 expectations airflow sensor",
    "cc1296bb1c9fba07": "dbt freshness fct_orders warn 6h error 24h loaded_at",
    "d88ec0b47cef4011": "monte carlo raw_events 500k-2m row count jira data-ops",
    "e912ea6bbf7e7992": "customer_dimension fivetran salesforce-prod nightly 01:30",
    "e4092b70c7250af5": "revenue_reconciliation backfill 2026-01-01 2026-02-28 59 dag runs",
    "2a5e787a369bca1a": "clickstream kafka 24 partitions 3 days retention analytics-platform",
    "788542a9fa58bd2f": "fraud-model-v3 shadow mode fraud_shadow_log auc 0.94",
    # business/ops docs
    "d48e245e50dcef49": "sunset legacy checkout flow single-page checkout converts 12%",
    "c88dc3396b0254f8": "roadmap mobile push notifications q2 2026 60% mobile users",
    "fb65619230c2cdf6": "recommendation carousel feature flag 20% users 3% uplift aov",
    "046c26ef72094936": "deprecate api v1 external partners 6-month migration 2026-06-01",
    "9611b9ee15e10bb2": "acme corp checkout timing out european users 200 orders",
    "07656c9b1cf570ed": "login failing intermittently tuesday release customer success p1",
    "7f4181b3a165605c": "bigretail bulk order import failing silently 2026-03-05 inventory",
    "6d47a494e4b79e78": "nps dropped 42 to 31 february page load times product listings",
    "9210eec4581c224b": "sprint retro deployment slow ci pipeline 8 minutes build caching",
    "7df08e8cad10bf42": "architecture review payment processing own database isolate failures q2",
    "757a5f58da5e7ab4": "security audit service-to-service not encrypted 3 services march",
    "b70fe95367719bcd": "qbr ceo payments outage march 12 monitoring coverage",
    "ad0e30b2f2b52a53": "conversion rate dropped 8% 2026-03-13 payments incident $45k revenue",
    "96a8bb70b3be9243": "latency sla breach march payment p95 500ms bigretail contractual",
    "9c54ee59e47380f1": "mau grew 15% february referral program server costs 22% recommendation",
    "b74b468527867ab4": "hired maria chen sre observability mean time detection incidents",
    "a301035ca4a1e7b5": "data platform team expanded 3 to 5 january pipeline data quality",
    "38a8fd548af893fe": "chose datadog over new relic kubernetes integration per-host pricing",
    "508aca21f29c0584": "gdpr audit 2026-05-01 data retention anonymize customer data 2 years",
    "e307d34c1f7e6520": "soc2 type ii 5 control gaps access reviews audit trail schema changes",
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

# infra/ops aliases
K8S_HPA         = "e20fb0d54dd00b68"
RESOURCE_LIMITS = "da0ef2072d863866"
K8S_RBAC        = "37ccad1e28c5bfcd"
TF_STATE        = "370db464ee7fb506"
TF_MODULES      = "0fc646377b06340a"
TF_WORKSPACES   = "ee9086aa47e88593"
GH_ACTIONS_CI   = "d5e2dd3a32e26bd3"
BUILD_CACHE     = "514580a4390d606c"
DEPLOY_GATES    = "949e9f75d0cc55e7"
VAULT_SECRETS   = "4a6cd8de1781b393"
SEALED_SECRETS  = "0403784b9c613cba"
SECRET_ROTATION = "9e5090f3f46ecf06"
ISTIO_MTLS      = "68645b0bf4713c5e"
CIRCUIT_BREAKER = "723f12a447fa24d9"
OTEL_COLLECTOR  = "65999b4a7d2a41c1"
GRAFANA_DASH    = "0cd5afa2f85b40da"
LOG_AGG         = "7784a35655c2f9b3"
NS_STRATEGY     = "025e3843df69aa56"
ARGOCD          = "f8f78d25f8ede6e1"
NODE_AFFINITY   = "8bcb0c5ad1115947"

# data/backend aliases
KAFKA_USER_EVENTS = "50f1e8613d70ebb9"
KAFKA_ORDER_LAG   = "6a93ac4883bdcca2"
AIRFLOW_ETL       = "74f7f6e517252c4a"
DBT_MODELS        = "cb2839d2f8135169"
FEAST_FEATURES    = "b7330951c5efcd66"
AB_SHADOW         = "be461927d65e33df"
MLFLOW_MODEL      = "9e47d10b624e54c1"
BIGQUERY_PART     = "df4c8084fd7510dd"
SNOWFLAKE_ROLES   = "3a22c2a67d5dfc32"
SNOWFLAKE_WH      = "c1232523c63aabf9"
CELERY_QUEUES     = "504967e93b26b36d"
CELERY_DLQ        = "c1735b58a4eb924a"
CELERY_RETRY      = "8c64e6eec31b6724"
GX_SUITE          = "e988899f20df5a7f"
DBT_FRESHNESS     = "cc1296bb1c9fba07"
MONTE_CARLO       = "d88ec0b47cef4011"
FIVETRAN_CUSTOMER = "e912ea6bbf7e7992"
AIRFLOW_BACKFILL  = "e4092b70c7250af5"
KAFKA_CLICKSTREAM = "2a5e787a369bca1a"
FRAUD_SHADOW      = "788542a9fa58bd2f"

# business/ops aliases
BIZ_CHECKOUT_SUNSET = "d48e245e50dcef49"
BIZ_MOBILE_ROADMAP  = "c88dc3396b0254f8"
BIZ_REC_CAROUSEL    = "fb65619230c2cdf6"
BIZ_APIV1_DEPREC    = "046c26ef72094936"
BIZ_ACME_TIMEOUT    = "9611b9ee15e10bb2"
BIZ_LOGIN_FAILING   = "07656c9b1cf570ed"
BIZ_BIGRETAIL_IMPORT = "7f4181b3a165605c"
BIZ_NPS_DROP        = "6d47a494e4b79e78"
BIZ_RETRO           = "9210eec4581c224b"
BIZ_ARCH_REVIEW     = "7df08e8cad10bf42"
BIZ_SECURITY_AUDIT  = "757a5f58da5e7ab4"
BIZ_QBR             = "b70fe95367719bcd"
BIZ_CONVERSION_DROP = "ad0e30b2f2b52a53"
BIZ_SLA_BREACH      = "96a8bb70b3be9243"
BIZ_MAU_GROWTH      = "9c54ee59e47380f1"
BIZ_HIRE_SRE        = "b74b468527867ab4"
BIZ_DATA_TEAM       = "a301035ca4a1e7b5"
BIZ_VENDOR_DATADOG  = "38a8fd548af893fe"
BIZ_GDPR            = "508aca21f29c0584"
BIZ_SOC2            = "e307d34c1f7e6520"

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
    # infra/ops exact
    TestQuery("HPA configured for api-gateway: min 2 replicas, max 10, target CPU 70%", "exact", [K8S_HPA]),
    TestQuery("tf-state-prod-us-east-1 with DynamoDB table tf-locks", "exact", [TF_STATE]),
    TestQuery("Vault agent sidecar injects secrets at pod startup", "exact", [VAULT_SECRETS]),
    TestQuery("mTLS STRICT mode", "exact", [ISTIO_MTLS]),
    TestQuery("5 consecutive 5xx errors triggers 30s host ejection", "exact", [CIRCUIT_BREAKER]),
    TestQuery("OpenTelemetry Collector v0.96.0 as DaemonSet", "exact", [OTEL_COLLECTOR]),
    TestQuery("grafana.internal/d/slo-overview", "exact", [GRAFANA_DASH]),
    TestQuery("Loki v2.9.4 + Promtail DaemonSet", "exact", [LOG_AGG]),
    TestQuery("Argo CD v2.9.5 GitOps deployments", "exact", [ARGOCD]),
    TestQuery("Sealed Secrets controller v0.24.1", "exact", [SEALED_SECRETS]),
    # data/backend exact
    TestQuery("user-events Kafka topic 12 partitions ingestion-team lag 50K", "exact", [KAFKA_USER_EVENTS]),
    TestQuery("etl_user_activity Airflow DAG depends_on_past max_active_runs=1", "exact", [AIRFLOW_ETL]),
    TestQuery("fct_orders incremental model order_id unique key full refresh", "exact", [DBT_MODELS]),
    TestQuery("MLflow rec-model/production run ID a4f82c91", "exact", [MLFLOW_MODEL]),
    TestQuery("ANALYTICS_WH X-Small AUTO_SUSPEND=60 data-team@company.com budget", "exact", [SNOWFLAKE_WH]),
    TestQuery("email-dlq 3 retries exponential backoff queue depth 100", "exact", [CELERY_DLQ]),
    TestQuery("process_payment_webhook max_retries=5 payments-dlq SQS 90s SLA", "exact", [CELERY_RETRY]),
    TestQuery("dbt freshness fct_orders warn 6h error 24h loaded_at", "exact", [DBT_FRESHNESS]),
    TestQuery("clickstream Kafka 24 partitions retention.ms=259200000 acks=1", "exact", [KAFKA_CLICKSTREAM]),
    TestQuery("fraud-model-v3 AUC 0.94 rolling 7-day holdout ml-review-board", "exact", [FRAUD_SHADOW]),
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
    # infra/ops paraphrase
    TestQuery("horizontal pod autoscaler settings for the API gateway", "paraphrase", [K8S_HPA]),
    TestQuery("where is our Terraform remote state stored", "paraphrase", [TF_STATE]),
    TestQuery("how secrets get injected into pods at runtime", "paraphrase", [VAULT_SECRETS]),
    TestQuery("encrypted traffic between services in the mesh", "paraphrase", [ISTIO_MTLS]),
    TestQuery("automatic service ejection after repeated failures", "paraphrase", [CIRCUIT_BREAKER]),
    TestQuery("distributed tracing pipeline and where traces go", "paraphrase", [OTEL_COLLECTOR]),
    TestQuery("central observability dashboard for service SLOs", "paraphrase", [GRAFANA_DASH]),
    TestQuery("how are application logs collected and retained", "paraphrase", [LOG_AGG]),
    TestQuery("GitOps-based production deployment approval process", "paraphrase", [ARGOCD]),
    TestQuery("how often do we rotate database credentials", "paraphrase", [SECRET_ROTATION]),
    # data/backend paraphrase
    TestQuery("how long does Kafka keep messages on the user events topic", "paraphrase", [KAFKA_USER_EVENTS]),
    TestQuery("scheduled data pipeline for user activity runs at night", "paraphrase", [AIRFLOW_ETL]),
    TestQuery("how to rebuild dbt orders model after upstream schema change", "paraphrase", [DBT_MODELS]),
    TestQuery("where is the current production recommendation model stored", "paraphrase", [MLFLOW_MODEL]),
    TestQuery("what triggers a cost alert on our data warehouse", "paraphrase", [SNOWFLAKE_WH]),
    TestQuery("how many times does a failed notification job retry", "paraphrase", [CELERY_DLQ]),
    TestQuery("what happens when the payment webhook task exhausts all retries", "paraphrase", [CELERY_RETRY]),
    TestQuery("when does dbt raise a data freshness error on orders", "paraphrase", [DBT_FRESHNESS]),
    TestQuery("what AUC score must the fraud model reach before going live", "paraphrase", [FRAUD_SHADOW]),
    TestQuery("how is the clickstream Kafka topic configured for producers", "paraphrase", [KAFKA_CLICKSTREAM]),
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
    # new
    TestQuery("customer-facing product changes and roadmap items", "tangential", [BIZ_CHECKOUT_SUNSET, BIZ_MOBILE_ROADMAP, BIZ_REC_CAROUSEL]),
    TestQuery("business impact incidents and revenue losses", "tangential", [BIZ_CONVERSION_DROP, BIZ_SLA_BREACH, BIZ_LOGIN_FAILING]),
    TestQuery("infrastructure and health service level agreements", "tangential", [K8S_HPA, RESOURCE_LIMITS]),
    TestQuery("operational metrics and performance baselines", "tangential", [OTEL_COLLECTOR, GRAFANA_DASH]),
    TestQuery("security compliance and audit requirements", "tangential", [BIZ_GDPR, BIZ_SOC2]),
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
    # infra/ops adversarial — confusable across new docs
    TestQuery("Kubernetes pod scheduling and resource allocation", "adversarial", [K8S_HPA, RESOURCE_LIMITS, NODE_AFFINITY]),
    TestQuery("secrets and credentials management in production", "adversarial", [VAULT_SECRETS, SECRET_ROTATION, SEALED_SECRETS]),
    TestQuery("service mesh traffic policies and failure handling", "adversarial", [ISTIO_MTLS, CIRCUIT_BREAKER]),
    TestQuery("CI/CD pipeline and build speed optimizations", "adversarial", [GH_ACTIONS_CI, BUILD_CACHE, DEPLOY_GATES]),
    TestQuery("observability stack: logs, metrics, and traces", "adversarial", [LOG_AGG, OTEL_COLLECTOR, GRAFANA_DASH]),
    # data/backend adversarial — confusable across new docs
    TestQuery("Kafka consumer lag alerting", "adversarial", [KAFKA_USER_EVENTS, KAFKA_ORDER_LAG],),
    TestQuery("our data quality monitoring", "adversarial", [GX_SUITE, DBT_FRESHNESS, MONTE_CARLO]),
    TestQuery("model promotion criteria", "adversarial", [AB_SHADOW, MLFLOW_MODEL, FRAUD_SHADOW]),
    TestQuery("Airflow DAG scheduling and backfill", "adversarial", [AIRFLOW_ETL, AIRFLOW_BACKFILL]),
    TestQuery("Celery task failure handling", "adversarial", [CELERY_DLQ, CELERY_RETRY]),
]

NEGATIVE = [
    TestQuery("recipe for sourdough bread", "negative", [], is_negative=True),
    TestQuery("World Cup soccer results", "negative", [], is_negative=True),
    TestQuery("stock market performance in 2023", "negative", [], is_negative=True),
    TestQuery("how do volcanoes form", "negative", [], is_negative=True),
    TestQuery("Beethoven symphony compositions", "negative", [], is_negative=True),
    # new
    TestQuery("best practices for maintaining a coal mining operation", "negative", [], is_negative=True),
    TestQuery("how to bake a chocolate chip cookie", "negative", [], is_negative=True),
    TestQuery("baseball statistics for the 2025 World Series", "negative", [], is_negative=True),
    TestQuery("how to care for a pet hamster", "negative", [], is_negative=True),
    TestQuery("climate change policy in Scandinavian countries", "negative", [], is_negative=True),
    TestQuery("Renaissance art movements and the Sistine Chapel", "negative", [], is_negative=True),
    TestQuery("quantum entanglement in theoretical physics", "negative", [], is_negative=True),
    TestQuery("how to write a haiku poem", "negative", [], is_negative=True),
    TestQuery("organic gardening tips for tomato plants", "negative", [], is_negative=True),
    TestQuery("history of jazz music in New Orleans", "negative", [], is_negative=True),
]

MULTIHOP = [
    TestQuery("why did the payments service go down", "multihop", [INCIDENT, REPLICATION]),
    TestQuery("what security layers protect our APIs", "multihop", [JWT, RATELIMIT, ENVOY, MFA]),
    TestQuery("how did we improve database query speed", "multihop", [MATVIEW, INDEX]),
    TestQuery("what happens when a user session expires", "multihop", [SESSION, BUG, JWT]),
    TestQuery("our complete deployment and monitoring pipeline", "multihop", [CART, AUTOSCALE, DATADOG]),
    # new
    TestQuery("what caused the conversion rate drop after the payments outage", "multihop", [BIZ_CONVERSION_DROP, INCIDENT]),
    TestQuery("how does the RBAC setup connect to the SOC2 control gaps on access reviews", "multihop", [K8S_RBAC, BIZ_SOC2]),
    TestQuery("could the Redis cache TTL explain the NPS complaints about page load times", "multihop", [REDIS, BIZ_NPS_DROP]),
    TestQuery("what monitoring would alert us if the dbt freshness check fails and data quality degrades", "multihop", [DBT_FRESHNESS, MONTE_CARLO]),
    TestQuery("how did the new SRE hire relate to improving detection time the CEO asked about", "multihop", [BIZ_HIRE_SRE, BIZ_QBR]),
    TestQuery("what Terraform approval process protects the Vault secrets infrastructure", "multihop", [TF_WORKSPACES, VAULT_SECRETS]),
    TestQuery("if the Kafka order lag alert fires how does that affect the payment webhook retries", "multihop", [KAFKA_ORDER_LAG, CELERY_RETRY]),
    TestQuery("how does the circuit breaker on payments relate to the Acme Corp checkout timeout issue", "multihop", [CIRCUIT_BREAKER, BIZ_ACME_TIMEOUT]),
    TestQuery("did the materialized view optimization help with the BigRetail import performance problem", "multihop", [MATVIEW, BIZ_BIGRETAIL_IMPORT]),
    TestQuery("what's the relationship between the migration audit log and the GDPR data retention requirement", "multihop", [MIGRATION, BIZ_GDPR]),
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
    # new
    TestQuery("customer growth and business expansion", "lexical_trap",
              [BIZ_MAU_GROWTH],
              "'growth' and 'expansion' match startup news everywhere"),
    TestQuery("hiring and team scaling", "lexical_trap",
              [BIZ_HIRE_SRE, BIZ_DATA_TEAM],
              "'hiring' matches every tech company press release"),
    TestQuery("security and compliance review", "lexical_trap",
              [BIZ_SECURITY_AUDIT, BIZ_SOC2, BIZ_GDPR],
              "'security audit' matches enterprise security news"),
    TestQuery("vendor selection and infrastructure choice", "lexical_trap",
              [BIZ_VENDOR_DATADOG],
              "'vendor selection' matches enterprise IT news"),
    TestQuery("container orchestration and cluster management", "lexical_trap",
              [K8S_HPA, K8S_RBAC, NS_STRATEGY],
              "'cluster management' matches generic k8s news"),
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
    # new
    TestQuery("our kubernetes cluster setup and access control", "compositional",
              [K8S_HPA, K8S_RBAC, NS_STRATEGY],
              "HPA + RBAC + namespaces"),
    TestQuery("complete observability stack for all services", "compositional",
              [OTEL_COLLECTOR, GRAFANA_DASH, LOG_AGG],
              "telemetry collector + grafana dashboard + log aggregation"),
    TestQuery("secrets management and rotation across the platform", "compositional",
              [VAULT_SECRETS, SEALED_SECRETS, SECRET_ROTATION],
              "Vault + Sealed Secrets + rotation schedule"),
    TestQuery("terraform infrastructure as code with state management", "compositional",
              [TF_STATE, TF_MODULES, TF_WORKSPACES],
              "S3 state + module versions + workspace setup"),
    TestQuery("end-to-end CI/CD pipeline with caching and gates", "compositional",
              [GH_ACTIONS_CI, BUILD_CACHE, DEPLOY_GATES],
              "GitHub Actions + Docker cache + Argo CD gates"),
]

# 9. Temporal — time-specific queries requiring date awareness
TEMPORAL = [
    TestQuery("what happened in January 2026", "temporal",
              [MFA],
              "MFA rolled out 2026-01-10"),
    TestQuery("changes made in February 2026", "temporal",
              [MIGRATION, REPLICATION, INDEX],
              "migration ran 2026-02-18, replication tested 2026-02-25"),
    TestQuery("most recent production deployment", "temporal",
              [CART],
              "cart-service v2.3.1 deployed"),
    TestQuery("what was the last incident", "temporal",
              [INCIDENT],
              "2026-03-12 payments outage"),
    TestQuery("things that changed this quarter", "temporal",
              [MFA, MIGRATION, REPLICATION, INCIDENT, CART],
              "Q1 2026 events"),
    TestQuery("oldest configuration still in use", "temporal",
              [JWT, RATELIMIT],
              "JWT 24h expiry and rate limit 100req/min — no change dates"),
    TestQuery("when was the database schema last modified", "temporal",
              [MIGRATION],
              "migration 047 on 2026-02-18"),
    TestQuery("March 2026 events", "temporal",
              [INCIDENT],
              "2026-03-12 payments outage"),
    TestQuery("schedule changes this year", "temporal",
              [SPRINT],
              "sprint planning moved to Tuesdays"),
    TestQuery("what was set up before the outage", "temporal",
              [REPLICATION, MIGRATION, INDEX],
              "replication + migration + index all before 2026-03-12"),
    # Temporal queries for new docs — test hypergraph period node injection
    TestQuery("January 2026 security changes", "temporal",
              [MFA, SEALED_SECRETS],
              "MFA 2026-01-10 + Sealed Secrets key rotated 2026-01-15"),
    TestQuery("what happened in February 2026", "temporal",
              [MIGRATION, REPLICATION, SECRET_ROTATION],
              "migration 2026-02-18 + replication tested 2026-02-25 + rotation 2026-02-01"),
    TestQuery("Q1 2026 infrastructure changes", "temporal",
              [MFA, SEALED_SECRETS, SECRET_ROTATION, MIGRATION, REPLICATION, CODEREVIEW, INCIDENT],
              "all docs with dates in Jan-Mar 2026"),
    TestQuery("March 2026 policy and process updates", "temporal",
              [CODEREVIEW, INCIDENT],
              "code review policy 2026-03-01 + incident postmortem 2026-03-12"),
    TestQuery("data pipeline changes in early 2026", "temporal",
              [AIRFLOW_ETL, AIRFLOW_BACKFILL],
              "etl DAG has 2026-01-01 backfill range + revenue backfill 2026-01-01 to 2026-02-28"),
]

# 10. Conversational — messy, vague, human-like phrasing
CONVERSATIONAL = [
    TestQuery("that thing with the database that was really slow", "conversational",
              [MATVIEW],
              "6-table join taking 800ms"),
    TestQuery("the login bug we had", "conversational",
              [BUG],
              "expired JWT refresh flow bug"),
    TestQuery("didn't we set up some kind of caching recently", "conversational",
              [REDIS],
              "Redis cache for catalog-service"),
    TestQuery("what's our on-call schedule again", "conversational",
              [ONCALL],
              "weekly on-call rotation"),
    TestQuery("how many requests can someone make", "conversational",
              [RATELIMIT],
              "100 req/min per user"),
    TestQuery("something about tokens expiring too fast", "conversational",
              [JWT, BUG],
              "JWT 24h expiry + refresh bug"),
    TestQuery("who pages who when payments break", "conversational",
              [DATADOG, ONCALL],
              "PagerDuty alert to #payments-oncall"),
    TestQuery("what proxy thing are we using", "conversational",
              [ENVOY],
              "Envoy proxy sidecar"),
    TestQuery("the thing where we made the query faster", "conversational",
              [MATVIEW, INDEX],
              "materialized view + composite index"),
    TestQuery("do we have any audit logging", "conversational",
              [MIGRATION],
              "047_add_audit_log_table"),
    # new
    TestQuery("wait, didn't we just spin down that old checkout thing", "conversational", [BIZ_CHECKOUT_SUNSET]),
    TestQuery("there's something about people on phones not getting notified right", "conversational", [BIZ_MOBILE_ROADMAP]),
    TestQuery("wasn't there a customer complaining about timeouts in Europe", "conversational", [BIZ_ACME_TIMEOUT]),
    TestQuery("our NPS tanked like a month ago, something about loading", "conversational", [BIZ_NPS_DROP]),
    TestQuery("the big box store import just stopped working out of nowhere", "conversational", [BIZ_BIGRETAIL_IMPORT]),
]

# 11. Synonym hell — ZERO lexical overlap with source memories
SYNONYM_HELL = [
    TestQuery("credential validity duration", "synonym_hell",
              [JWT],
              "JWT tokens with 24-hour expiry — no shared words"),
    TestQuery("request throttling implementation", "synonym_hell",
              [RATELIMIT],
              "rate limiting token bucket — different vocabulary"),
    TestQuery("precomputed aggregate tables for catalog lookups", "synonym_hell",
              [MATVIEW],
              "materialized view — domain synonym"),
    TestQuery("identity federation for workforce applications", "synonym_hell",
              [SSO],
              "SSO with Okta — enterprise jargon"),
    TestQuery("observability stack for financial transactions", "synonym_hell",
              [DATADOG],
              "Datadog monitors for payments-api — different framing"),
    TestQuery("horizontal pod autoscaler configuration", "synonym_hell",
              [AUTOSCALE],
              "auto-scaling min 3 max 20 CPU 65% — k8s terminology"),
    TestQuery("write-ahead log shipping for disaster recovery", "synonym_hell",
              [REPLICATION],
              "streaming replication — PostgreSQL internals terminology"),
    TestQuery("structured fault report for service degradation", "synonym_hell",
              [INCIDENT],
              "incident postmortem — formal language"),
    TestQuery("opaque continuation tokens for paging", "synonym_hell",
              [PAGINATION],
              "cursor-based base64 pagination — different vocabulary"),
    TestQuery("ephemeral credential rotation defect", "synonym_hell",
              [BUG],
              "login bug expired JWT refresh — security jargon"),
    # new
    TestQuery("legacy transaction checkout module sunset", "synonym_hell", [BIZ_CHECKOUT_SUNSET]),
    TestQuery("mobile engagement activation for low-bandwidth devices", "synonym_hell", [BIZ_MOBILE_ROADMAP]),
    TestQuery("geographical latency mitigation for transatlantic commerce", "synonym_hell", [BIZ_ACME_TIMEOUT]),
    TestQuery("customer satisfaction metric deterioration investigation", "synonym_hell", [BIZ_NPS_DROP]),
    TestQuery("supply chain integration failure escalation", "synonym_hell", [BIZ_BIGRETAIL_IMPORT]),
]

# 12. Precision — must return exactly ONE specific memory, not related ones
PRECISION = [
    TestQuery("what algorithm does our rate limiter use", "precision",
              [RATELIMIT],
              "must find token bucket, not Envoy proxy or generic rate limiting"),
    TestQuery("what's the TTL on our product cache", "precision",
              [REDIS],
              "must find Redis TTL 5min, not generic caching mentions"),
    TestQuery("what columns does our audit log table have", "precision",
              [MIGRATION],
              "must find user_id, action, resource, timestamp"),
    TestQuery("what is our API pagination max page size", "precision",
              [PAGINATION],
              "must find max 100, default 25"),
    TestQuery("which cluster is cart-service deployed to", "precision",
              [CART],
              "must find east-1, 3 replicas"),
    TestQuery("what is the p99 latency threshold for payments alerts", "precision",
              [DATADOG],
              "must find 400ms"),
    TestQuery("what CPU target does our autoscaler use", "precision",
              [AUTOSCALE],
              "must find 65% CPU target"),
    TestQuery("which database is our primary on", "precision",
              [REPLICATION],
              "must find db-prod-01"),
    TestQuery("what replaced the 6-table join", "precision",
              [MATVIEW],
              "must find materialized view"),
    TestQuery("what PKCE flow did we migrate to", "precision",
              [OAUTH],
              "must find OAuth 2.0 with PKCE"),
    # hard precision: must distinguish between nearly identical new docs
    TestQuery("which Kafka topic has 6 partitions", "precision",
              [KAFKA_ORDER_LAG],
              "3 Kafka docs — only order-completed has 6 partitions (user-events=12, clickstream=24)"),
    TestQuery("which Kafka topic has 24 partitions", "precision",
              [KAFKA_CLICKSTREAM],
              "3 Kafka docs — only clickstream has 24 partitions"),
    TestQuery("which secrets rotate every 30 days", "precision",
              [SECRET_ROTATION],
              "3 secrets docs — rotation policy doc specifies 30d for API keys"),
    TestQuery("which Terraform resource needs 2-person approval before apply", "precision",
              [TF_WORKSPACES],
              "3 terraform docs — only workspaces doc mentions Atlantis 2-person approval"),
    TestQuery("which Celery queue has concurrency 16", "precision",
              [CELERY_QUEUES],
              "2 Celery docs — bulk queue concurrency=16 is in queues config, not DLQ or retry"),
    TestQuery("which Airflow DAG uses depends_on_past", "precision",
              [AIRFLOW_ETL],
              "2 Airflow docs — etl_user_activity has depends_on_past=True, backfill doc does not"),
    TestQuery("which dbt check warns after 6 hours", "precision",
              [DBT_FRESHNESS],
              "2 dbt docs — freshness check has warn=6h; models doc is about incremental strategy"),
    TestQuery("which Snowflake warehouse auto-suspends after 60 seconds", "precision",
              [SNOWFLAKE_WH],
              "2 Snowflake docs — WH config has AUTO_SUSPEND=60; roles doc has no suspend setting"),
    TestQuery("what is the minimum HPA replica count for api-gateway", "precision",
              [K8S_HPA],
              "must find min=2 — confusable with resource limits or autoscale docs"),
    TestQuery("which Vault secret type has a 15-minute TTL", "precision",
              [VAULT_SECRETS],
              "3 secrets docs — only Vault doc specifies AWS IAM TTL 15m"),
]

# 13. Negation — should return specific memories while EXCLUDING a topic
NEGATION = [
    TestQuery("auth changes besides OAuth and SSO", "negation",
              [JWT, SESSION, MFA, BUG],
              "should find JWT/session/MFA/bug, NOT OAuth or SSO"),
    TestQuery("infrastructure that isn't about databases", "negation",
              [CART, AUTOSCALE, REDIS, ENVOY, DATADOG],
              "should find non-DB infra memories"),
    TestQuery("API standards other than pagination", "negation",
              [VERSIONING, ERRORS, ENVOY],
              "should find versioning/errors/envoy, NOT pagination"),
    TestQuery("process changes that aren't sprint planning", "negation",
              [ONCALL, CODEREVIEW],
              "should find on-call/code review, NOT sprint"),
    TestQuery("database work besides the audit log migration", "negation",
              [MATVIEW, INDEX, REPLICATION],
              "should find matview/index/replication, NOT migration"),
    # new
    TestQuery("authentication methods besides OAuth and SSO", "negation", [JWT, SESSION, MFA]),
    TestQuery("caching solutions that aren't Redis", "negation", [MATVIEW, FEAST_FEATURES]),
    TestQuery("security tools other than Vault and Sealed Secrets", "negation", [ISTIO_MTLS, K8S_RBAC]),
    TestQuery("monitoring not using Datadog", "negation", [GRAFANA_DASH, OTEL_COLLECTOR]),
    TestQuery("Kafka topics not related to user activity or orders", "negation", [KAFKA_CLICKSTREAM]),
    TestQuery("deployment strategies excluding canary and manual approval", "negation", [DEPLOY_GATES, ARGOCD]),
    TestQuery("incident response without PagerDuty", "negation", [DATADOG, INCIDENT]),
    TestQuery("data pipeline tools excluding Airflow and dbt", "negation", [FIVETRAN_CUSTOMER]),
    TestQuery("API versioning without v1 sunset", "negation", [VERSIONING, BIZ_APIV1_DEPREC]),
    TestQuery("Kubernetes networking not using Istio", "negation", [NS_STRATEGY]),
]

# 14. Cross-domain — connecting facts across different engineering areas
CROSS_DOMAIN = [
    TestQuery("what could cause cascading failures in our system", "cross_domain",
              [REPLICATION, INCIDENT, AUTOSCALE, RATELIMIT],
              "DB failover + outage + scaling + rate limits"),
    TestQuery("how does a user request flow through our infrastructure", "cross_domain",
              [ENVOY, RATELIMIT, JWT, REDIS, CART],
              "proxy → rate limit → auth → cache → service"),
    TestQuery("what would we need to change for SOC2 compliance", "cross_domain",
              [MFA, MIGRATION, CODEREVIEW, ONCALL],
              "MFA + audit log + code review + incident response"),
    TestQuery("single points of failure in our architecture", "cross_domain",
              [REPLICATION, REDIS, AUTOSCALE],
              "DB primary + cache + single-pod risk"),
    TestQuery("what happens if we lose east-1", "cross_domain",
              [CART, REPLICATION, DATADOG],
              "cart-service on east-1 + DB primary + monitoring"),
    TestQuery("onboarding a new developer: what do they need to know", "cross_domain",
              [VERSIONING, ERRORS, PAGINATION, CODEREVIEW, SPRINT],
              "API standards + process"),
    TestQuery("debugging a slow API response end to end", "cross_domain",
              [ENVOY, REDIS, MATVIEW, INDEX, DATADOG],
              "proxy → cache → query → index → monitoring"),
    TestQuery("disaster recovery readiness", "cross_domain",
              [REPLICATION, AUTOSCALE, DATADOG, ONCALL, INCIDENT],
              "DB replica + scaling + monitoring + on-call + postmortem"),
    TestQuery("attack surface of our public APIs", "cross_domain",
              [JWT, MFA, RATELIMIT, ENVOY, OAUTH],
              "auth + rate limiting + proxy + OAuth"),
    TestQuery("what would break if Redis went down", "cross_domain",
              [REDIS, MATVIEW],
              "cache layer + would need direct queries"),
    # cross-domain spanning all 3 doc sets (core + infra + data)
    TestQuery("end-to-end journey of a payment: from API to data pipeline", "cross_domain",
              [ENVOY, JWT, INCIDENT, CELERY_RETRY, DATADOG, DBT_FRESHNESS],
              "proxy → auth → payments outage → webhook task → monitoring → dbt freshness"),
    TestQuery("how do we deploy a new ML model safely", "cross_domain",
              [AB_SHADOW, MLFLOW_MODEL, FRAUD_SHADOW, DEPLOY_GATES, ARGOCD],
              "shadow mode + MLflow staging + fraud AUC threshold + deploy gates + GitOps"),
    TestQuery("our secrets management from code to pod", "cross_domain",
              [VAULT_SECRETS, SEALED_SECRETS, SECRET_ROTATION, K8S_RBAC],
              "Vault injection + sealed secrets in git + rotation policy + RBAC"),
    TestQuery("what monitors would fire during a data pipeline outage", "cross_domain",
              [DATADOG, MONTE_CARLO, DBT_FRESHNESS, GX_SUITE],
              "Datadog payments + Monte Carlo row count + dbt freshness + Great Expectations"),
    TestQuery("everything that runs on Kubernetes in production", "cross_domain",
              [CART, AUTOSCALE, K8S_HPA, OTEL_COLLECTOR, LOG_AGG, VAULT_SECRETS, ISTIO_MTLS],
              "services + HPA + OTel DaemonSet + Promtail + Vault sidecar + Istio mesh"),
]

# 15. Needle in haystack — unique detail buried in a longer memory
NEEDLE = [
    TestQuery("envoy-ratelimit ConfigMap", "needle",
              [ENVOY],
              "specific detail: envoy-ratelimit ConfigMap"),
    TestQuery("15-minute access token", "needle",
              [SESSION],
              "specific detail: 15-min access token in session refactor"),
    TestQuery("47 minutes of downtime", "needle",
              [INCIDENT],
              "specific detail: SEV-1, 47 min downtime"),
    TestQuery("connection pool exhaustion", "needle",
              [INCIDENT],
              "specific detail: root cause of payments outage"),
    TestQuery("v1 API sunset date", "needle",
              [VERSIONING],
              "specific detail: v1 sunset 2026-06-01"),
    # new
    TestQuery("what is the exact CPU limit per pod", "needle", [RESOURCE_LIMITS]),
    TestQuery("what is the AWS IAM dynamic secret TTL in Vault", "needle", [VAULT_SECRETS]),
    TestQuery("what was the Sealed Secrets key rotation date", "needle", [SEALED_SECRETS]),
    TestQuery("what is the default cursor pagination page size", "needle", [PAGINATION]),
    TestQuery("how many replicas does the cart-service have", "needle", [CART]),
    TestQuery("what is the BigRetail SLA p95 threshold that was breached", "needle", [BIZ_SLA_BREACH]),
    TestQuery("how many story points in the sprint plan", "needle", [SPRINT]),
    TestQuery("what's the response error code structure", "needle", [ERRORS]),
    TestQuery("what is the Envoy rate limit ConfigMap name", "needle", [ENVOY]),
    TestQuery("how long is the scale-down stabilization for HPA", "needle", [K8S_HPA]),
]

# ── Business/ops queries ──

BIZ_EXACT = [
    TestQuery("checkout keeps timing out for European office users", "biz_exact", [BIZ_ACME_TIMEOUT]),
    TestQuery("NPS score dropped from 42 to 31 in February", "biz_exact", [BIZ_NPS_DROP]),
    TestQuery("chose Datadog over New Relic for monitoring", "biz_exact", [BIZ_VENDOR_DATADOG]),
    TestQuery("GDPR audit scheduled for 2026-05-01", "biz_exact", [BIZ_GDPR]),
    TestQuery("sprint retro: CI pipeline takes over 8 minutes", "biz_exact", [BIZ_RETRO]),
    # new
    TestQuery("Acme Corp ticket #4891 checkout timing out European users", "biz_exact", [BIZ_ACME_TIMEOUT]),
    TestQuery("login failing intermittently since Tuesday release P1", "biz_exact", [BIZ_LOGIN_FAILING]),
    TestQuery("BigRetail bulk order import failing since 2026-03-05 3 days inventory lost", "biz_exact", [BIZ_BIGRETAIL_IMPORT]),
    TestQuery("conversion rate dropped 8 percent on 2026-03-13 45K revenue impact", "biz_exact", [BIZ_CONVERSION_DROP]),
    TestQuery("p95 response time exceeded 500 milliseconds 4 days in March BigRetail contractual review", "biz_exact", [BIZ_SLA_BREACH]),
    TestQuery("sunset legacy checkout 2026-04-15 new single-page converts 12 percent better", "biz_exact", [BIZ_CHECKOUT_SUNSET]),
    TestQuery("recommendation carousel 20 percent users 2026-02-10 3 percent uplift AOV", "biz_exact", [BIZ_REC_CAROUSEL]),
    TestQuery("Q2 2026 push notifications 60 percent mobile users engagement drops", "biz_exact", [BIZ_MOBILE_ROADMAP]),
    TestQuery("MAU increased 15 percent February 2026 referral program server costs up 22 percent", "biz_exact", [BIZ_MAU_GROWTH]),
    TestQuery("SOC2 Type II 5 control gaps access reviews audit trail schema changes", "biz_exact", [BIZ_SOC2]),
]

BIZ_PARAPHRASE = [
    TestQuery("why are European customers having trouble placing orders", "biz_paraphrase",
              [BIZ_ACME_TIMEOUT],
              "business language for checkout timeout — maps to Redis cache TTL issue"),
    TestQuery("which monitoring vendor did we pick and why", "biz_paraphrase",
              [BIZ_VENDOR_DATADOG],
              "rephrased vendor decision"),
    TestQuery("what compliance certifications are we working toward", "biz_paraphrase",
              [BIZ_SOC2, BIZ_GDPR],
              "paraphrase of SOC2 + GDPR prep"),
    TestQuery("what happened to our customer satisfaction scores recently", "biz_paraphrase",
              [BIZ_NPS_DROP],
              "NPS drop rephrased"),
    TestQuery("are we hiring for reliability engineering", "biz_paraphrase",
              [BIZ_HIRE_SRE],
              "SRE hire rephrased in business terms"),
    # new
    TestQuery("why are European customers unable to complete orders on time", "biz_paraphrase", [BIZ_ACME_TIMEOUT]),
    TestQuery("what authentication problems started after the recent production release", "biz_paraphrase", [BIZ_LOGIN_FAILING]),
    TestQuery("how did we lose three days worth of customer order data", "biz_paraphrase", [BIZ_BIGRETAIL_IMPORT]),
    TestQuery("what performance issue drove down customer satisfaction last month", "biz_paraphrase", [BIZ_NPS_DROP]),
    TestQuery("why did our sales revenue drop significantly in March", "biz_paraphrase", [BIZ_CONVERSION_DROP]),
    TestQuery("how did we fail to meet our SLA commitments with major customers", "biz_paraphrase", [BIZ_SLA_BREACH]),
    TestQuery("what is the timeline for transitioning to the new checkout system", "biz_paraphrase", [BIZ_CHECKOUT_SUNSET]),
    TestQuery("what business impact did the recommendation feature launch have", "biz_paraphrase", [BIZ_REC_CAROUSEL]),
    TestQuery("what should we prioritize for mobile users next quarter", "biz_paraphrase", [BIZ_MOBILE_ROADMAP]),
    TestQuery("how is the growth from our referral campaign affecting operations", "biz_paraphrase", [BIZ_MAU_GROWTH]),
]

BIZ_CROSS_DOMAIN = [
    TestQuery("what caused the checkout complaints from European customers", "biz_cross_domain",
              [BIZ_ACME_TIMEOUT, REDIS],
              "business complaint + engineering root cause: Redis TTL for EU traffic"),
    TestQuery("what was the business impact of the March 12 payments outage", "biz_cross_domain",
              [BIZ_CONVERSION_DROP, BIZ_QBR, INCIDENT],
              "revenue drop + CEO review + engineering postmortem"),
    TestQuery("why did our build times get flagged and what can we do about it", "biz_cross_domain",
              [BIZ_RETRO, GH_ACTIONS_CI, BUILD_CACHE],
              "sprint retro complaint + CI pipeline config + build cache setup"),
    TestQuery("what security gaps did auditors find and are we already fixing them", "biz_cross_domain",
              [BIZ_SECURITY_AUDIT, ISTIO_MTLS, BIZ_SOC2],
              "audit findings + Istio mTLS already deployed + SOC2 gaps"),
    TestQuery("login issues reported by customers and the technical root cause", "biz_cross_domain",
              [BIZ_LOGIN_FAILING, BUG],
              "customer complaint about login + engineering JWT refresh bug"),
    # new
    TestQuery("what technical issue caused the European checkout failures and how do we prevent it", "biz_cross_domain", [BIZ_ACME_TIMEOUT, REDIS, AUTOSCALE]),
    TestQuery("why did the post-release login failures hurt our customer satisfaction metrics", "biz_cross_domain", [BIZ_LOGIN_FAILING, JWT, SESSION]),
    TestQuery("what systems failed during the March 12 outage and how should we isolate payments", "biz_cross_domain", [BIZ_QBR, INCIDENT, CIRCUIT_BREAKER]),
    TestQuery("how did database performance issues contribute to the conversion rate drop", "biz_cross_domain", [BIZ_CONVERSION_DROP, MATVIEW, INDEX]),
    TestQuery("what caused the sustained performance degradation and SLA breaches", "biz_cross_domain", [BIZ_SLA_BREACH, DATADOG, K8S_HPA]),
    TestQuery("how are our caching and auto-scaling strategies affecting page load times", "biz_cross_domain", [BIZ_NPS_DROP, REDIS, AUTOSCALE]),
    TestQuery("what observability improvements would help us detect incidents faster", "biz_cross_domain", [BIZ_HIRE_SRE, GRAFANA_DASH, OTEL_COLLECTOR]),
    TestQuery("how does the import pipeline reliability affect inventory management for partners", "biz_cross_domain", [BIZ_BIGRETAIL_IMPORT, CELERY_DLQ, AIRFLOW_ETL]),
    TestQuery("what security controls are missing for our compliance requirements", "biz_cross_domain", [BIZ_SOC2, VAULT_SECRETS, ISTIO_MTLS]),
    TestQuery("how do our data quality systems support billing and reporting reliability", "biz_cross_domain", [BIZ_CONVERSION_DROP, MONTE_CARLO, DBT_FRESHNESS]),
]

BIZ_ADVERSARIAL = [
    TestQuery("customer complaints about website performance", "biz_adversarial",
              [BIZ_NPS_DROP, BIZ_ACME_TIMEOUT, BIZ_SLA_BREACH],
              "'website performance' matches tons of generic tech noise"),
    TestQuery("business impact of infrastructure decisions", "biz_adversarial",
              [BIZ_CONVERSION_DROP, BIZ_MAU_GROWTH, BIZ_VENDOR_DATADOG],
              "'business impact' + 'infrastructure' both noisy terms"),
    TestQuery("compliance requirements for data handling", "biz_adversarial",
              [BIZ_GDPR, BIZ_SOC2],
              "'compliance' and 'data handling' match regulatory news"),
    # new
    TestQuery("payment system failures and revenue impact during business hours", "biz_adversarial", [BIZ_CONVERSION_DROP, INCIDENT, BIZ_QBR]),
    TestQuery("security audit findings and compliance requirements for March deadline", "biz_adversarial", [BIZ_SECURITY_AUDIT, BIZ_SOC2, BIZ_GDPR]),
    TestQuery("user experience degradation and engagement metric declines", "biz_adversarial", [BIZ_NPS_DROP, BIZ_CONVERSION_DROP, BIZ_MOBILE_ROADMAP]),
    TestQuery("infrastructure reliability and service quality metrics", "biz_adversarial", [BIZ_SLA_BREACH, INCIDENT, K8S_HPA]),
    TestQuery("business continuity and disaster recovery capabilities", "biz_adversarial", [INCIDENT, BIZ_ARCH_REVIEW, REPLICATION]),
    TestQuery("growth metrics and cost management in operations", "biz_adversarial", [BIZ_MAU_GROWTH, BIZ_DATA_TEAM, RESOURCE_LIMITS]),
    TestQuery("incident detection and response capabilities", "biz_adversarial", [BIZ_QBR, INCIDENT, BIZ_HIRE_SRE]),
    TestQuery("monitoring and alerting for critical business processes", "biz_adversarial", [DATADOG, GRAFANA_DASH, BIZ_VENDOR_DATADOG]),
    TestQuery("data integrity and pipeline reliability for analytics", "biz_adversarial", [MONTE_CARLO, DBT_FRESHNESS, BIZ_DATA_TEAM]),
    TestQuery("team capacity and hiring for platform stability", "biz_adversarial", [BIZ_HIRE_SRE, BIZ_DATA_TEAM, BIZ_RETRO]),
    TestQuery("customer complaints about website performance and transaction processing", "biz_adversarial", [BIZ_ACME_TIMEOUT, BIZ_NPS_DROP, BIZ_SLA_BREACH]),
    TestQuery("platform updates and deprecation planning for legacy systems", "biz_adversarial", [BIZ_APIV1_DEPREC, BIZ_CHECKOUT_SUNSET, VERSIONING]),
]

BIZ_TEMPORAL = [
    TestQuery("business events in March 2026", "biz_temporal",
              [BIZ_RETRO, BIZ_QBR, BIZ_CONVERSION_DROP, BIZ_SLA_BREACH, BIZ_SECURITY_AUDIT, BIZ_BIGRETAIL_IMPORT],
              "sprint retro 3/15 + QBR 3/20 + conversion drop 3/13 + SLA breach March + audit remediation March + BigRetail 3/5"),
    TestQuery("what changed in January 2026 from a business perspective", "biz_temporal",
              [BIZ_APIV1_DEPREC, BIZ_SECURITY_AUDIT, BIZ_DATA_TEAM],
              "API v1 deprecation notice 1/20 + security audit 1/25 + data team expansion January"),
    # new
    TestQuery("what business metrics changed in February 2026", "biz_temporal", [BIZ_NPS_DROP, BIZ_MAU_GROWTH, BIZ_REC_CAROUSEL, AIRFLOW_BACKFILL]),
    TestQuery("what incidents and issues occurred in March 2026", "biz_temporal", [INCIDENT, BIZ_QBR, BIZ_CONVERSION_DROP, BIZ_BIGRETAIL_IMPORT, BIZ_SLA_BREACH]),
    TestQuery("what system failures happened between March 12 and March 20", "biz_temporal", [INCIDENT, BIZ_QBR, BIZ_CONVERSION_DROP]),
    TestQuery("what are the deadlines for Q2 2026 initiatives", "biz_temporal", [BIZ_MOBILE_ROADMAP, BIZ_ARCH_REVIEW, BIZ_CHECKOUT_SUNSET, BIZ_HIRE_SRE]),
    TestQuery("what needs to be completed by April 2026", "biz_temporal", [BIZ_CHECKOUT_SUNSET, BIZ_GDPR, BIZ_HIRE_SRE]),
    TestQuery("what compliance work is due by May 1 2026", "biz_temporal", [BIZ_GDPR, BIZ_SOC2]),
    TestQuery("what happened on Tuesday when the login system failed", "biz_temporal", [BIZ_LOGIN_FAILING, BUG, SESSION]),
    TestQuery("what business milestones occurred at the March 15 retrospective", "biz_temporal", [BIZ_RETRO, BIZ_QBR, INCIDENT]),
    TestQuery("what capacity changes occurred when the data team expanded", "biz_temporal", [BIZ_DATA_TEAM, AIRFLOW_ETL, FEAST_FEATURES, DBT_MODELS]),
    TestQuery("what infrastructure decisions were made at the February 20 architecture review", "biz_temporal", [BIZ_ARCH_REVIEW, CIRCUIT_BREAKER, REPLICATION]),
    TestQuery("what security work was mandated from the January 25 audit", "biz_temporal", [BIZ_SECURITY_AUDIT, VAULT_SECRETS, ISTIO_MTLS, BIZ_SOC2]),
    TestQuery("what operational changes happened in January 2026", "biz_temporal", [BIZ_DATA_TEAM, BIZ_APIV1_DEPREC, BIZ_SECURITY_AUDIT, MFA]),
    TestQuery("what product features launched in early 2026", "biz_temporal", [BIZ_REC_CAROUSEL, BIZ_MOBILE_ROADMAP, BIZ_APIV1_DEPREC]),
]

ALL_QUERIES = (EXACT + PARAPHRASE + TANGENTIAL + ADVERSARIAL + NEGATIVE +
               MULTIHOP + LEXICAL_TRAPS + COMPOSITIONAL +
               TEMPORAL + CONVERSATIONAL + SYNONYM_HELL + PRECISION +
               NEGATION + CROSS_DOMAIN + NEEDLE +
               BIZ_EXACT + BIZ_PARAPHRASE + BIZ_CROSS_DOMAIN +
               BIZ_ADVERSARIAL + BIZ_TEMPORAL)


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


def is_engineering(item: dict) -> bool:
    content = (item.get("l0", "") + " " + item.get("content", "")).lower()
    return any(fp in content for fp in _ID_FINGERPRINTS.values())


def dcg(rels, k):
    return sum(r / math.log2(i + 2) for i, r in enumerate(rels[:k]))


def ndcg(rels, k, n_rel):
    ideal = [1] * min(n_rel, k) + [0] * max(0, k - n_rel)
    idcg = dcg(ideal, k)
    return dcg(rels, k) / idcg if idcg > 0 else 0.0


def evaluate(queries, verbose=False, progress=None):
    precisions, hit_rates, mrrs, ndcgs, fp_rates = [], [], [], [], []
    base = progress[0] if progress else 0
    total = progress[1] if progress else len(queries)
    for qi, tq in enumerate(queries):
        cur = base + qi + 1
        if not verbose:
            print(f"\r  [{cur}/{total}]", end="", flush=True)
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
    if not verbose:
        print()  # newline after progress counter
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
        ("Temporal", TEMPORAL), ("Conversational", CONVERSATIONAL),
        ("Synonym hell", SYNONYM_HELL), ("Precision", PRECISION),
        ("Negation", NEGATION), ("Cross-domain", CROSS_DOMAIN),
        ("Needle", NEEDLE),
        ("Biz exact", BIZ_EXACT), ("Biz paraphrase", BIZ_PARAPHRASE),
        ("Biz cross-domain", BIZ_CROSS_DOMAIN), ("Biz adversarial", BIZ_ADVERSARIAL),
        ("Biz temporal", BIZ_TEMPORAL),
    ]
    total = len(ALL_QUERIES)
    n_positive = sum(1 for q in ALL_QUERIES if not q.is_negative)
    print("=" * 80)
    print(f"  ark BRUTAL benchmark — top-{K}, {total} queries ({n_positive}+/{total-n_positive}-), ~1173 docs")
    print(f"  23 eng + 150 general + 500 AG News + 500 tech news")
    print("=" * 80)
    rows = []
    done = 0
    for name, queries in cats:
        print(f"\n{name} ({len(queries)} queries)  [{done}/{total}]")
        m = evaluate(queries, verbose, progress=(done, total))
        done += len(queries)
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

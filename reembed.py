"""Re-embed all memories.

Clears old embeddings.db and tantivy index, then re-ingests all memories
with the configured embedding provider (default: fastembed nomic; set
OPENROUTER_EMBED_MODEL + OPENROUTER_API_KEY for pplx-embed).

Total corpus: 63 engineering docs (23 core + 20 infra + 20 data) + 1150 noise.
"""
import asyncio
import hashlib
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


ENGINEERING_INFRA = [
    "Kubernetes HPA configured for api-gateway: min 2 replicas, max 10, target CPU 70%, scale-down stabilization 300s. Managed by platform-team in k8s/overlays/prod/hpa.yaml.",
    "Resource limits enforced for all pods in prod namespace: CPU request 250m/limit 1000m, memory request 256Mi/limit 1Gi. OOMKilled events in ml-inference-service triggered refactor in Feb 2026.",
    "RBAC policy: engineers get view on all namespaces, write only on their team namespace. ServiceAccount tokens rotated weekly. ClusterRole 'eng-read-all' bound to group engineering@company.com via Okta.",
    "Terraform state stored in S3 bucket tf-state-prod-us-east-1 with DynamoDB table tf-locks for state locking. Backend config in terraform/backend.tf, owned by infra-team.",
    "Terraform modules pinned: vpc-module v2.1.0, rds-module v1.4.2, eks-module v3.0.1 in terraform/modules.tf. All modules sourced from internal registry at registry.internal/terraform.",
    "Terraform workspaces: prod, staging, dev-{engineer} per developer. Workspace variables in terraform/vars/{workspace}.tfvars. Prod workspace requires 2-person approval in Atlantis before apply.",
    "GitHub Actions CI pipeline for all services: lint -> unit tests -> integration tests -> docker build -> push to ECR. Defined in .github/workflows/ci.yml, runs on ubuntu-22.04, avg build time 4m32s.",
    "GitHub Actions build cache: Docker layer cache stored in ECR, npm/pip caches in GitHub Actions cache. Cache key is hash of lock files. Saves avg 2m10s per build across 12 repos.",
    "Deploy gates in Argo CD: staging canary at 10% for 30 min before full rollout. Production requires manual approval from service owner. Gate config in argocd/apps/{service}/rollout.yaml.",
    "HashiCorp Vault v1.15.2 on k8s in vault namespace. AppRole auth for services, OIDC for engineers. Dynamic secrets: PostgreSQL TTL 1h, AWS IAM TTL 15m. Vault agent sidecar injects secrets at pod startup.",
    "Sealed Secrets controller v0.24.1 encrypts secrets for Git storage. Public key rotated 2026-01-15. Cert from sealed-secrets-controller in kube-system namespace. bitnami/sealed-secrets chart version 2.15.3.",
    "Secrets rotation policy: database passwords every 90 days, API keys every 30 days, service tokens every 7 days. Rotation automated via Vault + secrets-rotation.yml GitHub Actions workflow. Last full rotation 2026-02-01.",
    "Istio service mesh v1.20.3 in prod cluster. All east-west traffic mTLS STRICT mode. PeerAuthentication policy applied cluster-wide. Ingress via Istio gateway on port 443 with cert-manager issued TLS.",
    "Istio circuit breaker for payments-service: 5 consecutive 5xx errors triggers 30s host ejection from load balancer pool. Retry policy: 3 retries, 500ms backoff, on 503s. Config in VirtualService payments-vs.yaml.",
    "OpenTelemetry Collector v0.96.0 as DaemonSet. Traces to Jaeger (5% sampling), metrics to Prometheus, logs to Loki. Auto-instrumentation for Java and Python via OpenTelemetry Operator in otel-system namespace.",
    "Grafana v10.2.3 dashboards for all services. SLO overview at grafana.internal/d/slo-overview, refresh 5s. Alert rules sync'd from grafana/provisioning/alerting/ in infra repo. 47 active alert rules across 8 service teams.",
    "Log aggregation: Loki v2.9.4 + Promtail DaemonSet. Retention 30 days prod, 7 days staging. Volume ~2TB/day. Structured JSON logs enforced for all services via OPA admission webhook log-format-policy.",
    "Kubernetes namespace strategy: prod, staging, dev-{team}. Network policies block cross-team traffic in dev. Prod has PodSecurityPolicy restricted. ResourceQuota per namespace: 64 CPU cores, 256Gi memory.",
    "Argo CD v2.9.5 GitOps deployments. Sync policy: automated for staging, manual for prod. ApplicationSet used for per-service apps. Custom health checks defined for all CRDs in argocd/apps/.",
    "Node affinity rules: ml-inference pods on GPU nodes (node.kubernetes.io/gpu=true), prod on on-demand, batch on spot. Taint spot=true:NoSchedule on spot node pool. Spot interruption handler DaemonSet in kube-system.",
]

ENGINEERING_DATA = [
    'user-events Kafka topic has 12 partitions, retention 7 days. Consumer group ingestion-team monitors lag via Burrow; PagerDuty alert fires when lag exceeds 50K messages for 3 consecutive minutes.',
    'analytics-consumer group reads from order-completed Kafka topic (6 partitions, replication-factor 3). Lag monitored via Prometheus kafka_consumer_group_lag metric; alert fires at lag > 10K messages for 5 minutes, routes to #data-oncall.',
    'etl_user_activity Airflow DAG runs daily at 02:00 UTC, max_active_runs=1, depends_on_past=True. Backfill triggered via: airflow dags backfill -s 2026-01-01 -e 2026-02-01 etl_user_activity.',
    'dbt_analytics project has 47 models. fct_orders incremental model uses order_id as unique key with merge strategy. Full refresh required after schema changes to raw_orders: dbt run --full-refresh --select fct_orders.',
    'ML feature store uses Feast 0.32. user_purchase_features feature view refreshed every 15 min via materialization job. Online store backed by Redis (ttl 1h), offline store in BigQuery dataset feast_offline.',
    'rec_model_v2 receives 20% of recommendation traffic via feature flag rec-ab-test in LaunchDarkly. Shadow mode logs predictions to rec_shadow_log without affecting users. Rollback threshold: CTR drops > 5% vs control.',
    'MLflow tracks experiments under recommendations project. Production model is run ID a4f82c91, registered as rec-model/production. Challenger model rec-model/staging promoted after A/B test with p < 0.05 lift in CTR.',
    'analytics.events BigQuery table partitioned by event_date (daily), clustered on user_id, event_type. Query cost capped at $50 per query via BI Engine reservation analytics-team-slot (100 slots). Partition expiry: 365 days.',
    'Snowflake roles: ANALYTICS_ROLE has SELECT on PROD_DB.PUBLIC.*; LOADER_ROLE has INSERT and COPY INTO; TRANSFORMER_ROLE owns DBT_PROD schema. Cross-account data share enabled for partner acme_corp (account: acme.us-east-1).',
    'Snowflake warehouse ANALYTICS_WH configured: size=X-Small, AUTO_SUSPEND=60, AUTO_RESUME=true, MAX_CLUSTER_COUNT=3. Monthly spend alert at $2000 sends email to data-team@company.com via Snowflake budget policy.',
    'Celery workers run 3 queues: default (concurrency 8, priority 5), high (concurrency 4, priority 10), bulk (concurrency 16, priority 1). Beat scheduler on worker-01. CELERY_TASK_SERIALIZER=json, broker=redis://redis-celery:6379/0.',
    'Failed tasks in email-notifications Celery queue moved to DLQ email-dlq after 3 retries with exponential backoff: countdown=1s, 2s, 4s. DLQ alert fires if queue depth > 100 messages; reviewed by data-platform team weekly.',
    'process_payment_webhook Celery task: max_retries=5, countdown=30, autoretry_for=(PaymentGatewayError, NetworkTimeout). On final failure, task serialized to payments-dlq SQS queue for manual replay. SLA: complete within 90s.',
    'Great Expectations suite orders_suite has 12 expectations on orders data source. expect_column_values_to_not_be_null on order_id, expect_column_values_to_be_between on amount (0, 100000). Suite runs post-ETL; failures block dbt models via Airflow sensor.',
    'dbt freshness check on fct_orders: warn after 6h, error after 24h, based on loaded_at column. Failures route alert to #data-quality Slack channel. Freshness checked every 30 min by dbt Cloud job freshness-monitor-prod.',
    'Monte Carlo monitors raw_events table: expected row count 500K-2M per day. Anomaly detector fires if < 400K or > 3M rows, or if null rate on user_id exceeds 0.1%. Incidents auto-created in Jira project DATA-OPS.',
    'customer_dimension table rebuilt nightly at 01:30 UTC from Salesforce via Fivetran connector salesforce-prod. Incremental sync runs every 2h (08:00-20:00 UTC). Schema change notifications sent to #fivetran-alerts channel.',
    'revenue_reconciliation Airflow DAG backfilled 2026-01-01 to 2026-02-28 after decimal precision migration on amount column. 59 DAG runs completed with max_active_runs=3, catchup=True. Validation: reconcile_task compared ETL output vs source totals.',
    'clickstream Kafka topic: 24 partitions, retention.ms=259200000 (3 days), log.cleanup.policy=delete, segment.bytes=1073741824. Producer config: acks=1, batch.size=65536, linger.ms=5. Topic owned by analytics-platform team.',
    'fraud-model-v3 deployed in shadow mode alongside fraud-model-v2. Shadow predictions written to fraud_shadow_log BigQuery table. Promotion to production requires AUC >= 0.94 on rolling 7-day holdout evaluated by ml-review-board.',
]


async def main():
    ark_home = os.environ.get("ARK_HOME", os.path.expanduser("~/.ark"))
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

    # Ingest infra/ops engineering memories
    print(f"Ingesting {len(ENGINEERING_INFRA)} infra/ops memories...")
    for i, text in enumerate(ENGINEERING_INFRA):
        await call_tool('ingest', {'content': text})
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(ENGINEERING_INFRA)}]")

    # Ingest data/backend engineering memories
    print(f"Ingesting {len(ENGINEERING_DATA)} data/backend memories...")
    for i, text in enumerate(ENGINEERING_DATA):
        await call_tool('ingest', {'content': text})
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(ENGINEERING_DATA)}]")

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

    total = len(ENGINEERING) + len(ENGINEERING_INFRA) + len(ENGINEERING_DATA) + len(NOISE) + len(ag_news) + len(tech_news)
    n_eng = len(ENGINEERING) + len(ENGINEERING_INFRA) + len(ENGINEERING_DATA)
    print(f"\nDone: {total} memories ingested ({n_eng} engineering + {len(NOISE)+len(ag_news)+len(tech_news)} noise)")

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

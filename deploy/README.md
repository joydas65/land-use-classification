# Cloud Run deployment and monitoring handoff

The production API is deployed in project `land-use-classification-502614` as service
`terraclass-api` in `asia-south1`. Future releases remain intentionally manual because publishing a
container or changing Cloud Run and Monitoring resources changes public infrastructure and can
affect billing.

## Inputs

- project: `land-use-classification-502614`
- region: `asia-south1`
- service: `terraclass-api`
- released image: `ghcr.io/joydas65/terraclass-api:api-v1.0.0`
- runtime identity: `terraclass-runtime@land-use-classification-502614.iam.gserviceaccount.com`

## Build and deploy

For a future semantic container release, deploy the immutable digest produced by the successful
GitHub workflow rather than a mutable tag:

```bash
gcloud run deploy terraclass-api \
  --project land-use-classification-502614 \
  --image "ghcr.io/joydas65/terraclass-api@sha256:IMMUTABLE_INDEX_DIGEST" \
  --region asia-south1 \
  --allow-unauthenticated \
  --service-account terraclass-runtime@land-use-classification-502614.iam.gserviceaccount.com \
  --cpu 2 \
  --memory 2Gi \
  --concurrency 4 \
  --min 0 \
  --max 3 \
  --timeout 30 \
  --set-env-vars "TERRACLASS_ALLOWED_ORIGINS=https://terraclass-land-use-classification.vercel.app,TERRACLASS_DEVICE=cpu,TERRACLASS_FAIL_ON_MODEL_ERROR=true,TERRACLASS_MAX_CONCURRENT_INFERENCES=1,TERRACLASS_QUEUE_TIMEOUT_SECONDS=5"
```

Record both the OCI index digest supplied to Cloud Run and the Linux/AMD64 digest resolved by the
revision. Keep the public-invoker, no-project-role runtime identity, resource, probe, and CORS
contracts unchanged unless a new reviewed requirement justifies the change.

## Acceptance gate

1. `/api/v1/health/live`, `/api/v1/health/ready`, and `/api/v1/model` return their versioned shapes.
2. The artifact SHA-256 equals
   `b4e8522aa702ef8d6670acd58e37ef2dd8948148a4fa9f07b88c23953473e523`.
3. The production load probe has zero failures at concurrency 1, 2, and 4.
4. Scale-from-zero client time, steady-state p50/p95, revision, region, and image digests are recorded.
5. Browser preflight and prediction requests succeed only from the Vercel production origin.

Only after this gate passes should `NEXT_PUBLIC_TERRACLASS_API_URL` be added to Vercel and the
frontend redeployed. Do not store Google credentials, access tokens, or service-account keys in this
repository.

## Monitoring

`configs/monitoring/observability_v1.json` defines the privacy allowlist, candidate objectives, and
drift claim boundary. The alert policies under `deploy/monitoring/` monitor the Cloud Run 5xx ratio
and warm-container p95 latency. Both were deployed and read back on 17 July 2026. They intentionally
omit notification channels until the owner chooses and verifies a destination. See
`reports/cloud_monitoring_deployment_2026-07-17.json` and `docs/OBSERVABILITY_AND_DRIFT.md` before
creating or editing a policy.

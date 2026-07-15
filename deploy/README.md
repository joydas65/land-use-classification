# Cloud Run deployment handoff

The repository is ready for a Google Cloud project, but deployment is intentionally not automatic:
creating Artifact Registry and Cloud Run resources can enable billing. Use a project owned by the
portfolio author and review the current Google Cloud pricing before continuing.

## Inputs

- `PROJECT_ID`: the selected Google Cloud project
- `REGION`: start with `asia-south1` for an India-focused portfolio audience, subject to current
  service availability and price review
- repository: `terraclass`
- service: `terraclass-api`
- image: `terraclass-api:1.0.0`

## Build and deploy

After the `model-v1.0.0` GitHub release asset is public and CI passes:

```bash
gcloud services enable artifactregistry.googleapis.com cloudbuild.googleapis.com run.googleapis.com
gcloud artifacts repositories create terraclass \
  --repository-format=docker \
  --location="$REGION"
gcloud builds submit \
  --tag "$REGION-docker.pkg.dev/$PROJECT_ID/terraclass/terraclass-api:1.0.0"
gcloud run deploy terraclass-api \
  --image "$REGION-docker.pkg.dev/$PROJECT_ID/terraclass/terraclass-api:1.0.0" \
  --region "$REGION" \
  --allow-unauthenticated \
  --cpu 2 \
  --memory 2Gi \
  --concurrency 4 \
  --min 0 \
  --max 3 \
  --timeout 30 \
  --set-env-vars "TERRACLASS_ALLOWED_ORIGINS=https://terraclass-land-use-classification.vercel.app,TERRACLASS_DEVICE=cpu,TERRACLASS_FAIL_ON_MODEL_ERROR=true,TERRACLASS_MAX_CONCURRENT_INFERENCES=1,TERRACLASS_QUEUE_TIMEOUT_SECONDS=5"
```

Record the immutable image digest from Artifact Registry and the Cloud Run revision. A mutable tag
is convenient for the build command, but the evidence register should identify the digest actually
resolved by the deployed revision.

## Acceptance gate

1. `/api/v1/health/live`, `/api/v1/health/ready`, and `/api/v1/model` return their versioned shapes.
2. The artifact SHA-256 equals
   `b4e8522aa702ef8d6670acd58e37ef2dd8948148a4fa9f07b88c23953473e523`.
3. The production load probe has zero failures at concurrency 1, 2, and 4.
4. Cold-start time, steady-state p50/p95, memory, revision, region, and image digest are recorded.
5. Browser preflight and prediction requests succeed only from the Vercel production origin.

Only after this gate passes should `NEXT_PUBLIC_TERRACLASS_API_URL` be added to Vercel and the
frontend redeployed. Do not store Google credentials, access tokens, or service-account keys in this
repository.

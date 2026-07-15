# Production Inference Phase — 16 July 2026

## Outcome

The verified ResNet18 inference service now has a portable production contract: a non-root CPU
container, checksum-pinned model distribution, bounded request capacity, automated CI definitions,
and a repeatable HTTP load test. The API is **not yet deployed** and the public Vercel interface
continues to show the model as offline. This keeps the portfolio claim aligned with what a visitor
can actually use.

## Why the API is separate from Vercel

Vercel remains the right home for the Tailwind CSS/Next.js frontend. Its standard Python function
bundle limit is 500 MB and its request/response body limit is 4.5 MB. The installed Torch package is
529 MB before adding torchvision, application dependencies, or the 44,795,275-byte model. Moving the
API into a container avoids an oversized function and preserves the API's 10 MB image contract.

The production target is Google Cloud Run because it runs OCI images, resolves a deployed image to
an immutable digest, accepts the injected `PORT`, and provides revision scaling and health probes.
The configuration remains portable to another container service if the hosting choice changes.

## Model distribution

`configs/serving/model_release_v1.json` binds model ID, model version, asset name, HTTPS URL, exact
byte count, and SHA-256. `scripts/fetch_serving_artifact.py` streams into a temporary file, rejects
wrong redirects, length, or content, and atomically replaces the destination only after verification.

The reserved release is `model-v1.0.0` with asset `resnet18_group_aware_v1.pt`. Publication is
pending explicit approval to upload the public 44.8 MB model asset. Until it exists, the production
container target is intentionally not buildable; the artifact-free `runtime-base` target remains
available for CI contract validation.

## Container and capacity contract

- Python 3.12 slim Linux base with patched pip 26.1.2 plus pinned PyTorch 2.13.0 and
  torchvision 0.28.0 CPU wheels.
- One non-root process and one loaded model copy per container.
- Startup fails if the model is missing, corrupt, or inconsistent with its serving configuration.
- A one-slot asynchronous semaphore prevents unbounded model execution; waiting requests time out
  with a structured `429 inference_capacity_exceeded` response and `Retry-After` header.
- Cloud Run is configured for 2 vCPU, 2 GiB memory, container concurrency 4, scale-to-zero, a maximum
  of three instances, startup readiness checks, and liveness checks.
- CORS permits the production Vercel origin rather than a wildcard.

## Local HTTP load evidence

The versioned report at `reports/api_load_test_2026-07-16.json` used the real hash-verified model and
`agricultural06.tif`. It issued five warm-up requests followed by **60 measured requests** across
three levels, with zero failures. These are steady-state local Apple Silicon CPU results, not Cloud
Run service-level objectives.

| Concurrency | Requests | Throughput | p50 total latency | p95 total latency |
|---:|---:|---:|---:|---:|
| 1 | 20 | 42.7 requests/second | 21.6 ms | 34.1 ms |
| 2 | 20 | **52.9 requests/second** | 35.4 ms | 44.8 ms |
| 4 | 20 | 49.9 requests/second | 78.9 ms | **84.1 ms** |

Throughput leveled off between concurrency 2 and 4 while latency rose, which is expected for a
single protected model execution slot. The evidence supports starting with container concurrency 4;
production measurements must still test cold starts, memory, scaling, and network latency.

## Automation

`.github/workflows/ci.yml` defines independent Python, frontend, and container-contract jobs. The
Python job also checks dependency consistency and known vulnerabilities before tests.
`.github/workflows/container-release.yml` builds the production image only after the model release is
available, publishes to GHCR, and requests provenance and an SBOM. GitHub CI run
[`29455400219`](https://github.com/joydas65/land-use-classification/actions/runs/29455400219)
completed successfully on 16 July: the Python, web, and artifact-free `runtime-base` container jobs
all passed. The model-bearing production image remains deliberately unbuilt until the approved
release asset exists.

## Remaining production handoff

1. Publish the checksum-pinned `model-v1.0.0` GitHub release asset.
2. Download the published asset through the release contract and verify its byte count and SHA-256.
3. Build and publish the production image, then deploy its exact digest to Cloud Run.
4. Run the same load probe against the HTTPS Cloud Run URL and record cold-start behavior.
5. Set `NEXT_PUBLIC_TERRACLASS_API_URL`, redeploy Vercel, verify CORS and predictions, and only then
   change the project status to a deployed end-to-end classifier.

The repository does not claim production API deployment, production SLOs, or an integrated public
classifier before all five steps pass.

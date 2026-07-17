# Observability and Drift-Readiness

## What this phase adds

TerraClass emits one structured `prediction_observation` event after each successful inference. The
event is designed for operational monitoring and deliberately contains only derived, explicitly
allowlisted fields: the model identity, predicted class, rounded confidence and confidence bucket,
inference latency, image dimensions, payload size, media type, and a random request ID for
correlation.

The service does **not** log the uploaded filename, image bytes, an image hash, client IP address, or
user agent in its application telemetry. It does not persist uploaded images. Google Cloud's managed
request logs are a separate platform facility governed by the project's logging policy; this
repository neither copies those fields into prediction telemetry nor changes the project's retention
settings.

Uvicorn's duplicate access log is disabled in the service entry point. Cloud Run's managed request
log remains the platform source for HTTP access diagnostics; prediction telemetry stays limited to
the application allowlist above.

The machine-readable contract is
`configs/monitoring/observability_v1.json`. Unit and API tests enforce the allowlist and prove that a
private upload filename cannot enter the prediction event.

This backward-compatible observability release advances the HTTP service to `1.1.0`. The deployed
ResNet18 artifact remains model version `1.0.0`; service and model versions are intentionally
independent.

## Candidate operating objectives

These are initial engineering objectives, not historical service-level guarantees:

| Signal | Candidate objective | Measurement boundary |
|---|---:|---|
| Availability | 99% over a rolling 30-day window | Container requests without a 5xx response; client-caused 4xx responses excluded |
| Steady-state latency | p95 at or below 1,000 ms over five minutes | GA `run.googleapis.com/request_latencies`; begins at the running container and excludes startup |
| Scale-from-zero request | Client total at or below 30,000 ms | One client request after verified zero-instance idleness; includes startup and network time |

The 16 July load probe remains the current warm production evidence: 60 measured requests, zero
failures, and 365.2 ms p95 at concurrency four. A 30-day measurement window has not elapsed, so the
99% target must not be described as an achieved production SLO.

Cloud Run documents that its GA request-latency metric excludes container startup. For that reason,
the cold-request measurement is recorded separately instead of mixing two different latency
boundaries. Cloud Run can keep an idle instance for up to 15 minutes, so a cold probe is accepted
only when minimum instances is zero, the preceding idle interval exceeds that window, and the
platform records a new traffic-triggered autoscaling instance.

The 17 July probe met all three conditions. Cloud Monitoring reported zero active and zero idle
instances at 19:24 UTC after a 1,119.990-second request gap. The next prediction caused a new
`AUTOSCALING` instance and returned the correct class in **11,013.115 ms** of client-observed time,
inside the 30,000 ms candidate threshold. This is one cold request, not a percentile or guarantee;
the exact evidence is in `reports/cloud_run_scale_to_zero_2026-07-17.json`.

## Alert policy templates

Two versioned Cloud Monitoring policies live under `deploy/monitoring/`:

- `cloud-run-5xx-ratio.json` opens an incident when 5xx responses exceed 5% in a five-minute
  alignment window. It is intentionally sensitive for the current low-volume portfolio service.
- `cloud-run-p95-latency.json` opens an incident when container p95 exceeds 1,000 ms for five
  minutes. This policy monitors warm container behavior and does not claim to measure cold startup.

Both policies were created and read back from project `land-use-classification-502614` on 17 July:

- 5xx ratio: `projects/land-use-classification-502614/alertPolicies/5310080937064810576`
- p95 latency: `projects/land-use-classification-502614/alertPolicies/5310080937064809962`

They are enabled and can create Cloud Monitoring incidents. They contain no notification channel;
email or another destination must be attached separately and verified by the project owner. This
avoids silently routing operational messages to an unconfirmed address. The readback evidence is
stored in `reports/cloud_monitoring_deployment_2026-07-17.json`.

The following commands are for a fresh project or deliberate policy recreation. Do not rerun them in
the current project without first checking for the existing policy IDs, or duplicate alerts will be
created:

```bash
gcloud monitoring policies create \
  --project land-use-classification-502614 \
  --policy-from-file deploy/monitoring/cloud-run-5xx-ratio.json
gcloud monitoring policies create \
  --project land-use-classification-502614 \
  --policy-from-file deploy/monitoring/cloud-run-p95-latency.json
```

## What the telemetry can and cannot say about drift

After at least 100 successful production predictions, the class distribution, confidence-bucket
distribution, and inference latency can be reviewed for obvious changes. That floor prevents a few
portfolio demonstrations from being presented as a stable population.

Those signals are drift-*ready*, not a validated drift detector. Changes can reflect genuine data
shift, seasonality, user-selection bias, a new image source, or simply a small sample. Credible drift
or accuracy claims require a representative production reference window, ground-truth labels or a
documented human-review sample, and versioned thresholds tested against expected variation. The
telemetry alone does not establish semantic drift, real-world accuracy, fairness, or label quality.

The scheduled 18 July implementation adds the strict offline analyzer described in
`docs/PRODUCTION_FEEDBACK_AND_DRIFT.md`. It aggregates events without retaining request IDs, compares
eligible windows with Jensen–Shannon divergence and latency/confidence signals, and calculates
accuracy and macro-F1 for owner-reviewed samples. The first real inventory contained one event, so
the 100-event floor prevented a comparison. This is implementation and refusal evidence, not drift
or production-accuracy evidence.

## Source basis

- [Cloud Run autoscaling](https://docs.cloud.google.com/run/docs/about-instance-autoscaling) defines
  scale-to-zero behavior and the possible 15-minute idle lifetime.
- [Cloud Run metric definitions](https://docs.cloud.google.com/monitoring/api/metrics_gcp_p_z)
  distinguish container request latency from end-to-end and pending latency.
- [Cloud Monitoring alert policies](https://docs.cloud.google.com/monitoring/alerts/policies-in-api)
  documents JSON/YAML policy creation and metric-ratio conditions.

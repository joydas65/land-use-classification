# Production Feedback and Drift Analysis

## Outcome

The scheduled 18 July phase was implemented and verified early on 17 July 2026. TerraClass now has
an offline, versioned workflow for turning privacy-allowlisted prediction logs and owner-controlled
human reviews into aggregate evidence. It can:

- validate raw `prediction_observation` events or Cloud Logging entries against the exact telemetry
  allowlist;
- build class, confidence-bucket, low-confidence, and inference-latency profiles without retaining
  request IDs or request-level events;
- compare sufficiently large reference and current windows with base-2 Jensen–Shannon divergence;
- detect candidate changes in low-confidence rate and p95 inference latency;
- validate private human-review records and calculate reviewed-sample accuracy, macro-F1, and
  per-class F1; and
- refuse drift or production-accuracy conclusions below explicit sample floors.

The implementation is in `src/terraclass/drift.py`, its machine-readable contract is
`configs/monitoring/drift_analysis_v1.json`, and `terraclass-drift` is the installed command-line
entry point.

## Privacy boundary

Prediction input is accepted only when its payload contains exactly the fields emitted by
`src/terraclass/telemetry.py`. A filename, image content/hash, IP address, user agent, or any
unrecognized field makes validation fail. The aggregate profile does not retain request IDs or
request-level observations.

Human review is an owner-controlled offline workflow rather than an unauthenticated public API. Each
record contains a random request ID, model identity, predicted class, reviewed class, review-source
category, and review timestamp. It cannot contain an image, filename, network identity, reviewer
name, or reviewer email. The summary discards request IDs and does not collect reviewer identity.

This design lets an authorized reviewer match a prediction temporarily while keeping personal
details and satellite imagery out of the versioned analysis report.

## Candidate signals

The first analysis contract uses the following engineering defaults:

| Signal | Candidate threshold |
|---|---:|
| Predicted-class Jensen–Shannon divergence | greater than 0.10 |
| Confidence-bucket Jensen–Shannon divergence | greater than 0.10 |
| Low-confidence-rate increase | greater than 0.10 |
| Inference-latency p95 ratio | greater than 2.0× |

Jensen–Shannon divergence is symmetric and bounded from zero to one when base-2 logarithms are used:
zero means the compared categorical distributions are identical, while one is their maximum
separation. These thresholds are intentionally named candidate signals. They have not been validated
against seasonal, source, geographic, or user-selection variation and therefore do not constitute a
validated drift detector.

Both the reference and current prediction windows require at least 100 events. Human-review metrics
also require 100 reviewed predictions before they are eligible for interpretation, and even then
they describe only that reviewed sample. A candidate signal should lead to data-quality and
human-review investigation, not an automatic claim that semantic drift occurred.

## Running the workflow

Export matching application events to a private path outside the repository:

```bash
gcloud logging read \
  'resource.type="cloud_run_revision" AND
   resource.labels.service_name="terraclass-api" AND
   jsonPayload.event="prediction_observation"' \
  --project=land-use-classification-502614 \
  --order=asc \
  --limit=1000 \
  --format=json > /secure/path/current-predictions.json
```

Build a current aggregate profile:

```bash
terraclass-drift \
  --current /secure/path/current-predictions.json \
  --output reports/current-production-profile.json
```

Add `--reference /secure/path/reference-predictions.json` only after a representative reference
window exists. Add `--reviews /secure/path/human-reviews.jsonl` only after the records pass the
documented privacy contract. Raw logs and review rows must remain outside Git; only the aggregate
report is eligible for versioning.

## First production inventory

The 17 July query returned one service-`1.1.0` `prediction_observation` for model `1.0.0`. The
analyzer produced an aggregate profile, removed the request ID, and reported:

- one prediction versus the 100-event minimum;
- no usable reference/current comparison;
- no human-reviewed labels; and
- no basis for production accuracy, semantic drift, or achieved-SLO claims.

The exact aggregate and claim boundary are stored in
`reports/production_drift_readiness_2026-07-17.json`. The raw Cloud Logging entry was not committed.

## Production dashboard

`deploy/monitoring/terraclass-operations-dashboard.json` passed the live Google Cloud
`--validate-only` check and was then created and read back as:

`projects/280836764570/dashboards/0c996266-70c0-4ad0-adc0-3e919225a4e4`

The dashboard shows request rate by response class, warm-container p95 latency, active/idle instance
counts, and the privacy-allowlisted prediction log stream. A prominent text panel preserves the
five-class scope and explains that the dashboard does not prove production accuracy, semantic drift,
or an achieved SLO.

The two alert policies remain enabled without a notification channel. Email routing will be attached
only after the owner explicitly confirms the destination.

## Source basis

- [Google Cloud dashboard API](https://docs.cloud.google.com/monitoring/dashboards/api-dashboard)
  documents dashboard layouts, widgets, creation, and readback.
- [Cloud Monitoring Dashboard resource](https://docs.cloud.google.com/monitoring/api/ref_v3/rest/v1/projects.dashboards)
  defines the JSON contract validated by the deployment.

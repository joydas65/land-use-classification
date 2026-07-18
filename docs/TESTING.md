# Testing and Audit Guide

The quality gate contains ten layers:

1. `terraclass-audit` cross-checks the immutable notebook, checksum, configuration, observed metrics, issue register, and documentation tokens.
2. `pytest` validates configuration, data discovery, deterministic and group-aware stratification,
   group isolation, manifest provenance, transforms, custom and transfer architectures, freezing
   policy, metrics, a gradient-update smoke test, deterministic notebook generation, security and
   IIT evidence, serving configuration, input limits, ranked predictions, checkpoint promotion,
   artifact distribution, benchmark statistics, bounded API capacity, deployment contracts, and the
   versioned FastAPI contract.
3. The web gate runs ESLint, a production vinext build, server-rendered HTML/contract tests, and the
   native Next.js build used by Vercel. Its browser contract includes decoding TIFF selections to a
   temporary PNG preview while retaining the original TIFF bytes for model inference.
4. `python -m compileall src scripts tests` catches syntax/import compilation failures.
5. GitHub CI builds the artifact-free container target; the production workflow separately fetches
   the checksum-pinned model and publishes an image with provenance and an SBOM.
6. The observability contract cross-checks the application telemetry allowlist, prohibited privacy
   fields, candidate service objectives, Cloud Monitoring policy metrics, and drift claim boundary.
7. The production-review gate validates strict prediction/review schemas, privacy exclusions,
   aggregate profiles, Jensen–Shannon calculations, insufficient-sample refusal, reviewed-sample
   accuracy/macro-F1, dashboard definition, and deployment readback evidence.
8. The model-quality gate validates calibration math, temperature-bound detection, fixed-width
   reliability bins, selective prediction, deterministic explainability selection, Grad-CAM output,
   and consistency among the protocol, evidence report, figures, résumé claims, and documentation.
9. The external-calibration gate validates pinned noncommercial dataset provenance, deterministic
   and role-disjoint sampling, bootstrap intervals, five-fold stability, untouched-test calibration
   metrics, UC Merced regression refusal, separate OOD metrics, and production claim boundaries.
10. The corruption-robustness gate validates deterministic corruptions, the 16-condition matrix,
    exact TTA tensor semantics, validation-only candidate selection, untouched-test refusal after
    rejection, figure/report hashes, résumé wording, and the synthetic-evidence claim boundary.

The complete local gate is:

```bash
PYTHONPATH=src python scripts/audit_consistency.py --project-root .
PYTHONPATH=src pytest
python -m pip check
python -m pip_audit
ruff check .
ruff format --check .
python -m compileall -q src scripts tests
cd web
# Use the Node.js 24 major pinned in package.json.
npm ci
npm run lint
npm test
npm run build:vercel
```

The container contract can be built without publishing or downloading the model:

```bash
docker build --target runtime-base --tag terraclass-api:contract-test .
```

Docker is not installed in the 16 July local environment, so that build is defined and
machine-checked but is not recorded as locally executed. GitHub CI run
[`29457675941`](https://github.com/joydas65/land-use-classification/actions/runs/29457675941)
successfully built the artifact-free `runtime-base` target alongside passing Python and web jobs.
The public model release was then verified through a fresh unauthenticated HTTPS download. Its
44,795,275-byte size and SHA-256 match the release and serving contracts; the result is committed in
`reports/model_release_verification_2026-07-16.json`.

The signed `api-v1.0.0` tag triggered public container-release run `29503393345`. The workflow
published `ghcr.io/joydas65/terraclass-api` successfully. A separate anonymous OCI pull verified
that the semantic tag and `sha-3b5b074` resolve to the same immutable index digest. Registry
inspection also verified the Linux/AMD64 image manifest and its two in-toto attestation layers: an
SPDX SBOM and SLSA provenance v1. The exact descriptors are committed in
`reports/container_release_verification_2026-07-16.json`.

For the observability release, GitHub CI run `29528731780` passed the Python, web, and
container-contract jobs. Tag `api-v1.1.0` triggered container-release run `29528840225`; anonymous
inspection verified both release tags against OCI index
`sha256:aee708b1d979a331f8f4f71ad9988ab01e6b04bc1cf2fc4420ad535328a06e41`, the resolved Linux/AMD64
manifest, and its SPDX/SLSA attestation layers. The current release evidence is committed in
`reports/container_release_verification_2026-07-17.json`.

The dependency review was rerun on 16 and 17 July with `npm audit --audit-level=high` for the
complete tree and `npm audit --omit=dev --audit-level=high` for production dependencies. Neither
tree has a high or critical advisory. The complete tree currently reports one low and five moderate
transitive findings. The production tree reports two moderate findings inherited from Next.js's
embedded PostCSS version; npm offered only a breaking downgrade, so they are documented rather than
hidden or force-fixed.
TerraClass does not accept user-authored CSS. The findings were reviewed for the integrated
deployment and remain accepted moderate transitive risk; they must be rechecked on dependency
updates.

The 16 July local Python audit initially identified pip 25.1.1 as vulnerable. After upgrading to
pip 26.1.2, `python -m pip_audit` reported no known third-party vulnerabilities on both 16 and 17
July; the local
`terraclass` package was skipped because it is not published on PyPI and is instead covered by the
repository's source, contract, and integration tests. The first GitHub run additionally exposed
runner-resolved Pillow 12.2.0 and setuptools 78.1.0 advisories. The project now requires Pillow
12.3 or newer, the build system requires setuptools 83 or newer, and CI upgrades setuptools before
auditing. The container pins the same patched Pillow and pip releases.

The workflow action majors were also refreshed to their current Node 24-based releases after the
first run reported Node 20 deprecation warnings.

Verify the submission notebook's acquisition and manifest cells against the real local dataset without running training:

```bash
python scripts/verify_submission_notebook.py \
  --archive data/raw/UCMerced_LandUse.zip \
  --dataset-root data/raw/UCMerced_LandUse/Images
```

Full training verification is performed on Colab Pro GPU according to `docs/COLAB_HANDOFF.md`; the returned results bundle is audited before any GPU metric is accepted.

The returned bundle was imported only after the strict validator checked archive safety and every cross-artifact invariant:

```bash
PYTHONPATH=src python scripts/import_colab_results.py \
  --bundle /path/to/terraclass_colab_results.zip \
  --verified-date 2026-07-13
```

`pytest` and `terraclass-audit` subsequently revalidate the committed report, comparison CSV, figure, manifests, bundle provenance, completed experiment matrix, and notebook PNG output without requiring the original ZIP.

The serving artifact is intentionally excluded from Git. When the local training checkpoint exists,
promote it through the verified export path and then run the recorded inference benchmark:

```bash
PYTHONPATH=src python scripts/export_serving_model.py \
  --source artifacts/resnet18_group_aware/best_model.pth \
  --output artifacts/serving/resnet18_group_aware_v1.pt \
  --expected-source-sha256 d3c22a5cf0e3f96c124f4c9e5b7b1200f696fb9b8bd95d6d79d8330035bf4067 \
  --model-id terraclass-resnet18-group-aware \
  --model-version 1.0.0
PYTHONPATH=src python scripts/benchmark_inference.py --project-root . --device cpu
```

A dataset-free test pass proves internal consistency; it does not reproduce the reported 74.67% accuracy. Reproduction is complete only after a full dataset run generates a hashed manifest and new `metrics.json`.

The real-model HTTP load report is reproducible after starting `terraclass-api`:

```bash
PYTHONPATH=src python scripts/load_test_api.py \
  --base-url http://127.0.0.1:8000 \
  --image data/raw/UCMerced_LandUse/Images/agricultural/agricultural06.tif \
  --content-type image/tiff \
  --concurrency 1 2 4 \
  --warmup-requests 5 \
  --requests-per-level 20 \
  --output reports/api_load_test_2026-07-16.json
```

The committed 16 July report contains 60 measured requests, zero failures, 52.9 requests/second
peak throughput, and 84.1 ms p95 latency at concurrency 4. It is local Apple Silicon steady-state
evidence and must not be described as a Cloud Run benchmark or production SLO.

The same protocol was run against the public Cloud Run service:

```bash
PYTHONPATH=src python scripts/load_test_api.py \
  --base-url https://terraclass-api-280836764570.asia-south1.run.app \
  --image data/raw/UCMerced_LandUse/Images/agricultural/agricultural06.tif \
  --content-type image/tiff \
  --concurrency 1 2 4 \
  --warmup-requests 5 \
  --requests-per-level 20 \
  --timeout-seconds 30 \
  --output reports/cloud_run_load_test_2026-07-16.json
```

The production report contains 60 measured requests and zero failures. Peak throughput was 13.3
requests/second; concurrency-4 p95 total latency was 365.2 ms. The deployment audit additionally
checks the exact OCI/platform digests, Cloud Run revision/resources/probes, HTTP 200 health and model
metadata, a dedicated runtime identity with no project roles, correct 99.30%-confidence production
prediction, origin-specific CORS, ready Vercel deployment, `Model ready` browser status, and zero
browser console warnings/errors. These are
point-in-time deployment measurements, not availability guarantees or a production SLO.

The 17 July telemetry tests additionally parse emitted JSON, check every confidence-bucket boundary,
reject invalid probabilities, and prove that an upload filename, image content/hash, IP address, and
user agent are absent from the prediction event. Deployment-contract tests parse both Cloud
Monitoring policy templates and require their metrics and service target to match
`configs/monitoring/observability_v1.json`. See `docs/OBSERVABILITY_AND_DRIFT.md` for the operational
claim boundary. Deployment tests also bind the `api-v1.1.0` release to Cloud Run revision
`terraclass-api-v1-1-0`, verify the exact structured log allowlist, and preserve the distinction
between deployed incident policies and unconfigured notification routing.

The scheduled 18 July feedback/drift-readiness tests use synthetic records only to exercise numerical
and privacy behavior. Production evidence remains separate: the first Cloud Logging inventory
contained one allowlisted prediction, and the committed aggregate correctly reports
`minimum_met=false` against the 100-event floor. The raw entry and request ID are not committed.
Dashboard configuration was checked through `gcloud monitoring dashboards create --validate-only`
before creation, then described by resource ID to verify the saved widgets and etag. See
`reports/production_drift_readiness_2026-07-17.json`.

The scheduled 19 July model-quality phase was run locally against the hash-verified serving artifact
and completed early on 17 July. Its 75 validation logits are the only input to scalar-temperature
fitting; the 75 test logits are used only for final evaluation. The fit reached the configured 0.05
lower bound, and the evidence therefore requires `deployment_approved=false` rather than promoting a
misleading calibration artifact. Seven focused unit tests cover the numerical and Grad-CAM
primitives. Reproduce the full model pass with:

```bash
PYTHONPATH=src python scripts/evaluate_model_quality.py \
  --project-root . \
  --device cpu
```

The committed report and two PNG figures are re-hashed by `terraclass-audit`; see
`docs/MODEL_QUALITY_AND_EXPLAINABILITY.md`.

The 18 July calibration-repair follow-up uses NWPU-RESISC45 without committing its 427 MB archive.
The downloader pins the archive and official split hashes, rejects unsafe or duplicate split rows,
and verifies the 31,500-image extracted inventory. Seven focused calibration/OOD tests and three
secure-download tests cover the new pipeline. Reproduce the real evaluation with:

```bash
PYTHONPATH=src python -m scripts.download_resisc45 --project-root .
PYTHONPATH=src python -m scripts.evaluate_external_calibration \
  --project-root . \
  --device cpu
```

The versioned manifest contains 500 independent calibration images, 500 untouched aligned external
test images, and 5,457 unmapped OOD images. The report requires the stable 2.697718 fit, its
500-replicate bootstrap interval, five-fold behavior, calibration improvements, and the failed UC
Merced NLL regression gate to remain consistent. See `docs/EXTERNAL_CALIBRATION_AND_OOD.md`.

The robustness phase scheduled for 20 July was completed early on 18 July. Nine focused test
functions cover configuration separation, deterministic sample seeds, five corruption families,
invalid parameters, exact TTA tensor semantics, the 16-condition matrix, aggregation, validation
selection, and production refusal. Reproduce the real model evaluation with:

```bash
PYTHONPATH=src python -m scripts.evaluate_robustness \
  --project-root . \
  --device cpu
```

The evaluator uses validation to decide whether four-view dihedral TTA may proceed. The candidate
reduced mean corruption macro F1 from 0.986390 to 0.985616, so it was rejected and candidate test
metrics were not computed. The single-view test baseline retained 0.991879 mean macro F1 across 15
synthetic corruption conditions; severe Gaussian blur was the worst condition at 0.918319 macro F1.
The JSON report and PNG figure are re-hashed by `terraclass-audit`; see
`docs/CORRUPTION_ROBUSTNESS.md`.

# Testing and Audit Guide

The quality gate contains five layers:

1. `terraclass-audit` cross-checks the immutable notebook, checksum, configuration, observed metrics, issue register, and documentation tokens.
2. `pytest` validates configuration, data discovery, deterministic and group-aware stratification, group isolation, manifest provenance, transforms, custom and transfer architectures, freezing policy, metrics, a gradient-update smoke test, deterministic notebook generation, security and IIT evidence, serving configuration, input limits, ranked predictions, checkpoint promotion, artifact distribution, benchmark statistics, bounded API capacity, deployment contracts, and the versioned FastAPI contract.
3. The web gate runs ESLint, a production vinext build, server-rendered HTML/contract tests, and the
   native Next.js build used by Vercel.
4. `python -m compileall src scripts tests` catches syntax/import compilation failures.
5. GitHub CI builds the artifact-free container target; the production workflow separately fetches
   the checksum-pinned model and publishes an image with provenance and an SBOM.

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

The dependency review was rerun on 16 July with `npm audit --audit-level=high` for the complete tree
and `npm audit --omit=dev --audit-level=high` for production dependencies. Neither tree has a high or
critical advisory. The complete tree currently reports one low and five moderate transitive
findings. The production tree reports two moderate findings inherited from Next.js's embedded
PostCSS version; npm offered only a breaking downgrade, so they are documented rather than hidden
or force-fixed.
TerraClass does not accept user-authored CSS, but the advisory must be reviewed again before the
integrated model API is deployed.

The 16 July local Python audit initially identified pip 25.1.1 as vulnerable. After upgrading to
pip 26.1.2, `python -m pip_audit` reported no known third-party vulnerabilities; the local
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

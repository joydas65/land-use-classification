# Testing and Audit Guide

The quality gate contains three layers:

1. `terraclass-audit` cross-checks the immutable notebook, checksum, configuration, observed metrics, issue register, and documentation tokens.
2. `pytest` validates configuration, data discovery, deterministic and group-aware stratification, group isolation, manifest provenance, transforms, custom and transfer architectures, freezing policy, metrics, a gradient-update smoke test, deterministic notebook generation, security and IIT evidence, serving configuration, input limits, ranked predictions, checkpoint promotion, benchmark statistics, and the versioned FastAPI contract.
3. The web gate runs ESLint, a production vinext build, and server-rendered HTML/contract tests.
4. `python -m compileall src scripts tests` catches syntax/import compilation failures.

The complete local gate is:

```bash
PYTHONPATH=src python scripts/audit_consistency.py --project-root .
PYTHONPATH=src pytest
ruff check .
ruff format --check .
python -m compileall -q src scripts tests
cd web && npm ci && npm run lint && npm test
```

The 15 July dependency review also ran `npm audit --omit=dev --audit-level=high`. It found no high
or critical production advisory. Two moderate findings are inherited from Next.js's embedded
PostCSS version; npm offered only a breaking downgrade, so they are documented rather than hidden or
force-fixed. TerraClass does not accept user-authored CSS, but the advisory must be reviewed again
before public deployment.

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

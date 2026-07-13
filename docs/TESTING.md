# Testing and Audit Guide

The quality gate contains three layers:

1. `terraclass-audit` cross-checks the immutable notebook, checksum, configuration, observed metrics, issue register, and documentation tokens.
2. `pytest` validates configuration, data discovery, deterministic and group-aware stratification, group isolation, manifest provenance, transforms, custom and transfer architectures, freezing policy, metrics, a gradient-update smoke test, and deterministic notebook generation, compilation, security, and IIT evidence.
3. `python -m compileall src scripts tests` catches syntax/import compilation failures.

The complete local gate is:

```bash
PYTHONPATH=src python scripts/audit_consistency.py --project-root .
PYTHONPATH=src pytest
ruff check .
ruff format --check .
python -m compileall -q src scripts tests
```

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

`pytest` and `terraclass-audit` subsequently revalidate the committed report, comparison CSV, figure, manifests, bundle provenance, completed experiment matrix, and notebook attachment without requiring the original ZIP.

A dataset-free test pass proves internal consistency; it does not reproduce the reported 74.67% accuracy. Reproduction is complete only after a full dataset run generates a hashed manifest and new `metrics.json`.

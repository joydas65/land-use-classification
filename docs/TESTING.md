# Testing and Audit Guide

The quality gate contains three layers:

1. `terraclass-audit` cross-checks the immutable notebook, checksum, configuration, observed metrics, issue register, and documentation tokens.
2. `pytest` validates configuration, data discovery, deterministic stratification, non-overlap, manifests, transforms, architecture, parameter count, metrics, and a gradient-update smoke test.
3. `python -m compileall src scripts tests` catches syntax/import compilation failures.

The complete local gate is:

```bash
PYTHONPATH=src python scripts/audit_consistency.py --project-root .
PYTHONPATH=src pytest
python -m compileall -q src scripts tests
```

A dataset-free test pass proves internal consistency; it does not reproduce the reported 74.67% accuracy. Reproduction is complete only after a full dataset run generates a hashed manifest and new `metrics.json`.


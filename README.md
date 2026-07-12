# TerraClass

TerraClass is the reproducible engineering wrapper around the supplied IIT Kanpur **Land Use Classification** sample project. The preserved notebook is evidence of the starting point; reusable code, configuration, manifests, tests, and later transfer-learning experiments live outside that immutable copy.

## Accepted baseline

The supplied dataset contains 2,100 images across 21 classes, but the saved baseline trains on **5 classes** and **500 images**. Its deterministic target split is **350/75/75** for training, validation, and testing.

| Item | Accepted value |
|---|---:|
| Model | Lightweight custom CNN |
| Trainable parameters | 102,277 |
| Epochs | 10 |
| Test accuracy | 74.67% |
| Test macro F1 | 0.733 |

These are observed values from the original notebook, not newly reproduced results. A new run must write its own manifest, checkpoint, and metrics file before it can be described as reproduced.

## Repository contract

- `notebooks/original/` is immutable and checksum-protected.
- `configs/baseline_5class.json` is the machine-readable baseline specification.
- `src/terraclass/` contains reusable dataset, transform, model, training, and audit code.
- `data/manifests/` is reserved for versioned split manifests.
- `artifacts/` contains untracked checkpoints and run metrics.
- `tests/` protects consistency among code, configuration, documentation, and the original notebook.
- `docs/BASELINE_AUDIT.md` records known defects in the supplied notebook.
- `docs/EXPERIMENT_PROTOCOL.md` defines the fair comparison required for the IIT submission.

## Local setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

Run the repository audit and tests:

```bash
terraclass-audit --project-root .
pytest
```

The original dataset host uses unencrypted HTTP and timed out during the 12 July audit. The downloader therefore defaults to TorchGeo's HTTPS mirror, which states that the archive is redistributed without modification. The expected 332,468,434-byte archive and SHA-256 are pinned in configuration:

```bash
python scripts/download_dataset.py
```

The original URL remains available only for provenance testing through `--source original --allow-insecure-http`.

Run the baseline after the dataset is available:

```bash
terraclass-baseline \
  --dataset-root data/raw/UCMerced_LandUse/Images \
  --config configs/baseline_5class.json \
  --output-dir artifacts/baseline_5class
```

The run produces `split_manifest.csv`, `best_baseline_model.pth`, and `metrics.json`. Data and model artifacts are intentionally excluded from Git.

The controlled 12 July CPU reproduction completed at 78.67% test accuracy and 0.777 macro F1 on the identical manifest. This is recorded as parity evidence, not a model improvement; see `docs/REPRODUCTION_RUN.md`.

## Scope boundary

The first improvement experiment must use the same five classes and split protocol. Scaling to all 21 classes is a separate extension and must not be presented as a direct accuracy comparison with the five-class baseline.

The historical split is checksum-preserved for assignment parity, but it contains visually related scene candidates across split boundaries. Portfolio claims will therefore include a separate group-aware five-class sensitivity run and use group-aware splitting for the 21-class extension.

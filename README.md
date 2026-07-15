# TerraClass

TerraClass is the reproducible engineering wrapper around the supplied IIT Kanpur **Land Use Classification** sample project. The preserved notebook is evidence of the starting point; reusable code, configuration, manifests, tests, and later transfer-learning experiments live outside that immutable copy.

## Delivery status

- **12 July 2026 — complete:** credential audit, repository setup, immutable baseline, verified dataset provenance, deterministic historical manifest, controlled CPU reproduction, tests, and consistency audit.
- **13 July 2026 milestone — complete for model selection:** conservative reviewed-scene groups, group-aware five-class manifest, transfer-learning infrastructure, and completed ResNet18 historical/group-aware experiments. ResNet18 is frozen as the selected model.
- **14 July 2026 collaboration milestone — complete:** self-contained IIT Kanpur Colab notebook, four-entry GPU matrix, secure results export, notebook tests, and local pre-training verification.
- **15 July 2026 submission milestone — complete:** returned NVIDIA L4 bundle validated, all four GPU runs verified, ResNet18 selected through documented tradeoff analysis, results embedded into the notebook, and IIT submission evidence finalized.
- **Next:** submit the updated notebook by 20 July 2026, then start the inference web-app phase.

Kaggle is not used by this repository. Dataset acquisition uses the UC Merced source or the checksum-pinned TorchGeo HTTPS mirror and requires no credential.

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
- `data/grouping/` records manually reviewed related-scene groups used for leakage control.
- `configs/transfer/` defines the ResNet18/EfficientNet-B0 by historical/group-aware experiment matrix.
- `artifacts/` contains untracked checkpoints and run metrics.
- `tests/` protects consistency among code, configuration, documentation, and the original notebook.
- `docs/BASELINE_AUDIT.md` records known defects in the supplied notebook.
- `docs/EXPERIMENT_PROTOCOL.md` defines the fair comparison required for the IIT submission.
- `notebooks/Improved_Land_Use_Classification_IITK.ipynb` is the self-contained GPU collaboration and submission notebook.
- `docs/COLAB_HANDOFF.md` defines the credential-free user/Codex results handoff.
- `reports/colab/VERIFICATION.json` is the canonical audited NVIDIA L4 evidence.
- `reports/figures/training_and_confusion_colab_l4.png` preserves the verified curves and confusion matrices.

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

The verified NVIDIA L4 matrix completed without failures. ResNet18 and EfficientNet-B0 both reached 100.00% test accuracy and 1.000 macro F1 on historical and group-aware manifests. ResNet18 remains selected because it had lower test loss on both manifests, faster historical L4 training, and independent local CPU support. On the historical manifest it improves accuracy by +25.33 percentage points and macro F1 by +0.267 over the supplied notebook. These results apply only to 500 images across 5 classes; see `docs/TRANSFER_RESULTS.md`.

Create and verify the leakage-controlled manifest:

```bash
PYTHONPATH=src python scripts/prepare_group_aware_manifest.py \
  --dataset-root data/raw/UCMerced_LandUse/Images
```

Run one transfer-learning experiment (replace the config and output directory for the other three matrix entries):

```bash
PYTHONPATH=src python -m terraclass.transfer_training \
  --dataset-root data/raw/UCMerced_LandUse/Images \
  --config configs/transfer/resnet18_historical.json \
  --output-dir artifacts/resnet18_historical
```

## Scope boundary

The first improvement experiment must use the same five classes and split protocol. Scaling to all 21 classes is a separate extension and must not be presented as a direct accuracy comparison with the five-class baseline.

The historical split is checksum-preserved for assignment parity, but it contains visually related scene candidates across split boundaries. Portfolio claims will therefore include a separate group-aware five-class sensitivity run and use group-aware splitting for the 21-class extension.

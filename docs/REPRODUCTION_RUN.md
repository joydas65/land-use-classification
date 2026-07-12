# Five-Class Baseline Reproduction — 12 July 2026

## Outcome

The configuration-driven runner completed all 10 epochs on CPU and used the exact versioned 350/75/75 manifest. The run selected epoch 9.

| Metric | Historical notebook | Controlled reproduction |
|---|---:|---:|
| Test accuracy | 74.67% | 78.67% |
| Test macro F1 | 0.733 | 0.777 |
| Best validation accuracy | 81.33% | 76.00% |

The controlled run correctly classified 59 of 75 test images; the historical run classified 56. This is **not** a model-improvement claim: model architecture, data, and training recipe are unchanged, while runtime versions and randomness controls differ. With only 75 test images, three predictions move accuracy by four percentage points.

## Provenance

- Download source: pinned TorchGeo HTTPS mirror of the original archive
- Archive size: 332,468,434 bytes
- Archive SHA-256: `06c539ef28703a58fb07bd2837991ac7c48b813b00bb12ac197efd813a18daeb`
- Manifest: `data/manifests/baseline_5class_seed42.csv`
- Manifest SHA-256: `73d19e048e742fdf616cbbc1f037efa009ea329ec600acef329f2a5bc7df87ea`
- Checkpoint and full metrics: untracked under `artifacts/baseline_5class_reproduction/`
- Tracked summary: `reports/baseline_reproduction_2026-07-12.json`

## Environment

- Python 3.13.5
- PyTorch 2.12.0
- Torchvision 0.27.0
- scikit-learn 1.9.0
- NumPy 2.4.5
- Pillow 12.2.0
- Device: CPU
- DataLoader workers: 0

A two-worker attempt failed because the execution sandbox disallowed PyTorch shared-memory management. Zero workers is the documented portable default and completed successfully.

## Dataset audit summary

- 2,100 RGB TIFF images in 21 balanced classes
- 2,056 images are 256×256; 44 have nearby but different dimensions
- 2,100 unique SHA-256 hashes; no exact duplicate groups
- 31 global perceptual-similarity candidates at difference-hash distance ≤4
- 14 candidates involve the historical five classes
- 8 five-class candidates cross historical split boundaries

Visual inspection confirmed at least one same-scene aircraft pair and multiple closely related beach scenes. The exact historical split remains necessary for assignment parity, but résumé-grade results require a group-aware sensitivity split.

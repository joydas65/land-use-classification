# Supplied Notebook Baseline Audit

Audit date: 12 July 2026  
Source: `notebooks/original/Land_Use_Classification.ipynb`  
Immutable SHA-256: `11ce096e0b2a0fd007bd2c9a50c7896cf4f85a024e60153b84086655d108456f`

## Verified baseline facts

- Dataset description: 2,100 TIFF images and 21 classes, 100 images per class.
- Executed model scope: 5 classes and 500 images.
- Selected classes: agricultural, airplane, baseballdiamond, beach, buildings.
- Split: 350/75/75 using seed 42 and stratification.
- Model: three convolution blocks followed by adaptive pooling and two linear layers.
- Trainable parameters: 102,277.
- Training configuration: Adam, learning rate 0.001, batch size 32, 10 epochs.
- Observed test accuracy: 74.67%.
- Observed test macro F1: 0.733.

The observed metrics come from a test set of only 75 images. They are accepted as historical baseline evidence, not as a high-confidence estimate of generalization.

## Findings requiring remediation

### ART-001 — Run artifacts lack provenance

The checkpoint stores only weights at a hard-coded location. It omits configuration, label mapping, selected epoch, validation metric, dataset fingerprint, framework version, and split manifest. The reusable runner stores this metadata alongside a hashed manifest and metrics JSON.

### CODE-001 — Notebook structure duplicates and shadows state

The notebook imports unused APIs, defines unused image-loading helpers, overwrites the 21-class `classes` value with the five-class subset, and later reuses `all_labels` for prediction labels. This makes cell-order mistakes harder to detect. Reusable modules now use explicit names and typed values.

### DATA-001 — The executed task is narrower than the dataset description

The introduction prominently describes 21 classes, while `NUM_CLASSES_TO_USE = 5` silently narrows the trained model to the alphabetically first five. All result descriptions must explicitly say “five-class baseline.” The future 21-class model is a separate experiment.

### DATA-002 — Image dimensions are not uniform in the verified archive

All 2,100 files are RGB TIFF images, but only 2,056 are exactly 256×256. The remaining 44 span several nearby dimensions from 242×256 through 257×257. The baseline resize transform prevents a tensor-shape failure, but the notebook's unconditional 256×256 description is inaccurate. The dataset audit records the complete distribution.

### DOC-001 — Narrative and comments are incomplete or unverifiable

The markdown fragment “For five” is incomplete. The estimated runtime is produced by an arithmetic heuristic rather than measurement. “Best Training Accuracy” is simply the maximum training epoch and is not the selected checkpoint criterion. Documentation now distinguishes observed facts, estimates, and future targets.

### EVAL-001 — Evaluation uncertainty is not quantified

The test set contains 75 images, 15 per class. The notebook reports a single split and seed without confidence intervals, repeated runs, calibration, duplicate checks, or robustness analysis. Test data is otherwise kept separate from training and validation, which is a sound part of the baseline.

### LEAK-001 — The historical split separates visually related scenes

All 2,100 files have unique SHA-256 hashes, but 64-bit difference-hash screening produced 31 perceptual-similarity candidates. Fourteen involve the selected five classes and eight cross historical split boundaries. Visual review confirmed that `airplane01.tif` and `airplane02.tif` depict the same aircraft/runway scene with different crop content; several beach pairs show nearly adjacent shoreline scenes. The historical manifest is retained only for exact baseline parity. Credible portfolio results must also use a group-aware split that keeps verified related scenes together.

### PATH-001 — Runtime paths are environment-specific

The dataset and checkpoint are written under `/home`, which is not portable across Colab, macOS, Windows, or container execution. The new runner accepts paths through command-line arguments and uses `map_location` when loading checkpoints.

### PORT-001 — Environment setup mutates the active interpreter

The notebook installs packages at runtime with `--break-system-packages`. Dependency versions are not pinned or recorded. The repository declares dependencies in `pyproject.toml`; run metadata will record the actual environment before submission.

### REPRO-001 — Randomness and splits are only partially controlled

NumPy and CPU PyTorch seeds are set, but Python randomness, CUDA seeds, worker seeds, deterministic algorithm behavior, and the exact file split are not persisted. The new runner controls these sources and writes a per-file split manifest with SHA-256 hashes.

### SEC-001 — Dataset acquisition is not integrity-protected

The original notebook downloads over unencrypted HTTP, does not call `raise_for_status`, has no timeout, trusts the archive paths, and does not record a checksum. The host timed out during the 12 July reproduction attempt. The repository now defaults to TorchGeo's HTTPS mirror, which documents the archive as redistributed without modification. The exact 332,468,434-byte size and SHA-256 `06c539ef28703a58fb07bd2837991ac7c48b813b00bb12ac197efd813a18daeb` are pinned and verified before extraction. The original HTTP URL requires explicit acknowledgement.

### TEST-001 — No automated tests exist in the supplied project

The notebook has no tests for configuration, class ordering, split overlap, stratification, transform output, architecture, parameter count, checkpoint metadata, or documentation consistency. The repository test suite covers these contracts and includes a training smoke test.

## Accepted baseline versus future improvements

The baseline reproduction may improve portability and determinism, but it must not change the model, five selected classes, preprocessing, split fractions, optimizer, scheduler, or epoch count. Transfer learning, stronger augmentation, calibration, near-duplicate grouping, Grad-CAM, and all-21-class training belong in separately named experiments.

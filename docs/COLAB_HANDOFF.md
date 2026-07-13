# Colab Pro Collaboration Handoff

## Current status

The collaboration handoff is complete. The self-contained notebook ran on an NVIDIA L4, the returned bundle contained all four experiments with no failures, and Codex imported only the validated report, comparison CSV, and evidence figure.

## Completed user run

1. Download the notebook from the repository and upload it to Google Colab.
2. Choose **Runtime → Change runtime type → GPU**. A T4, L4, A100, or other CUDA GPU is suitable.
3. Run all cells in order. Do not change the seed, manifests, class list, or test logic.
4. Confirm the device cell prints `Device: cuda` and a GPU name.
5. Let all four experiment entries finish. A failure is recorded rather than hidden.
6. The final cell downloads `terraclass_colab_results.zip`.
7. The returned ZIP had SHA-256 `2c834a31ad37e07de11681f0e3596040d60f1c18e31142dfcdaa97b7a38837ae`.

## Codex return step

Codex verified archive paths, file hashes, environment metadata, manifest hashes, completed and failed runs, checkpoint-selection logic, test metrics, per-class supports, confusion matrices, comparison CSV values, and figure integrity. Evidence commit `414233c8471ea961bfd9406a33f54b427e75ab49` anchors the imported results.

## Claim control

EfficientNet-B0 is now claimable as a completed comparison: it matched ResNet18 classification metrics with about 64% fewer parameters. ResNet18 remains the final IIT model for the documented loss, historical-runtime, and cross-runtime reproducibility reasons. Every résumé bullet must retain the “500-image, five-class subset” scope.

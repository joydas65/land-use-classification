# Colab Pro Collaboration Handoff

## Current status

Codex has built and locally audited `notebooks/TerraClass_IITK_Colab_Submission.ipynb`. The notebook is self-contained: it does not clone the private repository and does not request a Kaggle, GitHub, or Google credential. Its only network downloads are the checksum-pinned public dataset archive and official torchvision ImageNet weights.

## User run

1. Download the notebook from the repository and upload it to Google Colab.
2. Choose **Runtime → Change runtime type → GPU**. A T4, L4, A100, or other CUDA GPU is suitable.
3. Run all cells in order. Do not change the seed, manifests, class list, or test logic.
4. Confirm the device cell prints `Device: cuda` and a GPU name.
5. Let all four experiment entries finish. A failure is recorded rather than hidden.
6. The final cell downloads `terraclass_colab_results.zip`.
7. Attach that ZIP to this Codex task. Do not send a checkpoint yet; the small results bundle is enough for metric verification.

## Codex return step

Codex will verify archive contents, environment metadata, manifest hashes, completed and failed runs, test metrics, confusion matrices, and result/documentation consistency. It will then update the final notebook outputs, replace the submission-commit placeholder, rerun the complete repository gate, and push the final IIT submission commit.

## Claim control

No new EfficientNet result is claimable until it appears in the returned `colab_run_report.json`. The existing ResNet18 results remain the selected evidence unless the complete GPU comparison supports a different choice. Every résumé bullet must retain the “500-image, five-class subset” scope.

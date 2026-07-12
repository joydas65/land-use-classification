# Transfer-Learning Results

## Selected model

ResNet18 with ImageNet weights is the selected model for the IIT Kanpur notebook and first web-app version. It completed both required five-class tracks and reached the same ceiling metrics; the simpler completed evidence is preferable to an unfinished alternative.

| Model | Split | Selected checkpoint | Test accuracy | Test macro F1 | Balanced accuracy | Top-3 accuracy |
|---|---|---:|---:|---:|---:|---:|
| Supplied custom CNN | Historical | Supplied epoch 10 output | 74.67% | 0.733 | Not reported | Not reported |
| ResNet18 transfer learning | Historical | Global epoch 5 | 100.00% | 1.000 | 100.00% | 100.00% |
| ResNet18 transfer learning | Group-aware | Global epoch 4 | 100.00% | 1.000 | 100.00% | 100.00% |

The direct historical improvement is **+25.33 percentage points accuracy** and **+0.267 macro F1**. The group-aware result uses a different manifest and is robustness evidence, not another direct baseline comparison.

Both test partitions contain 75 images, 15 per class. Both ResNet18 confusion matrices are diagonal with 15 correct predictions for each of agricultural, airplane, baseballdiamond, beach, and buildings.

## EfficientNet-B0 feasibility result

The historical EfficientNet-B0 CPU run was stopped without test metrics after exceeding the reasonable local execution budget; CUDA and MPS were unavailable. Its group-aware run was therefore not started locally. These are explicitly incomplete experiments and provide no numerical result claim. The checked-in configurations remain ready for a Colab GPU benchmark.

## Interpretation boundary

Perfect performance on this small, visually distinct, balanced five-class subset is not evidence of perfect satellite-scene classification in general. It must never be described as a 21-class UC Merced score. The final notebook should disclose the 500-image scope, the 75-image test size, ImageNet pretraining, and the manually reviewed grouping policy.

The canonical machine-readable evidence is `reports/transfer_learning_results_2026-07-12.json`; checkpoints and raw run artifacts remain untracked.

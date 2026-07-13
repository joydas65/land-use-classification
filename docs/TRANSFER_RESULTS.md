# Transfer-Learning Results

## Selected model

ResNet18 with ImageNet weights is the selected IIT Kanpur model. The complete NVIDIA L4 comparison found no classification-metric difference between ResNet18 and EfficientNet-B0, so selection uses secondary evidence rather than overstating an accuracy advantage.

| Model | Split | Parameters | Epoch | Test loss | Accuracy | Macro F1 |
|---|---|---:|---:|---:|---:|---:|
| Supplied custom CNN | Historical | 102,277 | 10 | Not reported | 74.67% | 0.733 |
| ResNet18 | Historical | 11,179,077 | 4 | 0.0247 | 100.00% | 1.000 |
| ResNet18 | Group-aware | 11,179,077 | 4 | 0.0294 | 100.00% | 1.000 |
| EfficientNet-B0 | Historical | 4,013,953 | 5 | 0.0457 | 100.00% | 1.000 |
| EfficientNet-B0 | Group-aware | 4,013,953 | 4 | 0.0804 | 100.00% | 1.000 |

The direct historical improvement is **+25.33 percentage points accuracy** and **+0.267 macro F1**. The group-aware result uses a different manifest and is robustness evidence, not another direct baseline comparison.

All GPU runs also reached 100.00% balanced accuracy and top-3 accuracy. Every test confusion matrix is diagonal with 15 correct predictions for each of agricultural, airplane, baseballdiamond, beach, and buildings.

## Architecture decision

EfficientNet-B0 matched every classification score with 4,013,953 parameters—about 64% fewer than ResNet18—and is therefore a credible size-efficient alternative. ResNet18 remains selected because it produced lower test loss on both split protocols, completed the historical L4 run in 11.2 seconds versus 36.1 seconds, and agrees with an independent local CPU run. Training duration is not inference latency; the web-app phase must benchmark inference before making a production-serving decision.

## Interpretation boundary

Perfect performance on this small, visually distinct, balanced five-class subset is not evidence of perfect satellite-scene classification in general. It must never be described as a 21-class UC Merced score. The final notebook should disclose the 500-image scope, the 75-image test size, ImageNet pretraining, and the manually reviewed grouping policy.

The canonical GPU evidence is `reports/colab/VERIFICATION.json`, derived from bundle SHA-256 `2c834a31ad37e07de11681f0e3596040d60f1c18e31142dfcdaa97b7a38837ae`. The earlier `reports/transfer_learning_results_2026-07-12.json` remains an immutable record of local CPU feasibility, including the interrupted EfficientNet attempt.

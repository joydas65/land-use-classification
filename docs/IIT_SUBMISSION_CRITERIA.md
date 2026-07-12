# IIT Kanpur Submission Criteria Traceability

## Stated course requirements

The Google Classroom announcement permits one of the eight official sample projects and asks the participant to **improve or modify the model/performance** and submit the **updated Jupyter Notebook** by **20 July 2026**. Land Use Classification is one of those eight projects.

This repository does not invent additional college rules. The items below are engineering evidence that makes the two stated requirements reviewable.

| Stated requirement | Repository evidence | Submission status |
|---|---|---|
| Use an official sample project | Immutable supplied notebook and its SHA-256 | Complete |
| Improve or modify model/performance | Selected ResNet18 replaces the custom CNN and improves historical test accuracy from 74.67% to 100.00% and macro F1 from 0.733 to 1.000 | Complete |
| Submit an updated Jupyter Notebook | A clean Colab notebook will present acquisition, manifests, training, evaluation, and limitations | Pending final experiment results |
| Submit by 20 July 2026 | Dated execution plan and Git history | In progress |

## Acceptance evidence for the final notebook

- Run in a fresh Colab runtime without a local absolute path or embedded credential.
- Identify the official UC Merced dataset source, verified mirror, archive size, and SHA-256.
- Preserve the historical 350/75/75 five-class manifest for the only direct comparison to the supplied 74.67% accuracy and 0.733 macro F1.
- Select checkpoints using validation macro F1, then evaluate the test set once.
- Report accuracy, macro F1, balanced accuracy, top-3 accuracy, confusion matrix, and per-class scores.
- Present the group-aware sensitivity run separately; do not substitute it for the historical comparison.
- Include limitations and explain that the five-class experiment is not a 21-class benchmark.
- Include the submitted Git commit identifier and exact experiment configuration.

The controlled 78.67% reproduction is parity evidence, not the claimed model improvement. The improvement claim will be calculated from the chosen transfer model on the same historical manifest.

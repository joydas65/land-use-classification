# IIT Kanpur Final Submission Checklist

## Submission file

- **Status: submitted by email on 15 July 2026, before the 20 July 2026 deadline.**
- Submitted `notebooks/Improved_Land_Use_Classification_IITK.ipynb` as the email attachment.
- The notebook is the updated version of the official Land Use Classification sample project.

## Internal verification record

These identifiers preserve traceability in the repository and are not required in the notebook or
submission email.

- Evidence commit: `414233c8471ea961bfd9406a33f54b427e75ab49`.
- Returned Colab bundle SHA-256: `2c834a31ad37e07de11681f0e3596040d60f1c18e31142dfcdaa97b7a38837ae`.

## Verified contents

- Checksum-pinned UC Merced acquisition without credentials
- Same historical 5-class, 500-image, 350/75/75 split for direct comparison
- Separate group-aware manifest for leakage sensitivity
- ImageNet-pretrained ResNet18 and EfficientNet-B0 with two-stage fine-tuning
- Validation macro-F1 checkpoint selection and test-once evaluation
- Accuracy, macro F1, balanced accuracy, top-3 accuracy, per-class report, curves, and confusion matrices
- Explicit baseline improvement and architecture decision
- Limitations stating that the result is not a 21-class benchmark or universal satellite-classification claim

## Final result statement

The selected ResNet18 improves the supplied historical test accuracy from 74.67% to 100.00% and macro F1 from 0.733 to 1.000. It also reaches 1.000 macro F1 on the group-aware sensitivity split. Both test sets contain 75 images. EfficientNet-B0 matches the classification metrics with fewer parameters but is retained as the alternative after the documented selection tradeoff.

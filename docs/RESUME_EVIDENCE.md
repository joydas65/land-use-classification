# Résumé Evidence Register

## Claim policy

Only completed, versioned experiment outputs may become numerical résumé claims. Planned architectures, expected gains, and the controlled reproduction must not be presented as the final improvement.

## Skills already evidenced

- Reproducible computer-vision pipeline using PyTorch and torchvision
- Dataset provenance with a pinned 332,468,434-byte archive and SHA-256 verification
- Deterministic stratified manifests for 5 classes and 500 images with a 350/75/75 split
- Leakage analysis using exact hashes, perceptual-hash candidates, manual review, and group-aware splitting
- Two-stage ImageNet transfer-learning design for ResNet18 and EfficientNet-B0
- Validation macro-F1 checkpoint selection and multi-metric test evaluation
- Comparative L4 GPU benchmarking of ResNet18 and EfficientNet-B0 with parameter/loss/runtime tradeoff analysis
- Automated tests and a cross-artifact audit covering code, configurations, manifests, comments, and documentation

## Supported numerical claim

> Built a reproducible PyTorch land-use classifier on a balanced 500-image, five-class UC Merced subset; improved historical test accuracy from 74.67% to 100.00% and macro F1 from 0.733 to 1.000 using ImageNet-pretrained ResNet18, then reproduced 1.000 macro F1 on a separately generated, leakage-controlled group split.

Optional second bullet:

> Benchmarked ResNet18 and EfficientNet-B0 across historical and leakage-controlled splits on an NVIDIA L4; both achieved 1.000 macro F1, and selected ResNet18 using test loss, runtime, reproducibility, and model-size tradeoffs rather than accuracy alone.

Keep the dataset scope in the same bullet as the perfect score. Do not shorten this into a generic “100% satellite classifier” claim.

## Senior-engineering extension

After the model comparison is frozen, expose the selected checkpoint through a tested inference service and a small web interface. The deployable story should include model/version metadata, input validation, latency measurement, structured error handling, containerization, CI, and drift-ready telemetry. These are future deliverables and are not yet résumé claims.

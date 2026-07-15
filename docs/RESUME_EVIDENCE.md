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
- Hash-verified promotion from a rich training checkpoint to a minimal weights-only serving artifact
- Bounded, thread-safe image inference with strict model/provenance validation and structured top-k output
- Reproducible local CPU latency benchmarking over all 75 leakage-controlled test images
- Typed, versioned FastAPI inference service with bounded multipart uploads, structured errors, CORS, request IDs, structured request logs, and liveness/readiness probes
- Responsive TypeScript browser interface with client-side validation, model readiness, ranked confidence visualization, request provenance, and explicit model-scope communication
- Production web build, server-render contract tests, API integration tests, and dependency security review

## Supported numerical claim

> Built a reproducible PyTorch land-use classifier on a balanced 500-image, five-class UC Merced subset; improved historical test accuracy from 74.67% to 100.00% and macro F1 from 0.733 to 1.000 using ImageNet-pretrained ResNet18, then reproduced 1.000 macro F1 on a separately generated, leakage-controlled group split.

Optional second bullet:

> Benchmarked ResNet18 and EfficientNet-B0 across historical and leakage-controlled splits on an NVIDIA L4; both achieved 1.000 macro F1, and selected ResNet18 using test loss, runtime, reproducibility, and model-size tradeoffs rather than accuracy alone.

Optional serving bullet:

> Built a hash-verified, weights-only PyTorch inference layer with bounded image validation and model-version metadata; measured 14.6 ms median and 24.0 ms p95 local CPU latency across 75 leakage-controlled test requests.

Optional application bullet:

> Engineered a versioned FastAPI and TypeScript inference application around a PyTorch ResNet18 model, including bounded uploads, structured errors, health probes, request tracing, ranked probabilities, responsive UI, and automated API/render tests.

Keep the dataset scope in the same bullet as the perfect score. Do not shorten this into a generic “100% satellite classifier” claim.

## Senior-engineering extension

The model/version contract, input validation, restricted artifact loading, local latency benchmark,
typed HTTP service, structured API errors, request tracing, health probes, and browser interface are
complete. Containerization, CI, concurrency/load tests, deployed API evidence, production SLOs, and
drift-ready telemetry remain future deliverables and must not yet be claimed.

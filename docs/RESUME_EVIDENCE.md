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
- Responsive Tailwind CSS/Next.js browser interface with client-side validation, model readiness, ranked confidence visualization, request provenance, and explicit model-scope communication
- Public Vercel frontend deployment, native Next.js and worker builds, server-render contract tests, API integration tests, and dependency security review
- Published, versioned GitHub model release with checksum-pinned HTTPS distribution, bounded streaming, byte-count verification, and atomic promotion
- Non-root CPU container and Cloud Run resource/probe contracts with explicit CORS and bounded inference capacity
- Real-model HTTP concurrency testing across 60 measured requests with unique request tracing and zero failures
- Separate Python, web, container-contract, image-provenance, and SBOM workflow definitions

## Supported numerical claim

> Built a reproducible PyTorch land-use classifier on a balanced 500-image, five-class UC Merced subset; improved historical test accuracy from 74.67% to 100.00% and macro F1 from 0.733 to 1.000 using ImageNet-pretrained ResNet18, then reproduced 1.000 macro F1 on a separately generated, leakage-controlled group split.

Optional second bullet:

> Benchmarked ResNet18 and EfficientNet-B0 across historical and leakage-controlled splits on an NVIDIA L4; both achieved 1.000 macro F1, and selected ResNet18 using test loss, runtime, reproducibility, and model-size tradeoffs rather than accuracy alone.

Optional serving bullet:

> Built a hash-verified, weights-only PyTorch inference layer with bounded image validation and model-version metadata; measured 14.6 ms median and 24.0 ms p95 local CPU latency across 75 leakage-controlled test requests.

Optional application bullet:

> Engineered a versioned FastAPI inference service around a PyTorch ResNet18 model and deployed its responsive Tailwind CSS/Next.js frontend to Vercel, with bounded uploads, structured errors, health probes, request tracing, ranked probabilities, and automated API/render tests.

Optional production-readiness bullet:

> Hardened a PyTorch inference API with checksum-pinned model distribution, a non-root CPU container contract, bounded asynchronous capacity, and repeatable HTTP load testing; completed 60 steady-state requests without failures and measured 52.9 requests/second peak local throughput.

Optional model-release bullet:

> Published a versioned ResNet18 serving artifact through a public GitHub release and verified its 44,795,275-byte download against a pinned SHA-256 before allowing production-image assembly.

Keep the dataset scope in the same bullet as the perfect score. Do not shorten this into a generic “100% satellite classifier” claim.

## Senior-engineering extension

The model/version contract, input validation, restricted artifact loading, local sequential and HTTP
concurrency benchmarks, typed service, bounded capacity, container configuration, release contract,
workflow definitions, health probes, and public Vercel frontend are complete. GitHub CI has built the
artifact-free runtime container and passed the Python and web gates. The public, checksum-verified
model release is complete. The model-bearing production image, Cloud Run API, production SLOs, and
drift-ready telemetry remain future deliverables and must not yet be claimed.

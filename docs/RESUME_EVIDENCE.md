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
- Public Linux/AMD64 OCI image with semantic/source tags, immutable digest verification, SPDX SBOM, and SLSA v1 provenance
- Immutable-digest Cloud Run deployment with a dedicated no-role runtime identity, bounded CPU/memory/concurrency, scale-to-zero, startup/liveness probes, restricted Vercel CORS, and production request evidence
- Integrated Vercel-to-Cloud Run deployment verified through browser readiness, a correct production prediction, and a zero-failure 60-request production load probe
- Metric-confirmed scale-to-zero verification correlated across client timing, Cloud Run request logs, and an autoscaling instance-start event
- Privacy-allowlisted JSON prediction telemetry with independent service/model versioning and tests that exclude filenames, image content/hashes, network identity, and user agents
- Deployed Cloud Monitoring 5xx-ratio and warm-container p95 policies, with candidate objectives kept separate from achieved SLO and drift claims
- Privacy-preserving production ML review pipeline with strict log/review schemas, aggregate class/confidence/latency profiles, Jensen–Shannon comparison, sample-size gates, and reviewed-sample accuracy/macro-F1
- API-validated and deployed Cloud Monitoring dashboard for request rate, p95 latency, instance state, and allowlisted prediction logs
- Validation-only confidence calibration analysis using NLL, multiclass Brier score, 10-bin ECE, predictive entropy, and selective-risk curves on the untouched group-aware test set
- Deterministic Grad-CAM review across all five classes, with artifact/manifest hashes and an explicit distinction between qualitative localization and causal explanation
- Model-governance decision to reject temperature promotion after the fit reached its lower bound on a small, perfectly classified validation split
- Independent cross-domain calibration evaluation using 500 RESISC45 calibration images, a separate 500-image test set, a 500-replicate bootstrap, and five-fold stability analysis
- Separate OOD benchmark over 5,457 unmapped land-use scenes, with explicit evidence that temperature scaling and softmax confidence are insufficient OOD detectors
- Model-governance decision to reject a statistically stable external calibration candidate after it regressed UC Merced NLL and failed domain, mapping, and noncommercial-license promotion boundaries
- Deterministic corruption benchmark covering brightness, contrast, blur, noise, and JPEG compression at three severities, with clean, mean, per-family, per-severity, and worst-condition metrics
- Leakage-safe TTA selection that evaluated four-view logit averaging on validation, rejected the candidate before test evaluation, and preserved the production inference policy

## Supported numerical claim

> Built a reproducible PyTorch land-use classifier on a balanced 500-image, five-class UC Merced subset; improved historical test accuracy from 74.67% to 100.00% and macro F1 from 0.733 to 1.000 using ImageNet-pretrained ResNet18, then reproduced 1.000 macro F1 on a separately generated, leakage-controlled group split.

Optional second bullet:

> Benchmarked ResNet18 and EfficientNet-B0 across historical and leakage-controlled splits on an NVIDIA L4; both achieved 1.000 macro F1, and selected ResNet18 using test loss, runtime, reproducibility, and model-size tradeoffs rather than accuracy alone.

Optional serving bullet:

> Built a hash-verified, weights-only PyTorch inference layer with bounded image validation and model-version metadata; measured 14.6 ms median and 24.0 ms p95 local CPU latency across 75 leakage-controlled test requests.

Optional application bullet:

> Engineered and deployed an end-to-end ResNet18 inference application using FastAPI on Google Cloud Run and a responsive Tailwind CSS/Next.js frontend on Vercel, with bounded uploads, structured errors, health probes, request tracing, ranked probabilities, and automated API/render tests.

Optional production-readiness bullet:

> Hardened a PyTorch inference API with checksum-pinned model distribution, a non-root CPU container contract, bounded asynchronous capacity, and repeatable HTTP load testing; completed 60 steady-state requests without failures and measured 52.9 requests/second peak local throughput.

Optional model-release bullet:

> Published a versioned ResNet18 serving artifact through a public GitHub release and verified its 44,795,275-byte download against a pinned SHA-256 before allowing production-image assembly.

Optional container-release bullet:

> Published a public Linux/AMD64 ML inference container to GHCR, pinned semantic and source-commit tags to one immutable OCI digest, and verified attached SPDX SBOM and SLSA v1 provenance attestations.

Optional production-deployment bullet:

> Deployed the immutable ResNet18 container to Cloud Run under a dedicated least-privilege runtime identity, with scale-to-zero, bounded concurrency, health probes, and origin-restricted CORS; connected the Vercel UI and completed a 60-request production load probe with zero failures and 365.2 ms p95 latency at concurrency four.

Optional production-observability bullet:

> Added privacy-allowlisted structured prediction telemetry and Cloud Monitoring 5xx/p95 alert policies to a versioned ML API; verified one scale-from-zero request at 11.013 seconds by correlating a zero-instance metric, client timing, and the Cloud Run autoscaling log.

Optional production-ML monitoring bullet:

> Built a privacy-preserving production review pipeline that validates Cloud Logging and human-label records, aggregates class/confidence/latency profiles, compares eligible windows with Jensen–Shannon divergence, and calculates reviewed-sample accuracy and macro-F1 behind explicit 100-event evidence gates.

Optional model-quality bullet:

> Evaluated a ResNet18 classifier with validation-only temperature scaling, NLL/Brier/ECE,
> predictive entropy, selective-risk curves, and deterministic Grad-CAM across five land-use
> classes; rejected calibration deployment when the bounded fit proved unstable on the small,
> perfectly classified validation split.

Optional calibration-governance bullet:

> Built an independent calibration and OOD evaluation pipeline for a ResNet18 land-use classifier;
> fit a stable temperature on 500 cross-domain images, reduced untouched-test ECE from 0.220 to
> 0.066 without changing accuracy, quantified uncertainty with 500 bootstrap fits and five-fold
> validation, and rejected global deployment after detecting a 10.9× UC Merced NLL regression.

Optional robustness bullet:

> Built a deterministic corruption benchmark for a ResNet18 land-use classifier across 1,125
> corrupted test evaluations; measured 0.992 mean macro F1 and 0.918 worst-case macro F1 under
> severe Gaussian blur, and rejected four-view test-time augmentation on validation before opening
> candidate test metrics.

Keep the dataset scope in the same bullet as the perfect score. Do not shorten this into a generic “100% satellite classifier” claim.

## Senior-engineering extension

The model/version contract, input validation, restricted artifact loading, local sequential and HTTP
concurrency benchmarks, typed service, bounded capacity, container configuration, release contract,
workflow definitions, health probes, and public Vercel frontend are complete. GitHub CI has built the
artifact-free runtime container and passed the Python and web gates. The public, checksum-verified
model release, model-bearing production image, Cloud Run API, production load probe, integrated
Vercel deployment, scale-from-zero measurement, structured prediction telemetry, and incident
policies are complete. The production review analyzer and operations dashboard are also complete,
including strict privacy schemas, sample-size refusal, Jensen–Shannon comparison, and reviewed-sample
metrics. Notification routing still needs an owner-approved destination. The candidate objectives
need sufficient historical traffic before they can be described as achieved SLOs, and credible drift
validation still requires two representative windows plus at least 100 labeled or human-reviewed
production examples. The model-quality phase is also complete: the original softmax remains in
production. An external RESISC45 follow-up identified a statistically stable candidate, but it is
not deployed because it materially regresses UC Merced calibration, uses proxy class mappings, and
depends on evaluation data whose redistributor states a noncommercial license.
The corruption-robustness phase is also complete. The single-view model retained 0.991879 mean
macro F1 across the 15 synthetic test conditions, while four-view TTA was rejected on validation
and was not evaluated as a candidate on the test split. These are small five-class synthetic-stress
results, not production-traffic or adversarial-robustness claims.

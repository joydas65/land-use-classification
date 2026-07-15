# Inference Foundation — 15 July 2026

## Outcome

The selected leakage-controlled ResNet18 checkpoint is now promoted through a separate serving
contract instead of being loaded directly by an application. The contract fixes the model ID,
semantic version, architecture, class order, preprocessing, training-manifest provenance, selected
epoch, input limits, and both checkpoint hashes.

The IIT submission notebook remains unchanged. This is a separate portfolio engineering phase.

## Artifact promotion

The original training checkpoint contains optimizer-independent model weights plus rich experiment
metadata. Before reading it, the promotion script verifies its SHA-256 and uses PyTorch's restricted
weights-only loader with one explicitly allowed PyTorch version type. It then writes a minimal
artifact atomically and proves that the result can be loaded with the default restricted type set.

| Artifact | SHA-256 |
|---|---|
| Group-aware training checkpoint | `d3c22a5cf0e3f96c124f4c9e5b7b1200f696fb9b8bd95d6d79d8330035bf4067` |
| ResNet18 serving artifact v1.0.0 | `b4e8522aa702ef8d6670acd58e37ef2dd8948148a4fa9f07b88c23953473e523` |

The binary artifacts remain excluded from Git. The 16 July phase added a versioned GitHub release
contract and an atomic downloader that verifies HTTPS, byte count, and the same SHA-256 before the
model can enter a production image. Public release publication is still pending approval.

Recreate the local serving artifact from the audited training checkpoint:

```bash
PYTHONPATH=src python scripts/export_serving_model.py \
  --source artifacts/resnet18_group_aware/best_model.pth \
  --output artifacts/serving/resnet18_group_aware_v1.pt \
  --expected-source-sha256 d3c22a5cf0e3f96c124f4c9e5b7b1200f696fb9b8bd95d6d79d8330035bf4067 \
  --model-id terraclass-resnet18-group-aware \
  --model-version 1.0.0
```

## Inference contract

`TerraClassPredictor` verifies the serving-artifact hash before deserialization, loads only a
weights-only artifact, checks every identity and provenance field, reconstructs the model without a
network download, validates the state dictionary strictly, and applies the versioned evaluation
transform. It also provides:

- bounded encoded-image size and decoded-pixel limits;
- Pillow-based format detection and decoding rather than trusting a filename;
- top-k validation and calibrated softmax probabilities;
- a lock around model execution for safe use by a threaded API process;
- structured model/version, image-dimension, and latency fields in each prediction.

The classifier supports only `agricultural`, `airplane`, `baseballdiamond`, `beach`, and
`buildings`. It must reject or clearly qualify images outside that five-class scope.

## Local CPU benchmark

The benchmark used all 75 test images from the leakage-controlled manifest. Ten warm-up requests
were excluded, followed by 75 sequential measured requests. Input files were read into memory before
measurement. Request latency covers image decoding, validation, preprocessing, and inference; it
does not include network transport or concurrent server overhead.

| Measurement | Result |
|---|---:|
| Median request latency | 14.6 ms |
| p95 request latency | 24.0 ms |
| Mean request latency | 16.2 ms |
| Sequential throughput | 61.6 requests/second |
| Model load time | 180.5 ms |
| Prediction accuracy sanity check | 100.0% over 75 requests |

These are local CPU measurements on the recorded macOS/PyTorch environment, not production service
SLOs. API serialization, network latency, concurrency, memory, cold-container startup, and sustained
load still require measurement in the deployment environment.

Re-run the benchmark:

```bash
PYTHONPATH=src python scripts/benchmark_inference.py \
  --project-root . \
  --device cpu \
  --warmup 10 \
  --iterations 75
```

## Production-readiness continuation

The 15 July application phase added a typed HTTP API, responsive browser interface, request IDs,
structured logs, and health/readiness probes. The 16 July phase added the non-root container
contract, bounded inference queue, release manifest, CI definitions, Cloud Run template, and real
HTTP concurrency benchmark; see `docs/PRODUCTION_INFERENCE.md`. Publishing the release and image,
deploying the API, collecting Cloud Run evidence, and connecting Vercel remain the next handoff.

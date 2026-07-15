"use client";

import { ChangeEvent, DragEvent, FormEvent, useEffect, useMemo, useState } from "react";

type Benchmark = {
  device: string;
  measured_requests: number;
  p50_request_latency_ms: number;
  p95_request_latency_ms: number;
  throughput_requests_per_second: number;
};

type ModelMetadata = {
  model_id: string;
  model_version: string;
  architecture: string;
  class_names: string[];
  selected_epoch: number;
  verified_test_accuracy: number;
  verified_test_macro_f1: number;
  default_top_k: number;
  max_image_bytes: number;
  benchmark: Benchmark;
  scope: string;
};

type RankedPrediction = {
  rank: number;
  class_name: string;
  probability: number;
};

type PredictionResult = {
  request_id: string;
  model_id: string;
  model_version: string;
  predicted_class: string;
  confidence: number;
  predictions: RankedPrediction[];
  latency_ms: number;
  image_width: number;
  image_height: number;
};

type ApiError = {
  error?: {
    message?: string;
  };
};

const FALLBACK_LIMIT_BYTES = 10 * 1024 * 1024;
const API_BASE_URL = (
  process.env.NEXT_PUBLIC_TERRACLASS_API_URL ?? "http://localhost:8000"
).replace(/\/$/, "");

const formatClassName = (value: string) =>
  value
    .replace(/([a-z])([A-Z])/g, "$1 $2")
    .replace(/[_-]+/g, " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());

const formatPercentage = (value: number) => `${(value * 100).toFixed(1)}%`;

export function TerraClassApp() {
  const [metadata, setMetadata] = useState<ModelMetadata | null>(null);
  const [modelStatus, setModelStatus] = useState<"checking" | "ready" | "offline">(
    "checking",
  );
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [topK, setTopK] = useState(3);
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);

  useEffect(() => {
    const controller = new AbortController();

    async function loadModelMetadata() {
      try {
        const [modelResponse, readinessResponse] = await Promise.all([
          fetch(`${API_BASE_URL}/api/v1/model`, { signal: controller.signal }),
          fetch(`${API_BASE_URL}/api/v1/health/ready`, { signal: controller.signal }),
        ]);
        if (!modelResponse.ok || !readinessResponse.ok) {
          throw new Error("The model service is not ready.");
        }
        const model = (await modelResponse.json()) as ModelMetadata;
        setMetadata(model);
        setTopK(model.default_top_k);
        setModelStatus("ready");
      } catch (loadError) {
        if ((loadError as Error).name !== "AbortError") {
          setModelStatus("offline");
        }
      }
    }

    void loadModelMetadata();
    return () => controller.abort();
  }, []);

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  const allowedTypes = useMemo(
    () => new Set(["image/jpeg", "image/png", "image/tiff", "image/webp"]),
    [],
  );

  function selectFile(candidate: File | null) {
    setPrediction(null);
    setError(null);
    if (!candidate) return;

    if (!allowedTypes.has(candidate.type)) {
      setError("Choose a PNG, JPEG, TIFF, or WebP satellite image.");
      return;
    }
    const maxBytes = metadata?.max_image_bytes ?? FALLBACK_LIMIT_BYTES;
    if (candidate.size > maxBytes) {
      setError(`The image must be smaller than ${(maxBytes / 1_000_000).toFixed(0)} MB.`);
      return;
    }

    setFile(candidate);
    setPreviewUrl(URL.createObjectURL(candidate));
  }

  function handleFileInput(event: ChangeEvent<HTMLInputElement>) {
    selectFile(event.target.files?.[0] ?? null);
  }

  function handleDrop(event: DragEvent<HTMLLabelElement>) {
    event.preventDefault();
    setIsDragging(false);
    selectFile(event.dataTransfer.files?.[0] ?? null);
  }

  async function runPrediction(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!file || modelStatus !== "ready") return;

    setIsSubmitting(true);
    setError(null);
    setPrediction(null);
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/predictions?top_k=${topK}`, {
        method: "POST",
        body: formData,
      });
      const payload = (await response.json()) as PredictionResult | ApiError;
      if (!response.ok) {
        const apiError = payload as ApiError;
        throw new Error(apiError.error?.message ?? "The image could not be classified.");
      }
      setPrediction(payload as PredictionResult);
    } catch (requestError) {
      setError(
        requestError instanceof Error
          ? requestError.message
          : "The image could not be classified.",
      );
    } finally {
      setIsSubmitting(false);
    }
  }

  const classes = metadata?.class_names ?? [
    "agricultural",
    "airplane",
    "baseballdiamond",
    "beach",
    "buildings",
  ];

  return (
    <main>
      <header className="site-header">
        <a className="brand" href="#top" aria-label="TerraClass home">
          <span className="brand-mark" aria-hidden="true">TC</span>
          <span>
            <strong>TerraClass</strong>
            <small>Land intelligence, made visible</small>
          </span>
        </a>
        <nav aria-label="Primary navigation">
          <a href="#classifier">Classifier</a>
          <a href="#method">Method</a>
          <a href="#evidence">Evidence</a>
        </nav>
        <div className={`service-state service-state--${modelStatus}`} role="status">
          <span aria-hidden="true" />
          {modelStatus === "ready"
            ? "Model ready"
            : modelStatus === "offline"
              ? "API offline"
              : "Checking model"}
        </div>
      </header>

      <section className="hero" id="top">
        <div className="hero-copy">
          <p className="eyebrow">Satellite scene intelligence</p>
          <h1>See what the model sees.</h1>
          <p className="hero-summary">
            Upload an aerial scene and inspect how a leakage-aware ResNet18 distinguishes
            five land-use classes. Every result includes confidence, ranked alternatives,
            and model provenance.
          </p>
          <a className="text-link" href="#classifier">Try the classifier <span aria-hidden="true">↓</span></a>
        </div>
        <div className="hero-map" aria-label="Abstract satellite landscape illustration">
          <div className="map-coordinate map-coordinate--top">26.4499° N</div>
          <div className="map-block map-block--fields" />
          <div className="map-block map-block--coast" />
          <div className="map-block map-block--built" />
          <div className="map-reticle" aria-hidden="true"><span /></div>
          <div className="map-coordinate map-coordinate--bottom">80.3319° E</div>
        </div>
        <dl className="hero-metrics" id="evidence">
          <div><dt>500</dt><dd>balanced images</dd></div>
          <div><dt>5</dt><dd>land-use classes</dd></div>
          <div><dt>100%</dt><dd>scoped test accuracy</dd></div>
          <div><dt>75</dt><dd>held-out test images</dd></div>
        </dl>
      </section>

      <section className="classifier-section" id="classifier">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Interactive inference</p>
            <h2>Classify a satellite scene</h2>
          </div>
          <p>
            Supported formats: PNG, JPEG, TIFF and WebP. Images remain in memory only for
            the duration of the request.
          </p>
        </div>

        <div className="workspace">
          <form className="upload-panel" onSubmit={runPrediction}>
            <label
              className={`drop-zone ${isDragging ? "drop-zone--active" : ""}`}
              onDragEnter={() => setIsDragging(true)}
              onDragLeave={() => setIsDragging(false)}
              onDragOver={(event) => event.preventDefault()}
              onDrop={handleDrop}
            >
              <input
                type="file"
                accept="image/png,image/jpeg,image/tiff,image/webp"
                onChange={handleFileInput}
              />
              {previewUrl ? (
                // A blob URL is required for a local preview before upload.
                // eslint-disable-next-line @next/next/no-img-element
                <img src={previewUrl} alt="Selected satellite scene preview" />
              ) : (
                <span className="drop-zone-empty">
                  <strong>Drop an aerial image here</strong>
                  <span>or click to choose a file</span>
                </span>
              )}
              {file && <span className="file-chip">{file.name}</span>}
            </label>

            <div className="form-controls">
              <label>
                Ranked results
                <select value={topK} onChange={(event) => setTopK(Number(event.target.value))}>
                  {classes.map((_, index) => (
                    <option key={index + 1} value={index + 1}>{index + 1}</option>
                  ))}
                </select>
              </label>
              <button type="submit" disabled={!file || modelStatus !== "ready" || isSubmitting}>
                {isSubmitting ? "Analysing scene…" : "Run classification"}
              </button>
            </div>
            {modelStatus === "offline" && (
              <p className="notice notice--warning" role="alert">
                The interface is ready, but the inference API is not reachable at
                {` ${API_BASE_URL}`}.
              </p>
            )}
            {error && <p className="notice notice--error" role="alert">{error}</p>}
          </form>

          <section className="result-panel" aria-live="polite" aria-busy={isSubmitting}>
            {prediction ? (
              <>
                <div className="result-lead">
                  <p className="eyebrow">Top prediction</p>
                  <h3>{formatClassName(prediction.predicted_class)}</h3>
                  <strong>{formatPercentage(prediction.confidence)}</strong>
                  <span>model confidence</span>
                </div>
                <ol className="ranked-list">
                  {prediction.predictions.map((item) => (
                    <li key={item.class_name}>
                      <div>
                        <span>{String(item.rank).padStart(2, "0")}</span>
                        <strong>{formatClassName(item.class_name)}</strong>
                        <b>{formatPercentage(item.probability)}</b>
                      </div>
                      <span className="probability-track">
                        <span style={{ width: `${item.probability * 100}%` }} />
                      </span>
                    </li>
                  ))}
                </ol>
                <dl className="request-facts">
                  <div><dt>Inference</dt><dd>{prediction.latency_ms.toFixed(1)} ms</dd></div>
                  <div><dt>Image</dt><dd>{prediction.image_width} × {prediction.image_height}</dd></div>
                  <div><dt>Model</dt><dd>{prediction.model_version}</dd></div>
                  <div><dt>Request</dt><dd title={prediction.request_id}>{prediction.request_id.slice(0, 8)}</dd></div>
                </dl>
              </>
            ) : (
              <div className="result-empty">
                <span className="result-index">01—05</span>
                <h3>Your analysis will appear here.</h3>
                <p>
                  Select a scene to see the predicted class, calibrated probabilities,
                  dimensions and request-level latency.
                </p>
                <div className="class-key">
                  {classes.map((className, index) => (
                    <span key={className}><b>{String(index + 1).padStart(2, "0")}</b>{formatClassName(className)}</span>
                  ))}
                </div>
              </div>
            )}
          </section>
        </div>
      </section>

      <section className="method-section" id="method">
        <div>
          <p className="eyebrow">Built for credible evaluation</p>
          <h2>Strong results, stated at the right scope.</h2>
        </div>
        <div className="method-grid">
          <article>
            <span>01</span>
            <h3>Leakage-aware split</h3>
            <p>Perceptually related scenes are grouped before splitting, reducing optimistic evaluation from near-duplicate imagery.</p>
          </article>
          <article>
            <span>02</span>
            <h3>Transfer learning</h3>
            <p>ResNet18 was selected against EfficientNet-B0 using verified accuracy, macro F1, loss and runtime evidence.</p>
          </article>
          <article>
            <span>03</span>
            <h3>Reproducible serving</h3>
            <p>Versioned configuration, artifact hashes and typed responses connect the trained model to every prediction.</p>
          </article>
        </div>
        <aside className="scope-note">
          <strong>Scope matters.</strong>
          <p>{metadata?.scope ?? "This model covers a balanced five-class, 500-image subset of UC Merced. It is not a universal satellite classifier."}</p>
        </aside>
      </section>

      <footer>
        <div><strong>TerraClass</strong><span>Applied ML, from experiment to interface.</span></div>
        <p>ResNet18 · PyTorch · FastAPI · leakage-aware evaluation</p>
      </footer>
    </main>
  );
}

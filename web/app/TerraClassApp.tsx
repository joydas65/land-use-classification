"use client";

import { ChangeEvent, DragEvent, FormEvent, useEffect, useMemo, useRef, useState } from "react";

import { createImagePreviewUrl } from "@/lib/image-preview";

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
  process.env.NEXT_PUBLIC_TERRACLASS_API_URL ??
  (process.env.NODE_ENV === "development" ? "http://localhost:8000" : "")
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
    API_BASE_URL ? "checking" : "offline",
  );
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isPreparingPreview, setIsPreparingPreview] = useState(false);
  const [topK, setTopK] = useState(3);
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const previewRequestId = useRef(0);

  useEffect(() => {
    if (!API_BASE_URL) {
      return;
    }

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

  useEffect(() => {
    return () => {
      previewRequestId.current += 1;
    };
  }, []);

  const allowedTypes = useMemo(
    () => new Set(["image/jpeg", "image/png", "image/tiff", "image/webp"]),
    [],
  );

  async function selectFile(candidate: File | null) {
    const requestId = previewRequestId.current + 1;
    previewRequestId.current = requestId;
    setPrediction(null);
    setError(null);
    setFile(null);
    setPreviewUrl(null);
    setIsPreparingPreview(false);
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
    setIsPreparingPreview(true);
    try {
      const nextPreviewUrl = await createImagePreviewUrl(candidate);
      if (previewRequestId.current !== requestId) {
        URL.revokeObjectURL(nextPreviewUrl);
        return;
      }
      setPreviewUrl(nextPreviewUrl);
    } catch {
      if (previewRequestId.current === requestId) {
        setError("The image is ready for classification, but its preview could not be rendered.");
      }
    } finally {
      if (previewRequestId.current === requestId) {
        setIsPreparingPreview(false);
      }
    }
  }

  function handleFileInput(event: ChangeEvent<HTMLInputElement>) {
    void selectFile(event.target.files?.[0] ?? null);
  }

  function handleDrop(event: DragEvent<HTMLLabelElement>) {
    event.preventDefault();
    setIsDragging(false);
    void selectFile(event.dataTransfer.files?.[0] ?? null);
  }

  async function runPrediction(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!file || modelStatus !== "ready" || !API_BASE_URL) return;

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

  const serviceDotClass =
    modelStatus === "ready"
      ? "bg-[#36a56f]"
      : modelStatus === "offline"
        ? "bg-coral"
        : "bg-sun";

  return (
    <main className="min-h-screen bg-paper text-ink">
      <header className="sticky top-0 z-20 grid min-h-[84px] grid-cols-[1fr_auto_1fr] items-center border-b border-line bg-paper/95 px-[clamp(22px,5vw,78px)] py-3.5 backdrop-blur-md max-[980px]:grid-cols-[1fr_auto] max-[620px]:min-h-[70px]">
        <a className="flex items-center gap-3 no-underline" href="#top" aria-label="TerraClass home">
          <span className="flex size-[38px] items-center justify-center bg-ink text-[13px] font-extrabold tracking-[-0.03em] text-paper" aria-hidden="true">
            TC
          </span>
          <span>
            <strong className="block text-[17px] tracking-[-0.02em]">TerraClass</strong>
            <small className="mt-0.5 block text-[11px] text-muted max-[620px]:hidden">
              Land intelligence, made visible
            </small>
          </span>
        </a>
        <nav className="flex gap-[30px] max-[980px]:hidden" aria-label="Primary navigation">
          {[
            ["Classifier", "#classifier"],
            ["Method", "#method"],
            ["Evidence", "#evidence"],
          ].map(([label, href]) => (
            <a className="text-[13px] font-bold no-underline transition-colors hover:text-teal focus-visible:text-teal" href={href} key={href}>
              {label}
            </a>
          ))}
        </nav>
        <div className="flex items-center justify-self-end gap-2 text-xs font-bold max-[620px]:text-[0]" role="status">
          <span className={`size-2 ${serviceDotClass}`} aria-hidden="true" />
          {modelStatus === "ready"
            ? "Model ready"
            : modelStatus === "offline"
              ? "API offline"
              : "Checking model"}
        </div>
      </header>

      <section className="relative grid min-h-[650px] grid-cols-[minmax(0,1fr)_minmax(420px,0.86fr)] overflow-hidden max-[980px]:grid-cols-1 max-[980px]:pb-0" id="top">
        <div className="max-w-[760px] self-center px-[clamp(30px,7vw,112px)] pt-[90px] pb-[150px] max-[980px]:pb-20 max-[620px]:px-6 max-[620px]:py-[70px]">
          <p className="mb-5 text-xs font-extrabold tracking-[0.16em] text-teal uppercase">Satellite scene intelligence</p>
          <h1 className="mb-8 max-w-[670px] font-serif text-[clamp(54px,6vw,94px)] leading-[0.92] font-normal tracking-[-0.065em] max-[620px]:text-[56px]">
            See what the model sees.
          </h1>
          <p className="max-w-[650px] text-lg leading-[1.7] text-[#405052]">
            Upload an aerial scene and inspect how a leakage-aware ResNet18 distinguishes
            five land-use classes. Every result includes confidence, ranked alternatives,
            and model provenance.
          </p>
          <a className="mt-[18px] inline-block text-sm font-extrabold no-underline transition-colors hover:text-teal" href="#classifier">
            Try the classifier <span className="ml-2 inline-block" aria-hidden="true">↓</span>
          </a>
        </div>

        <div className="relative min-h-[610px] overflow-hidden bg-[#a2ad8b] max-[980px]:min-h-[420px]" aria-label="Abstract satellite landscape illustration">
          <div
            className="absolute inset-0 z-[1] opacity-100"
            style={{
              backgroundImage:
                "linear-gradient(rgba(255,255,255,.12) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,.12) 1px, transparent 1px)",
              backgroundSize: "52px 52px",
            }}
          />
          <div
            className="absolute -top-[8%] right-[44%] bottom-1/2 -left-[10%] -rotate-[8deg]"
            style={{
              background:
                "repeating-linear-gradient(116deg, #d0bd6c 0 24px, #c6a950 24px 52px, #7f946b 52px 80px)",
            }}
          />
          <div
            className="absolute top-[34%] -right-[25%] -bottom-[20%] left-1/4 -rotate-[13deg]"
            style={{
              background:
                "radial-gradient(ellipse at 0 50%, #d9c79d 0 29%, #7bb3b0 31% 54%, #286c70 56% 100%)",
            }}
          />
          <div
            className="absolute -top-[4%] -right-[12%] bottom-[60%] left-1/2 rotate-[7deg] bg-[#7b817c]"
            style={{
              backgroundImage:
                "linear-gradient(32deg, transparent 45%, #d7d2c4 46% 52%, transparent 53%), linear-gradient(-32deg, transparent 45%, #b8b3a5 46% 52%, transparent 53%)",
              backgroundSize: "46px 46px",
            }}
          />
          <div className="absolute top-7 right-[30px] z-[2] font-mono text-[11px] tracking-[0.08em] text-white/90">26.4499° N</div>
          <div className="absolute bottom-24 left-[30px] z-[2] font-mono text-[11px] tracking-[0.08em] text-white/90">80.3319° E</div>
          <div className="absolute top-[46%] left-[52%] z-[3] size-[120px] -translate-1/2 border border-white/85" aria-hidden="true">
            <span className="absolute top-1/2 left-1/2 h-px w-[150px] -translate-1/2 bg-white/85" />
            <span className="absolute top-1/2 left-1/2 h-[150px] w-px -translate-1/2 bg-white/85" />
            <span className="absolute top-1/2 left-1/2 size-2 -translate-1/2 bg-coral" />
          </div>
        </div>

        <dl className="absolute right-0 bottom-0 left-0 z-[5] grid grid-cols-4 bg-ink pl-[clamp(30px,7vw,112px)] text-white max-[980px]:relative max-[980px]:grid-cols-4 max-[980px]:pl-0 max-[620px]:grid-cols-2" id="evidence">
          {[
            ["500", "balanced images"],
            ["5", "land-use classes"],
            ["100%", "scoped test accuracy"],
            ["75", "held-out test images"],
          ].map(([value, label]) => (
            <div className="border-l border-white/20 px-7 py-[22px] max-[620px]:px-[22px] max-[620px]:py-[18px]" key={label}>
              <dt className="font-serif text-[28px]">{value}</dt>
              <dd className="mt-[5px] text-[11px] tracking-[0.08em] text-[#b9c6c5] uppercase">{label}</dd>
            </div>
          ))}
        </dl>
      </section>

      <section className="bg-cream px-[clamp(22px,5vw,78px)] py-[110px] max-[620px]:py-20" id="classifier">
        <div className="mx-auto mb-12 grid max-w-[1380px] grid-cols-[1fr_minmax(300px,0.7fr)] items-end gap-[60px] max-[980px]:grid-cols-1 max-[620px]:gap-6">
          <div>
            <p className="mb-5 text-xs font-extrabold tracking-[0.16em] text-teal uppercase">Interactive inference</p>
            <h2 className="font-serif text-[clamp(38px,4vw,62px)] leading-none font-normal tracking-[-0.045em]">Classify a satellite scene</h2>
          </div>
          <p className="mb-0.5 max-w-[520px] leading-[1.65] text-muted">
            Supported formats: PNG, JPEG, TIFF and WebP. Images remain in memory only for
            the duration of the request.
          </p>
        </div>

        <div className="mx-auto grid max-w-[1380px] grid-cols-[minmax(0,1.05fr)_minmax(390px,0.95fr)] border border-line max-[980px]:grid-cols-1">
          <form className="border-r border-line p-[clamp(24px,4vw,54px)] max-[980px]:border-r-0 max-[980px]:border-b" onSubmit={runPrediction}>
            <label
              className={`relative flex h-[430px] cursor-pointer items-center justify-center overflow-hidden border border-dashed transition-colors max-[620px]:h-[330px] ${
                isDragging
                  ? "border-teal bg-[#dfebe7]"
                  : "border-[#8b9995] bg-[#e8e7df] hover:border-teal hover:bg-[#dfebe7]"
              }`}
              onDragEnter={() => setIsDragging(true)}
              onDragLeave={() => setIsDragging(false)}
              onDragOver={(event) => event.preventDefault()}
              onDrop={handleDrop}
            >
              <input
                className="absolute size-px opacity-0"
                type="file"
                accept="image/png,image/jpeg,image/tiff,image/webp"
                onChange={handleFileInput}
              />
              {isPreparingPreview ? (
                <span className="text-center text-muted" role="status">
                  <strong className="mb-[7px] block font-serif text-2xl font-normal text-ink">Preparing preview…</strong>
                  <span className="text-[13px]">The original image remains unchanged for classification.</span>
                </span>
              ) : previewUrl ? (
                // Browser-compatible files use a local blob URL; TIFF files are decoded to a temporary PNG URL.
                // eslint-disable-next-line @next/next/no-img-element
                <img className="size-full object-cover" src={previewUrl} alt="Selected satellite scene preview" />
              ) : (
                <span className="text-center text-muted">
                  <span className="mx-auto mb-[18px] block size-14 border border-[#8b9995] text-[30px] leading-[54px] font-light">+</span>
                  <strong className="mb-[7px] block font-serif text-2xl font-normal text-ink">Drop an aerial image here</strong>
                  <span className="text-[13px]">or click to choose a file</span>
                </span>
              )}
              <span className="pointer-events-none absolute inset-[15px] border border-white/65" />
              {file && (
                <span className="absolute bottom-7 left-7 z-[2] max-w-[calc(100%-56px)] overflow-hidden bg-ink/90 px-3 py-[9px] text-xs text-ellipsis whitespace-nowrap text-white">
                  {file.name}
                </span>
              )}
            </label>

            <div className="mt-5 flex items-end justify-between gap-4 max-[620px]:flex-col max-[620px]:items-stretch">
              <label className="text-xs font-bold text-muted">
                Ranked results
                <select className="mt-2 block min-w-[110px] border-0 border-b border-ink bg-transparent px-0.5 py-2 text-ink" value={topK} onChange={(event) => setTopK(Number(event.target.value))}>
                  {classes.map((_, index) => (
                    <option key={index + 1} value={index + 1}>{index + 1}</option>
                  ))}
                </select>
              </label>
              <button className="min-h-12 cursor-pointer bg-coral px-[25px] text-[13px] font-extrabold text-[#211c18] transition-colors hover:bg-[#f17b61] disabled:cursor-not-allowed disabled:opacity-50 max-[620px]:w-full" type="submit" disabled={!file || modelStatus !== "ready" || isSubmitting}>
                {isSubmitting ? "Analysing scene…" : "Run classification"}
              </button>
            </div>
            {modelStatus === "offline" && (
              <p className="mt-[18px] bg-[#f2e9c8] px-[15px] py-[13px] text-[13px] leading-6" role="alert">
                {API_BASE_URL
                  ? `The interface is ready, but the inference API is not reachable at ${API_BASE_URL}.`
                  : "The interface is deployed. Model API deployment is the next engineering phase."}
              </p>
            )}
            {error && <p className="mt-[18px] bg-[#f5d9d2] px-[15px] py-[13px] text-[13px] leading-6 text-[#7c2e20]" role="alert">{error}</p>}
          </form>

          <section className="min-h-[566px] bg-[#e8eee9] p-[clamp(32px,4vw,56px)]" aria-live="polite" aria-busy={isSubmitting}>
            {prediction ? (
              <>
                <div className="mb-7 grid grid-cols-[1fr_auto] border-b border-line pb-[26px]">
                  <p className="col-span-full mb-3 text-xs font-extrabold tracking-[0.16em] text-teal uppercase">Top prediction</p>
                  <h3 className="m-0 font-serif text-[44px] font-normal tracking-[-0.04em]">{formatClassName(prediction.predicted_class)}</h3>
                  <strong className="font-serif text-[38px] font-normal text-teal">{formatPercentage(prediction.confidence)}</strong>
                  <span className="col-start-2 text-right text-[11px] text-muted uppercase">model confidence</span>
                </div>
                <ol className="m-0 list-none p-0">
                  {prediction.predictions.map((item) => (
                    <li className="mb-5" key={item.class_name}>
                      <div className="mb-2 grid grid-cols-[32px_1fr_auto] items-baseline">
                        <span className="font-mono text-[11px] text-muted">{String(item.rank).padStart(2, "0")}</span>
                        <strong className="text-sm">{formatClassName(item.class_name)}</strong>
                        <b className="text-[13px]">{formatPercentage(item.probability)}</b>
                      </div>
                      <span className="block h-[5px] overflow-hidden bg-[#cbd6d0]">
                        <span className="block h-full bg-teal transition-[width] duration-500 motion-reduce:transition-none" style={{ width: `${item.probability * 100}%` }} />
                      </span>
                    </li>
                  ))}
                </ol>
                <dl className="mt-[34px] grid grid-cols-4 border-t border-line pt-5 max-[620px]:grid-cols-2 max-[620px]:gap-y-[18px]">
                  {[
                    ["Inference", `${prediction.latency_ms.toFixed(1)} ms`],
                    ["Image", `${prediction.image_width} × ${prediction.image_height}`],
                    ["Model", prediction.model_version],
                    ["Request", prediction.request_id.slice(0, 8)],
                  ].map(([label, value]) => (
                    <div className="border-l border-line pl-3" key={label}>
                      <dt className="text-[10px] tracking-[0.08em] text-muted uppercase">{label}</dt>
                      <dd className="mt-[7px] overflow-hidden font-mono text-xs text-ellipsis" title={label === "Request" ? prediction.request_id : undefined}>{value}</dd>
                    </div>
                  ))}
                </dl>
              </>
            ) : (
              <div className="flex h-full flex-col justify-center">
                <span className="font-mono text-xs text-teal">01—05</span>
                <h3 className="my-6 max-w-[460px] font-serif text-[clamp(34px,4vw,54px)] leading-[1.06] font-normal tracking-[-0.04em]">Your analysis will appear here.</h3>
                <p className="max-w-[460px] leading-[1.6] text-muted">
                  Select a scene to see the predicted class, calibrated probabilities,
                  dimensions and request-level latency.
                </p>
                <div className="mt-[34px] grid grid-cols-2 border-t border-line pt-5 max-[620px]:grid-cols-1">
                  {classes.map((className, index) => (
                    <span className="py-2 text-xs" key={className}>
                      <b className="mr-2.5 inline-block font-mono text-teal">{String(index + 1).padStart(2, "0")}</b>
                      {formatClassName(className)}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </section>
        </div>
      </section>

      <section className="mx-auto grid max-w-[1536px] grid-cols-[0.75fr_1.25fr] gap-[70px] px-[clamp(22px,5vw,78px)] py-[120px] max-[980px]:grid-cols-1 max-[620px]:py-20" id="method">
        <div>
          <p className="mb-5 text-xs font-extrabold tracking-[0.16em] text-teal uppercase">Built for credible evaluation</p>
          <h2 className="font-serif text-[clamp(38px,4vw,62px)] leading-none font-normal tracking-[-0.045em]">Strong results, stated at the right scope.</h2>
        </div>
        <div className="border-t border-line">
          {[
            ["01", "Leakage-aware split", "Perceptually related scenes are grouped before splitting, reducing optimistic evaluation from near-duplicate imagery."],
            ["02", "Transfer learning", "ResNet18 was selected against EfficientNet-B0 using verified accuracy, macro F1, loss and runtime evidence."],
            ["03", "Reproducible serving", "Versioned configuration, artifact hashes and typed responses connect the trained model to every prediction."],
          ].map(([number, title, description]) => (
            <article className="grid grid-cols-[48px_180px_1fr] gap-5 border-b border-line py-7 max-[620px]:grid-cols-[34px_1fr] max-[620px]:gap-2.5" key={number}>
              <span className="font-mono text-[11px] text-coral">{number}</span>
              <h3 className="m-0 text-[15px] font-bold">{title}</h3>
              <p className="m-0 text-sm leading-[1.6] text-muted max-[620px]:col-start-2">{description}</p>
            </article>
          ))}
        </div>
        <aside className="col-start-2 border-l-4 border-sun py-1.5 pl-[22px] max-[980px]:col-start-1">
          <strong className="mb-[7px] block text-[13px]">Scope matters.</strong>
          <p className="m-0 text-sm leading-[1.6] text-muted">
            {metadata?.scope ?? "This model covers a balanced five-class, 500-image subset of UC Merced. It is not a universal satellite classifier."}
          </p>
        </aside>
      </section>

      <footer className="flex items-center justify-between bg-ink px-[clamp(22px,5vw,78px)] py-[38px] text-white max-[620px]:flex-col max-[620px]:items-start max-[620px]:gap-5">
        <div>
          <strong className="block">TerraClass</strong>
          <span className="mt-[5px] block text-xs text-[#afbfbd]">Applied ML, from experiment to interface.</span>
        </div>
        <p className="m-0 text-xs text-[#afbfbd]">ResNet18 · PyTorch · FastAPI · leakage-aware evaluation</p>
      </footer>
    </main>
  );
}

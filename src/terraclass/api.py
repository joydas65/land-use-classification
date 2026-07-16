"""Typed HTTP boundary for TerraClass model inference."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException

from terraclass.inference import (
    InferenceInputError,
    Prediction,
    ServingConfig,
    TerraClassPredictor,
    load_serving_config,
)
from terraclass.telemetry import emit_structured_event, prediction_observation

LOGGER = logging.getLogger("terraclass.api")
SERVICE_NAME = "terraclass-inference-api"
SERVICE_VERSION = "1.1.0"
ALLOWED_IMAGE_TYPES = {
    "image/jpeg",
    "image/png",
    "image/tiff",
    "image/webp",
}
SatelliteImageUpload = Annotated[UploadFile, File(description="Satellite image")]
TopKQuery = Annotated[int, Query(ge=1, le=5)]


class ModelUnavailableError(RuntimeError):
    """Raised when a request requires a model that did not become ready."""


class InferenceCapacityError(RuntimeError):
    """Raised when the bounded inference queue cannot accept another request."""


@dataclass(frozen=True)
class ApiSettings:
    project_root: Path
    serving_config_path: Path
    benchmark_path: Path
    device: str
    allowed_origins: tuple[str, ...]
    fail_on_model_error: bool = False
    max_concurrent_inferences: int = 1
    queue_timeout_seconds: float = 5.0

    def __post_init__(self) -> None:
        if self.max_concurrent_inferences <= 0:
            raise ValueError("max_concurrent_inferences must be positive")
        if self.queue_timeout_seconds <= 0:
            raise ValueError("queue_timeout_seconds must be positive")

    @classmethod
    def from_environment(cls) -> ApiSettings:
        default_root = Path(__file__).resolve().parents[2]
        project_root = Path(os.getenv("TERRACLASS_PROJECT_ROOT", default_root)).resolve()
        origins = tuple(
            value.strip()
            for value in os.getenv(
                "TERRACLASS_ALLOWED_ORIGINS",
                "http://localhost:3000,http://localhost:5173",
            ).split(",")
            if value.strip()
        )
        return cls(
            project_root=project_root,
            serving_config_path=project_root / "configs/serving/resnet18_group_aware_v1.json",
            benchmark_path=project_root / "reports/inference_benchmark_2026-07-15.json",
            device=os.getenv("TERRACLASS_DEVICE", "auto"),
            allowed_origins=origins,
            fail_on_model_error=os.getenv("TERRACLASS_FAIL_ON_MODEL_ERROR", "false").lower()
            in {"1", "true", "yes"},
            max_concurrent_inferences=int(os.getenv("TERRACLASS_MAX_CONCURRENT_INFERENCES", "1")),
            queue_timeout_seconds=float(os.getenv("TERRACLASS_QUEUE_TIMEOUT_SECONDS", "5")),
        )


class RankedPredictionResponse(BaseModel):
    rank: int = Field(ge=1)
    class_name: str
    probability: float = Field(ge=0, le=1)


class PredictionResponse(BaseModel):
    request_id: str
    model_id: str
    model_version: str
    predicted_class: str
    confidence: float = Field(ge=0, le=1)
    predictions: list[RankedPredictionResponse]
    latency_ms: float = Field(gt=0)
    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)


class HealthResponse(BaseModel):
    service: str
    version: str
    status: str
    model_ready: bool


class BenchmarkResponse(BaseModel):
    device: str
    measured_requests: int
    p50_request_latency_ms: float
    p95_request_latency_ms: float
    throughput_requests_per_second: float


class ModelMetadataResponse(BaseModel):
    model_id: str
    model_version: str
    architecture: str
    class_names: list[str]
    selected_epoch: int
    training_manifest_sha256: str
    serving_artifact_sha256: str
    verified_test_accuracy: float
    verified_test_macro_f1: float
    default_top_k: int
    max_image_bytes: int
    max_image_pixels: int
    benchmark: BenchmarkResponse
    scope: str


class ErrorDetail(BaseModel):
    code: str
    message: str
    request_id: str


class ErrorResponse(BaseModel):
    error: ErrorDetail


PredictorLoader = Callable[[ServingConfig, Path, str], TerraClassPredictor]


def _default_predictor_loader(
    config: ServingConfig,
    project_root: Path,
    device: str,
) -> TerraClassPredictor:
    return TerraClassPredictor.load(config, project_root, device=device)


def _request_id(request: Request) -> str:
    return str(getattr(request.state, "request_id", "unknown"))


def _error_response(request: Request, status_code: int, code: str, message: str) -> JSONResponse:
    body = ErrorResponse(
        error=ErrorDetail(code=code, message=message, request_id=_request_id(request))
    )
    return JSONResponse(status_code=status_code, content=body.model_dump())


def _load_benchmark(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def create_app(
    settings: ApiSettings | None = None,
    *,
    predictor_loader: PredictorLoader = _default_predictor_loader,
) -> FastAPI:
    settings = settings or ApiSettings.from_environment()
    serving_config = load_serving_config(settings.serving_config_path)
    benchmark = _load_benchmark(settings.benchmark_path)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.predictor = None
        app.state.model_error = None
        try:
            app.state.predictor = await run_in_threadpool(
                predictor_loader,
                serving_config,
                settings.project_root,
                settings.device,
            )
        except Exception as error:
            app.state.model_error = type(error).__name__
            LOGGER.exception("TerraClass model failed to initialize")
            if settings.fail_on_model_error:
                raise
        yield
        app.state.predictor = None

    app = FastAPI(
        title="TerraClass Inference API",
        summary="Versioned five-class satellite land-use inference",
        version=SERVICE_VERSION,
        lifespan=lifespan,
    )
    app.state.inference_slots = asyncio.Semaphore(settings.max_concurrent_inferences)
    if settings.allowed_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.allowed_origins),
            allow_credentials=False,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["Accept", "Content-Type", "X-Request-ID"],
            expose_headers=["X-Request-ID"],
        )

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request.state.request_id = str(uuid.uuid4())
        started = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - started) * 1000
        response.headers["X-Request-ID"] = request.state.request_id
        emit_structured_event(
            {
                "event": "http_request",
                "schema_version": 1,
                "request_id": request.state.request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": round(duration_ms, 3),
            }
        )
        return response

    @app.exception_handler(InferenceInputError)
    async def inference_input_handler(request: Request, error: InferenceInputError):
        return _error_response(request, 422, "invalid_image", str(error))

    @app.exception_handler(ModelUnavailableError)
    async def unavailable_handler(request: Request, _: ModelUnavailableError):
        return _error_response(
            request,
            503,
            "model_unavailable",
            "The inference model is not ready. Please try again shortly.",
        )

    @app.exception_handler(InferenceCapacityError)
    async def capacity_handler(request: Request, _: InferenceCapacityError):
        response = _error_response(
            request,
            429,
            "inference_capacity_exceeded",
            "The inference service is busy. Please retry shortly.",
        )
        response.headers["Retry-After"] = "1"
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, _: RequestValidationError):
        return _error_response(request, 422, "invalid_request", "The request is not valid.")

    @app.exception_handler(StarletteHTTPException)
    async def http_handler(request: Request, error: StarletteHTTPException):
        message = str(error.detail) if error.status_code < 500 else "The request failed."
        return _error_response(request, error.status_code, "http_error", message)

    @app.exception_handler(Exception)
    async def unexpected_handler(request: Request, _: Exception):
        LOGGER.exception("Unhandled TerraClass API error")
        return _error_response(
            request,
            500,
            "internal_error",
            "An unexpected error occurred while processing the request.",
        )

    def predictor(request: Request) -> TerraClassPredictor:
        loaded = getattr(request.app.state, "predictor", None)
        if loaded is None:
            raise ModelUnavailableError
        return loaded

    @app.get("/api/v1/health/live", response_model=HealthResponse, tags=["health"])
    async def live(request: Request) -> HealthResponse:
        ready = getattr(request.app.state, "predictor", None) is not None
        return HealthResponse(
            service=SERVICE_NAME,
            version=SERVICE_VERSION,
            status="alive",
            model_ready=ready,
        )

    @app.get(
        "/api/v1/health/ready",
        response_model=HealthResponse,
        responses={503: {"model": ErrorResponse}},
        tags=["health"],
    )
    async def ready(request: Request) -> HealthResponse:
        predictor(request)
        return HealthResponse(
            service=SERVICE_NAME,
            version=SERVICE_VERSION,
            status="ready",
            model_ready=True,
        )

    @app.get("/api/v1/model", response_model=ModelMetadataResponse, tags=["model"])
    async def model_metadata() -> ModelMetadataResponse:
        latency = benchmark["request_latency_ms"]
        return ModelMetadataResponse(
            model_id=serving_config.model_id,
            model_version=serving_config.model_version,
            architecture=serving_config.architecture,
            class_names=list(serving_config.class_names),
            selected_epoch=serving_config.selected_epoch,
            training_manifest_sha256=serving_config.training_manifest_sha256,
            serving_artifact_sha256=serving_config.serving_artifact.sha256,
            verified_test_accuracy=serving_config.test_accuracy,
            verified_test_macro_f1=serving_config.test_macro_f1,
            default_top_k=serving_config.limits.default_top_k,
            max_image_bytes=serving_config.limits.max_image_bytes,
            max_image_pixels=serving_config.limits.max_image_pixels,
            benchmark=BenchmarkResponse(
                device=benchmark["environment"]["device"],
                measured_requests=benchmark["protocol"]["measured_requests"],
                p50_request_latency_ms=latency["p50"],
                p95_request_latency_ms=latency["p95"],
                throughput_requests_per_second=benchmark["throughput_requests_per_second"],
            ),
            scope=(
                "Balanced 500-image UC Merced subset covering five classes; "
                "not a universal satellite classifier."
            ),
        )

    @app.post(
        "/api/v1/predictions",
        response_model=PredictionResponse,
        responses={
            429: {"model": ErrorResponse},
            415: {"model": ErrorResponse},
            422: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
        },
        tags=["inference"],
    )
    async def predict_image(
        request: Request,
        file: SatelliteImageUpload,
        top_k: TopKQuery = serving_config.limits.default_top_k,
    ) -> PredictionResponse | JSONResponse:
        loaded_predictor = predictor(request)
        if file.content_type not in ALLOWED_IMAGE_TYPES:
            return _error_response(
                request,
                415,
                "unsupported_media_type",
                "Upload a PNG, JPEG, TIFF, or WebP image.",
            )
        payload = await file.read(serving_config.limits.max_image_bytes + 1)
        await file.close()
        if len(payload) > serving_config.limits.max_image_bytes:
            raise InferenceInputError(
                f"Image exceeds the {serving_config.limits.max_image_bytes}-byte limit"
            )
        try:
            await asyncio.wait_for(
                request.app.state.inference_slots.acquire(),
                timeout=settings.queue_timeout_seconds,
            )
        except TimeoutError as error:
            raise InferenceCapacityError from error
        try:
            prediction: Prediction = await run_in_threadpool(
                loaded_predictor.predict_bytes,
                payload,
                top_k=top_k,
            )
        finally:
            request.app.state.inference_slots.release()
        emit_structured_event(
            prediction_observation(
                request_id=_request_id(request),
                model_id=prediction.model_id,
                model_version=prediction.model_version,
                predicted_class=prediction.predicted_class,
                confidence=prediction.confidence,
                inference_latency_ms=prediction.latency_ms,
                image_width=prediction.image_width,
                image_height=prediction.image_height,
                payload_bytes=len(payload),
                content_type=file.content_type,
            )
        )
        return PredictionResponse(
            request_id=_request_id(request),
            **prediction.to_dict(),
        )

    return app


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run(
        "terraclass.api:app",
        host=os.getenv("TERRACLASS_HOST", "127.0.0.1"),
        port=int(os.getenv("TERRACLASS_PORT", os.getenv("PORT", "8000"))),
        reload=False,
        access_log=False,
    )


if __name__ == "__main__":
    main()

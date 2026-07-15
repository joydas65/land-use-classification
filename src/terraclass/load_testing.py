"""Dependency-free HTTP load probe for the versioned inference API."""

from __future__ import annotations

import hashlib
import json
import platform
import statistics
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class RequestMeasurement:
    request_id: str
    status_code: int
    total_latency_ms: float
    model_pipeline_latency_ms: float


def percentile(values: list[float], percentage: float) -> float:
    if not values:
        raise ValueError("Cannot calculate a percentile of an empty sequence")
    if not 0 <= percentage <= 100:
        raise ValueError("percentage must be in [0, 100]")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentage / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "minimum": min(values),
        "maximum": max(values),
    }


def get_json(url: str, *, timeout_seconds: float) -> dict[str, Any]:
    request = Request(url, headers={"Accept": "application/json", "User-Agent": "TerraClass/1.0"})
    with urlopen(request, timeout=timeout_seconds) as response:
        return json.load(response)


def post_prediction(
    base_url: str,
    payload: bytes,
    content_type: str,
    *,
    timeout_seconds: float,
) -> RequestMeasurement:
    boundary = f"terraclass-{uuid.uuid4().hex}"
    prefix = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="file"; filename="scene"\r\n'
        f"Content-Type: {content_type}\r\n\r\n"
    ).encode()
    body = prefix + payload + f"\r\n--{boundary}--\r\n".encode()
    request = Request(
        f"{base_url.rstrip('/')}/api/v1/predictions?top_k=3",
        data=body,
        method="POST",
        headers={
            "Accept": "application/json",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(len(body)),
            "User-Agent": "TerraClass-load-test/1.0",
        },
    )
    started = time.perf_counter()
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            response_body = json.load(response)
            status_code = response.status
            header_request_id = response.headers.get("X-Request-ID")
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Inference request returned HTTP {error.code}: {detail}") from error
    total_latency_ms = (time.perf_counter() - started) * 1000
    request_id = str(response_body.get("request_id", ""))
    if status_code != 200 or not request_id or header_request_id != request_id:
        raise RuntimeError("Inference response violated the request-ID contract")
    return RequestMeasurement(
        request_id=request_id,
        status_code=status_code,
        total_latency_ms=total_latency_ms,
        model_pipeline_latency_ms=float(response_body["latency_ms"]),
    )


RequestFunction = Callable[[str, bytes, str], RequestMeasurement]


def run_load_level(
    base_url: str,
    payload: bytes,
    content_type: str,
    *,
    concurrency: int,
    request_count: int,
    request_function: RequestFunction,
) -> dict[str, Any]:
    if concurrency <= 0 or request_count <= 0:
        raise ValueError("concurrency and request_count must be positive")
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        measurements = list(
            executor.map(
                lambda _: request_function(base_url, payload, content_type),
                range(request_count),
            )
        )
    elapsed_seconds = time.perf_counter() - started
    request_ids = [measurement.request_id for measurement in measurements]
    if len(set(request_ids)) != request_count:
        raise RuntimeError("Load test received duplicate request IDs")
    return {
        "concurrency": concurrency,
        "requests": request_count,
        "failures": 0,
        "elapsed_seconds": elapsed_seconds,
        "throughput_requests_per_second": request_count / elapsed_seconds,
        "total_latency_ms": summarize(
            [measurement.total_latency_ms for measurement in measurements]
        ),
        "model_pipeline_latency_ms": summarize(
            [measurement.model_pipeline_latency_ms for measurement in measurements]
        ),
    }


def run_api_load_test(
    base_url: str,
    image_path: Path,
    *,
    content_type: str,
    concurrency_levels: tuple[int, ...],
    warmup_requests: int,
    requests_per_level: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    if not concurrency_levels or len(set(concurrency_levels)) != len(concurrency_levels):
        raise ValueError("concurrency_levels must contain unique values")
    if warmup_requests < 0:
        raise ValueError("warmup_requests must be non-negative")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    payload = image_path.read_bytes()
    normalized_url = base_url.rstrip("/")
    health = get_json(f"{normalized_url}/api/v1/health/ready", timeout_seconds=timeout_seconds)
    model = get_json(f"{normalized_url}/api/v1/model", timeout_seconds=timeout_seconds)
    if health.get("status") != "ready" or not health.get("model_ready"):
        raise RuntimeError("Inference API did not pass its readiness contract")

    def request_function(url: str, body: bytes, media_type: str) -> RequestMeasurement:
        return post_prediction(
            url,
            body,
            media_type,
            timeout_seconds=timeout_seconds,
        )

    for _ in range(warmup_requests):
        request_function(normalized_url, payload, content_type)

    levels = [
        run_load_level(
            normalized_url,
            payload,
            content_type,
            concurrency=concurrency,
            request_count=requests_per_level,
            request_function=request_function,
        )
        for concurrency in concurrency_levels
    ]
    return {
        "schema_version": 1,
        "measured_at": datetime.now(UTC).isoformat(),
        "target": normalized_url,
        "model": {
            "model_id": model["model_id"],
            "model_version": model["model_version"],
            "serving_artifact_sha256": model["serving_artifact_sha256"],
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor() or None,
        },
        "protocol": {
            "image_name": image_path.name,
            "image_sha256": hashlib.sha256(payload).hexdigest(),
            "image_bytes": len(payload),
            "content_type": content_type,
            "concurrency_levels": list(concurrency_levels),
            "warmup_requests": warmup_requests,
            "requests_per_level": requests_per_level,
            "timeout_seconds": timeout_seconds,
        },
        "levels": levels,
    }

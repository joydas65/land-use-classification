import io
import uuid
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from terraclass.api import ApiSettings, create_app
from terraclass.inference import Prediction, RankedPrediction, load_serving_config


class FakePredictor:
    def predict_bytes(self, payload: bytes, *, top_k: int | None = None) -> Prediction:
        with Image.open(io.BytesIO(payload)) as image:
            width, height = image.size
        ranked = (
            RankedPrediction(rank=1, class_name="beach", probability=0.91),
            RankedPrediction(rank=2, class_name="agricultural", probability=0.06),
            RankedPrediction(rank=3, class_name="buildings", probability=0.03),
        )[:top_k]
        return Prediction(
            model_id="terraclass-resnet18-group-aware",
            model_version="1.0.0",
            predicted_class="beach",
            confidence=0.91,
            predictions=ranked,
            latency_ms=12.5,
            image_width=width,
            image_height=height,
        )


def _settings(project_root: Path) -> ApiSettings:
    return ApiSettings(
        project_root=project_root,
        serving_config_path=project_root / "configs/serving/resnet18_group_aware_v1.json",
        benchmark_path=project_root / "reports/inference_benchmark_2026-07-15.json",
        device="cpu",
        allowed_origins=("http://localhost:5173",),
    )


def _png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (32, 24), color=(229, 201, 134)).save(buffer, format="PNG")
    return buffer.getvalue()


def _client(project_root: Path) -> TestClient:
    app = create_app(
        _settings(project_root),
        predictor_loader=lambda config, root, device: FakePredictor(),
    )
    return TestClient(app)


def test_health_and_model_metadata_are_typed(project_root: Path) -> None:
    with _client(project_root) as client:
        live = client.get("/api/v1/health/live")
        ready = client.get("/api/v1/health/ready")
        metadata = client.get("/api/v1/model")
    assert live.status_code == 200
    assert live.json()["model_ready"] is True
    assert ready.json()["status"] == "ready"
    assert metadata.status_code == 200
    assert metadata.json()["model_id"] == "terraclass-resnet18-group-aware"
    assert metadata.json()["class_names"] == [
        "agricultural",
        "airplane",
        "baseballdiamond",
        "beach",
        "buildings",
    ]
    assert metadata.json()["benchmark"]["measured_requests"] == 75


def test_prediction_returns_request_id_and_top_k(project_root: Path) -> None:
    with _client(project_root) as client:
        response = client.post(
            "/api/v1/predictions?top_k=2",
            files={"file": ("scene.png", _png_bytes(), "image/png")},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["predicted_class"] == "beach"
    assert [item["class_name"] for item in body["predictions"]] == [
        "beach",
        "agricultural",
    ]
    assert body["image_width"] == 32
    assert body["image_height"] == 24
    assert response.headers["X-Request-ID"] == body["request_id"]
    uuid.UUID(body["request_id"])


def test_api_returns_structured_media_and_validation_errors(project_root: Path) -> None:
    with _client(project_root) as client:
        media = client.post(
            "/api/v1/predictions",
            files={"file": ("scene.txt", b"not an image", "text/plain")},
        )
        invalid_top_k = client.post(
            "/api/v1/predictions?top_k=8",
            files={"file": ("scene.png", _png_bytes(), "image/png")},
        )
    assert media.status_code == 415
    assert media.json()["error"]["code"] == "unsupported_media_type"
    assert media.headers["X-Request-ID"] == media.json()["error"]["request_id"]
    assert invalid_top_k.status_code == 422
    assert invalid_top_k.json()["error"]["code"] == "invalid_request"


def test_api_stays_live_when_model_is_unavailable(project_root: Path) -> None:
    def unavailable(config, root, device):
        raise FileNotFoundError("model unavailable")

    app = create_app(_settings(project_root), predictor_loader=unavailable)
    with TestClient(app) as client:
        live = client.get("/api/v1/health/live")
        ready = client.get("/api/v1/health/ready")
        prediction = client.post(
            "/api/v1/predictions",
            files={"file": ("scene.png", _png_bytes(), "image/png")},
        )
    assert live.status_code == 200
    assert live.json()["model_ready"] is False
    assert ready.status_code == 503
    assert ready.json()["error"]["code"] == "model_unavailable"
    assert prediction.status_code == 503


def test_openapi_exposes_versioned_contract(project_root: Path) -> None:
    config = load_serving_config(project_root / "configs/serving/resnet18_group_aware_v1.json")
    with _client(project_root) as client:
        schema = client.get("/openapi.json").json()
    assert schema["info"]["version"] == "1.0.0"
    assert "/api/v1/predictions" in schema["paths"]
    assert config.model_version == "1.0.0"

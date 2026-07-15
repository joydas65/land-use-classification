"""Validated model loading and image inference for the TerraClass serving track."""

from __future__ import annotations

import hashlib
import io
import json
import re
import threading
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image, UnidentifiedImageError
from torch import nn

from terraclass.config import load_config
from terraclass.devices import select_device
from terraclass.transfer import SUPPORTED_ARCHITECTURES, build_transfer_model
from terraclass.transforms import build_eval_transform

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class ModelArtifactError(RuntimeError):
    """Raised when a serving artifact violates its versioned contract."""


class InferenceInputError(ValueError):
    """Raised when an inference request contains an invalid image or option."""


@dataclass(frozen=True)
class ArtifactReference:
    path: str
    sha256: str


@dataclass(frozen=True)
class InferenceLimits:
    max_image_bytes: int
    max_image_pixels: int
    default_top_k: int


@dataclass(frozen=True)
class ServingConfig:
    schema_version: int
    model_id: str
    model_version: str
    architecture: str
    dropout: float
    class_names: tuple[str, ...]
    baseline_config_path: str
    training_manifest_path: str
    training_manifest_sha256: str
    source_checkpoint: ArtifactReference
    serving_artifact: ArtifactReference
    selected_epoch: int
    test_accuracy: float
    test_macro_f1: float
    limits: InferenceLimits

    def validate(self) -> None:
        errors: list[str] = []
        if self.schema_version != 1:
            errors.append(f"unsupported schema_version={self.schema_version}")
        if not self.model_id.strip() or not self.model_version.strip():
            errors.append("model_id and model_version must not be blank")
        if self.architecture not in SUPPORTED_ARCHITECTURES:
            errors.append(f"unsupported architecture={self.architecture!r}")
        if not 0 <= self.dropout < 1:
            errors.append("dropout must be in [0, 1)")
        if len(self.class_names) < 2 or len(set(self.class_names)) != len(self.class_names):
            errors.append("class_names must contain at least two unique values")
        for field, value in (
            ("training_manifest_sha256", self.training_manifest_sha256),
            ("source_checkpoint.sha256", self.source_checkpoint.sha256),
            ("serving_artifact.sha256", self.serving_artifact.sha256),
        ):
            if not _SHA256_PATTERN.fullmatch(value):
                errors.append(f"{field} must be a lowercase SHA-256")
        for field, value in (
            ("baseline_config_path", self.baseline_config_path),
            ("training_manifest_path", self.training_manifest_path),
            ("source_checkpoint.path", self.source_checkpoint.path),
            ("serving_artifact.path", self.serving_artifact.path),
        ):
            path = Path(value)
            if path.is_absolute() or ".." in path.parts:
                errors.append(f"{field} must be a project-relative path")
        if self.selected_epoch <= 0:
            errors.append("selected_epoch must be positive")
        if not 0 <= self.test_accuracy <= 1 or not 0 <= self.test_macro_f1 <= 1:
            errors.append("test metrics must be in [0, 1]")
        if self.limits.max_image_bytes <= 0 or self.limits.max_image_pixels <= 0:
            errors.append("image limits must be positive")
        if not 1 <= self.limits.default_top_k <= len(self.class_names):
            errors.append("default_top_k must be within the class count")
        if errors:
            raise ValueError("Invalid serving configuration: " + "; ".join(errors))


@dataclass(frozen=True)
class RankedPrediction:
    rank: int
    class_name: str
    probability: float


@dataclass(frozen=True)
class Prediction:
    model_id: str
    model_version: str
    predicted_class: str
    confidence: float
    predictions: tuple[RankedPrediction, ...]
    latency_ms: float
    image_width: int
    image_height: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_serving_config(path: str | Path) -> ServingConfig:
    with Path(path).open(encoding="utf-8") as handle:
        raw = json.load(handle)
    model = raw["model"]
    source = raw["source_checkpoint"]
    artifact = raw["serving_artifact"]
    metrics = raw["verified_test_metrics"]
    limits = raw["limits"]
    config = ServingConfig(
        schema_version=int(raw["schema_version"]),
        model_id=str(raw["model_id"]),
        model_version=str(raw["model_version"]),
        architecture=str(model["architecture"]),
        dropout=float(model["dropout"]),
        class_names=tuple(model["class_names"]),
        baseline_config_path=str(raw["baseline_config_path"]),
        training_manifest_path=str(raw["training_manifest_path"]),
        training_manifest_sha256=str(raw["training_manifest_sha256"]),
        source_checkpoint=ArtifactReference(path=str(source["path"]), sha256=str(source["sha256"])),
        serving_artifact=ArtifactReference(
            path=str(artifact["path"]), sha256=str(artifact["sha256"])
        ),
        selected_epoch=int(raw["selected_epoch"]),
        test_accuracy=float(metrics["accuracy"]),
        test_macro_f1=float(metrics["macro_f1"]),
        limits=InferenceLimits(
            max_image_bytes=int(limits["max_image_bytes"]),
            max_image_pixels=int(limits["max_image_pixels"]),
            default_top_k=int(limits["default_top_k"]),
        ),
    )
    config.validate()
    return config


class TerraClassPredictor:
    """Thread-safe predictor with bounded input handling and top-k probabilities."""

    def __init__(
        self,
        *,
        model: nn.Module,
        transform: Any,
        config: ServingConfig,
        device: torch.device,
    ) -> None:
        self.model = model.to(device).eval()
        self.transform = transform
        self.config = config
        self.device = device
        self._lock = threading.Lock()

    @classmethod
    def load(
        cls,
        config: ServingConfig,
        project_root: str | Path,
        *,
        device: str = "auto",
    ) -> TerraClassPredictor:
        root = Path(project_root).resolve()
        artifact_path = root / config.serving_artifact.path
        if not artifact_path.is_file():
            raise FileNotFoundError(f"Serving artifact does not exist: {artifact_path}")
        actual_hash = _sha256(artifact_path)
        if actual_hash != config.serving_artifact.sha256:
            raise ModelArtifactError(
                f"Serving artifact SHA-256 {actual_hash} differs from the configured value"
            )
        serving_device = select_device(device)
        try:
            artifact = torch.load(
                artifact_path,
                map_location=serving_device,
                weights_only=True,
            )
        except Exception as error:
            raise ModelArtifactError(
                f"Serving artifact could not be loaded safely: {error}"
            ) from error
        cls._validate_artifact(artifact, config)

        baseline = load_config(root / config.baseline_config_path)
        if baseline.dataset.selected_classes != config.class_names:
            raise ModelArtifactError("Baseline class order differs from the serving contract")
        preprocessing = json.loads(json.dumps(asdict(baseline.preprocessing)))
        if artifact["preprocessing"] != preprocessing:
            raise ModelArtifactError("Artifact preprocessing differs from the baseline config")

        model = build_transfer_model(
            config.architecture,
            len(config.class_names),
            pretrained=False,
            dropout=config.dropout,
        )
        try:
            model.load_state_dict(artifact["model_state_dict"], strict=True)
        except RuntimeError as error:
            raise ModelArtifactError(
                f"Artifact state dictionary is incompatible: {error}"
            ) from error
        return cls(
            model=model,
            transform=build_eval_transform(baseline.preprocessing),
            config=config,
            device=serving_device,
        )

    @staticmethod
    def _validate_artifact(artifact: Any, config: ServingConfig) -> None:
        if not isinstance(artifact, dict):
            raise ModelArtifactError("Serving artifact must be a dictionary")
        required = {
            "schema_version",
            "model_id",
            "model_version",
            "architecture",
            "class_names",
            "dropout",
            "manifest_sha256",
            "selected_epoch",
            "source_checkpoint_sha256",
            "preprocessing",
            "model_state_dict",
        }
        missing = required - set(artifact)
        if missing:
            raise ModelArtifactError(f"Serving artifact is missing fields: {sorted(missing)}")
        expected = {
            "schema_version": config.schema_version,
            "model_id": config.model_id,
            "model_version": config.model_version,
            "architecture": config.architecture,
            "class_names": list(config.class_names),
            "dropout": config.dropout,
            "manifest_sha256": config.training_manifest_sha256,
            "selected_epoch": config.selected_epoch,
            "source_checkpoint_sha256": config.source_checkpoint.sha256,
        }
        for field, value in expected.items():
            if artifact.get(field) != value:
                raise ModelArtifactError(
                    f"Serving artifact {field}={artifact.get(field)!r} differs from {value!r}"
                )
        if not isinstance(artifact["model_state_dict"], dict):
            raise ModelArtifactError("model_state_dict must be a dictionary")

    def predict_bytes(self, payload: bytes, *, top_k: int | None = None) -> Prediction:
        if not isinstance(payload, bytes):
            raise InferenceInputError("Image payload must be bytes")
        if not payload:
            raise InferenceInputError("Image payload must not be empty")
        if len(payload) > self.config.limits.max_image_bytes:
            raise InferenceInputError(
                f"Image exceeds the {self.config.limits.max_image_bytes}-byte limit"
            )
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(io.BytesIO(payload)) as source:
                    width, height = source.size
                    if width * height > self.config.limits.max_image_pixels:
                        raise InferenceInputError(
                            f"Image exceeds the {self.config.limits.max_image_pixels}-pixel limit"
                        )
                    source.load()
                    image = source.convert("RGB")
        except (
            Image.DecompressionBombError,
            Image.DecompressionBombWarning,
            OSError,
            UnidentifiedImageError,
        ) as error:
            raise InferenceInputError("Payload is not a supported, decodable image") from error
        return self.predict_image(image, top_k=top_k)

    def predict_image(self, image: Image.Image, *, top_k: int | None = None) -> Prediction:
        if not isinstance(image, Image.Image):
            raise InferenceInputError("image must be a PIL image")
        width, height = image.size
        if width <= 0 or height <= 0:
            raise InferenceInputError("Image dimensions must be positive")
        if width * height > self.config.limits.max_image_pixels:
            raise InferenceInputError(
                f"Image exceeds the {self.config.limits.max_image_pixels}-pixel limit"
            )
        requested_top_k = self.config.limits.default_top_k if top_k is None else top_k
        if not 1 <= requested_top_k <= len(self.config.class_names):
            raise InferenceInputError(f"top_k must be between 1 and {len(self.config.class_names)}")

        started = time.perf_counter()
        tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        with self._lock, torch.inference_mode():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1)
            values, indices = probabilities.topk(requested_top_k, dim=1)
        latency_ms = (time.perf_counter() - started) * 1000
        ranked = tuple(
            RankedPrediction(
                rank=rank,
                class_name=self.config.class_names[index],
                probability=float(probability),
            )
            for rank, (index, probability) in enumerate(
                zip(indices[0].cpu().tolist(), values[0].cpu().tolist(), strict=True),
                start=1,
            )
        )
        return Prediction(
            model_id=self.config.model_id,
            model_version=self.config.model_version,
            predicted_class=ranked[0].class_name,
            confidence=ranked[0].probability,
            predictions=ranked,
            latency_ms=latency_ms,
            image_width=width,
            image_height=height,
        )

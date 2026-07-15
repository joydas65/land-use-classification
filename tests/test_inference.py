import hashlib
import io
import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch
from PIL import Image
from torch import nn

import terraclass.inference as inference_module
from terraclass.inference import (
    ArtifactReference,
    InferenceInputError,
    ModelArtifactError,
    ServingConfig,
    TerraClassPredictor,
    load_serving_config,
)


class FixedLogitModel(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = torch.tensor([[0.0, 3.0, 1.0, -1.0, 2.0]], device=inputs.device)
        return logits.repeat(inputs.shape[0], 1)


def _predictor(config: ServingConfig) -> TerraClassPredictor:
    return TerraClassPredictor(
        model=FixedLogitModel(),
        transform=lambda image: torch.zeros(3, 8, 8),
        config=config,
        device=torch.device("cpu"),
    )


def _png_bytes(size: tuple[int, int] = (16, 12)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color=(32, 64, 128)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_serving_config_matches_selected_model(project_root: Path) -> None:
    config = load_serving_config(project_root / "configs/serving/resnet18_group_aware_v1.json")
    assert config.model_id == "terraclass-resnet18-group-aware"
    assert config.architecture == "resnet18"
    assert config.selected_epoch == 4
    assert config.test_accuracy == 1.0
    assert config.training_manifest_path == "data/manifests/five_class_group_aware_seed42.csv"
    assert config.class_names == (
        "agricultural",
        "airplane",
        "baseballdiamond",
        "beach",
        "buildings",
    )


def test_predict_bytes_returns_ranked_probabilities(project_root: Path) -> None:
    config = load_serving_config(project_root / "configs/serving/resnet18_group_aware_v1.json")
    prediction = _predictor(config).predict_bytes(_png_bytes())
    assert prediction.predicted_class == "airplane"
    assert prediction.image_width == 16
    assert prediction.image_height == 12
    assert [item.class_name for item in prediction.predictions] == [
        "airplane",
        "buildings",
        "baseballdiamond",
    ]
    assert [item.rank for item in prediction.predictions] == [1, 2, 3]
    assert prediction.confidence == pytest.approx(prediction.predictions[0].probability)
    assert 0 < prediction.latency_ms
    assert json.loads(json.dumps(prediction.to_dict()))["model_version"] == "1.0.0"


@pytest.mark.parametrize("payload", (b"", b"not an image"))
def test_predict_bytes_rejects_invalid_payload(project_root: Path, payload: bytes) -> None:
    config = load_serving_config(project_root / "configs/serving/resnet18_group_aware_v1.json")
    with pytest.raises(InferenceInputError):
        _predictor(config).predict_bytes(payload)


def test_predictor_enforces_size_and_top_k_limits(project_root: Path) -> None:
    config = load_serving_config(project_root / "configs/serving/resnet18_group_aware_v1.json")
    small_limit = replace(
        config,
        limits=replace(config.limits, max_image_bytes=8, max_image_pixels=100),
    )
    predictor = _predictor(small_limit)
    with pytest.raises(InferenceInputError, match="byte limit"):
        predictor.predict_bytes(_png_bytes())
    with pytest.raises(InferenceInputError, match="pixel limit"):
        predictor.predict_image(Image.new("RGB", (11, 10)))
    pixel_limit = replace(
        config,
        limits=replace(config.limits, max_image_bytes=1024, max_image_pixels=100),
    )
    with pytest.raises(InferenceInputError, match="pixel limit"):
        _predictor(pixel_limit).predict_bytes(_png_bytes((11, 10)))
    with pytest.raises(InferenceInputError, match="top_k"):
        _predictor(config).predict_bytes(_png_bytes(), top_k=6)


def test_serving_config_rejects_unsafe_artifact_path(project_root: Path, tmp_path: Path) -> None:
    source = project_root / "configs/serving/resnet18_group_aware_v1.json"
    raw = json.loads(source.read_text(encoding="utf-8"))
    raw["serving_artifact"]["path"] = "../outside.pt"
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="project-relative"):
        load_serving_config(path)


def test_predictor_loads_hash_verified_weights_only_artifact(
    project_root: Path,
    baseline_config,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_serving_config(project_root / "configs/serving/resnet18_group_aware_v1.json")
    artifact = {
        "schema_version": 1,
        "model_id": config.model_id,
        "model_version": config.model_version,
        "architecture": config.architecture,
        "class_names": list(config.class_names),
        "dropout": config.dropout,
        "manifest_sha256": config.training_manifest_sha256,
        "selected_epoch": config.selected_epoch,
        "source_checkpoint_sha256": config.source_checkpoint.sha256,
        "preprocessing": json.loads(json.dumps(asdict(baseline_config.preprocessing))),
        "model_state_dict": {},
    }
    artifact_path = tmp_path / "artifact.pt"
    torch.save(artifact, artifact_path)
    artifact_hash = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    test_config = replace(
        config,
        serving_artifact=ArtifactReference(path="artifact.pt", sha256=artifact_hash),
    )
    monkeypatch.setattr(inference_module, "load_config", lambda _: baseline_config)
    monkeypatch.setattr(
        inference_module, "build_transfer_model", lambda *args, **kwargs: FixedLogitModel()
    )
    monkeypatch.setattr(
        inference_module,
        "build_eval_transform",
        lambda _: lambda image: torch.zeros(3, 8, 8),
    )

    predictor = TerraClassPredictor.load(test_config, tmp_path, device="cpu")
    assert predictor.predict_bytes(_png_bytes()).predicted_class == "airplane"

    tampered = replace(
        test_config,
        serving_artifact=ArtifactReference(path="artifact.pt", sha256="f" * 64),
    )
    with pytest.raises(ModelArtifactError, match="SHA-256"):
        TerraClassPredictor.load(tampered, tmp_path, device="cpu")

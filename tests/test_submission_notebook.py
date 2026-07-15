import base64
import hashlib
import json
from pathlib import Path

from scripts.build_submission_notebook import build_notebook

NOTEBOOK_NAME = "Improved_Land_Use_Classification_IITK.ipynb"


def _source(notebook: dict) -> str:
    return "\n".join(str(cell.get("source", "")) for cell in notebook["cells"])


def test_submission_notebook_is_deterministically_generated(project_root: Path) -> None:
    notebook_path = project_root / "notebooks" / NOTEBOOK_NAME
    committed = json.loads(notebook_path.read_text(encoding="utf-8"))
    assert committed == build_notebook()
    assert committed["nbformat"] == 4
    assert committed["metadata"]["accelerator"] == "GPU"
    assert len(committed["cells"]) == 21
    verified_output_cells = [
        cell
        for cell in committed["cells"]
        if "verified-gpu-output" in cell.get("metadata", {}).get("tags", [])
    ]
    assert len(verified_output_cells) == 1
    verified_output_cell = verified_output_cells[0]
    assert verified_output_cell["execution_count"] == 1
    assert len(verified_output_cell["outputs"]) == 1
    output = verified_output_cell["outputs"][0]
    assert output["output_type"] == "display_data"
    assert hashlib.sha256(base64.b64decode(output["data"]["image/png"])).hexdigest() == (
        "c5d1ba44d1d202bd2e259b3a2959f9876a37f438ce340ba6a0c23f884759faf0"
    )


def test_submission_notebook_has_required_iit_and_ml_evidence(project_root: Path) -> None:
    notebook = json.loads((project_root / "notebooks" / NOTEBOOK_NAME).read_text(encoding="utf-8"))
    source = _source(notebook)
    required = (
        "74.67%",
        "0.733",
        "ResNet18",
        "EfficientNet-B0",
        "validation macro F1",
        "balanced accuracy",
        "top-3 accuracy",
        "confusion",
        "classification_report",
        "historical",
        "group-aware",
        "Limitations and scope",
        "files.download",
        "73d19e048e742fdf616cbbc1f037efa009ea329ec600acef329f2a5bc7df87ea",
        "26bc3503f6a16e841286771b727e1f1f14a58c623deafe26c45e52d68b88081d",
        "NVIDIA L4",
        "Selected final architecture: ResNet18",
    )
    for token in required:
        assert token in source
    assert source.count("test_metrics = evaluate(model, test_loader, criterion)") == 1
    assert "Submission target:" not in source
    assert "Deadline:" not in source
    assert "GPU experiment matrix" not in source
    assert "Use only the generated comparison table" not in source
    assert "Collaboration hand-off" not in source
    assert "2c834a31ad37e07de11681f0e3596040d60f1c18e31142dfcdaa97b7a38837ae" not in source
    assert "414233c8471ea961bfd9406a33f54b427e75ab49" not in source
    assert not any(cell.get("attachments") for cell in notebook["cells"])


def test_submission_notebook_contains_only_verified_saved_output(project_root: Path) -> None:
    notebook = json.loads((project_root / "notebooks" / NOTEBOOK_NAME).read_text(encoding="utf-8"))
    source = _source(notebook)
    forbidden = (
        "/Users/",
        "joydas.0111@gmail.com",
        "kaggle.json",
        "KAGGLE_KEY",
        "GITHUB_TOKEN",
        "ghp_",
        "TO_BE_FILLED",
    )
    for token in forbidden:
        assert token not in source
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        if "verified-gpu-output" in cell.get("metadata", {}).get("tags", []):
            assert cell.get("execution_count") == 1
            assert len(cell.get("outputs", [])) == 1
        else:
            assert cell.get("execution_count") is None
            assert not cell.get("outputs")


def test_submission_notebook_python_cells_compile(project_root: Path) -> None:
    notebook = json.loads((project_root / "notebooks" / NOTEBOOK_NAME).read_text(encoding="utf-8"))
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        python_source = "\n".join(
            line for line in str(cell["source"]).splitlines() if not line.lstrip().startswith("%")
        )
        compile(python_source, f"{NOTEBOOK_NAME}:cell-{index}", "exec")

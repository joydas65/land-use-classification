import json
from pathlib import Path

from scripts.build_submission_notebook import build_notebook

NOTEBOOK_NAME = "TerraClass_IITK_Colab_Submission.ipynb"


def _source(notebook: dict) -> str:
    return "\n".join(str(cell.get("source", "")) for cell in notebook["cells"])


def test_submission_notebook_is_deterministically_generated(project_root: Path) -> None:
    notebook_path = project_root / "notebooks" / NOTEBOOK_NAME
    committed = json.loads(notebook_path.read_text(encoding="utf-8"))
    assert committed == build_notebook()
    assert committed["nbformat"] == 4
    assert committed["metadata"]["accelerator"] == "GPU"
    assert len(committed["cells"]) == 20
    assert all(
        not cell.get("outputs") for cell in committed["cells"] if cell["cell_type"] == "code"
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
        "2c834a31ad37e07de11681f0e3596040d60f1c18e31142dfcdaa97b7a38837ae",
        "414233c8471ea961bfd9406a33f54b427e75ab49",
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
    attachments = [cell.get("attachments", {}) for cell in notebook["cells"]]
    assert sum(bool(value) for value in attachments) == 1
    assert "training_and_confusion_colab_l4.png" in next(value for value in attachments if value)


def test_submission_notebook_contains_no_local_secret_or_saved_output(project_root: Path) -> None:
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
    assert all(
        cell.get("execution_count") is None
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )


def test_submission_notebook_python_cells_compile(project_root: Path) -> None:
    notebook = json.loads((project_root / "notebooks" / NOTEBOOK_NAME).read_text(encoding="utf-8"))
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        python_source = "\n".join(
            line for line in str(cell["source"]).splitlines() if not line.lstrip().startswith("%")
        )
        compile(python_source, f"{NOTEBOOK_NAME}:cell-{index}", "exec")

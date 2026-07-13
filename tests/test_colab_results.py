import json
from pathlib import Path

import pytest

from terraclass.colab_results import BundleValidationError, audit_versioned_evidence


def _audit(project_root: Path, report_dir: Path | None = None) -> dict:
    return audit_versioned_evidence(
        report_dir or project_root / "reports/colab",
        project_root / "reports/figures/training_and_confusion_colab_l4.png",
        project_root / "data/manifests/baseline_5class_seed42.csv",
        project_root / "data/manifests/five_class_group_aware_seed42.csv",
    )


def test_versioned_colab_gpu_evidence_passes_exhaustive_audit(project_root: Path) -> None:
    verification = _audit(project_root)
    assert verification["hardware"] == {"device": "cuda", "gpu": "NVIDIA L4"}
    assert verification["failures"] == []
    assert len(verification["runs"]) == 4
    assert {run["test"]["macro_f1"] for run in verification["runs"]} == {1.0}
    assert verification["selected_architecture"] == "resnet18"


def test_versioned_evidence_rejects_metric_tampering(tmp_path: Path, project_root: Path) -> None:
    source = project_root / "reports/colab"
    report_dir = tmp_path / "colab"
    report_dir.mkdir()
    for name in ("VERIFICATION.json", "model_comparison.csv"):
        (report_dir / name).write_bytes((source / name).read_bytes())
    report = json.loads((source / "colab_run_report.json").read_text(encoding="utf-8"))
    report["results"][0]["test"]["accuracy"] = 0.99
    (report_dir / "colab_run_report.json").write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(BundleValidationError, match="report hash mismatch"):
        _audit(project_root, report_dir)

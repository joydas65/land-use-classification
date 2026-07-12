from pathlib import Path

from terraclass.audit import KNOWN_ISSUE_IDS, audit_project


def test_cross_artifact_consistency_audit_passes(project_root: Path) -> None:
    report = audit_project(project_root)
    assert report.ok, report.errors
    assert report.observed["test_accuracy"] == 0.7467
    assert report.observed["test_macro_f1"] == 0.733
    assert report.observed["parameter_count"] == 102_277
    assert report.observed["known_issue_ids"] == list(KNOWN_ISSUE_IDS)

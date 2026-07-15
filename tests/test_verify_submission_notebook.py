import json
from pathlib import Path

import pytest

from scripts.verify_submission_notebook import verification_sources


def test_pretraining_verifier_selects_semantic_cells(project_root: Path) -> None:
    notebook = json.loads(
        (project_root / "notebooks/Improved_Land_Use_Classification_IITK.ipynb").read_text(
            encoding="utf-8"
        )
    )
    sources = verification_sources(notebook)
    assert len(sources) == 4
    assert all(not source.lstrip().startswith("%") for source in sources)
    assert "SEED = 42" in sources[0]
    assert "Verified dataset" in sources[1]
    assert "def group_aware_split" in sources[2]
    assert "Verified manifest hashes" in sources[3]


def test_pretraining_verifier_rejects_missing_cell() -> None:
    with pytest.raises(ValueError, match="Expected one submission cell"):
        verification_sources({"cells": []})

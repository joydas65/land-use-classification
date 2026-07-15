"""Execute the submission notebook's acquisition and manifest cells without training."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import tempfile
import types
from pathlib import Path

VERIFICATION_CELL_MARKERS = (
    ("SEED = 42", "ARCHIVE_SHA256", "WORK_DIR = Path.cwd()"),
    ("def sha256(path: Path)", "def safe_extract", "Verified dataset"),
    ("class Sample", "def historical_split", "def group_aware_split"),
    ("def write_manifest", "Verified manifest hashes"),
)


def verification_sources(notebook: dict) -> list[str]:
    """Select pre-training cells by stable semantic markers, never by position."""
    code_sources = [
        str(cell.get("source", ""))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    ]
    selected: list[str] = []
    for markers in VERIFICATION_CELL_MARKERS:
        matches = [source for source in code_sources if all(token in source for token in markers)]
        if len(matches) != 1:
            raise ValueError(
                f"Expected one submission cell matching {markers!r}; found {len(matches)}"
            )
        selected.append(matches[0])
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--notebook",
        type=Path,
        default=Path("notebooks/Improved_Land_Use_Classification_IITK.ipynb"),
    )
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    notebook = json.loads(args.notebook.read_text(encoding="utf-8"))
    code_cells = verification_sources(notebook)
    original_directory = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="terraclass-notebook-check-") as temporary:
        temporary_root = Path(temporary)
        data_directory = temporary_root / "terraclass_colab" / "data"
        extracted_parent = data_directory / "UCMerced_LandUse"
        extracted_parent.mkdir(parents=True)
        (data_directory / "UCMerced_LandUse.zip").symlink_to(args.archive.resolve())
        (extracted_parent / "Images").symlink_to(
            args.dataset_root.resolve(), target_is_directory=True
        )
        namespace: dict[str, object] = {}
        try:
            os.environ["MPLCONFIGDIR"] = str(temporary_root / "matplotlib-cache")
            os.environ["XDG_CACHE_HOME"] = str(temporary_root / "xdg-cache")
            if importlib.util.find_spec("seaborn") is None:
                sys.modules["seaborn"] = types.ModuleType("seaborn")
            os.chdir(temporary_root)
            for cell_index, source in enumerate(code_cells):
                exec(compile(source, f"submission-cell-{cell_index}", "exec"), namespace)
        finally:
            os.chdir(original_directory)
    print("Submission notebook pre-training verification: PASS")


if __name__ == "__main__":
    main()

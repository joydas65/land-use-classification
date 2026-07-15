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
    code_cells = [cell["source"] for cell in notebook["cells"] if cell["cell_type"] == "code"]
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
            for cell_index in (1, 2, 3, 4):
                source = code_cells[cell_index]
                exec(compile(source, f"submission-cell-{cell_index}", "exec"), namespace)
        finally:
            os.chdir(original_directory)
    print("Submission notebook pre-training verification: PASS")


if __name__ == "__main__":
    main()

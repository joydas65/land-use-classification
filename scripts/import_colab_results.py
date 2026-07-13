"""Validate and import a returned Colab results bundle as versioned evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from terraclass.colab_results import audit_bundle, sha256_bytes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--verified-date", required=True)
    parser.add_argument("--report-dir", type=Path, default=Path("reports/colab"))
    parser.add_argument(
        "--figure-output",
        type=Path,
        default=Path("reports/figures/training_and_confusion_colab_l4.png"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    verification, files = audit_bundle(args.bundle, verified_date=args.verified_date)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    args.figure_output.parent.mkdir(parents=True, exist_ok=True)
    imported = {
        "colab_run_report.json": files["colab_run_report.json"],
        "model_comparison.csv": files["model_comparison.csv"],
    }
    for name, content in imported.items():
        (args.report_dir / name).write_bytes(content)
    args.figure_output.write_bytes(files["training_and_confusion.png"])
    (args.report_dir / "VERIFICATION.json").write_text(
        json.dumps(verification, indent=2) + "\n", encoding="utf-8"
    )
    checksums = [f"{sha256_bytes(content)}  {name}" for name, content in sorted(imported.items())]
    checksums.extend(
        [
            f"{verification['figure']['sha256']}  {args.figure_output.as_posix()}",
            f"{verification['source_bundle']['sha256']}  SOURCE::{args.bundle.name}",
        ]
    )
    (args.report_dir / "SHA256SUMS").write_text("\n".join(checksums) + "\n", encoding="utf-8")
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()

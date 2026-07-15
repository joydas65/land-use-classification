"""Measure the TerraClass HTTP API at controlled concurrency levels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from terraclass.load_testing import run_api_load_test


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--content-type", default="image/tiff")
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--warmup-requests", type=int, default=5)
    parser.add_argument("--requests-per-level", type=int, default=20)
    parser.add_argument("--timeout-seconds", type=float, default=10.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/api_load_test_2026-07-16.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_api_load_test(
        args.base_url,
        args.image,
        content_type=args.content_type,
        concurrency_levels=tuple(args.concurrency),
        warmup_requests=args.warmup_requests,
        requests_per_level=args.requests_per_level,
        timeout_seconds=args.timeout_seconds,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

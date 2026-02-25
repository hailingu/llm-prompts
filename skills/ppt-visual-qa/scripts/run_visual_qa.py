#!/usr/bin/env python3
"""Executable gate-runner for PPT visual QA.

Usage:
  # Ensure dependencies (first time only):
  # source .venv/bin/activate
  # pip install playwright beautifulsoup4 pyyaml
  # playwright install chromium

  python skills/ppt-visual-qa/scripts/run_visual_qa.py \
    --presentation-dir "docs/presentations/ai-report-Bain-style_20260216_v1" \
    --mode production \
    --strict

This runner executes a practical subset of gates and explicitly marks the rest
as `not_implemented` to avoid false green results.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

# Import from the lib module
from lib import VisualQaRunner


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run executable visual QA gate-runner for PPT HTML slides"
    )
    parser.add_argument(
        "--presentation-dir",
        required=True,
        help="Path to presentation directory containing slide-*.html",
    )
    parser.add_argument(
        "--mode",
        default="production",
        choices=["draft", "production"],
        help="QA mode",
    )
    parser.add_argument(
        "--report-out",
        default=None,
        help="Deprecated: report path is fixed to <presentation-dir>/qa/layout-runtime-report.json",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 1 if gates fail (and optionally if unimplemented exists)",
    )
    parser.add_argument(
        "--allow-unimplemented",
        action="store_true",
        help="Do not fail strict mode on not_implemented gates",
    )
    parser.add_argument(
        "--slides",
        nargs="+",
        type=int,
        help="Specific slide numbers to check (e.g. 1 2 5). If set, performs partial update of existing report.",
    )
    parser.add_argument(
        "--gates",
        nargs="+",
        type=str,
        help="Specific gate IDs to check (e.g. G01 G02 G11). If not set, checks all applicable gates.",
    )
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()
    presentation_dir = Path(args.presentation_dir).resolve()
    if not presentation_dir.exists() or not presentation_dir.is_dir():
        print(f"Invalid presentation directory: {presentation_dir}", file=sys.stderr)
        return 2

    canonical_report_out = (presentation_dir / "qa" / "layout-runtime-report.json").resolve()
    if args.report_out:
        requested_report_out = Path(args.report_out).resolve()
        if requested_report_out != canonical_report_out:
            print(
                "Invalid --report-out. QA report path is fixed to "
                f"{canonical_report_out}",
                file=sys.stderr,
            )
            return 2

    report_out = canonical_report_out

    runner = VisualQaRunner(
        presentation_dir=presentation_dir,
        mode=args.mode,
        report_out=report_out,
        strict=args.strict,
        allow_unimplemented=args.allow_unimplemented,
        target_slides=args.slides,
        target_gates=args.gates,
    )
    return runner.run()


if __name__ == "__main__":
    raise SystemExit(main())
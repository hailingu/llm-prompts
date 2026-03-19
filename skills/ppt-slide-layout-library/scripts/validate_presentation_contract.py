#!/usr/bin/env python3
"""Validate generated PPT HTML package contract for fixed-canvas rendering."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import List


def check_presentation_html(path: Path) -> List[str]:
    errors: List[str] = []
    text = path.read_text(encoding="utf-8")

    required_patterns = {
        "slide viewport": r"slide-viewport",
        "iframe fixed width": r"width\s*:\s*1920px|width=\"1920\"",
        "iframe fixed height": r"height\s*:\s*1080px|height=\"1080\"",
        "fitSlideFrame": r"function\s+fitSlideFrame\s*\(",
        "transform scale": r"transform\s*=\s*`scale\(\$\{scale\}\)`|scale\(",
        "centered translate scale": r"translate\(\s*-50%\s*,\s*-50%\s*\)\s*scale\(",
        "absolute center anchor": r"(left\s*:\s*50%[\s\S]*top\s*:\s*50%)|(top\s*:\s*50%[\s\S]*left\s*:\s*50%)",
    }

    for desc, pattern in required_patterns.items():
        if not re.search(pattern, text):
            errors.append(f"presentation.html missing {desc} contract")

    return errors


def check_slide_html(path: Path) -> List[str]:
    errors: List[str] = []
    text = path.read_text(encoding="utf-8")

    if not re.search(r"1920px|w-\[1920px\]|--slide-width\s*:\s*1920px", text):
        errors.append(f"{path.name}: missing fixed 1920 width contract")
    if not re.search(r"1080px|h-\[1080px\]|--slide-height\s*:\s*1080px", text):
        errors.append(f"{path.name}: missing fixed 1080 height contract")

    forbidden_patterns = {
        "responsive max-width": r"max-width\s*:\s*1280px|max-width\s*:\s*\d+px",
        "viewport min-height": r"min-height\s*:\s*100vh",
    }
    for desc, pattern in forbidden_patterns.items():
        if re.search(pattern, text):
            errors.append(f"{path.name}: forbidden {desc} detected")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate generated presentation HTML fixed-canvas contract.")
    parser.add_argument(
        "presentation_dir",
        help="Directory containing presentation.html and slide-*.html files",
    )
    args = parser.parse_args()

    presentation_dir = Path(args.presentation_dir).resolve()
    if not presentation_dir.exists():
        print(f"ERROR: missing presentation directory: {presentation_dir}")
        return 1

    presentation_html = presentation_dir / "presentation.html"
    if not presentation_html.exists():
        print(f"ERROR: missing {presentation_html}")
        return 1

    slide_files = sorted(presentation_dir.glob("slide-*.html"))
    if not slide_files:
        print(f"ERROR: no slide-*.html files under {presentation_dir}")
        return 1

    errors: List[str] = []
    errors.extend(check_presentation_html(presentation_html))
    for slide in slide_files:
        errors.extend(check_slide_html(slide))

    if errors:
        print("Presentation contract validation FAILED:")
        for item in errors:
            print(f"- {item}")
        return 1

    print(
        "Presentation contract validation PASSED: "
        f"{presentation_html.name} + {len(slide_files)} slide files"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

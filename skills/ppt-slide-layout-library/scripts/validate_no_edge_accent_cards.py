#!/usr/bin/env python3
"""Block edge-accent card styles in generated slide HTML files."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import List, Tuple

# Disallow strong edge accents commonly used as card emphasis.
FORBIDDEN_PATTERNS = [
    re.compile(r"border-l-(?:2|3|4|5|6|7|8)\b"),
    re.compile(r"border-t-(?:2|3|4|5|6|7|8)\b"),
    re.compile(r"border-left\s*:\s*(?:2|3|4|5|6|7|8)px\s+solid", re.IGNORECASE),
    re.compile(r"border-top\s*:\s*(?:2|3|4|5|6|7|8)px\s+solid", re.IGNORECASE),
]

# Allow structural lines and chart guides.
ALLOWLIST_PATTERNS = [
    re.compile(r"header|footer|nav|controls", re.IGNORECASE),
    re.compile(r"border-top\s*:\s*1px\s+solid", re.IGNORECASE),
    re.compile(r"border-left\s*:\s*1px\s+solid", re.IGNORECASE),
    re.compile(r"border-\w*\s*dashed", re.IGNORECASE),
    re.compile(r"chart|axis|grid|timeline", re.IGNORECASE),
]


def is_allowlisted(line: str) -> bool:
    return any(p.search(line) for p in ALLOWLIST_PATTERNS)


def scan_file(path: Path) -> List[Tuple[int, str]]:
    findings: List[Tuple[int, str]] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines, start=1):
        if is_allowlisted(line):
            continue
        if any(p.search(line) for p in FORBIDDEN_PATTERNS):
            findings.append((i, line.strip()))
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate no edge-accent card styles in slide HTML files.")
    parser.add_argument("presentation_dir", help="Directory containing slide-*.html")
    args = parser.parse_args()

    presentation_dir = Path(args.presentation_dir).resolve()
    if not presentation_dir.exists():
        print(f"ERROR: missing presentation directory: {presentation_dir}")
        return 1

    slide_files = sorted(presentation_dir.glob("slide-*.html"))
    if not slide_files:
        print(f"ERROR: no slide-*.html files found under {presentation_dir}")
        return 1

    errors: List[str] = []
    for slide in slide_files:
        findings = scan_file(slide)
        for line_no, content in findings:
            errors.append(f"{slide.name}:{line_no}: {content}")

    if errors:
        print("Edge-accent card validation FAILED:")
        for item in errors:
            print(f"- {item}")
        return 1

    print(f"Edge-accent card validation PASSED: {len(slide_files)} slide files checked")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Validate slide underflow risk (large blank areas caused by over-stretched containers)."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import List


def extract_main(html: str) -> str:
    m = re.search(r"<main[^>]*>(.*?)</main>", html, flags=re.IGNORECASE | re.DOTALL)
    return m.group(1) if m else html


def strip_tags(text: str) -> str:
    text = re.sub(r"<script[\s\S]*?</script>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<style[\s\S]*?</style>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def count_full_height_cards(main_html: str) -> int:
    # Heuristic: content cards that are stretched to full height are a common source of bottom blank blocks.
    pattern = re.compile(
        r"<div[^>]*class=\"[^\"]*(?:h-full|flex-1)[^\"]*(?:rounded|card|bg-)[^\"]*\"[^>]*>",
        flags=re.IGNORECASE,
    )
    return len(pattern.findall(main_html))


def count_flex_lists(main_html: str) -> int:
    # Heuristic: ul.flex-1 with few items tends to create large underflow voids.
    count = 0
    for m in re.finditer(r"<ul[^>]*class=\"[^\"]*flex-1[^\"]*\"[^>]*>(.*?)</ul>", main_html, flags=re.IGNORECASE | re.DOTALL):
        li_count = len(re.findall(r"<li\b", m.group(1), flags=re.IGNORECASE))
        if li_count <= 3:
            count += 1
    return count


def detect_three_col_skeleton(main_html: str) -> bool:
    if re.search(r"grid-cols-3", main_html):
        return True
    # common split patterns used in generated decks
    if len(re.findall(r"w-1/3", main_html)) >= 1:
        return True
    if len(re.findall(r"w-1/2", main_html)) >= 2:
        return True
    return False


def count_visual_blocks(main_html: str) -> int:
    patterns = [r"<canvas\b", r"<svg\b", r"echarts\.init", r"x6", r"id=\"chart", r"id='chart"]
    return sum(len(re.findall(p, main_html, flags=re.IGNORECASE)) for p in patterns)


def validate_slide(path: Path) -> List[str]:
    html = path.read_text(encoding="utf-8")
    main_html = extract_main(html)
    plain = strip_tags(main_html)

    text_len = len(plain)
    full_cards = count_full_height_cards(main_html)
    flex_lists = count_flex_lists(main_html)
    has_three_col = detect_three_col_skeleton(main_html)
    visual_blocks = count_visual_blocks(main_html)

    errors: List[str] = []

    # Gate U1: density floor for multi-column pages.
    if has_three_col and text_len < 1400 and visual_blocks < 2:
        errors.append(
            f"{path.name}: Underflow Density Gate failed (3-col skeleton with low content density: text_len={text_len}, visual_blocks={visual_blocks})"
        )

    # Gate U2: stretched container abuse.
    if full_cards >= 3 and text_len < 2000:
        errors.append(
            f"{path.name}: Underflow Density Gate failed (too many stretched cards: full_height_cards={full_cards}, text_len={text_len})"
        )

    if flex_lists >= 1:
        errors.append(
            f"{path.name}: Adaptive Fallback Gate failed (ul.flex-1 with <=3 items detected; compress or downgrade layout)"
        )

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate underflow density and fallback compliance for slide HTML files.")
    parser.add_argument("presentation_dir", help="Directory containing slide-*.html files")
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
        errors.extend(validate_slide(slide))

    if errors:
        print("Underflow density validation FAILED:")
        for e in errors:
            print(f"- {e}")
        return 1

    print(f"Underflow density validation PASSED: {len(slide_files)} slide files checked")
    return 0


if __name__ == "__main__":
    sys.exit(main())

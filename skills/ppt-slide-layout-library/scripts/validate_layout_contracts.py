#!/usr/bin/env python3
"""Validate ppt-slide-layout-library index and layout contract consistency."""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import yaml


REQUIRED_LAYOUT_CONTRACT_FIELDS = [
    "required_thinking_fields",
    "narrative_fit",
    "compatible_chart_families",
    "compatible_component_families",
    "compatible_map_archetypes",
    "fallback_layouts",
    "overflow_recovery_order",
]


def parse_index_pairs(index_file: Path) -> Tuple[Dict[str, str], List[str]]:
    """Parse id->file mapping from index.yml with regex (tolerates non-strict YAML sections)."""
    mapping: Dict[str, str] = {}
    errors: List[str] = []
    pending_id: str | None = None

    id_re = re.compile(r"^\s*-\s*id:\s*([A-Za-z0-9_\-]+)\s*$")
    file_re = re.compile(r"^\s*file:\s*([A-Za-z0-9_.\-]+)\s*$")

    for line_no, raw_line in enumerate(index_file.read_text(encoding="utf-8").splitlines(), start=1):
        id_match = id_re.match(raw_line)
        if id_match:
            pending_id = id_match.group(1)
            continue

        file_match = file_re.match(raw_line)
        if file_match and pending_id:
            file_name = file_match.group(1)
            if pending_id in mapping and mapping[pending_id] != file_name:
                errors.append(
                    f"duplicate id with conflicting file at {index_file}:{line_no}: {pending_id} -> {mapping[pending_id]} vs {file_name}"
                )
            mapping[pending_id] = file_name
            pending_id = None

    if not mapping:
        errors.append(f"no layout id/file pairs found in {index_file}")

    return mapping, errors


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"top-level YAML object is not a mapping in {path}")
    return data


def validate_layout_files(layout_dir: Path, index_pairs: Dict[str, str]) -> List[str]:
    errors: List[str] = []

    indexed_files = set(index_pairs.values())

    id_to_files = defaultdict(list)
    for layout_id, file_name in index_pairs.items():
        id_to_files[layout_id].append(file_name)

    for layout_id, files in id_to_files.items():
        unique_files = sorted(set(files))
        if len(unique_files) > 1:
            errors.append(f"layout id {layout_id} points to multiple files: {', '.join(unique_files)}")

    for layout_id, file_name in index_pairs.items():
        file_path = layout_dir / file_name
        if not file_path.exists():
            errors.append(f"index entry missing file: {layout_id} -> {file_name}")

    for layout_file in sorted(layout_dir.glob("*.yml")):
        if layout_file.name == "index.yml":
            continue

        if layout_file.name not in indexed_files:
            errors.append(f"layout file is not indexed: {layout_file.name}")

        try:
            data = load_yaml(layout_file)
        except Exception as exc:
            errors.append(f"failed to parse {layout_file.name}: {exc}")
            continue

        layout_id = None
        layout_meta = data.get("layout")
        if isinstance(layout_meta, dict):
            layout_id = layout_meta.get("id")
        if not layout_id and isinstance(data.get("name"), str):
            # Backward-compatible support for legacy layout assets.
            layout_id = data.get("name")
        if not isinstance(layout_id, str) or not layout_id:
            errors.append(f"{layout_file.name}: missing layout id (layout.id or top-level name)")
            continue

        if layout_id not in index_pairs:
            errors.append(f"{layout_file.name}: layout.id {layout_id} not found in index.yml")
        else:
            expected_file = index_pairs[layout_id]
            if expected_file != layout_file.name:
                errors.append(
                    f"{layout_file.name}: layout.id {layout_id} mapped to {expected_file} in index.yml"
                )

        contract = data.get("layout_contract")
        if not isinstance(contract, dict):
            errors.append(f"{layout_file.name}: missing layout_contract")
            continue

        for field in REQUIRED_LAYOUT_CONTRACT_FIELDS:
            if field not in contract:
                errors.append(f"{layout_file.name}: layout_contract missing field {field}")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate layout index and contract consistency.")
    parser.add_argument(
        "--layout-dir",
        default=str(Path(__file__).resolve().parents[1] / "assets" / "layouts"),
        help="Path to layouts directory containing index.yml and *.yml files",
    )
    args = parser.parse_args()

    layout_dir = Path(args.layout_dir).resolve()
    index_file = layout_dir / "index.yml"

    if not index_file.exists():
        print(f"ERROR: missing {index_file}")
        return 1

    index_pairs, parse_errors = parse_index_pairs(index_file)
    errors = list(parse_errors)
    errors.extend(validate_layout_files(layout_dir, index_pairs))

    if errors:
        print("Layout contract validation FAILED:")
        for item in errors:
            print(f"- {item}")
        return 1

    print(
        "Layout contract validation PASSED: "
        f"{len(index_pairs)} indexed layouts checked in {layout_dir}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

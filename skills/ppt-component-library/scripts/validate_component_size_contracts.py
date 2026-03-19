#!/usr/bin/env python3
"""Validate component estimated heights used by layout examples against component contracts."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import yaml


def parse_range(height_range: str) -> Tuple[float, float] | None:
    if not isinstance(height_range, str) or "-" not in height_range:
        return None
    left, right = height_range.split("-", 1)
    try:
        return float(left.strip()), float(right.strip())
    except ValueError:
        return None


def to_number(value: object) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if value is None:
        return None
    text = str(value)
    number = re.sub(r"[^0-9.]", "", text)
    if not number:
        return None
    try:
        return float(number)
    except ValueError:
        return None


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"top-level YAML object is not a mapping in {path}")
    return data


def collect_component_ranges(core_components_path: Path) -> Tuple[Dict[Tuple[str, str], Tuple[float, float]], List[str]]:
    errors: List[str] = []
    data = load_yaml(core_components_path)
    components = data.get("components")
    if not isinstance(components, dict):
        return {}, [f"missing components mapping in {core_components_path}"]

    ranges: Dict[Tuple[str, str], Tuple[float, float]] = {}
    for component_name, component_data in components.items():
        if not isinstance(component_data, dict):
            continue
        variants = ((component_data.get("component_spec") or {}).get("variants") or {})
        if not isinstance(variants, dict):
            continue
        for variant_name, variant_data in variants.items():
            if not isinstance(variant_data, dict):
                continue
            parsed = parse_range(variant_data.get("empirical_height_px"))
            if parsed is None:
                errors.append(
                    f"invalid empirical_height_px for {component_name}.{variant_name}"
                )
                continue
            ranges[(component_name, variant_name)] = parsed

    return ranges, errors


def walk(node: object):
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from walk(value)
    elif isinstance(node, list):
        for item in node:
            yield from walk(item)


def validate_layout_usage(
    layout_dir: Path,
    ranges: Dict[Tuple[str, str], Tuple[float, float]],
) -> Tuple[List[str], int]:
    errors: List[str] = []
    checked = 0

    for layout_file in sorted(layout_dir.glob("*.yml")):
        if layout_file.name == "index.yml":
            continue

        data = load_yaml(layout_file)

        for node in walk(data):
            if not isinstance(node, dict):
                continue

            family = node.get("component_family")
            variant = node.get("variant")
            estimated_height = node.get("estimated_height_px")
            if not (family and variant and estimated_height is not None):
                continue

            checked += 1
            key = (str(family), str(variant))
            numeric_height = to_number(estimated_height)
            if numeric_height is None:
                errors.append(
                    f"{layout_file.name}: cannot parse estimated_height_px for {family}.{variant}: {estimated_height}"
                )
                continue

            if key not in ranges:
                errors.append(
                    f"{layout_file.name}: missing component variant contract for {family}.{variant}"
                )
                continue

            min_h, max_h = ranges[key]
            if not (min_h <= numeric_height <= max_h):
                errors.append(
                    f"{layout_file.name}: {family}.{variant} estimated_height_px={numeric_height} out of range {min_h}-{max_h}"
                )

    return errors, checked


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate layout estimated heights against component contracts.")
    parser.add_argument(
        "--component-core",
        default=str(Path(__file__).resolve().parents[1] / "assets" / "core_components.yml"),
        help="Path to core_components.yml",
    )
    parser.add_argument(
        "--layout-dir",
        default=str(Path(__file__).resolve().parents[2] / "ppt-slide-layout-library" / "assets" / "layouts"),
        help="Path to layout assets directory",
    )
    args = parser.parse_args()

    component_core = Path(args.component_core).resolve()
    layout_dir = Path(args.layout_dir).resolve()

    if not component_core.exists():
        print(f"ERROR: missing component core file: {component_core}")
        return 1
    if not layout_dir.exists():
        print(f"ERROR: missing layout directory: {layout_dir}")
        return 1

    ranges, errors = collect_component_ranges(component_core)
    usage_errors, checked_count = validate_layout_usage(layout_dir, ranges)
    errors.extend(usage_errors)

    if errors:
        print("Component size contract validation FAILED:")
        for item in errors:
            print(f"- {item}")
        return 1

    print(
        "Component size contract validation PASSED: "
        f"{checked_count} layout component estimates checked against {len(ranges)} variant ranges"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

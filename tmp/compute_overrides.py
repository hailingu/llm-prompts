#!/usr/bin/env python3
"""Compute slide_overrides for MFT design_spec.json based on content size estimation."""
import json
import math

with open("docs/presentations/MFT-20260210/slides_semantic.json") as f:
    sem = json.load(f)

# Constants from design_spec
bar_h_standard = 0.55
bar_h_narrow = 0.40
content_margin_top = 0.12
bottom_bar = 0.35
slide_h = 7.5

# Narrow title_bar types in design_spec
narrow_types = {"sequence", "process", "data-process", "flow", "flowchart", "data-heavy"}
no_bar_types = {"title", "section_divider"}


def card_h(attrs_count):
    return 0.35 + 0.02 + attrs_count * 0.30 + 0.24


def estimate_content_h(sd):
    comps = sd.get("components", {})
    h = 0.0

    # comparison_items
    if comps.get("comparison_items"):
        items = comps["comparison_items"]
        max_attrs = max(len(item.get("attributes", {})) for item in items) if items else 0
        h += card_h(max_attrs)

    # decisions
    if comps.get("decisions"):
        h += len(comps["decisions"]) * (0.65 + 0.12)

    # kpis
    if comps.get("kpis"):
        h += 1.2

    # risks (treated like comparison_items)
    if comps.get("risks"):
        items = comps["risks"]
        max_attrs = max(len(item.get("attributes", {})) for item in items) if items else 0
        h += card_h(max_attrs)

    # bullets from content or components
    content_bullets = sd.get("content", [])
    comp_bullets = comps.get("bullets", [])
    if content_bullets:
        h += len(content_bullets[:8]) * 0.48
    elif comp_bullets:
        h += len(comp_bullets[:8]) * 0.48

    # callouts
    if comps.get("callouts"):
        h += 0.8 * len(comps["callouts"])

    # timeline_items
    if comps.get("timeline_items"):
        h += 1.0

    return h


print("{:>3}  {:<16}  {:>5}  {:>6}  {:>8}  {:>6}  {:<20}".format(
    "ID", "Type", "Bar", "Avail", "Content", "Ratio", "Decision"))
print("-" * 75)

overrides = {}

for sd in sem["slides"]:
    sid = sd["id"]
    stype = sd.get("slide_type", "bullet-list")

    if stype in no_bar_types:
        continue

    bar_h = bar_h_narrow if stype in narrow_types else bar_h_standard
    avail_h = slide_h - bar_h - content_margin_top - bottom_bar
    content_h = estimate_content_h(sd)
    fill_ratio = content_h / avail_h if avail_h > 0 else 0

    if fill_ratio < 0.35:
        decision = "center"
        override = {
            "content_fill": "center",
            "estimated_fill_ratio": round(fill_ratio, 2),
            "rationale": "content_h={:.1f} vs avail_h={:.1f}, too sparse for expand".format(content_h, avail_h),
        }
    elif fill_ratio < 0.55:
        max_h = round(content_h * 1.5, 1)
        decision = "expand cap={:.1f}".format(max_h)
        override = {
            "content_fill": "expand",
            "max_card_h": max_h,
            "estimated_fill_ratio": round(fill_ratio, 2),
            "rationale": "moderate density, cap at {:.1f} prevents over-stretch".format(max_h),
        }
    elif fill_ratio < 0.80:
        max_h = round(avail_h * 0.9, 1)
        decision = "expand cap={:.1f}".format(max_h)
        override = {
            "content_fill": "expand",
            "max_card_h": max_h,
            "estimated_fill_ratio": round(fill_ratio, 2),
            "rationale": "good density, gentle expand to {:.1f}".format(max_h),
        }
    else:
        decision = "top-align"
        override = {
            "content_fill": "top-align",
            "estimated_fill_ratio": round(fill_ratio, 2),
            "rationale": "content_h={:.1f} nearly fills avail_h={:.1f}".format(content_h, avail_h),
        }

    overrides[str(sid)] = override
    print("{:>3}  {:<16}  {:>5.2f}  {:>6.2f}  {:>8.2f}  {:>6.2f}  {:<20}".format(
        sid, stype, bar_h, avail_h, content_h, fill_ratio, decision))

print()
print(json.dumps(overrides, indent=2, ensure_ascii=False))

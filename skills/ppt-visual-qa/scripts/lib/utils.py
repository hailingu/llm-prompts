"""Utility functions for PPT Visual QA."""

import re


def strip_tags(text: str) -> str:
    """Remove HTML tags and normalize whitespace."""
    text = re.sub(r"<script[\s\S]*?</script>", " ", text, flags=re.I)
    text = re.sub(r"<style[\s\S]*?</style>", " ", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def count_structured_claims(plain_text: str) -> int:
    """Count structured claims like percentages, currency, years in text."""
    claims = re.findall(r"(\d+(?:\.\d+)?%|\$\d+|\d{4}|\d+[xXBMKT])", plain_text)
    return len(claims)


def infer_layout(html: str) -> str:
    """Infer the layout type from HTML content."""
    low = html.lower()
    has_timeline_container = any(token in low for token in [
        "timeline-item",
        "timeline-track",
        "milestone-timeline",
    ])
    has_timeline_nodes = "timeline-node" in low and any(token in low for token in ["dot", "year", "phase"])
    has_process_container = any(token in low for token in [
        "process-step",
        "step-process-container",
        "process-flow",
        "process-card",
    ])

    if "cover-slide" in low or "thank-you-slide" in low:
        return "cover"
    if "title slide" in low or "title section" in low:
        return "cover"
    if has_process_container:
        return "process"
    if has_timeline_container or has_timeline_nodes:
        return "milestone-timeline"
    if "grid-cols-2" in low or "w-1/2" in low or "w-7/12" in low or "w-5/12" in low or "w-8/12" in low or "w-4/12" in low or "w-2/3" in low or "w-1/3" in low:
        return "dual-column"
    if "grid-cols-3" in low or "w-1/3" in low:
        return "three-column"
    return "unknown"
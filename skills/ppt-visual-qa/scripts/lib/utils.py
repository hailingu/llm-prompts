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
    if "cover-slide" in low or "thank-you-slide" in low:
        return "cover"
    if "title slide" in low or "title section" in low:
        return "cover"
    if "timeline" in low or "milestone" in low or "year" in low and ("dot" in low or "phase" in low):
        return "milestone-timeline"
    if "process" in low or "step" in low:
        return "process"
    if "grid-cols-2" in low or "w-1/2" in low or "w-7/12" in low or "w-5/12" in low or "w-8/12" in low or "w-4/12" in low or "w-2/3" in low or "w-1/3" in low:
        return "dual-column"
    if "grid-cols-3" in low or "w-1/3" in low:
        return "three-column"
    return "unknown"
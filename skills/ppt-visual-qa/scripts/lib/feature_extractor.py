"""Feature extraction from slide HTML."""

import re
from pathlib import Path
from typing import Dict, List

from .models import SlideFeatures
from .utils import strip_tags, count_structured_claims, infer_layout
from .constants import MAIN_OUTER_AVAILABLE_PX, TAILWIND_MB_MAP, TAILWIND_GAP_MAP


class FeatureExtractor:
    """Extract static features from slide HTML files."""

    def __init__(self):
        self.slide_html_map: Dict[str, str] = {}
        self.deck_text_len_map: Dict[str, int] = {}

    def _is_special_exempt_slide(self, html: str) -> bool:
        """Check if slide is a special exempt (cover, thank-you)."""
        markers = ["cover-slide", "thank-you-slide", 'class="slide cover-slide"']
        if any(marker in html for marker in markers):
            return True
        if "感谢您的关注" in html or ">谢谢<" in html:
            return True
        return False

    def extract_features(self, slides: List[Path]) -> Dict[str, SlideFeatures]:
        """Extract features from all slides."""
        features_map: Dict[str, SlideFeatures] = {}

        for slide in slides:
            html = slide.read_text(encoding="utf-8", errors="ignore")
            self.slide_html_map[slide.name] = html
            plain = strip_tags(html)
            self.deck_text_len_map[slide.name] = len(plain)

            has_required_skeleton = all(
                token in html for token in ["slide-header", "slide-main", "slide-footer"]
            )
            special = self._is_special_exempt_slide(html)
            if (not has_required_skeleton) and special:
                has_required_skeleton = True

            # Count structural elements
            n_h2 = len(re.findall(r"<h2\b", html, flags=re.I))
            n_h3 = len(re.findall(r"<h3\b", html, flags=re.I))
            n_h4 = len(re.findall(r"<h4\b", html, flags=re.I))
            n_cards = len(re.findall(r'class="[^"]*\bcard\b', html))
            n_card_float = len(re.findall(r'class="[^"]*\bcard-float\b', html))
            n_insight = len(re.findall(r'class="[^"]*\binsight-card\b', html))
            n_li = len(re.findall(r"<li\b", html, flags=re.I))
            n_tr = len(re.findall(r"<tr\b", html, flags=re.I))
            n_progress = len(re.findall(r'class="[^"]*\b(step|progress|timeline)\b', html))

            # Flex penalty for stacked cards
            flex_penalty = 0
            if "flex-col" in html and n_cards > 2:
                flex_penalty = n_cards * 40

            # Canvas heights
            canvas_heights = [
                int(v) for v in re.findall(r'<canvas[^>]*height="(\d+)"', html, flags=re.I)
            ]
            canvas_sum = sum(canvas_heights)

            # Tailwind spacing
            mb_sum = sum(TAILWIND_MB_MAP.get(int(v), 0) for v in re.findall(r"\bmb-(\d+)\b", html))
            mt_sum = sum(TAILWIND_MB_MAP.get(int(v), 0) for v in re.findall(r"\bmt-(\d+)\b", html))
            gap_sum = sum(TAILWIND_GAP_MAP.get(int(v), 0) for v in re.findall(r"\bgap-(\d+)\b", html))

            # Estimate fixed block budget
            est_fixed = (
                18 + n_h2 * 30 + n_h3 * 24 + n_h4 * 18
                + n_cards * 14 + n_card_float * 14 + n_insight * 20
                + n_li * 9 + n_tr * 8 + n_progress * 8
                + canvas_sum * 0.55 + gap_sum * 0.35 + mb_sum * 0.2 + mt_sum * 0.2
                + flex_penalty
                + 80
            )
            m03_ok = est_fixed <= MAIN_OUTER_AVAILABLE_PX

            features_map[slide.name] = SlideFeatures(
                slide=slide.name,
                text_len=len(plain),
                has_required_skeleton=has_required_skeleton,
                is_special_exempt=special,
                has_chartjs_cdn="cdn.jsdelivr.net/npm/chart.js" in html,
                has_chartjs_usage="new Chart(" in html,
                has_echarts_usage="echarts.init(" in html,
                has_echarts_cdn="echarts" in html and "cdn" in html,
                has_chart_wrap_explicit_height=bool(
                    re.search(r"chart-wrap[^\"]*h-\[\d+px\]", html)
                    or re.search(r'chart-wrap[^>]*style="[^\"]*height\s*:\s*\d+px', html, flags=re.I)
                ),
                has_maintain_aspect_ratio_false="maintainAspectRatio: false" in html,
                has_domcontentloaded_init="DOMContentLoaded" in html,
                has_step_size="stepSize" in html,
                has_empty_data_fallback=("暂无数据" in html) or ("N/A" in html) or ("未提供" in html),
                has_nan_guard=("isNaN(" in html) or ("Number.isNaN" in html) or ("?? 0" in html) or ("|| 0" in html),
                has_labels_data_length_guard=("labels.length" in html and ".data.length" in html),
                has_debug_controls=any(k in html for k in ["brand-switcher", "debug", "grid-overlay", "dev-ruler"]),
                inferred_layout=infer_layout(html),
                is_analysis_like=(not special) and has_required_skeleton,
                structured_claim_count=count_structured_claims(plain),
                has_structured_keywords=any(all(k in plain.lower() for k in group) for group in [["结论", "原因", "建议"], ["insight", "driver", "action"], ["market", "risk", "strategy"], ["现状", "挑战", "对策"]]),
                m03_fixed_block_budget_ok=m03_ok,
                m03_est_fixed_px=float(est_fixed),
            )

        return features_map
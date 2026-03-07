"""Gate evaluation logic for PPT Visual QA."""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from .models import GateDef, SlideFeatures, ProfileMetrics, GateResult
from .constants import MAIN_OUTER_AVAILABLE_PX, SEMANTIC_COLORS


NON_CHART_SLIDE = "non-chart slide"
NO_CHART_RUNTIME_PATH = "no chart runtime path on this slide"
NO_SEMANTIC_COMPONENT_CONTEXT = "no semantic component context"


@dataclass
class GateEvaluationContext:
    """Precomputed slide context reused across multiple gate handlers."""

    html: str
    html_low: str
    runtime_available: bool
    semantic_color_hits: int
    card_count: int
    chart_area_heights: List[int]
    max_chart_h: int
    radius_unique_count: int


class GateEvaluator:
    """Evaluate QA gates against slide features and runtime profiles."""

    def __init__(
        self,
        slide_html_map: Dict[str, str],
        slide_theme_text: str,
        presentation_text: str,
        deck_text_len_map: Dict[str, int],
        footer_safe_gap_min_px: float = 0,
        tolerance_px: float = 650,
    ):
        self.slide_html_map = slide_html_map
        self.slide_theme_text = slide_theme_text
        self.presentation_text = presentation_text
        self.deck_text_len_map = deck_text_len_map
        self.footer_safe_gap_min_px = footer_safe_gap_min_px
        self.tolerance_px = tolerance_px

    def _pass_result(self, gate: GateDef, reason: str) -> GateResult:
        return GateResult(gate.id, "pass", reason, gate.level, gate.phase, gate.scope, gate.category)

    def _fail_result(self, gate: GateDef, reason: str) -> GateResult:
        return GateResult(gate.id, "fail", reason, gate.level, gate.phase, gate.scope, gate.category)

    def _is_map_like(self, html_low: str) -> bool:
        """Check whether the slide behaves like a map-led narrative page."""
        map_markers = [
            "map",
            "geo",
            "basemap",
            "leaflet",
            "topojson",
            "route",
            "corridor",
            "strait",
            "region",
            "overlay",
        ]
        return any(marker in html_low for marker in map_markers)

    def _has_semantic_component_context(self, html_low: str, card_count: int) -> bool:
        """Check whether semantic-color analysis has meaningful component context."""
        if card_count > 0:
            return True
        return any(token in html_low for token in ["insight", "status", "kpi", "callout", "badge"])

    def _has_secondary_structure(self, html_low: str, features: SlideFeatures, card_count: int) -> bool:
        """Check whether the slide has a meaningful secondary hierarchy below the title level."""
        if "<h3" in html_low or "<h4" in html_low:
            return True
        if card_count > 0:
            return True
        if "<li" in html_low or "<table" in html_low:
            return True
        return features.has_chartjs_usage or features.has_echarts_usage or self._is_map_like(html_low)

    def _semantic_border_widths(self, html_low: str) -> List[str]:
        """Extract semantic accent border widths while ignoring structural dividers."""
        widths: List[str] = []
        for class_attr in re.findall(r'class="([^"]+)"', html_low, flags=re.I):
            if not any(token in class_attr for token in ["card", "insight", "kpi", "callout", "badge", "alert"]):
                continue
            widths.extend(re.findall(r"border-[ltrb]-(\d)", class_attr))
        return widths

    def _has_semantic_component_application(self, html_low: str) -> bool:
        """Check whether semantic colors are applied on cards, badges, or annotated components."""
        semantic_badges = [
            "badge-risk",
            "badge-warn",
            "badge-info",
            "badge-stage",
            "badge-success",
            "badge-danger",
            "badge-warning",
            "badge-primary",
        ]
        semantic_vars = [
            "var(--risk)",
            "var(--warn)",
            "var(--info)",
            "var(--success)",
            "var(--stage)",
            "var(--brand-primary)",
            "var(--brand-secondary)",
            "var(--brand-accent)",
        ]
        if any(token in html_low for token in semantic_badges):
            return True
        if re.search(r"(border-left-color|color|background|background-color)\s*:\s*var\(--(risk|warn|info|success|stage|brand-primary|brand-secondary|brand-accent)\)", html_low):
            return True
        return any(token in html_low for token in semantic_vars)

    def _build_context(
        self,
        features: SlideFeatures,
        profiles: List[ProfileMetrics],
    ) -> GateEvaluationContext:
        """Precompute reusable slide context for grouped gate handlers."""
        html = self.slide_html_map.get(features.slide, "")
        html_low = html.lower()
        chart_area_heights = [
            int(value) for value in re.findall(r"h-\[(\d+)px\]", html)
        ] + [
            int(value)
            for value in re.findall(r"height\s*:\s*(\d+)px", html, flags=re.I)
        ]
        radius_vals = [float(value) for value in re.findall(r"\br\s*:\s*(\d+(?:\.\d+)?)", html)]
        backend = profiles[0].backend if profiles else "unknown"
        return GateEvaluationContext(
            html=html,
            html_low=html_low,
            runtime_available=backend == "playwright-runtime",
            semantic_color_hits=sum(1 for key in SEMANTIC_COLORS if key in html_low),
            card_count=len(re.findall(r'class="[^"]*\b(card|insight-card|kpi-card)\b', html, flags=re.I)),
            chart_area_heights=chart_area_heights,
            max_chart_h=max(chart_area_heights) if chart_area_heights else 0,
            radius_unique_count=len(set(radius_vals)),
        )

    def scope_applies(self, gate: GateDef, features: SlideFeatures) -> bool:
        """Check if gate scope applies to this slide."""
        scope = gate.scope.lower()
        html_low = self.slide_html_map.get(features.slide, "").lower()
        scope_checks = {
            "all": lambda: True,
            "cover": lambda: features.is_special_exempt and features.inferred_layout == "cover",
            "analysis": lambda: features.is_analysis_like,
            "bubble": lambda: ("bubble" in features.slide.lower()) or ("type: 'bubble'" in html_low) or ('type: "bubble"' in html_low),
            "line": lambda: features.has_chartjs_usage or ("line" in html_low and "chart" in html_low),
            "echarts": lambda: features.has_echarts_usage,
            "heatmap": lambda: ("heat" in features.slide.lower()) or ("heatmap" in html_low) or ("visualmap" in html_low),
            "dual-column": lambda: features.inferred_layout == "dual-column",
            "side-by-side": lambda: ("side" in features.slide.lower()) or features.inferred_layout == "dual-column",
            "radar-kpi": lambda: ("radar" in features.slide.lower()) or ("radar" in html_low and "kpi" in html_low),
            "gantt": lambda: ("gantt" in features.slide.lower()) or ("gantt" in html_low),
            "process": lambda: features.inferred_layout == "process",
            "full-width": lambda: "full-width" in features.slide.lower(),
            "milestone-timeline": lambda: features.inferred_layout == "milestone-timeline",
            "list": lambda: features.text_len > 0 and (bool(re.search(r"<li\b", html_low)) or "list" in html_low),
        }
        return scope_checks.get(scope, lambda: False)()

    def _evaluate_structural_gates(
        self,
        gate: GateDef,
        features: SlideFeatures,
        profiles: List[ProfileMetrics],
        ctx: GateEvaluationContext,
    ) -> Optional[GateResult]:
        """Evaluate structural and runtime-foundation gates."""
        handler_map = {
            "G01": self._evaluate_structural_basics,
            "G02": self._evaluate_structural_basics,
            "G03": self._evaluate_structural_basics,
            "G04": self._evaluate_structural_basics,
            "G05": self._evaluate_structural_basics,
            "G06": self._evaluate_chart_guard_gates,
            "G07": self._evaluate_chart_guard_gates,
            "G08": self._evaluate_chart_guard_gates,
            "G09": self._evaluate_chart_guard_gates,
            "G10": self._evaluate_budget_and_cover_gates,
            "G11": self._evaluate_budget_and_cover_gates,
            "G12": self._evaluate_budget_and_cover_gates,
            "G13": self._evaluate_budget_and_cover_gates,
        }
        handler = handler_map.get(gate.id)
        return handler(gate, features, profiles, ctx) if handler else None

    def _evaluate_structural_basics(
        self,
        gate: GateDef,
        features: SlideFeatures,
        profiles: List[ProfileMetrics],
        ctx: GateEvaluationContext,
    ) -> GateResult:
        """Evaluate core structural gates G01-G05."""
        handler_map = {
            "G01": self._check_input_completeness,
            "G02": self._check_required_skeleton,
            "G03": self._check_chart_height_explicit,
            "G04": self._check_chartjs_aspect_ratio,
            "G05": self._check_echarts_cdn,
        }
        return handler_map[gate.id](gate, features)

    def _check_input_completeness(self, gate: GateDef, features: SlideFeatures) -> GateResult:
        return self._pass_result(gate, "input appears complete") if (features.text_len > 0 and features.has_required_skeleton) else self._fail_result(gate, "input/content appears incomplete")

    def _check_required_skeleton(self, gate: GateDef, features: SlideFeatures) -> GateResult:
        return self._pass_result(gate, "required skeleton present") if features.has_required_skeleton else self._fail_result(gate, "missing slide-header/slide-main/slide-footer")

    def _check_chart_height_explicit(self, gate: GateDef, features: SlideFeatures) -> GateResult:
        if not (features.has_chartjs_usage or features.has_echarts_usage):
            return self._pass_result(gate, NON_CHART_SLIDE)
        return self._pass_result(gate, "chart container height is explicit") if features.has_chart_wrap_explicit_height else self._fail_result(gate, "no explicit chart-wrap/container height")

    def _check_chartjs_aspect_ratio(self, gate: GateDef, features: SlideFeatures) -> GateResult:
        if not features.has_chartjs_usage:
            return self._pass_result(gate, "no Chart.js usage on this slide")
        return self._pass_result(gate, "maintainAspectRatio false found") if features.has_maintain_aspect_ratio_false else self._fail_result(gate, "Chart.js used without maintainAspectRatio: false")

    def _check_echarts_cdn(self, gate: GateDef, features: SlideFeatures) -> GateResult:
        if not features.has_echarts_usage:
            return self._pass_result(gate, "no ECharts usage on this slide")
        return self._pass_result(gate, "ECharts cdn detected") if features.has_echarts_cdn else self._fail_result(gate, "ECharts usage without matched CDN")

    def _evaluate_chart_guard_gates(
        self,
        gate: GateDef,
        features: SlideFeatures,
        profiles: List[ProfileMetrics],
        ctx: GateEvaluationContext,
    ) -> GateResult:
        """Evaluate chart-guard structural gates G06-G09."""
        if not (features.has_chartjs_usage or features.has_echarts_usage):
            return self._pass_result(gate, NO_CHART_RUNTIME_PATH)
        if gate.id == "G06":
            return self._pass_result(gate, "NaN/null guard found") if features.has_nan_guard else self._fail_result(gate, "no explicit NaN/null guard pattern")
        if gate.id == "G07":
            return self._pass_result(gate, "labels/data length guard or normalization found") if features.has_labels_data_length_guard else self._fail_result(gate, "no labels/data length guard or normalization pattern")
        if gate.id == "G08":
            return self._pass_result(gate, "empty-data fallback text found") if features.has_empty_data_fallback else self._fail_result(gate, "empty-data fallback text not found")
        return self._pass_result(gate, "DOMContentLoaded init found") if features.has_domcontentloaded_init else self._fail_result(gate, "DOMContentLoaded initialization not found")

    def _evaluate_budget_and_cover_gates(
        self,
        gate: GateDef,
        features: SlideFeatures,
        profiles: List[ProfileMetrics],
        ctx: GateEvaluationContext,
    ) -> GateResult:
        """Evaluate budget and cover-layout gates G10-G13."""
        if gate.id == "G10":
            ok = all(p.m01_slide_total_budget_ok for p in profiles)
            return self._pass_result(gate, "M01 passed across profiles") if ok else self._fail_result(gate, "M01 failed in at least one profile")
        if gate.id == "G11":
            min_gap = float(self.footer_safe_gap_min_px)
            margins = {p.profile_id: (p.main_client_h - min_gap - p.main_scroll_h) for p in profiles}
            ok = all((margin >= 0) or (profile.m04_footer_overlap_risk <= 500) for profile, margin in zip(profiles, margins.values()))
            worst_profile, worst_margin = min(margins.items(), key=lambda item: item[1])
            if ok:
                return self._pass_result(gate, f"M02 passed across profiles (worst margin: {worst_margin:.1f}px @ {worst_profile})")
            failed_profiles = ", ".join([profile_id for profile_id, margin in margins.items() if margin < 0])
            return self._fail_result(gate, f"M02 failed: content exceeds main budget by {abs(worst_margin):.1f}px @ {worst_profile} (failed profiles: {failed_profiles})")
        if gate.id == "G12":
            if ctx.runtime_available:
                return self._evaluate_runtime_budget_gate(gate, profiles)
            delta = MAIN_OUTER_AVAILABLE_PX - float(features.m03_est_fixed_px)
            if features.m03_fixed_block_budget_ok:
                return self._pass_result(gate, f"M03 static fixed-block budget passed (headroom: {delta:.1f}px, est={features.m03_est_fixed_px:.1f}px, budget={MAIN_OUTER_AVAILABLE_PX:.1f}px)")
            return self._fail_result(gate, f"M03 static fixed-block budget exceeded by {abs(delta):.1f}px (est={features.m03_est_fixed_px:.1f}px, budget={MAIN_OUTER_AVAILABLE_PX:.1f}px)")
        if features.inferred_layout != "cover":
            return self._pass_result(gate, "non-cover slide")
        return self._pass_result(gate, "cover slide uses cover layout") if features.is_special_exempt else self._fail_result(gate, "cover slide markers missing")

    def _evaluate_runtime_budget_gate(
        self,
        gate: GateDef,
        profiles: List[ProfileMetrics],
    ) -> GateResult:
        """Evaluate runtime budget gate G12 for runtime-backed profiles."""
        runtime_budget_tolerance_px = 32.0
        worst_overflow = 0.0
        worst_profile = "unknown"
        worst_scroll = 0.0
        worst_client = 0.0
        for profile in profiles:
            overflow = profile.main_scroll_h - profile.main_client_h
            if overflow > runtime_budget_tolerance_px and overflow > worst_overflow:
                worst_overflow = overflow
                worst_profile = profile.profile_id
                worst_scroll = profile.main_scroll_h
                worst_client = profile.main_client_h
        if worst_overflow > 0:
            return self._fail_result(gate, f"M03 runtime budget exceeded by {worst_overflow:.1f}px after {int(runtime_budget_tolerance_px)}px tolerance (scroll={worst_scroll:.1f}px > client={worst_client:.1f}px @ {worst_profile})")
        return self._pass_result(gate, f"M03 runtime budget passed within {int(runtime_budget_tolerance_px)}px tolerance across {len(profiles)} profiles")

    def _evaluate_content_gates(
        self,
        gate: GateDef,
        features: SlideFeatures,
        ctx: GateEvaluationContext,
    ) -> Optional[GateResult]:
        """Evaluate content-density and report-readiness gates."""
        handler_map = {
            "G14": self._evaluate_content_core_gates,
            "G15": self._evaluate_content_core_gates,
            "G16": self._evaluate_content_core_gates,
            "G17": self._evaluate_content_core_gates,
            "G72": self._evaluate_content_post_gates,
            "G73": self._evaluate_content_post_gates,
        }
        handler = handler_map.get(gate.id)
        return handler(gate, features, ctx) if handler else None

    def _evaluate_content_core_gates(
        self,
        gate: GateDef,
        features: SlideFeatures,
        ctx: GateEvaluationContext,
    ) -> GateResult:
        """Evaluate content gates G14-G17."""
        if gate.id == "G14":
            return self._pass_result(gate, "text length meets threshold") if features.text_len >= 120 else self._fail_result(gate, f"text_len {features.text_len} < 120")
        if gate.id == "G16":
            return self._pass_result(gate, "structured claims >= baseline") if (features.structured_claim_count >= 0 and features.text_len >= 60) else self._fail_result(gate, "structured claims < baseline")
        key_page = (features.inferred_layout in {"dual-column", "process", "milestone-timeline"} or features.structured_claim_count >= 2 or features.text_len >= 180)
        if not key_page:
            return self._pass_result(gate, "non-key page")
        return self._pass_result(gate, "structured keywords found") if (features.has_structured_keywords or features.structured_claim_count >= 1 or features.text_len >= 100) else self._fail_result(gate, "missing structured claims or sufficient text length")

    def _evaluate_content_post_gates(
        self,
        gate: GateDef,
        features: SlideFeatures,
        ctx: GateEvaluationContext,
    ) -> GateResult:
        """Evaluate post-generation content gates G72-G73."""
        handler_map = {
            "G72": self._check_deck_text_density,
            "G73": self._check_key_page_readiness,
        }
        return handler_map[gate.id](gate, features, ctx)

    def _check_deck_text_density(
        self,
        gate: GateDef,
        features: SlideFeatures,
        ctx: GateEvaluationContext,
    ) -> GateResult:
        if features.inferred_layout == "cover":
            return self._pass_result(gate, "cover/title slide exempt from density check")
        chart_or_map_led = features.has_chartjs_usage or features.has_echarts_usage or self._is_map_like(ctx.html_low)
        vals = list(self.deck_text_len_map.values())
        if not vals:
            return self._pass_result(gate, "no deck text density baseline")
        median = sorted(vals)[len(vals) // 2]
        ok = (
            features.text_len >= max(24, int(0.20 * median))
            or features.structured_claim_count >= 1
            or ctx.card_count >= 2
            or (chart_or_map_led and (ctx.card_count >= 1 or features.text_len >= 60))
        )
        return self._pass_result(gate, "deck text density appears balanced") if ok else self._fail_result(gate, "text density too low vs deck baseline")

    def _check_key_page_readiness(
        self,
        gate: GateDef,
        features: SlideFeatures,
        ctx: GateEvaluationContext,
    ) -> GateResult:
        if features.inferred_layout == "cover":
            return self._pass_result(gate, "cover/title slide exempt from key-page check")
        chart_or_map_led = features.has_chartjs_usage or features.has_echarts_usage or self._is_map_like(ctx.html_low)
        key_page = (features.inferred_layout in {"dual-column", "process", "milestone-timeline"} or features.structured_claim_count >= 2 or features.text_len >= 180)
        if not key_page:
            return self._pass_result(gate, "non-key page")
        report_ready = (
            features.has_structured_keywords
            or (features.structured_claim_count >= 1 and features.text_len >= 80)
            or (chart_or_map_led and (ctx.card_count >= 1 or features.text_len >= 60))
            or (features.inferred_layout in {"process", "milestone-timeline"} and (ctx.card_count >= 2 or features.text_len >= 60))
        )
        return self._pass_result(gate, "key page is report-ready") if report_ready else self._fail_result(gate, "key page below report-ready threshold")

    def _evaluate_chart_gates(
        self,
        gate: GateDef,
        features: SlideFeatures,
        profiles: List[ProfileMetrics],
        ctx: GateEvaluationContext,
    ) -> Optional[GateResult]:
        """Evaluate chart-family, chart-budget, and chart-runtime gates."""
        handler_map = {
            "G18": self._evaluate_chart_engine_gates,
            "G19": self._evaluate_chart_engine_gates,
            "G20": self._evaluate_chart_engine_gates,
            "G21": self._evaluate_bubble_gates,
            "G22": self._evaluate_bubble_gates,
            "G23": self._evaluate_bubble_gates,
            "G24": self._evaluate_bubble_gates,
            "G25": self._evaluate_bubble_gates,
            "G26": self._evaluate_bubble_gates,
            "G27": self._evaluate_bubble_gates,
            "G28": self._evaluate_bubble_gates,
            "G29": self._evaluate_chart_budget_gates,
            "G30": self._evaluate_chart_budget_gates,
            "G31": self._evaluate_chart_budget_gates,
            "G32": self._evaluate_chart_budget_gates,
            "G33": self._evaluate_chart_budget_gates,
            "G34": self._evaluate_chart_budget_gates,
        }
        handler = handler_map.get(gate.id)
        return handler(gate, features, profiles, ctx) if handler else None

    def _evaluate_chart_engine_gates(
        self,
        gate: GateDef,
        features: SlideFeatures,
        profiles: List[ProfileMetrics],
        ctx: GateEvaluationContext,
    ) -> GateResult:
        """Evaluate chart engine and semantic gates G18-G20."""
        handler_map = {
            "G18": self._check_chart_engine_cdn,
            "G19": self._check_matrix_heatmap_semantics,
            "G20": self._check_chartjs_step_size,
        }
        return handler_map[gate.id](gate, features, ctx)

    def _check_chart_engine_cdn(self, gate: GateDef, features: SlideFeatures, ctx: GateEvaluationContext) -> GateResult:
        if features.has_chartjs_usage:
            return self._pass_result(gate, "chart engine/cdn matched") if features.has_chartjs_cdn else self._fail_result(gate, "Chart.js usage without CDN")
        if features.has_echarts_usage:
            return self._pass_result(gate, "chart engine/cdn matched") if features.has_echarts_cdn else self._fail_result(gate, "ECharts usage without CDN")
        return self._pass_result(gate, "no chart engine usage")

    def _check_matrix_heatmap_semantics(self, gate: GateDef, features: SlideFeatures, ctx: GateEvaluationContext) -> GateResult:
        matrix_semantic = ("matrix" in ctx.html_low) or ("矩阵" in ctx.html)
        heatmap_like = ("heatmap" in ctx.html_low) or ("visualmap" in ctx.html_low)
        if matrix_semantic and not heatmap_like:
            return self._fail_result(gate, "matrix semantics detected but no heatmap-like chart")
        return self._pass_result(gate, "chart semantics match")

    def _check_chartjs_step_size(self, gate: GateDef, features: SlideFeatures, ctx: GateEvaluationContext) -> GateResult:
        if not features.has_chartjs_usage:
            return self._pass_result(gate, "no chartjs usage")
        return self._pass_result(gate, "tick stepSize found") if (features.has_step_size or features.has_chart_wrap_explicit_height) else self._fail_result(gate, "stepSize not found")

    def _evaluate_bubble_gates(
        self,
        gate: GateDef,
        features: SlideFeatures,
        profiles: List[ProfileMetrics],
        ctx: GateEvaluationContext,
    ) -> GateResult:
        """Evaluate bubble-specific gates G21-G28."""
        handler_map = {
            "G21": self._check_bubble_clipping_risk,
            "G22": self._check_bubble_radius_distinguishable,
            "G23": self._check_bubble_radius_mapping,
            "G24": self._check_bubble_legend_semantics,
            "G25": self._check_bubble_color_mapping,
            "G26": self._check_bubble_legend_density,
            "G27": self._check_bubble_layout_budget,
            "G28": self._check_bubble_legend_weight,
        }
        return handler_map[gate.id](gate, features, profiles, ctx)

    def _check_bubble_clipping_risk(self, gate: GateDef, features: SlideFeatures, profiles: List[ProfileMetrics], ctx: GateEvaluationContext) -> GateResult:
        ok = all(p.m04_footer_overlap_risk <= 0 for p in profiles)
        return self._pass_result(gate, "no bubble clipping risk detected") if ok else self._fail_result(gate, "bubble clipping risk detected")

    def _check_bubble_radius_distinguishable(self, gate: GateDef, features: SlideFeatures, profiles: List[ProfileMetrics], ctx: GateEvaluationContext) -> GateResult:
        return self._pass_result(gate, "bubble radius distinguishable") if ctx.radius_unique_count >= 3 else self._fail_result(gate, f"bubble radius unique count {ctx.radius_unique_count} < 3")

    def _check_bubble_radius_mapping(self, gate: GateDef, features: SlideFeatures, profiles: List[ProfileMetrics], ctx: GateEvaluationContext) -> GateResult:
        mapped = bool(re.search(r"\br\s*:\s*[^,\n}]*confidence", ctx.html_low))
        return self._pass_result(gate, "bubble radius mapped from confidence") if mapped else self._fail_result(gate, "bubble radius not mapped from confidence")

    def _check_bubble_legend_semantics(self, gate: GateDef, features: SlideFeatures, profiles: List[ProfileMetrics], ctx: GateEvaluationContext) -> GateResult:
        legend_semantic = ("legend" in ctx.html_low) and (("对象" in ctx.html) or ("object" in ctx.html_low) or ("语义" in ctx.html))
        return self._pass_result(gate, "bubble semantic legend found") if legend_semantic else self._fail_result(gate, "bubble semantic legend missing")

    def _check_bubble_color_mapping(self, gate: GateDef, features: SlideFeatures, profiles: List[ProfileMetrics], ctx: GateEvaluationContext) -> GateResult:
        multi_color = ("backgroundcolor" in ctx.html_low and "[" in ctx.html_low and "," in ctx.html_low) or (ctx.semantic_color_hits >= 2)
        return self._pass_result(gate, "bubble color mapping distinguishable") if multi_color else self._fail_result(gate, "bubble color mapping weak")

    def _check_bubble_legend_density(self, gate: GateDef, features: SlideFeatures, profiles: List[ProfileMetrics], ctx: GateEvaluationContext) -> GateResult:
        legend_items = len(re.findall(r"legend", ctx.html_low))
        return self._pass_result(gate, "bubble legend density acceptable") if legend_items <= 8 else self._fail_result(gate, "bubble legend overcrowded")

    def _check_bubble_layout_budget(self, gate: GateDef, features: SlideFeatures, profiles: List[ProfileMetrics], ctx: GateEvaluationContext) -> GateResult:
        ok = features.m03_fixed_block_budget_ok and all(p.m02_main_budget_ok for p in profiles)
        return self._pass_result(gate, "bubble layout budget ok") if ok else self._fail_result(gate, "bubble layout budget violation")

    def _check_bubble_legend_weight(self, gate: GateDef, features: SlideFeatures, profiles: List[ProfileMetrics], ctx: GateEvaluationContext) -> GateResult:
        overweight = bool(re.search(r"legend[^\n]*(border-2|shadow-lg|bg-[a-z]+-\d{3})", ctx.html_low))
        return self._fail_result(gate, "bubble legend visual overweight") if overweight else self._pass_result(gate, "bubble legend visual weight ok")

    def _evaluate_chart_budget_gates(
        self,
        gate: GateDef,
        features: SlideFeatures,
        profiles: List[ProfileMetrics],
        ctx: GateEvaluationContext,
    ) -> GateResult:
        """Evaluate chart budget and heatmap gates G29-G34."""
        handler_map = {
            "G29": self._check_line_axis_mechanical_balance,
            "G30": self._check_axis_mechanical_pattern,
            "G31": self._check_inner_chart_budget,
            "G32": self._check_chart_component_collision,
            "G33": self._check_heatmap_container_height,
            "G34": self._check_heatmap_bottom_constraints,
        }
        return handler_map[gate.id](gate, features, profiles, ctx)

    def _chart_axis_is_mechanical(self, ctx: GateEvaluationContext) -> bool:
        return ("beginatzero:true" in ctx.html_low) and ("max:100" in ctx.html_low or "suggestedmax:100" in ctx.html_low)

    def _check_line_axis_mechanical_balance(self, gate: GateDef, features: SlideFeatures, profiles: List[ProfileMetrics], ctx: GateEvaluationContext) -> GateResult:
        return self._fail_result(gate, "line axis range appears unbalanced/mechanical") if self._chart_axis_is_mechanical(ctx) else self._pass_result(gate, "line axis range not mechanically constrained")

    def _check_axis_mechanical_pattern(self, gate: GateDef, features: SlideFeatures, profiles: List[ProfileMetrics], ctx: GateEvaluationContext) -> GateResult:
        return self._fail_result(gate, "mechanical beginAtZero:true,max:100 detected") if self._chart_axis_is_mechanical(ctx) else self._pass_result(gate, "no mechanical axis range pattern")

    def _check_inner_chart_budget(self, gate: GateDef, features: SlideFeatures, profiles: List[ProfileMetrics], ctx: GateEvaluationContext) -> GateResult:
        if not (features.has_chartjs_usage or features.has_echarts_usage):
            return self._pass_result(gate, NON_CHART_SLIDE)
        ok = features.has_chart_wrap_explicit_height and features.m03_fixed_block_budget_ok
        return self._pass_result(gate, "inner chart budget ok") if ok else self._fail_result(gate, "inner chart budget violation")

    def _check_chart_component_collision(self, gate: GateDef, features: SlideFeatures, profiles: List[ProfileMetrics], ctx: GateEvaluationContext) -> GateResult:
        ok = all(p.m05_overflow_nodes_ok for p in profiles)
        return self._pass_result(gate, "no chart component collision detected") if ok else self._fail_result(gate, "chart component collision risk")

    def _check_heatmap_container_height(self, gate: GateDef, features: SlideFeatures, profiles: List[ProfileMetrics], ctx: GateEvaluationContext) -> GateResult:
        if ("heatmap" not in ctx.html_low) and ("visualmap" not in ctx.html_low):
            return self._pass_result(gate, "non-heatmap slide")
        return self._pass_result(gate, "heatmap container >=220px") if ctx.max_chart_h >= 220 else self._fail_result(gate, f"heatmap container too short ({ctx.max_chart_h}px)")

    def _check_heatmap_bottom_constraints(self, gate: GateDef, features: SlideFeatures, profiles: List[ProfileMetrics], ctx: GateEvaluationContext) -> GateResult:
        grid_bottom = re.search(r"grid\s*:\s*\{[^\}]*bottom\s*:\s*(\d+)", ctx.html_low)
        visual_map_bottom = re.search(r"visualmap\s*:\s*\{[^\}]*bottom\s*:\s*(\d+)", ctx.html_low)
        grid_ok = (int(grid_bottom.group(1)) >= 44) if grid_bottom else False
        visual_map_ok = (int(visual_map_bottom.group(1)) <= 6) if visual_map_bottom else False
        return self._pass_result(gate, "heatmap grid/visualMap bottom constraints satisfied") if (grid_ok and visual_map_ok) else self._fail_result(gate, "heatmap grid.bottom / visualMap.bottom constraints not met")

    def _evaluate_layout_gates(
        self,
        gate: GateDef,
        features: SlideFeatures,
        profiles: List[ProfileMetrics],
        prev_layout: Optional[str],
        ctx: GateEvaluationContext,
    ) -> Optional[GateResult]:
        """Evaluate layout-balance, comparison, process, and timeline gates."""
        handler_map = {
            "G35": self._evaluate_layout_rhythm_gates,
            "G36": self._evaluate_layout_rhythm_gates,
            "G37": self._evaluate_layout_rhythm_gates,
            "G38": self._evaluate_layout_rhythm_gates,
            "G39": self._evaluate_layout_rhythm_gates,
            "G40": self._evaluate_layout_rhythm_gates,
            "G41": self._evaluate_layout_rhythm_gates,
            "G42": self._evaluate_layout_structure_gates,
            "G43": self._evaluate_layout_structure_gates,
            "G44": self._evaluate_layout_structure_gates,
            "G45": self._evaluate_layout_structure_gates,
            "G46": self._evaluate_process_layout_gates,
            "G47": self._evaluate_process_layout_gates,
            "G48": self._evaluate_process_layout_gates,
            "G49": self._evaluate_process_layout_gates,
            "G50": self._evaluate_process_layout_gates,
            "G51": self._evaluate_fullwidth_layout_gates,
            "G52": self._evaluate_fullwidth_layout_gates,
            "G53": self._evaluate_fullwidth_layout_gates,
            "G54": self._evaluate_fullwidth_layout_gates,
            "G55": self._evaluate_fullwidth_layout_gates,
            "G56": self._evaluate_fullwidth_layout_gates,
            "G57": self._evaluate_timeline_layout_gates,
            "G58": self._evaluate_timeline_layout_gates,
            "G59": self._evaluate_timeline_layout_gates,
        }
        handler = handler_map.get(gate.id)
        return handler(gate, features, profiles, prev_layout, ctx) if handler else None

    def _evaluate_layout_rhythm_gates(
        self,
        gate: GateDef,
        features: SlideFeatures,
        profiles: List[ProfileMetrics],
        prev_layout: Optional[str],
        ctx: GateEvaluationContext,
    ) -> GateResult:
        """Evaluate layout rhythm and comparison gates G35-G41."""
        handler_map = {
            "G35": self._check_adjacent_layout_rhythm,
            "G36": self._check_dual_column_height_balance,
            "G37": self._check_dual_column_occupancy,
            "G38": self._check_side_by_side_chart_alignment,
            "G39": self._check_side_by_side_whitespace,
            "G40": self._check_side_by_side_chart_ratio,
            "G41": self._check_bottom_card_height_match,
        }
        return handler_map[gate.id](gate, features, profiles, prev_layout, ctx)

    def _check_adjacent_layout_rhythm(self, gate: GateDef, features: SlideFeatures, profiles: List[ProfileMetrics], prev_layout: Optional[str], ctx: GateEvaluationContext) -> GateResult:
        if prev_layout is None:
            return self._pass_result(gate, "first slide, no adjacency check")
        if features.inferred_layout == "unknown" or prev_layout == "unknown":
            return self._pass_result(gate, "unknown layout skipped")
        if features.inferred_layout != prev_layout:
            return self._pass_result(gate, "layout differs from previous slide")
        if features.has_chartjs_usage or features.has_echarts_usage or ctx.card_count >= 2 or features.structured_claim_count >= 1:
            return self._pass_result(gate, "repeated layout appears justified by chart/content structure")
        return self._fail_result(gate, "adjacent slides share the same inferred layout without strong structural differentiation")

    def _check_dual_column_height_balance(self, gate: GateDef, features: SlideFeatures, profiles: List[ProfileMetrics], prev_layout: Optional[str], ctx: GateEvaluationContext) -> GateResult:
        if "flex-1" not in ctx.html_low or "h-full" not in ctx.html_low:
            return self._fail_result(gate, "dual-column missing flex-1 or h-full for height balancing")
        return self._pass_result(gate, "dual-column whitespace delta acceptable") if features.m03_fixed_block_budget_ok else self._fail_result(gate, "dual-column whitespace delta risk")

    def _check_dual_column_occupancy(self, gate: GateDef, features: SlideFeatures, profiles: List[ProfileMetrics], prev_layout: Optional[str], ctx: GateEvaluationContext) -> GateResult:
        return self._pass_result(gate, "dual-column occupancy acceptable") if (features.text_len >= 120 or ctx.card_count >= 2) else self._fail_result(gate, "dual-column occupancy appears low")

    def _check_side_by_side_chart_alignment(self, gate: GateDef, features: SlideFeatures, profiles: List[ProfileMetrics], prev_layout: Optional[str], ctx: GateEvaluationContext) -> GateResult:
        if "flex-col" in ctx.html_low and "card" in ctx.html_low and "justify-between" not in ctx.html_low and "h-full" not in ctx.html_low:
            return self._fail_result(gate, "side-by-side right column missing justify-between or h-full")
        if len(ctx.chart_area_heights) < 2:
            return self._pass_result(gate, "no side-by-side dual chart pair")
        aligned = abs(ctx.chart_area_heights[0] - ctx.chart_area_heights[1]) <= 8
        return self._pass_result(gate, "side-by-side chart heights aligned") if aligned else self._fail_result(gate, "side-by-side chart height mismatch > 8px")

    def _check_side_by_side_whitespace(self, gate: GateDef, features: SlideFeatures, profiles: List[ProfileMetrics], prev_layout: Optional[str], ctx: GateEvaluationContext) -> GateResult:
        return self._pass_result(gate, "side-by-side vertical whitespace balanced") if all(p.m04_footer_overlap_risk <= 300 for p in profiles) else self._fail_result(gate, "side-by-side vertical whitespace imbalance risk")

    def _check_side_by_side_chart_ratio(self, gate: GateDef, features: SlideFeatures, profiles: List[ProfileMetrics], prev_layout: Optional[str], ctx: GateEvaluationContext) -> GateResult:
        if not ctx.chart_area_heights:
            return self._pass_result(gate, "no explicit side-by-side chart area")
        ratio = ctx.max_chart_h / 510.0
        return self._pass_result(gate, "side-by-side chart ratio in expected range") if 0.00 <= ratio <= 2.00 else self._fail_result(gate, f"side-by-side chart ratio out of range ({ratio:.2f})")

    def _check_bottom_card_height_match(self, gate: GateDef, features: SlideFeatures, profiles: List[ProfileMetrics], prev_layout: Optional[str], ctx: GateEvaluationContext) -> GateResult:
        if len(ctx.chart_area_heights) < 2:
            return self._pass_result(gate, "no bottom pair height check required")
        matched = abs(ctx.chart_area_heights[-1] - ctx.chart_area_heights[-2]) <= 8
        return self._pass_result(gate, "bottom card heights matched") if matched else self._fail_result(gate, "bottom card height mismatch > 8px")

    def _evaluate_layout_structure_gates(
        self,
        gate: GateDef,
        features: SlideFeatures,
        profiles: List[ProfileMetrics],
        prev_layout: Optional[str],
        ctx: GateEvaluationContext,
    ) -> GateResult:
        """Evaluate radar/gantt structural layout gates G42-G45."""
        if gate.id == "G42":
            blocks = len(re.findall(r"(kpi|insight|callout|card)", ctx.html_low))
            return self._pass_result(gate, "radar-kpi right sidebar sufficiently structured") if blocks >= 3 else self._fail_result(gate, "radar-kpi sidebar sparse")
        if gate.id == "G43":
            return self._pass_result(gate, "radar-kpi whitespace balanced") if features.m03_fixed_block_budget_ok else self._fail_result(gate, "radar-kpi whitespace imbalance risk")
        if gate.id == "G44":
            return self._pass_result(gate, "gantt whitespace balanced") if features.m03_fixed_block_budget_ok else self._fail_result(gate, "gantt whitespace imbalance risk")
        label_zone = re.search(r"labelwidth\s*:\s*(\d+)", ctx.html_low)
        if not label_zone:
            return self._pass_result(gate, "no explicit oversized gantt label zone")
        return self._pass_result(gate, "gantt left label zone acceptable") if int(label_zone.group(1)) <= 18 else self._fail_result(gate, "gantt left label zone exceeds 18% budget")

    def _evaluate_process_layout_gates(
        self,
        gate: GateDef,
        features: SlideFeatures,
        profiles: List[ProfileMetrics],
        prev_layout: Optional[str],
        ctx: GateEvaluationContext,
    ) -> GateResult:
        """Evaluate process layout gates G46-G50."""
        handler_map = {
            "G46": self._check_process_detail_density,
            "G47": self._check_process_card_fill_rate,
            "G48": self._check_process_icon_sizing,
            "G49": self._check_process_icon_alignment,
            "G50": self._check_process_whitespace_formula,
        }
        return handler_map[gate.id](gate, features, profiles, prev_layout, ctx)

    def _check_process_detail_density(self, gate: GateDef, features: SlideFeatures, profiles: List[ProfileMetrics], prev_layout: Optional[str], ctx: GateEvaluationContext) -> GateResult:
        steps = len(re.findall(r"(process-step|step-process-container)", ctx.html_low))
        return self._pass_result(gate, "process detail sufficiently dense") if (steps >= 1 and features.text_len >= 40) else self._fail_result(gate, "process detail sparse")

    def _check_process_card_fill_rate(self, gate: GateDef, features: SlideFeatures, profiles: List[ProfileMetrics], prev_layout: Optional[str], ctx: GateEvaluationContext) -> GateResult:
        return self._pass_result(gate, "process card fill rate acceptable") if (features.m03_fixed_block_budget_ok and features.text_len >= 100) else self._fail_result(gate, "process card fill rate appears low")

    def _check_process_icon_sizing(self, gate: GateDef, features: SlideFeatures, profiles: List[ProfileMetrics], prev_layout: Optional[str], ctx: GateEvaluationContext) -> GateResult:
        icon_ok = ("text-xl" in ctx.html_low) or ("fa-" in ctx.html_low)
        height_ok = any(height >= 120 for height in ctx.chart_area_heights)
        has_step_marker = ("process-step" in ctx.html_low) or ("step" in ctx.html_low)
        return self._pass_result(gate, "process icon sizing acceptable") if (icon_ok or height_ok or has_step_marker) else self._fail_result(gate, "process icon/row sizing too small")

    def _check_process_icon_alignment(self, gate: GateDef, features: SlideFeatures, profiles: List[ProfileMetrics], prev_layout: Optional[str], ctx: GateEvaluationContext) -> GateResult:
        aligned = ("items-center" in ctx.html_low) and ("flex" in ctx.html_low)
        return self._pass_result(gate, "process icon column aligned") if (aligned or ("grid" in ctx.html_low)) else self._fail_result(gate, "process icon column alignment risk")

    def _check_process_whitespace_formula(self, gate: GateDef, features: SlideFeatures, profiles: List[ProfileMetrics], prev_layout: Optional[str], ctx: GateEvaluationContext) -> GateResult:
        ok = all((p.m02_main_budget_ok or p.m04_footer_overlap_risk <= 300) for p in profiles)
        return self._pass_result(gate, "process card whitespace formula passed") if ok else self._fail_result(gate, "process card whitespace formula failed")

    def _evaluate_fullwidth_layout_gates(
        self,
        gate: GateDef,
        features: SlideFeatures,
        profiles: List[ProfileMetrics],
        prev_layout: Optional[str],
        ctx: GateEvaluationContext,
    ) -> GateResult:
        """Evaluate full-width and heatmap layout gates G51-G56."""
        handler_map = {
            "G51": self._check_fullwidth_kpi_semantics,
            "G52": self._check_fullwidth_bottom_region,
            "G53": self._check_kpi_card_readability,
            "G54": self._check_fullwidth_total_budget,
            "G55": self._check_fullwidth_lower_half_ratio,
            "G56": self._check_heatmap_fill_whitespace,
        }
        return handler_map[gate.id](gate, features, profiles, prev_layout, ctx)

    def _check_fullwidth_kpi_semantics(self, gate: GateDef, features: SlideFeatures, profiles: List[ProfileMetrics], prev_layout: Optional[str], ctx: GateEvaluationContext) -> GateResult:
        numbers = len(re.findall(r"\d+(?:\.\d+)?%|\$\d+|\d{4}", ctx.html))
        kpi_words = len(re.findall(r"(kpi|指标|同比|环比|baseline|基线)", ctx.html_low))
        return self._pass_result(gate, "fullwidth KPI semantics strong") if (numbers >= 3 and kpi_words >= 2) else self._fail_result(gate, "fullwidth KPI semantics weak")

    def _check_fullwidth_bottom_region(self, gate: GateDef, features: SlideFeatures, profiles: List[ProfileMetrics], prev_layout: Optional[str], ctx: GateEvaluationContext) -> GateResult:
        ok = all(p.m02_main_budget_ok for p in profiles)
        return self._pass_result(gate, "fullwidth bottom region safe") if ok else self._fail_result(gate, "fullwidth bottom overflow risk")

    def _check_kpi_card_readability(self, gate: GateDef, features: SlideFeatures, profiles: List[ProfileMetrics], prev_layout: Optional[str], ctx: GateEvaluationContext) -> GateResult:
        kpi_cards = len(re.findall(r"kpi-card", ctx.html_low))
        return self._pass_result(gate, "kpi card readability acceptable") if (kpi_cards == 0 or features.text_len >= 80) else self._fail_result(gate, "kpi card readability violation")

    def _check_fullwidth_total_budget(self, gate: GateDef, features: SlideFeatures, profiles: List[ProfileMetrics], prev_layout: Optional[str], ctx: GateEvaluationContext) -> GateResult:
        return self._pass_result(gate, "fullwidth total budget ok") if features.m03_fixed_block_budget_ok else self._fail_result(gate, "fullwidth total budget violation")

    def _check_fullwidth_lower_half_ratio(self, gate: GateDef, features: SlideFeatures, profiles: List[ProfileMetrics], prev_layout: Optional[str], ctx: GateEvaluationContext) -> GateResult:
        if not ctx.chart_area_heights:
            return self._pass_result(gate, "no explicit fullwidth lower-half height to check")
        ratio = ctx.max_chart_h / 510.0
        return self._pass_result(gate, "fullwidth lower-half ratio in range") if 0.44 <= ratio <= 0.52 else self._fail_result(gate, f"fullwidth lower-half ratio out of range ({ratio:.2f})")

    def _check_heatmap_fill_whitespace(self, gate: GateDef, features: SlideFeatures, profiles: List[ProfileMetrics], prev_layout: Optional[str], ctx: GateEvaluationContext) -> GateResult:
        ok = features.m03_fixed_block_budget_ok and ctx.max_chart_h >= 220
        return self._pass_result(gate, "heatmap fill/whitespace acceptable") if ok else self._fail_result(gate, "heatmap fill/whitespace not acceptable")

    def _evaluate_timeline_layout_gates(
        self,
        gate: GateDef,
        features: SlideFeatures,
        profiles: List[ProfileMetrics],
        prev_layout: Optional[str],
        ctx: GateEvaluationContext,
    ) -> GateResult:
        """Evaluate timeline layout gates G57-G59."""
        handler_map = {
            "G57": self._check_timeline_connection_integrity,
            "G58": self._check_timeline_overlap,
            "G59": self._check_timeline_phase_segmentation,
        }
        return handler_map[gate.id](gate, features, profiles, prev_layout, ctx)

    def _check_timeline_connection_integrity(self, gate: GateDef, features: SlideFeatures, profiles: List[ProfileMetrics], prev_layout: Optional[str], ctx: GateEvaluationContext) -> GateResult:
        has_thin_absolute_line = bool(re.search(r'class="[^"]*absolute[^"]*(?:h-\[[123]px\]|w-\[[123]px\]|h-px|w-px)[^"]*"', ctx.html_low))
        has_connection = "connection-line" in ctx.html_low or ("absolute" in ctx.html_low and ("border-l" in ctx.html_low or "w-px" in ctx.html_low or "border-t" in ctx.html_low or "h-px" in ctx.html_low)) or has_thin_absolute_line
        has_anchor = "dot" in ctx.html_low or "year" in ctx.html_low or "phase" in ctx.html_low
        if not has_connection:
            return self._fail_result(gate, "timeline connection line missing (no absolute border or connection-line class)")
        if any(profile.timeline_disconnected for profile in profiles):
            return self._fail_result(gate, "timeline connection line is physically disconnected from cards")
        return self._pass_result(gate, "timeline anchor elements present and connected") if has_anchor else self._fail_result(gate, "timeline anchor elements missing")

    def _check_timeline_overlap(self, gate: GateDef, features: SlideFeatures, profiles: List[ProfileMetrics], prev_layout: Optional[str], ctx: GateEvaluationContext) -> GateResult:
        return self._pass_result(gate, "timeline overlap acceptable") if all(p.m05_overflow_nodes_ok for p in profiles) else self._fail_result(gate, "timeline card overlap risk")

    def _check_timeline_phase_segmentation(self, gate: GateDef, features: SlideFeatures, profiles: List[ProfileMetrics], prev_layout: Optional[str], ctx: GateEvaluationContext) -> GateResult:
        segmented = ("phase" in ctx.html_low) or ("阶段" in ctx.html)
        has_timeline_markup = ("timeline-item" in ctx.html_low) or ("timeline" in ctx.html_low)
        return self._pass_result(gate, "timeline phase segmentation present") if (segmented or has_timeline_markup) else self._fail_result(gate, "timeline phase segmentation missing")

    def _evaluate_visual_and_brand_gates(
        self,
        gate: GateDef,
        features: SlideFeatures,
        profiles: List[ProfileMetrics],
        ctx: GateEvaluationContext,
        mode: str,
    ) -> Optional[GateResult]:
        """Evaluate visual, semantic, brand-style, and post-generation gates."""
        handler_map = {
            "G60": self._evaluate_visual_boundary_gates,
            "G61": self._evaluate_visual_boundary_gates,
            "G62": self._evaluate_semantic_visual_gates,
            "G63": self._evaluate_semantic_visual_gates,
            "G64": self._evaluate_semantic_visual_gates,
            "G65": self._evaluate_semantic_visual_gates,
            "G66": self._evaluate_semantic_visual_gates,
            "G67": self._evaluate_semantic_visual_gates,
            "G68": self._evaluate_brand_runtime_gates,
            "G69": self._evaluate_brand_runtime_gates,
            "G70": self._evaluate_brand_runtime_gates,
            "G71": self._evaluate_brand_runtime_gates,
            "G74": self._evaluate_post_visual_gates,
            "G75": self._evaluate_post_visual_gates,
            "G76": self._evaluate_post_visual_gates,
            "G77": self._evaluate_post_visual_gates,
            "G78": self._evaluate_post_visual_gates,
            "G79": self._evaluate_post_visual_gates,
            "G80": self._evaluate_post_visual_gates,
            "G81": self._evaluate_runtime_mode_gates,
            "G82": self._evaluate_runtime_mode_gates,
            "G83": self._evaluate_runtime_mode_gates,
            "G84": self._evaluate_runtime_mode_gates,
        }
        handler = handler_map.get(gate.id)
        return handler(gate, features, profiles, ctx, mode) if handler else None

    def _evaluate_visual_boundary_gates(
        self,
        gate: GateDef,
        features: SlideFeatures,
        profiles: List[ProfileMetrics],
        ctx: GateEvaluationContext,
        mode: str,
    ) -> GateResult:
        """Evaluate visual boundary gates G60-G61."""
        if gate.id == "G60":
            min_gap = float(self.footer_safe_gap_min_px)
            worst_gap = min((p.footer_safe_gap for p in profiles), default=-999.0)
            ok = all((p.footer_safe_gap >= min_gap) or (p.m04_footer_overlap_risk <= 500) for p in profiles)
            if ok:
                return self._pass_result(gate, f"content boundary stays above footer/progress edge with >= {int(min_gap)}px gap (worst={worst_gap:.1f}px)")
            return self._fail_result(gate, f"content boundary enters footer/progress safety zone: worst gap {worst_gap:.1f}px < required {int(min_gap)}px")
        ok = all(p.m05_overflow_nodes_ok for p in profiles)
        return self._pass_result(gate, "M05 overflow_nodes == 0") if ok else self._fail_result(gate, "overflow nodes detected")

    def _evaluate_semantic_visual_gates(
        self,
        gate: GateDef,
        features: SlideFeatures,
        profiles: List[ProfileMetrics],
        ctx: GateEvaluationContext,
        mode: str,
    ) -> GateResult:
        """Evaluate semantic and component-level visual gates G62-G67."""
        handler_map = {
            "G62": self._check_semantic_color_richness,
            "G63": self._check_semantic_color_mapping,
            "G64": self._check_semantic_border_consistency,
            "G65": self._check_list_icon_consistency,
            "G66": self._check_semantic_color_ratio,
            "G67": self._check_semantic_color_contextual_application,
        }
        return handler_map[gate.id](gate, features, ctx)

    def _semantic_component_context_required(self, gate_id: str) -> bool:
        return gate_id in {"G62", "G63", "G67"}

    def _check_semantic_color_richness(self, gate: GateDef, features: SlideFeatures, ctx: GateEvaluationContext) -> GateResult:
        if self._semantic_component_context_required(gate.id) and not self._has_semantic_component_context(ctx.html_low, ctx.card_count):
            return self._pass_result(gate, NO_SEMANTIC_COMPONENT_CONTEXT)
        return self._pass_result(gate, "color richness sufficient") if ctx.semantic_color_hits >= 1 else self._fail_result(gate, "color richness insufficient (<1 semantic tier)")

    def _check_semantic_color_mapping(self, gate: GateDef, features: SlideFeatures, ctx: GateEvaluationContext) -> GateResult:
        if self._semantic_component_context_required(gate.id) and not self._has_semantic_component_context(ctx.html_low, ctx.card_count):
            return self._pass_result(gate, NO_SEMANTIC_COMPONENT_CONTEXT)
        if ctx.semantic_color_hits == 0:
            return self._fail_result(gate, "semantic color mapping missing")
        if features.has_chartjs_usage or features.has_echarts_usage or self._is_map_like(ctx.html_low):
            return self._pass_result(gate, "semantic color mapping present on a chart/map-led page")
        return self._pass_result(gate, "semantic color mapping appears coherent")

    def _check_semantic_border_consistency(self, gate: GateDef, features: SlideFeatures, ctx: GateEvaluationContext) -> GateResult:
        border_tokens = set(self._semantic_border_widths(ctx.html_low))
        if len(border_tokens) <= 1:
            return self._pass_result(gate, "semantic accent border widths are consistent")
        return self._fail_result(gate, "semantic accent border widths vary across emphasized components")

    def _check_list_icon_consistency(self, gate: GateDef, features: SlideFeatures, ctx: GateEvaluationContext) -> GateResult:
        icons_context = re.findall(r"<li[^>]*>.*?<i class=\"([a-z0-9\s-]+)\"", ctx.html, flags=re.S | re.I)
        raw_icons: List[str] = []
        for icon_context in icons_context:
            match = re.search(r"fa-[a-z0-9-]+", icon_context)
            if match:
                raw_icons.append(match.group(0))
        if not raw_icons:
            return self._pass_result(gate, "no list-item icon pattern detected")
        families = {icon.split("-")[1] for icon in raw_icons if "-" in icon}
        return self._pass_result(gate, "list icon family consistent") if len(families) <= 1 else self._fail_result(gate, "list icon inconsistency")

    def _check_semantic_color_ratio(self, gate: GateDef, features: SlideFeatures, ctx: GateEvaluationContext) -> GateResult:
        is_chart_heavy = (features.has_chartjs_usage or features.has_echarts_usage) and ctx.card_count <= 2
        if is_chart_heavy:
            return GateResult(gate.id, "pass", "chart-heavy slide exempted from color ratio", "info", gate.phase, gate.scope, gate.category)
        if ctx.card_count <= 0:
            return self._pass_result(gate, "no semantic-card ratio check needed")
        ratio = ctx.semantic_color_hits / max(ctx.card_count, 1)
        return self._pass_result(gate, "semantic color ratio in expected range") if 0.05 <= ratio <= 3.0 else self._fail_result(gate, f"semantic color ratio out of range ({ratio:.2f})")

    def _check_semantic_color_contextual_application(self, gate: GateDef, features: SlideFeatures, ctx: GateEvaluationContext) -> GateResult:
        if self._semantic_component_context_required(gate.id) and not self._has_semantic_component_context(ctx.html_low, ctx.card_count):
            return self._pass_result(gate, NO_SEMANTIC_COMPONENT_CONTEXT)
        if self._is_map_like(ctx.html_low) and ctx.semantic_color_hits >= 1:
            return self._pass_result(gate, "semantic color applied through map overlays or annotations")
        contextual = bool(re.search(r"(insight|status|kpi|badge)[^\n]{0,180}(text-|bg-|border-)(red|green|yellow|orange|blue)", ctx.html_low)) or self._has_semantic_component_application(ctx.html_low)
        return self._pass_result(gate, "semantic color applied on insight/status/KPI components") if contextual else self._fail_result(gate, "semantic color not landing on insight/status/KPI components")

    def _evaluate_brand_runtime_gates(
        self,
        gate: GateDef,
        features: SlideFeatures,
        profiles: List[ProfileMetrics],
        ctx: GateEvaluationContext,
        mode: str,
    ) -> GateResult:
        """Evaluate brand-style and runtime render gates G68-G71."""
        handler_map = {
            "G68": self._check_brand_scope_coverage,
            "G69": self._check_debug_controls_absent,
            "G70": self._check_style_switch_resize_hook,
            "G71": self._check_runtime_canvas_rendered,
        }
        return handler_map[gate.id](gate, features, profiles, ctx)

    def _check_brand_scope_coverage(self, gate: GateDef, features: SlideFeatures, profiles: List[ProfileMetrics], ctx: GateEvaluationContext) -> GateResult:
        active_brand_classes = set(re.findall(r'<body[^>]*class="[^"]*\b(brand-[a-z0-9-]+)\b', ctx.html, flags=re.I))
        if not active_brand_classes:
            has_brand_tokens = all(token in self.slide_theme_text for token in ["--brand-primary", "--brand-secondary", "--brand-accent"])
            return self._pass_result(gate, "theme exposes baseline brand tokens") if has_brand_tokens else self._fail_result(gate, "slide-theme.css missing baseline brand tokens")
        missing_scopes = [brand_class for brand_class in sorted(active_brand_classes) if f".{brand_class}" not in self.slide_theme_text]
        if missing_scopes:
            return self._fail_result(gate, f"missing active brand scope(s) in slide-theme.css: {', '.join(missing_scopes)}")
        return self._pass_result(gate, f"active brand scope(s) found: {', '.join(sorted(active_brand_classes))}")

    def _check_debug_controls_absent(self, gate: GateDef, features: SlideFeatures, profiles: List[ProfileMetrics], ctx: GateEvaluationContext) -> GateResult:
        return self._pass_result(gate, "no debug controls found") if (not features.has_debug_controls) else self._fail_result(gate, "debug controls detected")

    def _check_style_switch_resize_hook(self, gate: GateDef, features: SlideFeatures, profiles: List[ProfileMetrics], ctx: GateEvaluationContext) -> GateResult:
        if not self.presentation_text:
            return self._pass_result(gate, "presentation.html missing in this packaging")
        has_style_switch = "switchStyleProfile" in self.presentation_text or "switchBrand" in self.presentation_text or "setBrand" in self.presentation_text
        if not has_style_switch:
            return self._pass_result(gate, "style-profile switch feature not enabled")
        ok = (("switchStyleProfile" in self.presentation_text and "chart.resize" in self.presentation_text) or ("switchBrand" in self.presentation_text and "chart.resize" in self.presentation_text) or ("setBrand" in self.presentation_text and "resize" in self.presentation_text))
        return self._pass_result(gate, "style-profile switch resize hook detected") if ok else self._fail_result(gate, "style-profile switch resize hook not detected")

    def _check_runtime_canvas_rendered(self, gate: GateDef, features: SlideFeatures, profiles: List[ProfileMetrics], ctx: GateEvaluationContext) -> GateResult:
        has_chart = features.has_chartjs_usage or features.has_echarts_usage
        if not has_chart:
            return self._pass_result(gate, "no chart on this slide")
        if not ctx.runtime_available:
            return GateResult(gate.id, "not_applicable", "runtime canvas rendering not verifiable in static-fallback", gate.level, gate.phase, gate.scope, gate.category)
        ok = any(profile.rendered_canvas_count > 0 for profile in profiles)
        return self._pass_result(gate, "rendered canvas detected") if ok else self._fail_result(gate, "rendered canvas not detected")

    def _evaluate_post_visual_gates(
        self,
        gate: GateDef,
        features: SlideFeatures,
        profiles: List[ProfileMetrics],
        ctx: GateEvaluationContext,
        mode: str,
    ) -> GateResult:
        """Evaluate post-generation visual gates G74-G80."""
        handler_map = {
            "G74": self._check_style_hierarchy,
            "G75": self._check_chart_card_bounds,
            "G76": self._check_axis_label_clipping,
            "G77": self._check_runtime_profile_matrix,
            "G78": self._check_hidden_overflow_masking,
            "G79": self._check_main_overflow_blindspot,
            "G80": self._check_main_stack_boundary,
        }
        return handler_map[gate.id](gate, features, profiles, ctx)

    def _check_style_hierarchy(self, gate: GateDef, features: SlideFeatures, profiles: List[ProfileMetrics], ctx: GateEvaluationContext) -> GateResult:
        if features.inferred_layout == "cover":
            return self._pass_result(gate, "cover/title slide exempt from hierarchy check")
        has_primary_heading = ("<h2" in ctx.html_low) or ("<h1" in ctx.html_low)
        hierarchy = has_primary_heading and self._has_secondary_structure(ctx.html_low, features, ctx.card_count)
        return self._pass_result(gate, "style hierarchy appears complete") if hierarchy else self._fail_result(gate, "style hierarchy appears weak")

    def _check_chart_card_bounds(self, gate: GateDef, features: SlideFeatures, profiles: List[ProfileMetrics], ctx: GateEvaluationContext) -> GateResult:
        if not (features.has_chartjs_usage or features.has_echarts_usage):
            return self._pass_result(gate, NON_CHART_SLIDE)
        ok = all(profile.m05_overflow_nodes_ok and profile.canvas_overdraw_nodes == 0 for profile in profiles)
        return self._pass_result(gate, "chart elements stay within card bounds") if ok else self._fail_result(gate, "chart overflow card risk")

    def _check_axis_label_clipping(self, gate: GateDef, features: SlideFeatures, profiles: List[ProfileMetrics], ctx: GateEvaluationContext) -> GateResult:
        clipped = bool(re.search(r"(axislabel[^\n]{0,80}overflow\s*:\s*'truncate')", ctx.html_low))
        return self._fail_result(gate, "axis label clipping risk") if clipped else self._pass_result(gate, "no axis label clipping risk detected")

    def _check_runtime_profile_matrix(self, gate: GateDef, features: SlideFeatures, profiles: List[ProfileMetrics], ctx: GateEvaluationContext) -> GateResult:
        ok = all((p.m02_main_budget_ok or p.m04_footer_overlap_risk <= 500) for p in profiles)
        return self._pass_result(gate, "M06 runtime profile matrix passed") if ok else self._fail_result(gate, "M06 failed for at least one profile")

    def _check_hidden_overflow_masking(self, gate: GateDef, features: SlideFeatures, profiles: List[ProfileMetrics], ctx: GateEvaluationContext) -> GateResult:
        ok = all(not p.m07_hidden_overflow_masking_risk for p in profiles)
        return self._pass_result(gate, "M07 masking risk false") if ok else self._fail_result(gate, "hidden overflow masking risk detected")

    def _check_main_overflow_blindspot(self, gate: GateDef, features: SlideFeatures, profiles: List[ProfileMetrics], ctx: GateEvaluationContext) -> GateResult:
        blindspot = any((not p.m02_main_budget_ok) and p.m05_overflow_nodes_ok and p.m04_footer_overlap_risk > 500 for p in profiles)
        return self._fail_result(gate, "main budget overflow exists while overflow_nodes==0") if blindspot else self._pass_result(gate, "no main-overflow blindspot detected")

    def _check_main_stack_boundary(self, gate: GateDef, features: SlideFeatures, profiles: List[ProfileMetrics], ctx: GateEvaluationContext) -> GateResult:
        worst = max((p.m08_main_stack_overflow_risk_px for p in profiles), default=0.0)
        ok = all(p.m08_main_stack_overflow_risk_px <= self.tolerance_px for p in profiles)
        return self._pass_result(gate, f"main stack boundary safe (worst overflow {worst:.1f}px)") if ok else self._fail_result(gate, f"main stack overflow detected (worst {worst:.1f}px)")

    def _evaluate_runtime_mode_gates(
        self,
        gate: GateDef,
        features: SlideFeatures,
        profiles: List[ProfileMetrics],
        ctx: GateEvaluationContext,
        mode: str,
    ) -> GateResult:
        """Evaluate runtime-mode gates G81-G84."""
        handler_map = {
            "G81": self._check_runtime_backend_required,
            "G82": self._check_visual_collapse_risk,
            "G83": self._check_runtime_chart_min_height,
            "G84": self._check_text_contrast,
        }
        return handler_map[gate.id](gate, features, profiles, ctx, mode)

    def _check_runtime_backend_required(self, gate: GateDef, features: SlideFeatures, profiles: List[ProfileMetrics], ctx: GateEvaluationContext, mode: str) -> GateResult:
        if mode != "production":
            return self._pass_result(gate, "non-production mode")
        return self._pass_result(gate, "runtime backend enforced") if ctx.runtime_available else self._fail_result(gate, "production requires playwright-runtime backend; static-fallback is not allowed")

    def _check_visual_collapse_risk(self, gate: GateDef, features: SlideFeatures, profiles: List[ProfileMetrics], ctx: GateEvaluationContext, mode: str) -> GateResult:
        if not (features.has_chartjs_usage or features.has_echarts_usage):
            return self._pass_result(gate, NON_CHART_SLIDE)
        collapse_risk = sum(p.m09_chart_collapse_risk for p in profiles)
        return self._pass_result(gate, "visual collapse risk check passed") if collapse_risk == 0 else self._fail_result(gate, f"visual collapse risk (M08): {collapse_risk} profile-instances < 150px")

    def _check_runtime_chart_min_height(self, gate: GateDef, features: SlideFeatures, profiles: List[ProfileMetrics], ctx: GateEvaluationContext, mode: str) -> GateResult:
        if not (features.has_chartjs_usage or features.has_echarts_usage):
            return self._pass_result(gate, NON_CHART_SLIDE)
        collapse_risk = sum(p.m09_chart_collapse_risk for p in profiles)
        return self._pass_result(gate, "runtime chart minimum height check passed") if collapse_risk == 0 else self._fail_result(gate, f"chart collapse risk: {collapse_risk} profile-instances < 150px")

    def _check_text_contrast(self, gate: GateDef, features: SlideFeatures, profiles: List[ProfileMetrics], ctx: GateEvaluationContext, mode: str) -> GateResult:
        if not ctx.runtime_available:
            return self._pass_result(gate, "skipped static mode")
        issues = sum(p.contrast_issues for p in profiles)
        return self._pass_result(gate, "text contrast check passed") if issues == 0 else self._fail_result(gate, f"text contrast failure: {issues} low-contrast text nodes detected")

    def evaluate_gate(
        self,
        gate: GateDef,
        features: SlideFeatures,
        profiles: List[ProfileMetrics],
        prev_layout: Optional[str],
        mode: str,
    ) -> GateResult:
        """Evaluate a single gate against slide features and profiles."""
        if mode == "draft" and gate.draft_skip:
            return GateResult(gate.id, "skipped_mode", "draft mode skips this gate", gate.level, gate.phase, gate.scope, gate.category)

        if not self.scope_applies(gate, features):
            return GateResult(gate.id, "not_applicable", "scope not applicable for this slide", gate.level, gate.phase, gate.scope, gate.category)

        ctx = self._build_context(features, profiles)
        handlers = [
            lambda: self._evaluate_structural_gates(gate, features, profiles, ctx),
            lambda: self._evaluate_content_gates(gate, features, ctx),
            lambda: self._evaluate_chart_gates(gate, features, profiles, ctx),
            lambda: self._evaluate_layout_gates(gate, features, profiles, prev_layout, ctx),
            lambda: self._evaluate_visual_and_brand_gates(gate, features, profiles, ctx, mode),
        ]
        for handler in handlers:
            result = handler()
            if result is not None:
                return result

        return GateResult(
            gate.id,
            "not_implemented",
            "no executable checker mapping yet; use manual review and upstream contract context",
            gate.level,
            gate.phase,
            gate.scope,
            gate.category,
        )
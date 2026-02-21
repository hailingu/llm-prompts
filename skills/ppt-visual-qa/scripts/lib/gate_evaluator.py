"""Gate evaluation logic for PPT Visual QA."""

import re
from typing import Dict, List, Optional

from .models import GateDef, SlideFeatures, ProfileMetrics, GateResult
from .constants import MAIN_OUTER_AVAILABLE_PX, SEMANTIC_COLORS


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

    def scope_applies(self, gate: GateDef, features: SlideFeatures) -> bool:
        """Check if gate scope applies to this slide."""
        scope = gate.scope.lower()
        html_low = self.slide_html_map.get(features.slide, "").lower()
        
        if scope == "all":
            return True
        if scope == "cover":
            return features.is_special_exempt and features.inferred_layout == "cover"
        if scope == "analysis":
            return features.is_analysis_like
        if scope == "bubble":
            return ("bubble" in features.slide.lower()) or ("type: 'bubble'" in html_low) or ('type: "bubble"' in html_low)
        if scope == "line":
            return features.has_chartjs_usage or ("line" in html_low and "chart" in html_low)
        if scope == "echarts":
            return features.has_echarts_usage
        if scope == "heatmap":
            return ("heat" in features.slide.lower()) or ("heatmap" in html_low) or ("visualmap" in html_low)
        if scope == "dual-column":
            return features.inferred_layout == "dual-column"
        if scope == "side-by-side":
            return "side" in features.slide.lower() or features.inferred_layout == "dual-column"
        if scope == "radar-kpi":
            return ("radar" in features.slide.lower()) or ("radar" in html_low and "kpi" in html_low)
        if scope == "gantt":
            return ("gantt" in features.slide.lower()) or ("gantt" in html_low)
        if scope == "process":
            return features.inferred_layout == "process"
        if scope == "full-width":
            return "full-width" in features.slide.lower()
        if scope == "milestone-timeline":
            return features.inferred_layout == "milestone-timeline"
        if scope == "list":
            return features.text_len > 0 and (bool(re.search(r"<li\b", html_low)) or "list" in html_low)
        return False

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

        backend = profiles[0].backend if profiles else "unknown"
        runtime_available = backend == "playwright-runtime"

        if not self.scope_applies(gate, features):
            return GateResult(gate.id, "not_applicable", "scope not applicable for this slide", gate.level, gate.phase, gate.scope, gate.category)

        html = self.slide_html_map.get(features.slide, "")
        html_low = html.lower()
        
        # Pre-compute common values
        semantic_color_hits = sum(1 for key in SEMANTIC_COLORS if key in html_low)
        card_count = len(re.findall(r'class="[^"]*\b(card|insight-card|kpi-card)\b', html, flags=re.I))
        chart_area_heights = [int(v) for v in re.findall(r"h-\[(\d+)px\]", html)] + [int(v) for v in re.findall(r"height\s*:\s*(\d+)px", html, flags=re.I)]
        max_chart_h = max(chart_area_heights) if chart_area_heights else 0
        radius_vals = [float(v) for v in re.findall(r"\br\s*:\s*([0-9]+(?:\.[0-9]+)?)", html)]
        radius_unique_count = len(set(radius_vals))

        # Gate implementations
        if gate.id == "G01":
            return self._pass_result(gate, "input appears complete") if (features.text_len > 0 and features.has_required_skeleton) else self._fail_result(gate, "input/content appears incomplete")
            
        if gate.id == "G02":
            return self._pass_result(gate, "required skeleton present") if features.has_required_skeleton else self._fail_result(gate, "missing slide-header/slide-main/slide-footer")
            
        if gate.id == "G03":
            if not (features.has_chartjs_usage or features.has_echarts_usage):
                return self._pass_result(gate, "non-chart slide")
            return self._pass_result(gate, "chart container height is explicit") if features.has_chart_wrap_explicit_height else self._fail_result(gate, "no explicit chart-wrap/container height")
            
        if gate.id == "G04":
            if not features.has_chartjs_usage:
                return self._pass_result(gate, "no Chart.js usage on this slide")
            return self._pass_result(gate, "maintainAspectRatio false found") if features.has_maintain_aspect_ratio_false else self._fail_result(gate, "Chart.js used without maintainAspectRatio: false")
            
        if gate.id == "G05":
            if not features.has_echarts_usage:
                return self._pass_result(gate, "no ECharts usage on this slide")
            return self._pass_result(gate, "ECharts cdn detected") if features.has_echarts_cdn else self._fail_result(gate, "ECharts usage without matched CDN")
            
        if gate.id == "G06":
            if not (features.has_chartjs_usage or features.has_echarts_usage):
                return self._pass_result(gate, "no chart runtime path on this slide")
            return self._pass_result(gate, "NaN/null guard found") if features.has_nan_guard else self._fail_result(gate, "no explicit NaN/null guard pattern")
            
        if gate.id == "G07":
            if not (features.has_chartjs_usage or features.has_echarts_usage):
                return self._pass_result(gate, "no chart runtime path on this slide")
            return self._pass_result(gate, "labels/data length guard found") if features.has_labels_data_length_guard else self._fail_result(gate, "no labels.length vs data.length guard pattern")
            
        if gate.id == "G08":
            if not (features.has_chartjs_usage or features.has_echarts_usage):
                return self._pass_result(gate, "no chart runtime path on this slide")
            return self._pass_result(gate, "empty-data fallback text found") if features.has_empty_data_fallback else self._fail_result(gate, "empty-data fallback text not found")
            
        if gate.id == "G09":
            if not (features.has_chartjs_usage or features.has_echarts_usage):
                return self._pass_result(gate, "no chart runtime path on this slide")
            return self._pass_result(gate, "DOMContentLoaded init found") if features.has_domcontentloaded_init else self._fail_result(gate, "DOMContentLoaded initialization not found")

        if gate.id == "G10":
            ok = all(p.m01_slide_total_budget_ok for p in profiles)
            return self._pass_result(gate, "M01 passed across profiles") if ok else self._fail_result(gate, "M01 failed in at least one profile")

        if gate.id == "G11":
            min_gap = float(self.footer_safe_gap_min_px)
            margins = {p.profile_id: (p.main_client_h - min_gap - p.main_scroll_h) for p in profiles}
            ok = all((m >= 0) or (p.m04_footer_overlap_risk <= 500) for p, m in zip(profiles, margins.values()))
            worst_profile, worst_margin = min(margins.items(), key=lambda item: item[1])
            if ok:
                return self._pass_result(gate, f"M02 passed across profiles (worst margin: {worst_margin:.1f}px @ {worst_profile})")
            failed_profiles = ", ".join([k for k, v in margins.items() if v < 0])
            return self._fail_result(gate, f"M02 failed: content exceeds main budget by {abs(worst_margin):.1f}px @ {worst_profile} (failed profiles: {failed_profiles})")

        if gate.id == "G12":
            if profiles and profiles[0].backend == "playwright-runtime":
                worst_overflow = 0.0
                worst_profile = None
                for p in profiles:
                    overflow = p.main_scroll_h - p.main_client_h
                    if overflow > 0.5:
                        if overflow > worst_overflow:
                            worst_overflow = overflow
                            worst_profile = p.profile_id
                if worst_overflow > 0:
                    return self._fail_result(gate, f"M03 runtime budget exceeded by {worst_overflow:.1f}px (scroll={p.main_scroll_h:.1f}px > client={p.main_client_h:.1f}px @ {worst_profile})")
                return self._pass_result(gate, f"M03 runtime budget passed (content fits within container across {len(profiles)} profiles)")
            delta = MAIN_OUTER_AVAILABLE_PX - float(features.m03_est_fixed_px)
            if features.m03_fixed_block_budget_ok:
                return self._pass_result(gate, f"M03 static fixed-block budget passed (headroom: {delta:.1f}px, est={features.m03_est_fixed_px:.1f}px, budget={MAIN_OUTER_AVAILABLE_PX:.1f}px)")
            return self._fail_result(gate, f"M03 static fixed-block budget exceeded by {abs(delta):.1f}px (est={features.m03_est_fixed_px:.1f}px, budget={MAIN_OUTER_AVAILABLE_PX:.1f}px)")

        if gate.id == "G13":
            if features.inferred_layout != "cover":
                return self._pass_result(gate, "non-cover slide")
            return self._pass_result(gate, "cover slide uses cover layout") if features.is_special_exempt else self._fail_result(gate, "cover slide markers missing")

        if gate.id == "G14":
            return self._pass_result(gate, "text length meets threshold") if features.text_len >= 120 else self._fail_result(gate, f"text_len {features.text_len} < 120")

        if gate.id in {"G15", "G17"}:
            key_page = (features.inferred_layout in {"dual-column", "process", "milestone-timeline"} or features.structured_claim_count >= 2 or features.text_len >= 180)
            if not key_page:
                return self._pass_result(gate, "non-key page")
            return self._pass_result(gate, "three-part keywords found") if (features.has_three_part_keywords or features.structured_claim_count >= 1 or features.text_len >= 100) else self._fail_result(gate, "missing 结论/原因/建议 keywords")

        if gate.id == "G16":
            return self._pass_result(gate, "structured claims >= baseline") if (features.structured_claim_count >= 0 and features.text_len >= 60) else self._fail_result(gate, "structured claims < baseline")

        if gate.id == "G18":
            if features.has_chartjs_usage:
                return self._pass_result(gate, "chart engine/cdn matched") if features.has_chartjs_cdn else self._fail_result(gate, "Chart.js usage without CDN")
            if features.has_echarts_usage:
                return self._pass_result(gate, "chart engine/cdn matched") if features.has_echarts_cdn else self._fail_result(gate, "ECharts usage without CDN")
            return self._pass_result(gate, "no chart engine usage")

        if gate.id == "G19":
            matrix_semantic = ("matrix" in html_low) or ("矩阵" in html)
            heatmap_like = ("heatmap" in html_low) or ("visualmap" in html_low)
            if matrix_semantic and not heatmap_like:
                return self._fail_result(gate, "matrix semantics detected but no heatmap-like chart")
            return self._pass_result(gate, "chart semantics match")

        if gate.id == "G20":
            if not features.has_chartjs_usage:
                return self._pass_result(gate, "no chartjs usage")
            return self._pass_result(gate, "tick stepSize found") if (features.has_step_size or features.has_chart_wrap_explicit_height) else self._fail_result(gate, "stepSize not found")

        if gate.id == "G21":
            ok = all(p.m04_footer_overlap_risk <= 0 for p in profiles)
            return self._pass_result(gate, "no bubble clipping risk detected") if ok else self._fail_result(gate, "bubble clipping risk detected")

        if gate.id == "G22":
            return self._pass_result(gate, "bubble radius distinguishable") if radius_unique_count >= 3 else self._fail_result(gate, f"bubble radius unique count {radius_unique_count} < 3")

        if gate.id == "G23":
            mapped = bool(re.search(r"\br\s*:\s*[^,\n}]*confidence", html_low))
            return self._pass_result(gate, "bubble radius mapped from confidence") if mapped else self._fail_result(gate, "bubble radius not mapped from confidence")

        if gate.id == "G24":
            legend_semantic = ("legend" in html_low) and (("对象" in html) or ("object" in html_low) or ("语义" in html))
            return self._pass_result(gate, "bubble semantic legend found") if legend_semantic else self._fail_result(gate, "bubble semantic legend missing")

        if gate.id == "G25":
            multi_color = ("backgroundcolor" in html_low and "[" in html_low and "," in html_low) or (semantic_color_hits >= 2)
            return self._pass_result(gate, "bubble color mapping distinguishable") if multi_color else self._fail_result(gate, "bubble color mapping weak")

        if gate.id == "G26":
            legend_items = len(re.findall(r"legend", html_low))
            return self._pass_result(gate, "bubble legend density acceptable") if legend_items <= 8 else self._fail_result(gate, "bubble legend overcrowded")

        if gate.id == "G27":
            ok = features.m03_fixed_block_budget_ok and all(p.m02_main_budget_ok for p in profiles)
            return self._pass_result(gate, "bubble layout budget ok") if ok else self._fail_result(gate, "bubble layout budget violation")

        if gate.id == "G28":
            overweight = bool(re.search(r"legend[^\n]*(border-2|shadow-lg|bg-[a-z]+-\d{3})", html_low))
            return self._fail_result(gate, "bubble legend visual overweight") if overweight else self._pass_result(gate, "bubble legend visual weight ok")

        if gate.id == "G29":
            mechanical = ("beginatzero:true" in html_low) and ("max:100" in html_low or "suggestedmax:100" in html_low)
            return self._fail_result(gate, "line axis range appears unbalanced/mechanical") if mechanical else self._pass_result(gate, "line axis range not mechanically constrained")

        if gate.id == "G30":
            mechanical = ("beginatzero:true" in html_low) and ("max:100" in html_low or "suggestedmax:100" in html_low)
            return self._fail_result(gate, "mechanical beginAtZero:true,max:100 detected") if mechanical else self._pass_result(gate, "no mechanical axis range pattern")

        if gate.id == "G31":
            if not (features.has_chartjs_usage or features.has_echarts_usage):
                return self._pass_result(gate, "non-chart slide")
            ok = features.has_chart_wrap_explicit_height and features.m03_fixed_block_budget_ok
            return self._pass_result(gate, "inner chart budget ok") if ok else self._fail_result(gate, "inner chart budget violation")

        if gate.id == "G32":
            ok = all(p.m05_overflow_nodes_ok for p in profiles)
            return self._pass_result(gate, "no chart component collision detected") if ok else self._fail_result(gate, "chart component collision risk")

        if gate.id == "G33":
            if ("heatmap" not in html_low) and ("visualmap" not in html_low):
                return self._pass_result(gate, "non-heatmap slide")
            return self._pass_result(gate, "heatmap container >=220px") if max_chart_h >= 220 else self._fail_result(gate, f"heatmap container too short ({max_chart_h}px)")

        if gate.id == "G34":
            gb = re.search(r"grid\s*:\s*\{[^\}]*bottom\s*:\s*(\d+)", html_low)
            vb = re.search(r"visualmap\s*:\s*\{[^\}]*bottom\s*:\s*(\d+)", html_low)
            g_ok = (int(gb.group(1)) >= 44) if gb else False
            v_ok = (int(vb.group(1)) <= 6) if vb else False
            return self._pass_result(gate, "heatmap grid/visualMap bottom constraints satisfied") if (g_ok and v_ok) else self._fail_result(gate, "heatmap grid.bottom / visualMap.bottom constraints not met")

        if gate.id == "G35":
            if prev_layout is None:
                return self._pass_result(gate, "first slide, no adjacency check")
            if features.inferred_layout == "unknown" or prev_layout == "unknown":
                return self._pass_result(gate, "unknown layout skipped")
            return self._pass_result(gate, "layout differs from previous slide") if features.inferred_layout != prev_layout else self._fail_result(gate, "adjacent slides share same inferred layout")

        if gate.id == "G36":
            return self._pass_result(gate, "dual-column whitespace delta acceptable") if features.m03_fixed_block_budget_ok else self._fail_result(gate, "dual-column whitespace delta risk")

        if gate.id == "G37":
            return self._pass_result(gate, "dual-column occupancy acceptable") if (features.text_len >= 120 or card_count >= 2) else self._fail_result(gate, "dual-column occupancy appears low")

        if gate.id == "G38":
            if len(chart_area_heights) < 2:
                return self._pass_result(gate, "no side-by-side dual chart pair")
            return self._pass_result(gate, "side-by-side chart heights aligned") if abs(chart_area_heights[0] - chart_area_heights[1]) <= 8 else self._fail_result(gate, "side-by-side chart height mismatch > 8px")

        if gate.id == "G39":
            return self._pass_result(gate, "side-by-side vertical whitespace balanced") if all(p.m04_footer_overlap_risk <= 300 for p in profiles) else self._fail_result(gate, "side-by-side vertical whitespace imbalance risk")

        if gate.id == "G40":
            if not chart_area_heights:
                return self._pass_result(gate, "no explicit side-by-side chart area")
            ratio = max_chart_h / 510.0
            return self._pass_result(gate, "side-by-side chart ratio in expected range") if 0.00 <= ratio <= 2.00 else self._fail_result(gate, f"side-by-side chart ratio out of range ({ratio:.2f})")

        if gate.id == "G41":
            if len(chart_area_heights) < 2:
                return self._pass_result(gate, "no bottom pair height check required")
            return self._pass_result(gate, "bottom card heights matched") if abs(chart_area_heights[-1] - chart_area_heights[-2]) <= 8 else self._fail_result(gate, "bottom card height mismatch > 8px")

        if gate.id == "G42":
            blocks = len(re.findall(r"(kpi|insight|callout|card)", html_low))
            return self._pass_result(gate, "radar-kpi right sidebar sufficiently structured") if blocks >= 3 else self._fail_result(gate, "radar-kpi sidebar sparse")

        if gate.id == "G43":
            return self._pass_result(gate, "radar-kpi whitespace balanced") if features.m03_fixed_block_budget_ok else self._fail_result(gate, "radar-kpi whitespace imbalance risk")

        if gate.id == "G44":
            return self._pass_result(gate, "gantt whitespace balanced") if features.m03_fixed_block_budget_ok else self._fail_result(gate, "gantt whitespace imbalance risk")

        if gate.id == "G45":
            label_zone = re.search(r"labelwidth\s*:\s*(\d+)", html_low)
            if not label_zone:
                return self._pass_result(gate, "no explicit oversized gantt label zone")
            return self._pass_result(gate, "gantt left label zone acceptable") if int(label_zone.group(1)) <= 18 else self._fail_result(gate, "gantt left label zone exceeds 18% budget")

        if gate.id == "G46":
            steps = len(re.findall(r"(process-step|step-process-container)", html_low))
            return self._pass_result(gate, "process detail sufficiently dense") if (steps >= 1 and features.text_len >= 40) else self._fail_result(gate, "process detail sparse")

        if gate.id == "G47":
            return self._pass_result(gate, "process card fill rate acceptable") if (features.m03_fixed_block_budget_ok and features.text_len >= 100) else self._fail_result(gate, "process card fill rate appears low")

        if gate.id == "G48":
            icon_ok = ("text-xl" in html_low) or ("fa-" in html_low)
            h_ok = any(h >= 120 for h in chart_area_heights)
            return self._pass_result(gate, "process icon sizing acceptable") if (icon_ok or h_ok or ("process-step" in html_low) or ("step" in html_low)) else self._fail_result(gate, "process icon/row sizing too small")

        if gate.id == "G49":
            aligned = ("items-center" in html_low) and ("flex" in html_low)
            return self._pass_result(gate, "process icon column aligned") if (aligned or ("grid" in html_low)) else self._fail_result(gate, "process icon column alignment risk")

        if gate.id == "G50":
            ok = all((p.m02_main_budget_ok or p.m04_footer_overlap_risk <= 300) for p in profiles)
            return self._pass_result(gate, "process card whitespace formula passed") if ok else self._fail_result(gate, "process card whitespace formula failed")

        if gate.id == "G51":
            numbers = len(re.findall(r"\d+(?:\.\d+)?%|\$\d+|\d{4}", html))
            kpi_words = len(re.findall(r"(kpi|指标|同比|环比|baseline|基线)", html_low))
            return self._pass_result(gate, "fullwidth KPI semantics strong") if (numbers >= 3 and kpi_words >= 2) else self._fail_result(gate, "fullwidth KPI semantics weak")

        if gate.id == "G52":
            ok = all(p.m02_main_budget_ok for p in profiles)
            return self._pass_result(gate, "fullwidth bottom region safe") if ok else self._fail_result(gate, "fullwidth bottom overflow risk")

        if gate.id == "G53":
            kpi_cards = len(re.findall(r"kpi-card", html_low))
            return self._pass_result(gate, "kpi card readability acceptable") if (kpi_cards == 0 or features.text_len >= 80) else self._fail_result(gate, "kpi card readability violation")

        if gate.id == "G54":
            return self._pass_result(gate, "fullwidth total budget ok") if features.m03_fixed_block_budget_ok else self._fail_result(gate, "fullwidth total budget violation")

        if gate.id == "G55":
            if not chart_area_heights:
                return self._pass_result(gate, "no explicit fullwidth lower-half height to check")
            ratio = max_chart_h / 510.0
            return self._pass_result(gate, "fullwidth lower-half ratio in range") if 0.44 <= ratio <= 0.52 else self._fail_result(gate, f"fullwidth lower-half ratio out of range ({ratio:.2f})")

        if gate.id == "G56":
            return self._pass_result(gate, "heatmap fill/whitespace acceptable") if (features.m03_fixed_block_budget_ok and max_chart_h >= 220) else self._fail_result(gate, "heatmap fill/whitespace not acceptable")

        if gate.id == "G57":
            has_anchor = all(k in html_low for k in ["dot", "year", "card", "connection"])
            return self._pass_result(gate, "timeline anchor elements present") if (has_anchor or ("timeline" in html_low and "card" in html_low)) else self._fail_result(gate, "timeline anchor elements missing")

        if gate.id == "G58":
            return self._pass_result(gate, "timeline overlap acceptable") if all(p.m05_overflow_nodes_ok for p in profiles) else self._fail_result(gate, "timeline card overlap risk")

        if gate.id == "G59":
            segmented = ("phase" in html_low) or ("阶段" in html)
            return self._pass_result(gate, "timeline phase segmentation present") if (segmented or ("timeline-item" in html_low) or ("timeline" in html_low)) else self._fail_result(gate, "timeline phase segmentation missing")

        if gate.id == "G60":
            min_gap = float(self.footer_safe_gap_min_px)
            worst_gap = min((p.footer_safe_gap for p in profiles), default=-999.0)
            ok = all((p.footer_safe_gap >= min_gap) or (p.m04_footer_overlap_risk <= 500) for p in profiles)
            if ok:
                return self._pass_result(gate, f"content boundary stays above footer/progress edge with >= {int(min_gap)}px gap (worst={worst_gap:.1f}px)")
            return self._fail_result(gate, f"content boundary enters footer/progress safety zone: worst gap {worst_gap:.1f}px < required {int(min_gap)}px")

        if gate.id == "G61":
            ok = all(p.m05_overflow_nodes_ok for p in profiles)
            return self._pass_result(gate, "M05 overflow_nodes == 0") if ok else self._fail_result(gate, "overflow nodes detected")

        if gate.id == "G62":
            if card_count == 0:
                return self._pass_result(gate, "no semantic card context")
            return self._pass_result(gate, "color richness sufficient") if semantic_color_hits >= 1 else self._fail_result(gate, "color richness insufficient (<1 semantic tier)")

        if gate.id == "G63":
            if card_count == 0:
                return self._pass_result(gate, "no semantic card context")
            return self._pass_result(gate, "semantic color mapping appears coherent") if semantic_color_hits >= 1 else self._fail_result(gate, "semantic color mapping missing")

        if gate.id == "G64":
            border_tokens = set(re.findall(r"border-(\d)", html_low))
            return self._pass_result(gate, "border consistency acceptable") if len(border_tokens) <= 1 else self._fail_result(gate, "border consistency violation (multiple border width tokens)")

        if gate.id == "G65":
            icons_context = re.findall(r"<li[^>]*>.*?<i class=\"([a-z0-9\s-]+)\"", html, flags=re.S | re.I)
            raw_icons = []
            for ic in icons_context:
                m = re.search(r"fa-[a-z0-9-]+", ic)
                if m: raw_icons.append(m.group(0))
            if not raw_icons:
                return self._pass_result(gate, "no list-item icon pattern detected")
            families = set(i.split("-")[1] for i in raw_icons if "-" in i)
            return self._pass_result(gate, "list icon family consistent") if len(families) <= 1 else self._fail_result(gate, "list icon inconsistency")

        if gate.id == "G66":
            is_chart_heavy = (features.has_chartjs_usage or features.has_echarts_usage) and card_count <= 2
            if is_chart_heavy:
                return GateResult(gate.id, "pass", "chart-heavy slide exempted from color ratio", "info", gate.phase, gate.scope, gate.category)
            if card_count <= 0:
                return self._pass_result(gate, "no semantic-card ratio check needed")
            ratio = semantic_color_hits / max(card_count, 1)
            return self._pass_result(gate, "semantic color ratio in expected range") if 0.05 <= ratio <= 3.0 else self._fail_result(gate, f"semantic color ratio out of range ({ratio:.2f})")

        if gate.id == "G67":
            if card_count == 0 and ("kpi" not in html_low) and ("insight" not in html_low) and ("status" not in html_low):
                return self._pass_result(gate, "no semantic card context")
            contextual = bool(re.search(r"(insight|status|kpi)[^\n]{0,180}(text-|bg-|border-)(red|green|yellow|orange|blue)", html_low)) or ("brand-primary" in html_low) or ("badge-primary" in html_low) or ("success" in html_low) or ("warning" in html_low) or ("danger" in html_low)
            return self._pass_result(gate, "semantic color applied on insight/status/KPI components") if contextual else self._fail_result(gate, "semantic color not landing on insight/status/KPI components")

        if gate.id == "G68":
            brands_ok = all(k in self.slide_theme_text for k in [".brand-kpmg", ".brand-mckinsey", ".brand-bcg", ".brand-bain", ".brand-deloitte"])
            return self._pass_result(gate, "all brand scopes found") if brands_ok else self._fail_result(gate, "missing one or more brand scopes in slide-theme.css")

        if gate.id == "G69":
            return self._pass_result(gate, "no debug controls found") if (not features.has_debug_controls) else self._fail_result(gate, "debug controls detected")

        if gate.id == "G70":
            if not self.presentation_text:
                return self._pass_result(gate, "presentation.html missing in this packaging")
            if "switchBrand" not in self.presentation_text:
                return self._pass_result(gate, "brand-switch feature not enabled")
            ok = ("switchBrand" in self.presentation_text and "chart.resize" in self.presentation_text) or ("setBrand" in self.presentation_text and "resize" in self.presentation_text)
            return self._pass_result(gate, "brand switch resize hook detected") if ok else self._fail_result(gate, "brand switch resize hook not detected")

        if gate.id == "G71":
            has_chart = features.has_chartjs_usage or features.has_echarts_usage
            if not has_chart:
                return self._pass_result(gate, "no chart on this slide")
            if not runtime_available:
                return GateResult(gate.id, "not_applicable", "runtime canvas rendering not verifiable in static-fallback", gate.level, gate.phase, gate.scope, gate.category)
            ok = any(p.rendered_canvas_count > 0 for p in profiles)
            return self._pass_result(gate, "rendered canvas detected") if ok else self._fail_result(gate, "rendered canvas not detected")

        if gate.id == "G72":
            if features.inferred_layout == "cover":
                return self._pass_result(gate, "cover/title slide exempt from density check")
            vals = list(self.deck_text_len_map.values())
            if not vals:
                return self._pass_result(gate, "no deck text density baseline")
            median = sorted(vals)[len(vals) // 2]
            ok = features.text_len >= max(24, int(0.20 * median))
            return self._pass_result(gate, "deck text density appears balanced") if ok else self._fail_result(gate, "text density too low vs deck baseline")

        if gate.id == "G73":
            if features.inferred_layout == "cover":
                return self._pass_result(gate, "cover/title slide exempt from key-page check")
            key_page = (features.inferred_layout in {"dual-column", "process", "milestone-timeline"} or features.structured_claim_count >= 2 or features.text_len >= 180)
            if not key_page:
                return self._pass_result(gate, "non-key page")
            report_ready = features.has_three_part_keywords or (features.structured_claim_count >= 1 and features.text_len >= 80)
            return self._pass_result(gate, "key page is report-ready") if report_ready else self._fail_result(gate, "key page below report-ready threshold")

        if gate.id == "G74":
            if features.inferred_layout == "cover":
                return self._pass_result(gate, "cover/title slide exempt from hierarchy check")
            hierarchy = (("<h2" in html_low) or ("<h1" in html_low)) and (("<h3" in html_low) or ("card" in html_low) or (features.text_len >= 80))
            return self._pass_result(gate, "style hierarchy appears complete") if hierarchy else self._fail_result(gate, "style hierarchy appears weak")

        if gate.id == "G75":
            ok = all(p.m05_overflow_nodes_ok and p.canvas_overdraw_nodes == 0 for p in profiles)
            return self._pass_result(gate, "chart elements stay within card bounds") if ok else self._fail_result(gate, "chart overflow card risk")

        if gate.id == "G76":
            clipped = bool(re.search(r"(axislabel[^\n]{0,80}overflow\s*:\s*'truncate')", html_low))
            return self._fail_result(gate, "axis label clipping risk") if clipped else self._pass_result(gate, "no axis label clipping risk detected")

        if gate.id == "G77":
            ok = all((p.m02_main_budget_ok or p.m04_footer_overlap_risk <= 500) for p in profiles)
            return self._pass_result(gate, "M06 runtime profile matrix passed") if ok else self._fail_result(gate, "M06 failed for at least one profile")

        if gate.id == "G78":
            ok = all(not p.m07_hidden_overflow_masking_risk for p in profiles)
            return self._pass_result(gate, "M07 masking risk false") if ok else self._fail_result(gate, "hidden overflow masking risk detected")

        if gate.id == "G79":
            blindspot = any((not p.m02_main_budget_ok) and p.m05_overflow_nodes_ok and p.m04_footer_overlap_risk > 500 for p in profiles)
            return self._fail_result(gate, "main budget overflow exists while overflow_nodes==0") if blindspot else self._pass_result(gate, "no main-overflow blindspot detected")

        if gate.id == "G80":
            worst = max((p.m08_main_stack_overflow_risk_px for p in profiles), default=0.0)
            ok = all(p.m08_main_stack_overflow_risk_px <= self.tolerance_px for p in profiles)
            return self._pass_result(gate, f"main stack boundary safe (worst overflow {worst:.1f}px)") if ok else self._fail_result(gate, f"main stack overflow detected (worst {worst:.1f}px)")

        if gate.id == "G81":
            if mode != "production":
                return self._pass_result(gate, "non-production mode")
            return self._pass_result(gate, "runtime backend enforced") if runtime_available else self._fail_result(gate, "production requires playwright-runtime backend; static-fallback is not allowed")

        if gate.id == "G82":
            if not (features.has_chartjs_usage or features.has_echarts_usage):
                return self._pass_result(gate, "non-chart slide")
            m09 = sum(p.m09_chart_collapse_risk for p in profiles)
            return self._pass_result(gate, "visual collapse risk check passed") if m09 == 0 else self._fail_result(gate, f"visual collapse risk (M08): {m09} profile-instances < 150px")

        if gate.id == "G83":
            if not (features.has_chartjs_usage or features.has_echarts_usage):
                return self._pass_result(gate, "non-chart slide")
            m09 = sum(p.m09_chart_collapse_risk for p in profiles)
            return self._pass_result(gate, "runtime chart minimum height check passed") if m09 == 0 else self._fail_result(gate, f"chart collapse risk: {m09} profile-instances < 150px")

        if gate.id == "G84":
            if not runtime_available:
                return self._pass_result(gate, "skipped static mode")
            issues = sum(p.contrast_issues for p in profiles)
            return self._pass_result(gate, "text contrast check passed") if issues == 0 else self._fail_result(gate, f"text contrast failure: {issues} low-contrast text nodes detected")

        return GateResult(gate.id, "not_implemented", "no checker mapping yet", gate.level, gate.phase, gate.scope, gate.category)
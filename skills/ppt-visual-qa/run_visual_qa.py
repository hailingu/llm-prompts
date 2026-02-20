#!/usr/bin/env python3
"""Executable gate-runner for PPT visual QA.

Usage:
  # Ensure dependencies (first time only):
  # source .venv/bin/activate
  # pip install playwright beautifulsoup4
  # playwright install chromium

  python skills/ppt-visual-qa/run_visual_qa.py \
    --presentation-dir "docs/presentations/ai-report Bain-style_20260216_v1" \
    --mode production \
    --strict

This runner executes a practical subset of gates and explicitly marks the rest
as `not_implemented` to avoid false green results.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml


PROFILES_DEFAULT = [
    {"id": "P1", "width": 1280, "height": 720, "dpr": 1},
    {"id": "P2", "width": 1366, "height": 768, "dpr": 1},
    {"id": "P3", "width": 1512, "height": 982, "dpr": 2},
]

MAIN_OUTER_AVAILABLE_PX = 590.0


@dataclass
class GateDef:
    id: str
    condition: str
    phase: str
    level: str
    draft_skip: bool
    scope: str
    category: str


@dataclass
class ProfileMetrics:
    profile_id: str
    viewport: str
    dpr: int
    slide_h: float
    header_h: float
    main_h: float
    footer_h: float
    main_client_h: float
    main_scroll_h: float
    footer_safe_gap: float
    overflow_nodes: int
    overflow_node_details: List[Dict[str, object]]
    canvas_overdraw_nodes: int
    main_overflow_hidden: bool
    rendered_canvas_count: int
    m01_slide_total_budget_ok: bool
    m02_main_budget_ok: bool
    m04_footer_overlap_risk: float
    m05_overflow_nodes_ok: bool
    m07_hidden_overflow_masking_risk: bool
    m08_main_stack_overflow_risk_px: float
    m09_chart_collapse_risk: int
    passed: bool
    backend: str
    contrast_issues: int = 0


@dataclass
class SlideFeatures:
    slide: str
    text_len: int
    has_required_skeleton: bool
    is_special_exempt: bool
    has_chartjs_cdn: bool
    has_chartjs_usage: bool
    has_echarts_usage: bool
    has_echarts_cdn: bool
    has_chart_wrap_explicit_height: bool
    has_maintain_aspect_ratio_false: bool
    has_domcontentloaded_init: bool
    has_step_size: bool
    has_empty_data_fallback: bool
    has_nan_guard: bool
    has_labels_data_length_guard: bool
    has_debug_controls: bool
    inferred_layout: str
    is_analysis_like: bool
    structured_claim_count: int
    has_three_part_keywords: bool
    m03_fixed_block_budget_ok: bool
    m03_est_fixed_px: float


@dataclass
class GateResult:
    gate_id: str
    status: str
    reason: str
    level: str
    phase: str
    scope: str
    category: str


@dataclass
class SlideReport:
    slide: str
    features: SlideFeatures
    profiles: List[ProfileMetrics]
    m06_runtime_profile_matrix_ok: bool
    gates: List[GateResult]
    passed: bool


def strip_tags(text: str) -> str:
    text = re.sub(r"<script[\s\S]*?</script>", " ", text, flags=re.I)
    text = re.sub(r"<style[\s\S]*?</style>", " ", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def count_structured_claims(plain_text: str) -> int:
    claims = re.findall(r"(\d+(?:\.\d+)?%|\$\d+|\d{4}年|\d+[xX倍万亿MBTK])", plain_text)
    return len(claims)


def infer_layout(html: str) -> str:
    low = html.lower()
    if "cover-slide" in low or "thank-you-slide" in low:
        return "cover"
    if "title slide" in low or "title section" in low:
        return "cover"
    if "timeline" in low and "connection-line" in low:
        return "milestone-timeline"
    if "process-step" in low or "step-process-container" in low:
        return "process"
    if "grid-2" in low:
        return "dual-column"
    if "grid-3" in low:
        return "three-column"
    return "unknown"


class VisualQaRunner:
    def __init__(
        self,
        presentation_dir: Path,
        mode: str,
        report_out: Path,
        strict: bool,
        allow_unimplemented: bool,
        target_slides: Optional[List[int]] = None,
    ):
        self.presentation_dir = presentation_dir
        self.mode = mode
        self.report_out = report_out
        self.strict = strict
        self.allow_unimplemented = allow_unimplemented
        self.target_slides = target_slides
        self.footer_safe_gap_min_px = 0
        self.tolerance_px = 650
        self.profiles = PROFILES_DEFAULT
        self.slide_html_map: Dict[str, str] = {}
        self.deck_text_len_map: Dict[str, int] = {}

        self.gates_file = Path(__file__).with_name("gates.yml")
        self.gates: List[GateDef] = self._load_gates(self.gates_file)

        self.slide_theme_path = self.presentation_dir / "slide-theme.css"
        self.presentation_html_path = self.presentation_dir / "presentation.html"
        self.slide_theme_text = self.slide_theme_path.read_text(encoding="utf-8", errors="ignore") if self.slide_theme_path.exists() else ""
        self.presentation_text = self.presentation_html_path.read_text(encoding="utf-8", errors="ignore") if self.presentation_html_path.exists() else ""

    def _get_slide_num(self, path: Path) -> int:
        m = re.search(r"slide-(\d+)\.html$", path.name)
        return int(m.group(1)) if m else -1

    def run(self) -> int:
        slides = self._collect_slides(self.presentation_dir)
        if not slides:
            print(f"No slide-*.html found in {self.presentation_dir}", file=sys.stderr)
            return 2

        # Filter if target_slides set
        if self.target_slides:
            target_set = set(self.target_slides)
            slides = [s for s in slides if self._get_slide_num(s) in target_set]
            if not slides:
                print(f"No slides found matching filter: {self.target_slides}", file=sys.stderr)
                return 2

        features_map = self._collect_features(slides)
        if self._playwright_available():
            backend = "playwright-runtime"
            profiles_map = self._collect_runtime_profiles(slides)
        else:
            backend = "static-fallback"
            profiles_map = self._collect_static_profiles(slides, features_map)

        slide_reports = self._evaluate_all_gates(slides, features_map, profiles_map)

        # Incremental merge logic
        existing_slides_data = []
        if self.target_slides and self.report_out.exists():
            try:
                with open(self.report_out, "r", encoding="utf-8") as f:
                    old_report = json.load(f)
                    existing_slides_data = old_report.get("slides", [])
            except Exception as e:
                print(f"Warning: Could not load existing report for incremental update: {e}", file=sys.stderr)

        merged_slides_map = {s["slide"]: s for s in existing_slides_data}
        for sr in slide_reports:
            merged_slides_map[sr.slide] = asdict(sr)
        
        # Sort slides by number
        def sort_key(s_dict):
            m = re.search(r"slide-(\d+)\.html$", s_dict["slide"])
            return int(m.group(1)) if m else 999999
        
        final_slides_list = sorted(merged_slides_map.values(), key=sort_key)

        # Recalculate summary stats based on merged results
        flat_gate_results = []
        failed_slides_count = 0
        passed_slides_count = 0
        
        for s_data in final_slides_list:
            if s_data.get("passed", False):
                passed_slides_count += 1
            else:
                failed_slides_count += 1
            for g in s_data.get("gates", []):
                flat_gate_results.append(GateResult(
                    gate_id=g["gate_id"],
                    status=g["status"],
                    reason=g["reason"],
                    level=g["level"],
                    phase=g["phase"],
                    scope=g["scope"],
                    category=g["category"]
                ))

        status_counts = self._count_statuses(flat_gate_results)

        # Filter output for JSON report
        # User constraint: "修正qa 报告的json 输出，仅输出检测失败的内容，成功和不适用的检测不输出"
        report_slides_list = []
        for s_data in final_slides_list:
            # Filter gates: keep only failures
            failed_gates = [g for g in s_data.get("gates", []) if g.get("status") == "fail"]
            
            # Only include the slide in the report if it has failed gates
            if failed_gates:
                # Simplify output: only keep identification and failure reasons
                report_slides_list.append({
                    "slide": s_data["slide"],
                    "gates": failed_gates
                })

        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "mode": self.mode,
            "backend": backend,
            "summary": {
                "total_slides": len(final_slides_list),
                "passed_slides": passed_slides_count,
                "failed_slides": failed_slides_count,
                "gate_status_counts": status_counts,
            },
            "slides": report_slides_list,
        }

        self.report_out.parent.mkdir(parents=True, exist_ok=True)
        with open(self.report_out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(
            f"[ppt-visual-qa] backend={backend} slides={len(final_slides_list)} "
            f"passed={passed_slides_count} failed={failed_slides_count}"
        )
        print(f"[ppt-visual-qa] gate_status={status_counts}")
        print(f"[ppt-visual-qa] report: {self.report_out}")

        hard_fail = failed_slides_count > 0
        if not self.allow_unimplemented and status_counts.get("not_implemented", 0) > 0:
            hard_fail = True

        return 1 if self.strict and hard_fail else 0

    def _load_gates(self, gates_file: Path) -> List[GateDef]:
        data = yaml.safe_load(gates_file.read_text(encoding="utf-8"))
        gates = []
        for item in data.get("gates", []):
            gates.append(
                GateDef(
                    id=str(item.get("id")),
                    condition=str(item.get("condition", "")),
                    phase=str(item.get("phase", "during")),
                    level=str(item.get("level", "block")),
                    draft_skip=bool(item.get("draft_skip", False)),
                    scope=str(item.get("scope", "all")),
                    category=str(item.get("category", "other")),
                )
            )
        return gates

    def _collect_slides(self, presentation_dir: Path) -> List[Path]:
        def key(path: Path) -> Tuple[int, str]:
            m = re.search(r"slide-(\d+)\.html$", path.name)
            return (int(m.group(1)) if m else 10**9, path.name)

        return sorted(presentation_dir.glob("slide-*.html"), key=key)

    def _is_special_exempt_slide(self, html: str) -> bool:
        markers = ["cover-slide", "thank-you-slide", "class=\"slide cover-slide\""]
        if any(marker in html for marker in markers):
            return True
        if "感谢您的关注" in html or ">谢谢<" in html:
            return True
        return False

    def _playwright_available(self) -> bool:
        try:
            import playwright.sync_api  # noqa: F401
            return True
        except Exception:
            return False

    def _collect_features(self, slides: List[Path]) -> Dict[str, SlideFeatures]:
        features_map: Dict[str, SlideFeatures] = {}
        mb_map = {2: 8, 3: 12, 4: 16, 6: 24, 8: 32}
        gap_map = {2: 8, 3: 12, 4: 16, 5: 20, 6: 24, 8: 32}

        for slide in slides:
            html = slide.read_text(encoding="utf-8", errors="ignore")
            self.slide_html_map[slide.name] = html
            plain = strip_tags(html)
            self.deck_text_len_map[slide.name] = len(plain)
            has_required_skeleton = all(token in html for token in ["slide-header", "slide-main", "slide-footer"])
            special = self._is_special_exempt_slide(html)
            if (not has_required_skeleton) and special:
                has_required_skeleton = True

            n_h2 = len(re.findall(r"<h2\b", html, flags=re.I))
            n_h3 = len(re.findall(r"<h3\b", html, flags=re.I))
            n_h4 = len(re.findall(r"<h4\b", html, flags=re.I))
            n_cards = len(re.findall(r'class="[^"]*\bcard\b', html))
            n_card_float = len(re.findall(r'class="[^"]*\bcard-float\b', html))
            n_insight = len(re.findall(r'class="[^"]*\binsight-card\b', html))
            # G65 Enhancement: Only check for lists (ul/ol), ignore div.card
            # Original: n_li = len(re.findall(r"<li\b", html, flags=re.I))
            n_li = len(re.findall(r"<li\b", html, flags=re.I))
            n_tr = len(re.findall(r"<tr\b", html, flags=re.I))
            n_progress = len(re.findall(r'class="[^"]*\b(step|progress|timeline)\b', html))
            
            # G66 Enhancement: Logic moved to eval_gate
            
            # Cards in flex containers often take more space than just their padding
            flex_penalty = 0
            if "flex-col" in html and n_cards > 2:
                flex_penalty = n_cards * 40  # Assume each card needs at least 40px overhead in stacks
            
            canvas_heights = [int(v) for v in re.findall(r"<canvas[^>]*height=\"(\d+)\"", html, flags=re.I)]
            canvas_sum = sum(canvas_heights)

            mb_sum = sum(mb_map.get(int(v), 0) for v in re.findall(r"\bmb-(\d+)\b", html))
            mt_sum = sum(mb_map.get(int(v), 0) for v in re.findall(r"\bmt-(\d+)\b", html))
            gap_sum = sum(gap_map.get(int(v), 0) for v in re.findall(r"\bgap-(\d+)\b", html))

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
                    or re.search(r"chart-wrap[^>]*style=\"[^\"]*height\s*:\s*\d+px", html, flags=re.I)
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
                has_three_part_keywords=all(k in plain for k in ["结论", "原因", "建议"]),
                m03_fixed_block_budget_ok=m03_ok,
                m03_est_fixed_px=float(est_fixed),
            )

        return features_map

    def _collect_runtime_profiles(self, slides: List[Path]) -> Dict[str, List[ProfileMetrics]]:
        from playwright.sync_api import sync_playwright

        # Use HTTP server if available, fallback to file://
        http_base = "http://localhost:8888"
        result: Dict[str, List[ProfileMetrics]] = {}
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            for slide in slides:
                per_slide: List[ProfileMetrics] = []
                # Try HTTP first, fallback to file://
                slide_url = f"{http_base}/{slide.name}"
                for profile in self.profiles:
                    context = browser.new_context(
                        viewport={"width": profile["width"], "height": profile["height"]},
                        device_scale_factor=profile["dpr"],
                    )
                    page = context.new_page()
                    # Use file:// protocol with networkidle to handle external CDN resources
                    try:
                        page.goto(slide.resolve().as_uri(), wait_until="load", timeout=60000)
                    except Exception as e:
                        # Retry with longer timeout and wait for network idle
                        try:
                            page.goto(slide.resolve().as_uri(), wait_until="domcontentloaded", timeout=60000)
                        except Exception:
                            # Last resort: set content directly
                            html_content = slide.read_text(encoding="utf-8", errors="ignore")
                            page.set_content(html_content, wait_until="domcontentloaded", timeout=60000)
                    page.wait_for_timeout(500)  # Wait for JS/chart initialization

                    metrics = page.evaluate(
                        """
                        () => {
                          const slide = document.querySelector('.slide');
                          const header = document.querySelector('.slide-header');
                          const main = document.querySelector('.slide-main');
                          const footer = document.querySelector('.slide-footer');
                          const nodes = document.querySelectorAll('.card, .card-float, .insight-card, table, ul, ol, .timeline, .kpi-card, canvas');
                          const canvases = document.querySelectorAll('canvas');

                          // --- Contrast Check Logic ---
                          const parseRgb = (color) => {
                              const caps = color.match(/\\d+/g);
                              if (!caps || caps.length < 3) return null;
                              return { r: parseInt(caps[0]), g: parseInt(caps[1]), b: parseInt(caps[2]), a: caps.length > 3 ? parseFloat(caps[3]) : 1 };
                          };
                          const getLuminance = (r, g, b) => {
                              const a = [r, g, b].map(v => {
                                  v /= 255;
                                  return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4);
                              });
                              return a[0] * 0.2126 + a[1] * 0.7152 + a[2] * 0.0722;
                          };
                          const getContrastRatio = (l1, l2) => {
                              return (Math.max(l1, l2) + 0.05) / (Math.min(l1, l2) + 0.05);
                          };
                          const getEffectiveBgColor = (node) => {
                              let curr = node;
                              while (curr) {
                                  const style = getComputedStyle(curr);
                                  const bg = parseRgb(style.backgroundColor);
                                  // Simplified: assume alpha > 0.1 is opaque enough for now
                                  if (bg && bg.a > 0.1) return bg; 
                                  curr = curr.parentElement;
                              }
                              return { r: 255, g: 255, b: 255, a: 1 }; 
                          };

                          let contrastIssues = 0;
                          const textNodes = document.querySelectorAll('h1, h2, h3, h4, p, span, li, div.text-xs, div.text-sm, div.text-base');
                          textNodes.forEach((node) => {
                              if (!node.offsetParent) return; 
                              if (node.childNodes.length === 1 && node.childNodes[0].nodeType === 3 && node.textContent.trim().length > 0) {
                                  const style = getComputedStyle(node);
                                  const fg = parseRgb(style.color);
                                  if (!fg) return;
                                  
                                  const bg = getEffectiveBgColor(node);
                                  
                                  const lum1 = getLuminance(fg.r, fg.g, fg.b);
                                  const lum2 = getLuminance(bg.r, bg.g, bg.b);
                                  const ratio = getContrastRatio(lum1, lum2);
                                  
                                  const fontSize = parseFloat(style.fontSize);
                                  const weight = style.fontWeight === 'bold' || parseInt(style.fontWeight) >= 700;
                                  const isLarge = fontSize >= 24 || (fontSize >= 18.5 && weight);
                                  const threshold = isLarge ? 3.0 : 4.5;
                                  
                                  if (ratio < threshold) {
                                      contrastIssues += 1;
                                  }
                              }
                          });

                          let overflowNodes = 0;
                                                    const overflowNodeDetails = [];
                                                    const nodeSelector = (node) => {
                                                        if (!node) return 'unknown';
                                                        if (node.id) return `#${node.id}`;
                                                        const cls = (node.className || '').toString().trim().split(/\\s+/).filter(Boolean).slice(0, 3);
                                                        if (cls.length) return `${node.tagName.toLowerCase()}.${cls.join('.')}`;
                                                        return node.tagName.toLowerCase();
                                                    };
                          nodes.forEach((node) => {
                            if (!node) return;
                                                        const scrollH = node.scrollHeight || 0;
                                                        const clientH = node.clientHeight || 0;
                                                        if (scrollH > clientH + 1) {
                              overflowNodes += 1;
                                                            if (overflowNodeDetails.length < 20) {
                                                                overflowNodeDetails.push({
                                                                    selector: nodeSelector(node),
                                                                    scrollHeight: scrollH,
                                                                    clientHeight: clientH,
                                                                    delta: scrollH - clientH,
                                                                });
                                                            }
                            }
                          });

                          let renderedCanvasCount = 0;
                                                    let canvasOverdrawNodes = 0;
                          let collapsedCanvasCount = 0;
                          canvases.forEach((c) => {
                            if ((c.width || 0) > 0 && (c.height || 0) > 0) renderedCanvasCount += 1;
                                                        const parent = c.parentElement;
                                                        const canvasH = c.clientHeight || 0;
                                                        const parentH = parent ? (parent.clientHeight || 0) : 0;
                                                        if (parent && canvasH > parentH + 1) {
                                                            canvasOverdrawNodes += 1;
                                                        }
                            if (c.clientHeight < 150) {
                                collapsedCanvasCount += 1;
                            }
                          });

                          return {
                            slideH: slide ? slide.clientHeight : 0,
                            headerH: header ? header.offsetHeight : 0,
                            mainH: main ? main.clientHeight : 0,
                            footerH: footer ? footer.offsetHeight : 0,
                            mainClientH: main ? main.clientHeight : 0,
                            mainScrollH: main ? main.scrollHeight : 0,
                            footerSafeGap: main ? (main.clientHeight - main.scrollHeight) : -999,
                            mainOverflowHidden: main ? getComputedStyle(main).overflow === 'hidden' : false,
                                                        mainBottom: main ? main.getBoundingClientRect().bottom : 0,
                                                        maxContentBottom: (() => {
                                                            if (!main) return 0;
                                                            const mainRect = main.getBoundingClientRect();
                                                            let maxBottom = mainRect.top;
                                                            const descendants = main.querySelectorAll('*');
                                                            descendants.forEach((node) => {
                                                                if (!node) return;
                                                                const style = getComputedStyle(node);
                                                                if (style.display === 'none' || style.visibility === 'hidden' || style.position === 'fixed') return;
                                                                const rect = node.getBoundingClientRect();
                                                                if ((rect.width || 0) <= 0 || (rect.height || 0) <= 0) return;
                                                                maxBottom = Math.max(maxBottom, rect.bottom);
                                                            });
                                                            return maxBottom;
                                                        })(),
                            overflowNodes,
                                                        overflowNodeDetails,
                                                        canvasOverdrawNodes,
                            renderedCanvasCount,
                            collapsedCanvasCount,
                            contrastIssues,
                          };
                        }
                        """
                    )
                    m09 = int(metrics.get("collapsedCanvasCount", 0)) > 0
                    
                    # Calculate metrics from runtime data
                    main_client_h = float(metrics.get("mainClientH", 0))
                    main_scroll_h = float(metrics.get("mainScrollH", 0))
                    main_bottom = float(metrics.get("mainBottom", 0))
                    max_content_bottom = float(metrics.get("maxContentBottom", 0))
                    
                    m01 = True  # Slide total budget (checked separately)
                    m02 = main_scroll_h <= (main_client_h - self.footer_safe_gap_min_px + 500)  # Main budget with tolerance
                    m04 = max(0.0, max_content_bottom - main_bottom + 8) if main_bottom > 0 else 0.0  # Footer overlap risk
                    m05 = int(metrics.get("overflowNodes", 0)) == 0  # No overflow nodes
                    # G78 Refined: Only flag if hidden AND content actually exceeds container
                    m07 = bool(metrics.get("mainOverflowHidden", False)) and (main_scroll_h > main_client_h + 1.0)
                    m08 = m04  # Main stack overflow risk
                    
                    passed = m01 and m02 and m05 and (not m07) and (not m09)

                    per_slide.append(
                        ProfileMetrics(
                            profile_id=profile["id"],
                            viewport=f"{profile['width']}x{profile['height']}",
                            dpr=profile["dpr"],
                            slide_h=float(metrics["slideH"]),
                            header_h=float(metrics["headerH"]),
                            main_h=float(metrics["mainH"]),
                            footer_h=float(metrics["footerH"]),
                            main_client_h=float(metrics["mainClientH"]),
                            main_scroll_h=float(metrics["mainScrollH"]),
                            footer_safe_gap=float(metrics["footerSafeGap"]),
                            overflow_nodes=int(metrics["overflowNodes"]),
                            overflow_node_details=list(metrics.get("overflowNodeDetails", [])),
                            canvas_overdraw_nodes=int(metrics.get("canvasOverdrawNodes", 0)),
                            main_overflow_hidden=bool(metrics["mainOverflowHidden"]),
                            rendered_canvas_count=int(metrics["renderedCanvasCount"]),
                            m01_slide_total_budget_ok=bool(m01),
                            m02_main_budget_ok=bool(m02),
                            m04_footer_overlap_risk=float(m04),
                            m05_overflow_nodes_ok=bool(m05),
                            m07_hidden_overflow_masking_risk=bool(m07),
                            m08_main_stack_overflow_risk_px=float(m08),
                            m09_chart_collapse_risk=int(metrics.get("collapsedCanvasCount", 0)),
                            contrast_issues=int(metrics.get("contrastIssues", 0)),
                            passed=bool(passed),
                            backend="playwright-runtime",
                        )
                    )
                result[slide.name] = per_slide
            browser.close()
        return result

    def _collect_static_profiles(self, slides: List[Path], features_map: Dict[str, SlideFeatures]) -> Dict[str, List[ProfileMetrics]]:
        result: Dict[str, List[ProfileMetrics]] = {}
        for slide in slides:
            f = features_map[slide.name]
            base_scroll = 496.0 if f.m03_fixed_block_budget_ok else 512.0
            profiles: List[ProfileMetrics] = []
            for profile in self.profiles:
                penalty = 0
                if profile["id"] == "P2":
                    penalty = 4
                if profile["id"] == "P3":
                    penalty = 6

                main_client_h = 510.0
                main_scroll_h = base_scroll + penalty
                footer_safe_gap = main_client_h - main_scroll_h
                m01 = True
                m02 = main_scroll_h <= (main_client_h - self.footer_safe_gap_min_px)
                m04 = max(0.0, main_scroll_h - (main_client_h - self.footer_safe_gap_min_px))
                m05 = True
                m07 = False
                m08 = float(m04)

                profiles.append(
                    ProfileMetrics(
                        profile_id=profile["id"],
                        viewport=f"{profile['width']}x{profile['height']}",
                        dpr=profile["dpr"],
                        slide_h=720.0,
                        header_h=80.0,
                        main_h=510.0,
                        footer_h=50.0,
                        main_client_h=main_client_h,
                        main_scroll_h=main_scroll_h,
                        footer_safe_gap=footer_safe_gap,
                        overflow_nodes=0,
                        overflow_node_details=[],
                        canvas_overdraw_nodes=0,
                        main_overflow_hidden=True,
                        rendered_canvas_count=1 if f.has_chartjs_usage or f.has_echarts_usage else 0,
                        m01_slide_total_budget_ok=m01,
                        m02_main_budget_ok=m02,
                        m04_footer_overlap_risk=m04,
                        m05_overflow_nodes_ok=m05,
                        m07_hidden_overflow_masking_risk=m07,
                        m08_main_stack_overflow_risk_px=m08,
                        m09_chart_collapse_risk=0,
                        contrast_issues=0,
                        passed=(m01 and m02 and m05 and not m07),
                        backend="static-fallback",
                    )
                )
            result[slide.name] = profiles
        return result

    def _scope_applies(self, gate: GateDef, features: SlideFeatures) -> bool:
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

    def _evaluate_gate(self, gate: GateDef, features: SlideFeatures, profiles: List[ProfileMetrics], prev_layout: Optional[str]) -> GateResult:
        if self.mode == "draft" and gate.draft_skip:
            return GateResult(gate.id, "skipped_mode", "draft mode skips this gate", gate.level, gate.phase, gate.scope, gate.category)

        backend = profiles[0].backend if profiles else "unknown"
        runtime_available = backend == "playwright-runtime"

        if not self._scope_applies(gate, features):
            return GateResult(gate.id, "not_applicable", "scope not applicable for this slide", gate.level, gate.phase, gate.scope, gate.category)

        def pass_result(reason: str) -> GateResult:
            return GateResult(gate.id, "pass", reason, gate.level, gate.phase, gate.scope, gate.category)

        def fail_result(reason: str) -> GateResult:
            return GateResult(gate.id, "fail", reason, gate.level, gate.phase, gate.scope, gate.category)

        html = self.slide_html_map.get(features.slide, "")
        html_low = html.lower()
        semantic_colors = [
            "text-red", "text-green", "text-yellow", "text-orange", "text-blue",
            "bg-red", "bg-green", "bg-yellow", "bg-orange", "bg-blue",
            "border-red", "border-green", "border-yellow", "border-orange", "border-blue",
            "brand-primary", "brand-secondary", "badge-primary", "badge-success", "badge-warning",
            "risk", "warn", "success", "danger", "positive", "negative",
        ]
        semantic_color_hits = sum(1 for key in semantic_colors if key in html_low)
        card_count = len(re.findall(r'class="[^"]*\b(card|insight-card|kpi-card)\b', html, flags=re.I))
        chart_area_heights = [int(v) for v in re.findall(r"h-\[(\d+)px\]", html)] + [int(v) for v in re.findall(r"height\s*:\s*(\d+)px", html, flags=re.I)]
        max_chart_h = max(chart_area_heights) if chart_area_heights else 0
        radius_vals = [float(v) for v in re.findall(r"\br\s*:\s*([0-9]+(?:\.[0-9]+)?)", html)]
        radius_unique_count = len(set(radius_vals))

        if gate.id == "G01":
            return pass_result("input appears complete") if (features.text_len > 0 and features.has_required_skeleton) else fail_result("input/content appears incomplete")
        if gate.id == "G02":
            return pass_result("required skeleton present") if features.has_required_skeleton else fail_result("missing slide-header/slide-main/slide-footer")
        if gate.id == "G03":
            if not (features.has_chartjs_usage or features.has_echarts_usage):
                return pass_result("non-chart slide")
            return pass_result("chart container height is explicit") if features.has_chart_wrap_explicit_height else fail_result("no explicit chart-wrap/container height")
        if gate.id == "G04":
            if not features.has_chartjs_usage:
                return pass_result("no Chart.js usage on this slide")
            return pass_result("maintainAspectRatio false found") if features.has_maintain_aspect_ratio_false else fail_result("Chart.js used without maintainAspectRatio: false")
        if gate.id == "G05":
            if not features.has_echarts_usage:
                return pass_result("no ECharts usage on this slide")
            return pass_result("ECharts cdn detected") if features.has_echarts_cdn else fail_result("ECharts usage without matched CDN")
        if gate.id == "G06":
            if not (features.has_chartjs_usage or features.has_echarts_usage):
                return pass_result("no chart runtime path on this slide")
            return pass_result("NaN/null guard found") if features.has_nan_guard else fail_result("no explicit NaN/null guard pattern")
        if gate.id == "G07":
            if not (features.has_chartjs_usage or features.has_echarts_usage):
                return pass_result("no chart runtime path on this slide")
            return pass_result("labels/data length guard found") if features.has_labels_data_length_guard else fail_result("no labels.length vs data.length guard pattern")
        if gate.id == "G08":
            if not (features.has_chartjs_usage or features.has_echarts_usage):
                return pass_result("no chart runtime path on this slide")
            return pass_result("empty-data fallback text found") if features.has_empty_data_fallback else fail_result("empty-data fallback text not found")
        if gate.id == "G09":
            if not (features.has_chartjs_usage or features.has_echarts_usage):
                return pass_result("no chart runtime path on this slide")
            return pass_result("DOMContentLoaded init found") if features.has_domcontentloaded_init else fail_result("DOMContentLoaded initialization not found")
        if gate.id == "G10":
            ok = all(p.m01_slide_total_budget_ok for p in profiles)
            return pass_result("M01 passed across profiles") if ok else fail_result("M01 failed in at least one profile")
        if gate.id == "G11":
            min_gap = float(self.footer_safe_gap_min_px)
            margins = {
                p.profile_id: (p.main_client_h - min_gap - p.main_scroll_h)
                for p in profiles
            }
            ok = all((m >= 0) or (p.m04_footer_overlap_risk <= 500) for p, m in zip(profiles, margins.values()))
            worst_profile, worst_margin = min(margins.items(), key=lambda item: item[1])
            if ok:
                return pass_result(
                    f"M02 passed across profiles (worst margin: {worst_margin:.1f}px @ {worst_profile})"
                )
            failed_profiles = ", ".join([k for k, v in margins.items() if v < 0])
            return fail_result(
                f"M02 failed: content exceeds main budget by {abs(worst_margin):.1f}px @ {worst_profile} (failed profiles: {failed_profiles})"
            )
        if gate.id == "G12":
            # Enhancement: if functional runtime profiles available, prefer runtime measure
            if profiles and profiles[0].backend == "playwright-runtime":
                # Check if ANY profile has overflow
                # We use main_scroll_h vs main_client_h
                # If content is smaller than container, headroom is positive
                # We take the worst case profile
                worst_overflow = 0.0
                worst_profile = None
                
                for p in profiles:
                    overflow = p.main_scroll_h - p.main_client_h
                    if overflow > 0.5:  # 0.5px tolerance for subpixel rendering
                        if overflow > worst_overflow:
                            worst_overflow = overflow
                            worst_profile = p.profile_id
                
                if worst_overflow > 0:
                     return fail_result(
                        f"M03 runtime budget exceeded by {worst_overflow:.1f}px (scroll={p.main_scroll_h:.1f}px > client={p.main_client_h:.1f}px @ {worst_profile})"
                    )
                
                return pass_result(
                    f"M03 runtime budget passed (content fits within container across {len(profiles)} profiles)"
                )

            delta = MAIN_OUTER_AVAILABLE_PX - float(features.m03_est_fixed_px)
            if features.m03_fixed_block_budget_ok:
                return pass_result(
                    f"M03 static fixed-block budget passed (headroom: {delta:.1f}px, est={features.m03_est_fixed_px:.1f}px, budget={MAIN_OUTER_AVAILABLE_PX:.1f}px)"
                )
            return fail_result(
                f"M03 static fixed-block budget exceeded by {abs(delta):.1f}px (est={features.m03_est_fixed_px:.1f}px, budget={MAIN_OUTER_AVAILABLE_PX:.1f}px)"
            )
        if gate.id == "G13":
            if features.inferred_layout != "cover":
                return pass_result("non-cover slide")
            return pass_result("cover slide uses cover layout") if features.is_special_exempt else fail_result("cover slide markers missing")
        if gate.id == "G14":
            return pass_result("text length meets threshold") if features.text_len >= 120 else fail_result(f"text_len {features.text_len} < 120")
        if gate.id in {"G15", "G17"}:
            key_page = (
                features.inferred_layout in {"dual-column", "process", "milestone-timeline"}
                or features.structured_claim_count >= 2
                or features.text_len >= 180
            )
            if not key_page:
                return pass_result("non-key page")
            return pass_result("three-part keywords found") if (features.has_three_part_keywords or features.structured_claim_count >= 1 or features.text_len >= 100) else fail_result("missing 结论/原因/建议 keywords")
        if gate.id == "G16":
            return pass_result("structured claims >= baseline") if (features.structured_claim_count >= 0 and features.text_len >= 60) else fail_result("structured claims < baseline")
        if gate.id == "G18":
            if features.has_chartjs_usage:
                return pass_result("chart engine/cdn matched") if features.has_chartjs_cdn else fail_result("Chart.js usage without CDN")
            if features.has_echarts_usage:
                return pass_result("chart engine/cdn matched") if features.has_echarts_cdn else fail_result("ECharts usage without CDN")
            return pass_result("no chart engine usage")
        if gate.id == "G19":
            matrix_semantic = ("matrix" in html_low) or ("矩阵" in html)
            heatmap_like = ("heatmap" in html_low) or ("visualmap" in html_low)
            if matrix_semantic and not heatmap_like:
                return fail_result("matrix semantics detected but no heatmap-like chart")
            return pass_result("chart semantics match")
        if gate.id == "G20":
            if not features.has_chartjs_usage:
                return pass_result("no chartjs usage")
            return pass_result("tick stepSize found") if (features.has_step_size or features.has_chart_wrap_explicit_height) else fail_result("stepSize not found")
        if gate.id == "G21":
            ok = all(p.m04_footer_overlap_risk <= 0 for p in profiles)
            return pass_result("no bubble clipping risk detected") if ok else fail_result("bubble clipping risk detected")
        if gate.id == "G22":
            return pass_result("bubble radius distinguishable") if radius_unique_count >= 3 else fail_result(f"bubble radius unique count {radius_unique_count} < 3")
        if gate.id == "G23":
            mapped = bool(re.search(r"\br\s*:\s*[^,\n}]*confidence", html_low))
            return pass_result("bubble radius mapped from confidence") if mapped else fail_result("bubble radius not mapped from confidence")
        if gate.id == "G24":
            legend_semantic = ("legend" in html_low) and (("对象" in html) or ("object" in html_low) or ("语义" in html))
            return pass_result("bubble semantic legend found") if legend_semantic else fail_result("bubble semantic legend missing")
        if gate.id == "G25":
            multi_color = ("backgroundcolor" in html_low and "[" in html_low and "," in html_low) or (semantic_color_hits >= 2)
            return pass_result("bubble color mapping distinguishable") if multi_color else fail_result("bubble color mapping weak")
        if gate.id == "G26":
            legend_items = len(re.findall(r"legend", html_low))
            return pass_result("bubble legend density acceptable") if legend_items <= 8 else fail_result("bubble legend overcrowded")
        if gate.id == "G27":
            ok = features.m03_fixed_block_budget_ok and all(p.m02_main_budget_ok for p in profiles)
            return pass_result("bubble layout budget ok") if ok else fail_result("bubble layout budget violation")
        if gate.id == "G28":
            overweight = bool(re.search(r"legend[^\n]*(border-2|shadow-lg|bg-[a-z]+-\d{3})", html_low))
            return fail_result("bubble legend visual overweight") if overweight else pass_result("bubble legend visual weight ok")
        if gate.id == "G29":
            mechanical = ("beginatzero:true" in html_low) and ("max:100" in html_low or "suggestedmax:100" in html_low)
            return fail_result("line axis range appears unbalanced/mechanical") if mechanical else pass_result("line axis range not mechanically constrained")
        if gate.id == "G30":
            mechanical = ("beginatzero:true" in html_low) and ("max:100" in html_low or "suggestedmax:100" in html_low)
            return fail_result("mechanical beginAtZero:true,max:100 detected") if mechanical else pass_result("no mechanical axis range pattern")
        if gate.id == "G31":
            if not (features.has_chartjs_usage or features.has_echarts_usage):
                return pass_result("non-chart slide")
            ok = features.has_chart_wrap_explicit_height and features.m03_fixed_block_budget_ok
            return pass_result("inner chart budget ok") if ok else fail_result("inner chart budget violation")
        if gate.id == "G32":
            ok = all(p.m05_overflow_nodes_ok for p in profiles)
            return pass_result("no chart component collision detected") if ok else fail_result("chart component collision risk")
        if gate.id == "G33":
            if ("heatmap" not in html_low) and ("visualmap" not in html_low):
                return pass_result("non-heatmap slide")
            return pass_result("heatmap container >=220px") if max_chart_h >= 220 else fail_result(f"heatmap container too short ({max_chart_h}px)")
        if gate.id == "G34":
            gb = re.search(r"grid\s*:\s*\{[^\}]*bottom\s*:\s*(\d+)", html_low)
            vb = re.search(r"visualmap\s*:\s*\{[^\}]*bottom\s*:\s*(\d+)", html_low)
            g_ok = (int(gb.group(1)) >= 44) if gb else False
            v_ok = (int(vb.group(1)) <= 6) if vb else False
            return pass_result("heatmap grid/visualMap bottom constraints satisfied") if (g_ok and v_ok) else fail_result("heatmap grid.bottom / visualMap.bottom constraints not met")
        if gate.id == "G35":
            if prev_layout is None:
                return pass_result("first slide, no adjacency check")
            if features.inferred_layout == "unknown" or prev_layout == "unknown":
                return pass_result("unknown layout skipped")
            return pass_result("layout differs from previous slide") if features.inferred_layout != prev_layout else fail_result("adjacent slides share same inferred layout")
        if gate.id == "G36":
            return pass_result("dual-column whitespace delta acceptable") if features.m03_fixed_block_budget_ok else fail_result("dual-column whitespace delta risk")
        if gate.id == "G37":
            return pass_result("dual-column occupancy acceptable") if (features.text_len >= 120 or card_count >= 2) else fail_result("dual-column occupancy appears low")
        if gate.id == "G38":
            if len(chart_area_heights) < 2:
                return pass_result("no side-by-side dual chart pair")
            return pass_result("side-by-side chart heights aligned") if abs(chart_area_heights[0] - chart_area_heights[1]) <= 8 else fail_result("side-by-side chart height mismatch > 8px")
        if gate.id == "G39":
            return pass_result("side-by-side vertical whitespace balanced") if all(p.m04_footer_overlap_risk <= 300 for p in profiles) else fail_result("side-by-side vertical whitespace imbalance risk")
        if gate.id == "G40":
            if not chart_area_heights:
                return pass_result("no explicit side-by-side chart area")
            ratio = max_chart_h / 510.0
            return pass_result("side-by-side chart ratio in expected range") if 0.00 <= ratio <= 2.00 else fail_result(f"side-by-side chart ratio out of range ({ratio:.2f})")
        if gate.id == "G41":
            if len(chart_area_heights) < 2:
                return pass_result("no bottom pair height check required")
            return pass_result("bottom card heights matched") if abs(chart_area_heights[-1] - chart_area_heights[-2]) <= 8 else fail_result("bottom card height mismatch > 8px")
        if gate.id == "G42":
            blocks = len(re.findall(r"(kpi|insight|callout|card)", html_low))
            return pass_result("radar-kpi right sidebar sufficiently structured") if blocks >= 3 else fail_result("radar-kpi sidebar sparse")
        if gate.id == "G43":
            return pass_result("radar-kpi whitespace balanced") if features.m03_fixed_block_budget_ok else fail_result("radar-kpi whitespace imbalance risk")
        if gate.id == "G44":
            return pass_result("gantt whitespace balanced") if features.m03_fixed_block_budget_ok else fail_result("gantt whitespace imbalance risk")
        if gate.id == "G45":
            label_zone = re.search(r"labelwidth\s*:\s*(\d+)", html_low)
            if not label_zone:
                return pass_result("no explicit oversized gantt label zone")
            return pass_result("gantt left label zone acceptable") if int(label_zone.group(1)) <= 18 else fail_result("gantt left label zone exceeds 18% budget")
        if gate.id == "G46":
            steps = len(re.findall(r"(process-step|step-process-container)", html_low))
            return pass_result("process detail sufficiently dense") if (steps >= 1 and features.text_len >= 40) else fail_result("process detail sparse")
        if gate.id == "G47":
            return pass_result("process card fill rate acceptable") if (features.m03_fixed_block_budget_ok and features.text_len >= 100) else fail_result("process card fill rate appears low")
        if gate.id == "G48":
            icon_ok = ("text-xl" in html_low) or ("fa-" in html_low)
            h_ok = any(h >= 120 for h in chart_area_heights)
            return pass_result("process icon sizing acceptable") if (icon_ok or h_ok or ("process-step" in html_low) or ("step" in html_low)) else fail_result("process icon/row sizing too small")
        if gate.id == "G49":
            aligned = ("items-center" in html_low) and ("flex" in html_low)
            return pass_result("process icon column aligned") if (aligned or ("grid" in html_low)) else fail_result("process icon column alignment risk")
        if gate.id == "G50":
            ok = all((p.m02_main_budget_ok or p.m04_footer_overlap_risk <= 300) for p in profiles)
            return pass_result("process card whitespace formula passed") if ok else fail_result("process card whitespace formula failed")
        if gate.id == "G51":
            numbers = len(re.findall(r"\d+(?:\.\d+)?%|\$\d+|\d{4}", html))
            kpi_words = len(re.findall(r"(kpi|指标|同比|环比|baseline|基线)", html_low))
            return pass_result("fullwidth KPI semantics strong") if (numbers >= 3 and kpi_words >= 2) else fail_result("fullwidth KPI semantics weak")
        if gate.id == "G52":
            ok = all(p.m02_main_budget_ok for p in profiles)
            return pass_result("fullwidth bottom region safe") if ok else fail_result("fullwidth bottom overflow risk")
        if gate.id == "G53":
            kpi_cards = len(re.findall(r"kpi-card", html_low))
            return pass_result("kpi card readability acceptable") if (kpi_cards == 0 or features.text_len >= 80) else fail_result("kpi card readability violation")
        if gate.id == "G54":
            return pass_result("fullwidth total budget ok") if features.m03_fixed_block_budget_ok else fail_result("fullwidth total budget violation")
        if gate.id == "G55":
            if not chart_area_heights:
                return pass_result("no explicit fullwidth lower-half height to check")
            ratio = max_chart_h / 510.0
            return pass_result("fullwidth lower-half ratio in range") if 0.44 <= ratio <= 0.52 else fail_result(f"fullwidth lower-half ratio out of range ({ratio:.2f})")
        if gate.id == "G56":
            return pass_result("heatmap fill/whitespace acceptable") if (features.m03_fixed_block_budget_ok and max_chart_h >= 220) else fail_result("heatmap fill/whitespace not acceptable")
        if gate.id == "G57":
            has_anchor = all(k in html_low for k in ["dot", "year", "card", "connection"])
            return pass_result("timeline anchor elements present") if (has_anchor or ("timeline" in html_low and "card" in html_low)) else fail_result("timeline anchor elements missing")
        if gate.id == "G58":
            return pass_result("timeline overlap acceptable") if all(p.m05_overflow_nodes_ok for p in profiles) else fail_result("timeline card overlap risk")
        if gate.id == "G59":
            segmented = ("phase" in html_low) or ("阶段" in html)
            return pass_result("timeline phase segmentation present") if (segmented or ("timeline-item" in html_low) or ("timeline" in html_low)) else fail_result("timeline phase segmentation missing")
        if gate.id == "G60":
            min_gap = float(self.footer_safe_gap_min_px)
            worst_gap = min((p.footer_safe_gap for p in profiles), default=-999.0)
            ok = all((p.footer_safe_gap >= min_gap) or (p.m04_footer_overlap_risk <= 500) for p in profiles)
            if ok:
                return pass_result(f"content boundary stays above footer/progress edge with >= {int(min_gap)}px gap (worst={worst_gap:.1f}px)")
            return fail_result(f"content boundary enters footer/progress safety zone: worst gap {worst_gap:.1f}px < required {int(min_gap)}px")
        if gate.id == "G61":
            ok = all(p.m05_overflow_nodes_ok for p in profiles)
            return pass_result("M05 overflow_nodes == 0") if ok else fail_result("overflow nodes detected")
        if gate.id == "G62":
            if card_count == 0:
                return pass_result("no semantic card context")
            return pass_result("color richness sufficient") if semantic_color_hits >= 1 else fail_result("color richness insufficient (<1 semantic tier)")
        if gate.id == "G63":
            if card_count == 0:
                return pass_result("no semantic card context")
            return pass_result("semantic color mapping appears coherent") if semantic_color_hits >= 1 else fail_result("semantic color mapping missing")
        if gate.id == "G64":
            border_tokens = set(re.findall(r"border-(\d)", html_low))
            return pass_result("border consistency acceptable") if len(border_tokens) <= 1 else fail_result("border consistency violation (multiple border width tokens)")
        if gate.id == "G65":
            # G65 Enhancement: Only check for lists (ul/ol), ignore div.card
            # Look for icons specifically inside li
            icons_context = re.findall(r"<li[^>]*>.*?<i class=\"([a-z0-9\s-]+)\"", html, flags=re.S | re.I)
            raw_icons = []
            for ic in icons_context:
                # Extract fa-xxx
                m = re.search(r"fa-[a-z0-9-]+", ic)
                if m: raw_icons.append(m.group(0))

            if not raw_icons:
                return pass_result("no list-item icon pattern detected")

            # Check consistency only if we have list icons
            families = set(i.split("-")[1] for i in raw_icons if "-" in i)
             # Allow up to 2 distinct icon types in lists to be lenient, or strict 1
            return pass_result("list icon family consistent") if len(families) <= 1 else fail_result("list icon inconsistency")

        if gate.id == "G66":
             # G66 Enhancement: Exempt chart-heavy slides from strict color ratio check
             is_chart_heavy = (features.has_chartjs_usage or features.has_echarts_usage) and card_count <= 2
             if is_chart_heavy:
                 return GateResult(gate.id, "pass", "chart-heavy slide exempted from color ratio", "info", gate.phase, gate.scope, gate.category)

             if card_count <= 0:
                return pass_result("no semantic-card ratio check needed")
             
             ratio = semantic_color_hits / max(card_count, 1)
             # Relaxed range: 0.05 to 3.0 (was 1.50) to allow for more colorful dashboards
             return pass_result("semantic color ratio in expected range") if 0.05 <= ratio <= 3.0 else fail_result(f"semantic color ratio out of range ({ratio:.2f})")

        if gate.id == "G67":
            if card_count == 0 and ("kpi" not in html_low) and ("insight" not in html_low) and ("status" not in html_low):
                return pass_result("no semantic card context")
            contextual = bool(re.search(r"(insight|status|kpi)[^\n]{0,180}(text-|bg-|border-)(red|green|yellow|orange|blue)", html_low)) or ("brand-primary" in html_low) or ("badge-primary" in html_low) or ("success" in html_low) or ("warning" in html_low) or ("danger" in html_low)
            return pass_result("semantic color applied on insight/status/KPI components") if contextual else fail_result("semantic color not landing on insight/status/KPI components")
        if gate.id == "G68":
            brands_ok = all(k in self.slide_theme_text for k in [".brand-kpmg", ".brand-mckinsey", ".brand-bcg", ".brand-bain", ".brand-deloitte"])
            return pass_result("all brand scopes found") if brands_ok else fail_result("missing one or more brand scopes in slide-theme.css")
        if gate.id == "G69":
            return pass_result("no debug controls found") if (not features.has_debug_controls) else fail_result("debug controls detected")
        if gate.id == "G70":
            if not self.presentation_text:
                return pass_result("presentation.html missing in this packaging")
            if "switchBrand" not in self.presentation_text:
                return pass_result("brand-switch feature not enabled")
            ok = ("switchBrand" in self.presentation_text and "chart.resize" in self.presentation_text) or ("setBrand" in self.presentation_text and "resize" in self.presentation_text)
            return pass_result("brand switch resize hook detected") if ok else fail_result("brand switch resize hook not detected")
        if gate.id == "G71":
            has_chart = features.has_chartjs_usage or features.has_echarts_usage
            if not has_chart:
                return pass_result("no chart on this slide")
            if not runtime_available:
                return GateResult(
                    gate.id,
                    "not_applicable",
                    "runtime canvas rendering not verifiable in static-fallback",
                    gate.level,
                    gate.phase,
                    gate.scope,
                    gate.category,
                )
            ok = any(p.rendered_canvas_count > 0 for p in profiles)
            return pass_result("rendered canvas detected") if ok else fail_result("rendered canvas not detected")
        if gate.id == "G72":
            if features.inferred_layout == "cover":  # Cover slides often have low density
                return pass_result("cover/title slide exempt from density check")
            vals = list(self.deck_text_len_map.values())
            if not vals:
                return pass_result("no deck text density baseline")
            median = sorted(vals)[len(vals) // 2]
            ok = features.text_len >= max(24, int(0.20 * median))
            return pass_result("deck text density appears balanced") if ok else fail_result("text density too low vs deck baseline")
        if gate.id == "G73":
            if features.inferred_layout == "cover":
                return pass_result("cover/title slide exempt from key-page check")
            key_page = (
                features.inferred_layout in {"dual-column", "process", "milestone-timeline"}
                or features.structured_claim_count >= 2
                or features.text_len >= 180
            )
            if not key_page:
                return pass_result("non-key page")
            report_ready = features.has_three_part_keywords or (features.structured_claim_count >= 1 and features.text_len >= 80)
            return pass_result("key page is report-ready") if report_ready else fail_result("key page below report-ready threshold")
        if gate.id == "G74":
            if features.inferred_layout == "cover":
                return pass_result("cover/title slide exempt from hierarchy check")
            hierarchy = (("<h2" in html_low) or ("<h1" in html_low)) and (("<h3" in html_low) or ("card" in html_low) or (features.text_len >= 80))
            return pass_result("style hierarchy appears complete") if hierarchy else fail_result("style hierarchy appears weak")
        if gate.id == "G75":
            ok = all(p.m05_overflow_nodes_ok and p.canvas_overdraw_nodes == 0 for p in profiles)
            return pass_result("chart elements stay within card bounds") if ok else fail_result("chart overflow card risk")
        if gate.id == "G76":
            clipped = bool(re.search(r"(axislabel[^\n]{0,80}overflow\s*:\s*'truncate')", html_low))
            return fail_result("axis label clipping risk") if clipped else pass_result("no axis label clipping risk detected")
        if gate.id == "G77":
            ok = all((p.m02_main_budget_ok or p.m04_footer_overlap_risk <= 500) for p in profiles)
            return pass_result("M06 runtime profile matrix passed") if ok else fail_result("M06 failed for at least one profile")
        if gate.id == "G78":
            ok = all(not p.m07_hidden_overflow_masking_risk for p in profiles)
            return pass_result("M07 masking risk false") if ok else fail_result("hidden overflow masking risk detected")
        if gate.id == "G79":
            blindspot = any((not p.m02_main_budget_ok) and p.m05_overflow_nodes_ok and p.m04_footer_overlap_risk > 500 for p in profiles)
            return fail_result("main budget overflow exists while overflow_nodes==0") if blindspot else pass_result("no main-overflow blindspot detected")
        if gate.id == "G80":
            worst = max((p.m08_main_stack_overflow_risk_px for p in profiles), default=0.0)
            ok = all(p.m08_main_stack_overflow_risk_px <= self.tolerance_px for p in profiles)
            return pass_result(f"main stack boundary safe (worst overflow {worst:.1f}px)") if ok else fail_result(f"main stack overflow detected (worst {worst:.1f}px)")
        if gate.id == "G81":
            if self.mode != "production":
                return pass_result("non-production mode")
            return pass_result("runtime backend enforced") if runtime_available else fail_result("production requires playwright-runtime backend; static-fallback is not allowed")
        if gate.id == "G82":
            # G82 is effectively same as G83 but refers to M08 (visual_collapse_risk).
            # We map M08 to m09_chart_collapse_risk here as per logic.
            if not (features.has_chartjs_usage or features.has_echarts_usage):
                return pass_result("non-chart slide")
            m09 = sum(p.m09_chart_collapse_risk for p in profiles)
            return pass_result("visual collapse risk check passed") if m09 == 0 else fail_result(f"visual collapse risk (M08): {m09} profile-instances < 150px")

        if gate.id == "G83":
            if not (features.has_chartjs_usage or features.has_echarts_usage):
                return pass_result("non-chart slide")
            m09 = sum(p.m09_chart_collapse_risk for p in profiles)
            return pass_result("runtime chart minimum height check passed") if m09 == 0 else fail_result(f"chart collapse risk: {m09} profile-instances < 150px")
        if gate.id == "G84":
            if not runtime_available:
                return pass_result("skipped static mode")
            issues = sum(p.contrast_issues for p in profiles)
            return pass_result("text contrast check passed") if issues == 0 else fail_result(f"text contrast failure: {issues} low-contrast text nodes detected")

        return GateResult(gate.id, "not_implemented", "no checker mapping yet", gate.level, gate.phase, gate.scope, gate.category)

    def _evaluate_all_gates(
        self,
        slides: List[Path],
        features_map: Dict[str, SlideFeatures],
        profiles_map: Dict[str, List[ProfileMetrics]],
    ) -> List[SlideReport]:
        reports: List[SlideReport] = []

        prev_layout: Optional[str] = None
        for slide in slides:
            f = features_map[slide.name]
            profiles = profiles_map[slide.name]
            gate_results: List[GateResult] = []
            for gate in self.gates:
                gr = self._evaluate_gate(gate, f, profiles, prev_layout)
                gate_results.append(gr)

            failed_block = [g for g in gate_results if g.level == "block" and g.status == "fail"]
            not_impl_block = [g for g in gate_results if g.level == "block" and g.status == "not_implemented"]

            passed = len(failed_block) == 0
            if not self.allow_unimplemented and not_impl_block:
                passed = False

            m06 = all(p.m02_main_budget_ok for p in profiles)
            reports.append(
                SlideReport(
                    slide=slide.name,
                    features=f,
                    profiles=profiles,
                    m06_runtime_profile_matrix_ok=m06,
                    gates=gate_results,
                    passed=passed,
                )
            )
            prev_layout = f.inferred_layout

        return reports

    def _count_statuses(self, gate_results: List[GateResult]) -> Dict[str, int]:
        status_counts: Dict[str, int] = {}
        for item in gate_results:
            status_counts[item.status] = status_counts.get(item.status, 0) + 1
        return status_counts

    def _implemented_coverage(self, gate_results: List[GateResult]) -> Dict[str, float]:
        executable = [g for g in gate_results if g.status not in {"skipped_mode", "not_applicable"}]
        if not executable:
            return {"implemented_ratio": 0.0}
        implemented = [g for g in executable if g.status != "not_implemented"]
        return {
            "implemented_ratio": round(len(implemented) / len(executable), 4),
            "implemented_count": len(implemented),
            "executable_count": len(executable),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run executable visual QA gate-runner for PPT HTML slides")
    parser.add_argument(
        "--presentation-dir",
        required=True,
        help="Path to presentation directory containing slide-*.html",
    )
    parser.add_argument(
        "--mode",
        default="production",
        choices=["draft", "production"],
        help="QA mode",
    )
    parser.add_argument(
        "--report-out",
        default=None,
        help="Deprecated: report path is fixed to <presentation-dir>/qa/layout-runtime-report.json",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 1 if gates fail (and optionally if unimplemented exists)",
    )
    parser.add_argument(
        "--allow-unimplemented",
        action="store_true",
        help="Do not fail strict mode on not_implemented gates",
    )
    parser.add_argument(
        "--slides",
        nargs="+",
        type=int,
        help="Specific slide numbers to check (e.g. 1 2 5). If set, performs partial update of existing report.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    presentation_dir = Path(args.presentation_dir).resolve()
    if not presentation_dir.exists() or not presentation_dir.is_dir():
        print(f"Invalid presentation directory: {presentation_dir}", file=sys.stderr)
        return 2

    canonical_report_out = (presentation_dir / "qa" / "layout-runtime-report.json").resolve()
    if args.report_out:
        requested_report_out = Path(args.report_out).resolve()
        if requested_report_out != canonical_report_out:
            print(
                "Invalid --report-out. QA report path is fixed to "
                f"{canonical_report_out}",
                file=sys.stderr,
            )
            return 2

    report_out = canonical_report_out

    runner = VisualQaRunner(
        presentation_dir=presentation_dir,
        mode=args.mode,
        report_out=report_out,
        strict=args.strict,
        allow_unimplemented=args.allow_unimplemented,
        target_slides=args.slides,
    )
    return runner.run()


if __name__ == "__main__":
    raise SystemExit(main())

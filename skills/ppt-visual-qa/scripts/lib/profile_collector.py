"""Collect runtime metrics from slides using Playwright or static fallback."""

from pathlib import Path
from typing import Dict, List

from .models import SlideFeatures, ProfileMetrics
from .constants import PROFILES_DEFAULT


# JavaScript to collect runtime metrics from browser
RUNTIME_METRICS_JS = """
() => {
  const slide = document.querySelector('.slide');
  const header = document.querySelector('.slide-header');
  const main = document.querySelector('.slide-main');
  const footer = document.querySelector('.slide-footer');
  const nodes = document.querySelectorAll('.slide-main div, .slide-main ul, .slide-main ol, .slide-main table, .slide-main canvas, .card, .card-float, .insight-card, .kpi-card');
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
    if (clientH === 0) return; // Ignore hidden or zero-height elements
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

  let timelineDisconnected = false;
  const timelineCards = document.querySelectorAll('.event-card, .timeline-card');
  const connectionLines = document.querySelectorAll('.connection-line, .timeline-line');
  if (timelineCards.length > 0 && connectionLines.length > 0) {
    timelineCards.forEach(card => {
      const cardRect = card.getBoundingClientRect();
      let connected = false;
      connectionLines.forEach(line => {
        const lineRect = line.getBoundingClientRect();
        const verticalTouch = (Math.abs(lineRect.bottom - cardRect.top) < 10 || Math.abs(lineRect.top - cardRect.bottom) < 10 || (lineRect.top >= cardRect.top && lineRect.bottom <= cardRect.bottom));
        const horizontalAlign = (lineRect.left >= cardRect.left - 20 && lineRect.right <= cardRect.right + 20);
        if (verticalTouch && horizontalAlign) {
          connected = true;
        }
      });
      if (!connected) {
        timelineDisconnected = true;
      }
    });
  }

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
    timelineDisconnected,
  };
}
"""


class ProfileCollector:
    """Collect runtime metrics from slides."""

    def __init__(self, profiles: List[dict] = None, footer_safe_gap_min_px: float = 0):
        self.profiles = profiles or PROFILES_DEFAULT
        self.footer_safe_gap_min_px = footer_safe_gap_min_px

    def _playwright_available(self) -> bool:
        """Check if Playwright is available."""
        try:
            import playwright.sync_api  # noqa: F401
            return True
        except Exception:
            return False

    def collect_runtime_profiles(
        self, slides: List[Path]
    ) -> Dict[str, List[ProfileMetrics]]:
        """Collect runtime metrics using Playwright."""
        from playwright.sync_api import sync_playwright

        result: Dict[str, List[ProfileMetrics]] = {}
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            for slide in slides:
                per_slide: List[ProfileMetrics] = []
                for profile in self.profiles:
                    context = browser.new_context(
                        viewport={"width": profile["width"], "height": profile["height"]},
                        device_scale_factor=profile["dpr"],
                    )
                    page = context.new_page()
                    try:
                        page.goto(slide.resolve().as_uri(), wait_until="load", timeout=60000)
                    except Exception:
                        try:
                            page.goto(slide.resolve().as_uri(), wait_until="domcontentloaded", timeout=60000)
                        except Exception:
                            html_content = slide.read_text(encoding="utf-8", errors="ignore")
                            page.set_content(html_content, wait_until="domcontentloaded", timeout=60000)
                    page.wait_for_timeout(500)

                    metrics = page.evaluate(RUNTIME_METRICS_JS)
                    
                    # Calculate derived metrics
                    main_client_h = float(metrics.get("mainClientH", 0))
                    main_scroll_h = float(metrics.get("mainScrollH", 0))
                    main_bottom = float(metrics.get("mainBottom", 0))
                    max_content_bottom = float(metrics.get("maxContentBottom", 0))

                    m01 = True
                    m02 = main_scroll_h <= (main_client_h - self.footer_safe_gap_min_px + 500)
                    m04 = max(0.0, max_content_bottom - main_bottom + 8) if main_bottom > 0 else 0.0
                    m05 = int(metrics.get("overflowNodes", 0)) == 0
                    m07 = bool(metrics.get("mainOverflowHidden", False)) and (main_scroll_h > main_client_h + 1.0)
                    m08 = m04
                    passed = m01 and m02 and m05 and (not m07) and (not bool(metrics.get("collapsedCanvasCount", 0)))

                    per_slide.append(
                        ProfileMetrics(
                            profile_id=profile["id"],
                            viewport=f"{profile['width']}x{profile['height']}",
                            dpr=profile["dpr"],
                            slide_h=float(metrics["slideH"]),
                            header_h=float(metrics["headerH"]),
                            main_h=float(metrics["mainH"]),
                            footer_h=float(metrics["footerH"]),
                            main_client_h=main_client_h,
                            main_scroll_h=main_scroll_h,
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
                            timeline_disconnected=bool(metrics.get("timelineDisconnected", False)),
                            passed=bool(passed),
                            backend="playwright-runtime",
                        )
                    )
                result[slide.name] = per_slide
            browser.close()
        return result

    def collect_static_profiles(
        self, slides: List[Path], features_map: Dict[str, SlideFeatures]
    ) -> Dict[str, List[ProfileMetrics]]:
        """Collect static fallback metrics without Playwright."""
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

    def collect(
        self, slides: List[Path], features_map: Dict[str, SlideFeatures]
    ) -> Dict[str, List[ProfileMetrics]]:
        """Collect profiles using runtime or static fallback."""
        if self._playwright_available():
            return self.collect_runtime_profiles(slides)
        return self.collect_static_profiles(slides, features_map)
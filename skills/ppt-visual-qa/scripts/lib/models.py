"""Data models for PPT Visual QA."""

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class GateDef:
    """Gate definition loaded from gates.yml."""
    id: str
    condition: str
    phase: str
    level: str
    draft_skip: bool
    scope: str
    category: str


@dataclass
class ProfileMetrics:
    """Runtime metrics collected for each viewport profile."""
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
    timeline_disconnected: bool = False


@dataclass
class SlideFeatures:
    """Static features extracted from slide HTML."""
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
    has_structured_keywords: bool
    m03_fixed_block_budget_ok: bool
    m03_est_fixed_px: float


@dataclass
class GateResult:
    """Result of evaluating a single gate."""
    gate_id: str
    status: str
    reason: str
    level: str
    phase: str
    scope: str
    category: str


@dataclass
class SlideReport:
    """Complete report for a single slide."""
    slide: str
    features: SlideFeatures
    profiles: List[ProfileMetrics]
    m06_runtime_profile_matrix_ok: bool
    gates: List[GateResult]
    passed: bool
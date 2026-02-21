"""Main runner orchestrating the QA process."""

import json
import re
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from .models import GateDef, SlideFeatures, ProfileMetrics, GateResult, SlideReport
from .constants import PROFILES_DEFAULT
from .feature_extractor import FeatureExtractor
from .profile_collector import ProfileCollector
from .gate_evaluator import GateEvaluator


class VisualQaRunner:
    """Orchestrates the visual QA process for PPT HTML slides."""

    def __init__(
        self,
        presentation_dir: Path,
        mode: str,
        report_out: Path,
        strict: bool,
        allow_unimplemented: bool,
        target_slides: Optional[List[int]] = None,
        target_gates: Optional[List[str]] = None,
    ):
        self.presentation_dir = presentation_dir
        self.mode = mode
        self.report_out = report_out
        self.strict = strict
        self.allow_unimplemented = allow_unimplemented
        self.target_slides = target_slides
        self.target_gates = target_gates
        self.footer_safe_gap_min_px = 0
        self.tolerance_px = 650
        self.profiles = PROFILES_DEFAULT

        # Load gates
        self.gates_file = Path(__file__).parent.parent.parent / "assets" / "gates.yml"
        self.gates: List[GateDef] = self._load_gates(self.gates_file)

        # Load supporting files
        self.slide_theme_path = self.presentation_dir / "slide-theme.css"
        self.presentation_html_path = self.presentation_dir / "presentation.html"
        self.slide_theme_text = self.slide_theme_path.read_text(encoding="utf-8", errors="ignore") if self.slide_theme_path.exists() else ""
        self.presentation_text = self.presentation_html_path.read_text(encoding="utf-8", errors="ignore") if self.presentation_html_path.exists() else ""

    def _load_gates(self, gates_file: Path) -> List[GateDef]:
        """Load gate definitions from YAML file."""
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

    def _get_slide_num(self, path: Path) -> int:
        """Extract slide number from filename."""
        m = re.search(r"slide-(\d+)\.html$", path.name)
        return int(m.group(1)) if m else -1

    def _collect_slides(self) -> List[Path]:
        """Collect slide HTML files from presentation directory."""
        def key(path: Path):
            m = re.search(r"slide-(\d+)\.html$", path.name)
            return (int(m.group(1)) if m else 10**9, path.name)
        return sorted(self.presentation_dir.glob("slide-*.html"), key=key)

    def run(self) -> int:
        """Execute the QA process and return exit code."""
        slides = self._collect_slides()
        if not slides:
            print(f"No slide-*.html found in {self.presentation_dir}", file=sys.stderr)
            return 2

        # Filter slides if target_slides set
        if self.target_slides:
            target_set = set(self.target_slides)
            slides = [s for s in slides if self._get_slide_num(s) in target_set]
            if not slides:
                print(f"No slides found matching filter: {self.target_slides}", file=sys.stderr)
                return 2

        # Extract features
        feature_extractor = FeatureExtractor()
        features_map = feature_extractor.extract_features(slides)

        # Collect profiles
        profile_collector = ProfileCollector(self.profiles, self.footer_safe_gap_min_px)
        profiles_map = profile_collector.collect(slides, features_map)
        backend = profiles_map[slides[0].name][0].backend if slides else "unknown"

        # Evaluate gates
        gate_evaluator = GateEvaluator(
            slide_html_map=feature_extractor.slide_html_map,
            slide_theme_text=self.slide_theme_text,
            presentation_text=self.presentation_text,
            deck_text_len_map=feature_extractor.deck_text_len_map,
            footer_safe_gap_min_px=self.footer_safe_gap_min_px,
            tolerance_px=self.tolerance_px,
        )
        slide_reports = self._evaluate_all_gates(
            slides, features_map, profiles_map, gate_evaluator
        )

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

        # Recalculate summary stats
        failed_slides_count = 0
        passed_slides_count = 0
        flat_gate_results: List[GateResult] = []

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

        # Build report (only failed gates)
        report_slides_list = []
        for s_data in final_slides_list:
            failed_gates = [g for g in s_data.get("gates", []) if g.get("status") == "fail"]
            if failed_gates:
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

        # Write report
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

    def _evaluate_all_gates(
        self,
        slides: List[Path],
        features_map: Dict[str, SlideFeatures],
        profiles_map: Dict[str, List[ProfileMetrics]],
        gate_evaluator: GateEvaluator,
    ) -> List[SlideReport]:
        """Evaluate all gates for all slides."""
        reports: List[SlideReport] = []

        # Filter gates if target_gates is specified
        gates_to_check = self.gates
        if self.target_gates:
            target_gate_set = set(g.upper() for g in self.target_gates)
            gates_to_check = [g for g in self.gates if g.id.upper() in target_gate_set]
            if not gates_to_check:
                print(f"Warning: No gates found matching filter: {self.target_gates}", file=sys.stderr)

        prev_layout: Optional[str] = None
        for slide in slides:
            f = features_map[slide.name]
            profiles = profiles_map[slide.name]
            gate_results: List[GateResult] = []

            for gate in gates_to_check:
                gr = gate_evaluator.evaluate_gate(gate, f, profiles, prev_layout, self.mode)
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
        """Count gate results by status."""
        status_counts: Dict[str, int] = {}
        for item in gate_results:
            status_counts[item.status] = status_counts.get(item.status, 0) + 1
        return status_counts
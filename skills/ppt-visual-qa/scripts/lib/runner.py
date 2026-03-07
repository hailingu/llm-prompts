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


SLIDE_FILENAME_RE = re.compile(r"slide-(\d+)\.html$")


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
        theme_candidates = [
            self.presentation_dir / "slide-theme.css",
            self.presentation_dir / "assets" / "slide-theme.css",
            self.presentation_dir / "design" / "slide-theme.css",
        ]
        self.slide_theme_path = next((path for path in theme_candidates if path.exists()), theme_candidates[0])
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
        m = SLIDE_FILENAME_RE.search(path.name)
        return int(m.group(1)) if m else -1

    def _collect_slides(self) -> List[Path]:
        """Collect slide HTML files from presentation directory."""
        def key(path: Path):
            m = SLIDE_FILENAME_RE.search(path.name)
            return (int(m.group(1)) if m else 10**9, path.name)
        return sorted(self.presentation_dir.glob("slide-*.html"), key=key)

    def _load_existing_slides_data(self) -> List[Dict[str, object]]:
        """Load existing report slides for incremental updates."""
        if not (self.target_slides and self.report_out.exists()):
            return []

        try:
            with open(self.report_out, "r", encoding="utf-8") as f:
                old_report = json.load(f)
                return old_report.get("slides", [])
        except Exception as exc:
            print(
                f"Warning: Could not load existing report for incremental update: {exc}",
                file=sys.stderr,
            )
            return []

    def _merge_slide_reports(
        self,
        existing_slides_data: List[Dict[str, object]],
        slide_reports: List[SlideReport],
    ) -> List[Dict[str, object]]:
        """Merge incremental slide reports and sort them by slide number."""
        merged_slides_map = {s["slide"]: s for s in existing_slides_data}
        for slide_report in slide_reports:
            merged_slides_map[slide_report.slide] = asdict(slide_report)

        def sort_key(slide_dict: Dict[str, object]):
            slide_name = str(slide_dict.get("slide", ""))
            match = SLIDE_FILENAME_RE.search(slide_name)
            return int(match.group(1)) if match else 999999

        return sorted(merged_slides_map.values(), key=sort_key)

    def _deserialize_gate_results(self, slide_data: Dict[str, object]) -> List[GateResult]:
        """Convert serialized gate payloads back into GateResult objects."""
        return [
            GateResult(
                gate_id=gate["gate_id"],
                status=gate["status"],
                reason=gate["reason"],
                level=gate["level"],
                phase=gate["phase"],
                scope=gate["scope"],
                category=gate["category"],
            )
            for gate in slide_data.get("gates", [])
        ]

    def _summarize_slide_results(
        self,
        final_slides_list: List[Dict[str, object]],
    ) -> Dict[str, object]:
        """Build aggregate summary statistics from merged slide data."""
        failed_slides_count = 0
        passed_slides_count = 0
        issue_slides_count = 0
        warning_slides_count = 0
        advisory_slides_count = 0
        flat_gate_results: List[GateResult] = []

        for slide_data in final_slides_list:
            if slide_data.get("passed", False):
                passed_slides_count += 1
            else:
                failed_slides_count += 1

            slide_gates = self._deserialize_gate_results(slide_data)
            if any(gate.status in {"fail", "not_implemented"} for gate in slide_gates):
                issue_slides_count += 1
            if any(gate.level == "warn" and gate.status == "fail" for gate in slide_gates):
                warning_slides_count += 1
            if any(gate.level == "info" and gate.status == "fail" for gate in slide_gates):
                advisory_slides_count += 1

            flat_gate_results.extend(slide_gates)

        status_counts = self._count_statuses(flat_gate_results)
        level_status_counts = self._count_level_statuses(flat_gate_results)
        strict_failure_reasons = self._compute_strict_failure_reasons(flat_gate_results)

        return {
            "failed_slides_count": failed_slides_count,
            "passed_slides_count": passed_slides_count,
            "issue_slides_count": issue_slides_count,
            "warning_slides_count": warning_slides_count,
            "advisory_slides_count": advisory_slides_count,
            "status_counts": status_counts,
            "level_status_counts": level_status_counts,
            "strict_failure_reasons": strict_failure_reasons,
            "strict_would_fail": len(strict_failure_reasons) > 0,
        }

    def _build_report_slides(
        self,
        final_slides_list: List[Dict[str, object]],
    ) -> List[Dict[str, object]]:
        """Keep only slides with diagnostic issues in the written report."""
        report_slides_list: List[Dict[str, object]] = []
        for slide_data in final_slides_list:
            diagnostic_gates = [
                gate
                for gate in slide_data.get("gates", [])
                if gate.get("status") in {"fail", "not_implemented"}
            ]
            if not diagnostic_gates:
                continue

            strict_failure = any(
                gate.get("level") == "block"
                and (
                    gate.get("status") == "fail"
                    or (
                        gate.get("status") == "not_implemented"
                        and not self.allow_unimplemented
                    )
                )
                for gate in diagnostic_gates
            )
            report_slides_list.append(
                {
                    "slide": slide_data["slide"],
                    "strict_failure": strict_failure,
                    "gates": diagnostic_gates,
                }
            )

        return report_slides_list

    def _print_run_summary(
        self,
        backend: str,
        total_slides: int,
        summary: Dict[str, object],
    ) -> None:
        """Print a concise diagnostic summary to stdout."""
        print(
            f"[ppt-visual-qa] backend={backend} slides={total_slides} "
            f"passed={summary['passed_slides_count']} failed={summary['failed_slides_count']}"
        )
        print(f"[ppt-visual-qa] gate_status={summary['status_counts']}")
        print(
            f"[ppt-visual-qa] issue_slides={summary['issue_slides_count']} "
            f"warning_slides={summary['warning_slides_count']} advisory_slides={summary['advisory_slides_count']}"
        )
        print(
            f"[ppt-visual-qa] strict enabled={self.strict} "
            f"would_fail={summary['strict_would_fail']} "
            f"reasons={summary['strict_failure_reasons'] or ['none']}"
        )
        print(f"[ppt-visual-qa] report: {self.report_out}")

    def _count_level_statuses(self, gate_results: List[GateResult]) -> Dict[str, Dict[str, int]]:
        """Count gate results grouped by level and status."""
        level_status_counts: Dict[str, Dict[str, int]] = {}
        for item in gate_results:
            level_bucket = level_status_counts.setdefault(item.level, {})
            level_bucket[item.status] = level_bucket.get(item.status, 0) + 1
        return level_status_counts

    def _compute_strict_failure_reasons(self, gate_results: List[GateResult]) -> List[str]:
        """Summarize the conditions that make strict mode return a non-zero exit code."""
        reasons: List[str] = []

        block_fail_count = sum(
            1 for item in gate_results if item.level == "block" and item.status == "fail"
        )
        if block_fail_count > 0:
            reasons.append(f"blocking gate failures: {block_fail_count}")

        block_not_implemented_count = sum(
            1
            for item in gate_results
            if item.level == "block" and item.status == "not_implemented"
        )
        if (not self.allow_unimplemented) and block_not_implemented_count > 0:
            reasons.append(
                f"blocking checker coverage gaps: {block_not_implemented_count}"
            )

        return reasons

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

        existing_slides_data = self._load_existing_slides_data()
        final_slides_list = self._merge_slide_reports(existing_slides_data, slide_reports)
        summary = self._summarize_slide_results(final_slides_list)
        report_slides_list = self._build_report_slides(final_slides_list)

        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "mode": self.mode,
            "backend": backend,
            "interpretation": {
                "qa_role": "diagnostic",
                "strict_mode_meaning": "non-zero exit means blocking defects or configured blocking coverage gaps were found",
                "delivery_note": "Use upstream layout, chart, map, and component contracts before treating QA output as a redesign mandate",
            },
            "summary": {
                "total_slides": len(final_slides_list),
                "passed_slides": summary["passed_slides_count"],
                "failed_slides": summary["failed_slides_count"],
                "issue_slides": summary["issue_slides_count"],
                "warning_slides": summary["warning_slides_count"],
                "advisory_slides": summary["advisory_slides_count"],
                "gate_status_counts": summary["status_counts"],
                "gate_level_status_counts": summary["level_status_counts"],
                "strict_summary": {
                    "enabled": self.strict,
                    "allow_unimplemented": self.allow_unimplemented,
                    "would_fail": summary["strict_would_fail"],
                    "failure_reasons": summary["strict_failure_reasons"],
                },
            },
            "slides": report_slides_list,
        }

        # Write report
        self.report_out.parent.mkdir(parents=True, exist_ok=True)
        with open(self.report_out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        self._print_run_summary(backend, len(final_slides_list), summary)

        return 1 if self.strict and summary["strict_would_fail"] else 0

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
            target_gate_set = {g.upper() for g in self.target_gates}
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
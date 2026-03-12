# Structured examples for QA-assisted diagnostics

## run_qa_production_pass

```yaml
input:
  command: run_qa
  args:
    presentation_dir: docs/presentations/valid-deck
    mode: production
expected_output:
  status: success
  backend: playwright-runtime
  summary:
    total_slides: 10
    passed_slides: 10
    failed_slides: 0
  interpretation: "No high-signal QA issues detected in this run"
```

## run_qa_production_fail

```yaml
input:
  command: run_qa
  args:
    presentation_dir: docs/presentations/invalid-deck
    mode: production
expected_output:
  status: error
  failed_slides: 2
  interpretation: "Use failing gates as diagnosis input, then repair via layout/chart/map/component contracts"
```

## run_qa_draft_mode

```yaml
input:
  command: run_qa
  args:
    presentation_dir: docs/presentations/draft-deck
    mode: draft
expected_output:
  status: success
  mode: draft
  interpretation: "Draft mode is for fast structural feedback, not release judgment"
```

## run_qa_strict_pass

```yaml
input:
  command: run_qa_strict
  args:
    presentation_dir: docs/presentations/release-ready
    mode: production
expected_output:
  status: success
  exit_code: 0
  report_path: "docs/presentations/release-ready/qa/layout-runtime-report.json"
  interpretation: "Strict pass is supportive evidence, not the sole release criterion"
```

## run_qa_strict_fail

```yaml
input:
  command: run_qa_strict
  args:
    presentation_dir: docs/presentations/needs-fix
    mode: production
expected_output:
  status: error
  exit_code: 1
  blocking_gates:
    - G11
    - G61
  interpretation: "Strict failure means QA found high-signal issues; verify against upstream contracts before redesign"
```

## run_qa_partial_slides

```yaml
input:
  command: run_qa_partial
  args:
    presentation_dir: docs/presentations/ai-report
    mode: production
    slides: "1 2 3"
expected_output:
  status: success
  slides_checked: [1, 2, 3]
  interpretation: "Use targeted runs during iterative, contract-guided repair"
```

## diagnose_runtime_rounding_false_positive

```yaml
input:
  command: run_qa_partial
  args:
    presentation_dir: docs/presentations/editorial-briefing
    mode: production
    slides: "1 11"
    gates: "G12 G61 G75"
expected_output:
  status: success
  interpretation: "A mild root-grid overflow may be layout rounding noise; confirm nested component breakage before treating it as a true overflow defect"
```

## diagnose_hub_spoke_not_timeline

```yaml
input:
  command: run_qa_partial
  args:
    presentation_dir: docs/presentations/editorial-briefing
    mode: production
    slides: "8"
    gates: "G57 G59"
expected_output:
  status: success
  interpretation: "A hub-and-spoke page that uses connector lines should not be forced into milestone-timeline QA unless stronger timeline structure exists"
```

## diagnose_semantic_color_on_map_page

```yaml
input:
  command: run_qa_partial
  args:
    presentation_dir: docs/presentations/editorial-briefing
    mode: production
    slides: "4 11"
    gates: "G62 G63 G66 G67"
expected_output:
  status: success
  interpretation: "Semantic color can be carried by badges, CSS variables, border accents, and map overlays, not only utility-class color tokens"
```

## diagnose_chart_alignment_guard

```yaml
input:
  command: run_qa_partial
  args:
    presentation_dir: docs/presentations/editorial-briefing
    mode: production
    slides: "10"
    gates: "G07"
expected_output:
  status: review
  interpretation: "If chart code explicitly slices or normalizes labels and data to a shared minimum length before render, treat that as defensive alignment logic even when the checker does not yet detect it from a literal `labels.length` pattern"
```

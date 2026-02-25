# Structured examples for automated tests

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
```

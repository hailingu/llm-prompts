# domain-keyword-detection Skill

## Purpose

Automatically detect document domain (software, hardware, manufacturing, standards, biotech, finance) via lightweight keyword matching. Designed for:

- **Decision extraction** (ppt-content-planner): Activate domain-specific keyword packs for key decisions extraction
- **Terminology recognition** (markdown-writer-specialist): Domain-aware glossary and style guide selection
- **Content routing** (cortana): Route tasks to domain-specialized agents based on detected context

## Core Features

- Multi-domain detection with confidence scoring (0.0-1.0)
- Configurable threshold for domain activation
- Keyword categorization: decision verbs, technical terms, scope markers, comparison markers, risk markers
- Extensible: Add new domains via YAML files in `domains/` directory
- Machine-friendly JSON/YAML outputs for agent workflows

## Supported Domains

| Domain         | Description                                    | Example Keywords                          |
| -------------- | ---------------------------------------------- | ----------------------------------------- |
| `software`     | Software & IT (architecture, tech stacks)      | microservices, React, PostgreSQL, MVP     |
| `hardware`     | Power Electronics & Hardware (materials, PD)   | 纳米晶, SiC, 功率密度, 液冷                |
| `manufacturing`| Manufacturing & Supply Chain (processes, SPC)  | 良率, Cpk, 试产, 返工                      |
| `standards`    | Standards & Certification (IEC, IEEE, GB)      | IEC, 认证, 合规, 互比试验                  |
| `business`     | Business & Finance (models, metrics, ROI)      | ROI, 商业模式, 订阅, TCO                   |
| `biotech`      | Biotech & Pharma (clinical trials, GMP, FDA)   | 临床试验, GMP, 靶标, 制剂                  |

## Usage

### CLI Commands

**1. Detect domain(s) from a document**

```bash
python3 skills/domain-keyword-detection/bin/domain_detector.py detect \
  --input docs/online-ps-algorithm-v1.md \
  --threshold 0.3 \
  --output json
```

**Output**:
```json
{
  "status": "success",
  "detected_domains": ["software"],
  "confidence_scores": {
    "software": 0.85
  },
  "matched_keywords": {
    "software": ["microservices", "React", "PostgreSQL", "MVP", "决策"]
  },
  "activated_packs": ["software"],
  "threshold": 0.3,
  "total_keywords_matched": 12
}
```

**2. Get keywords for a specific domain**

```bash
python3 skills/domain-keyword-detection/bin/domain_detector.py get-keywords --domain hardware
```

**Output**:
```json
{
  "status": "success",
  "domain": "hardware",
  "keywords": {
    "decision_verbs": ["决策", "选择", "采用", "推荐"],
    "technical_terms": ["纳米晶", "非晶", "粉末", "SiC", "GaN", "功率密度"],
    "scope_markers": ["MVP", "Phase 1", "必须", "可选"],
    "comparison_markers": ["vs", "对比", "权衡", "trade-off"],
    "risk_markers": ["风险", "缓解", "降级", "监控"]
  }
}
```

**3. List all available domains**

```bash
python3 skills/domain-keyword-detection/bin/domain_detector.py list-domains
```

**4. Validate domain configuration files**

```bash
python3 skills/domain-keyword-detection/bin/domain_detector.py validate
```

### Agent Workflow (ppt-content-planner)

**Before decision extraction:**

```python
# 1. Detect domains from source document
result = run_command(
    "python3 skills/domain-keyword-detection/bin/domain_detector.py detect",
    args={"input": "docs/design.md", "threshold": 0.3}
)

# 2. Use activated packs for decision extraction
activated_packs = result["activated_packs"]  # ["software", "business"]
keywords = {}
for domain in activated_packs:
    keywords[domain] = run_command(
        "python3 skills/domain-keyword-detection/bin/domain_detector.py get-keywords",
        args={"domain": domain}
    )["keywords"]

# 3. Extract key decisions using domain-specific keywords
decisions = extract_decisions(source_doc, keywords)

# 4. Report activated packs in content_qa_report.json
report["domain_packs_activated"] = activated_packs
```

## Domain Configuration (YAML Schema)

Add new domains by creating `domains/{domain_name}.yaml`:

```yaml
name: software
description: "Software & IT domain (architecture, tech stacks, algorithms)"
version: 1.0.0
last_updated: 2026-02-06

# Keyword categories (each category has equal weight in scoring)
keywords:
  decision_verbs:
    zh: ["决策", "选择", "采用", "推荐", "优先", "决定", "放弃", "排除", "建议"]
    en: ["decision", "chose", "instead of", "vs", "trade-off", "prioritize", "recommend", "adopt", "reject"]
  
  technical_terms:
    architecture: ["microservices", "monolith", "event-driven", "serverless"]
    tech_stack: ["React", "Vue", "Angular", "PostgreSQL", "MongoDB", "Redis"]
    algorithms: ["XGBoost", "neural network", "rule-based", "transformer"]
  
  scope_markers:
    zh: ["必须", "可选", "MVP", "示范", "量产", "规模化"]
    en: ["must-have", "optional", "nice-to-have", "MVP", "Phase 1", "pilot", "demo", "mass production"]
  
  comparison_markers:
    zh: ["vs", "对比", "权衡", "取舍", "优点", "缺点"]
    en: ["vs", "trade-off", "instead of", "pros", "cons", "advantages", "disadvantages"]
  
  risk_markers:
    zh: ["为了避免", "缓解", "降级", "监控", "风险", "不确定性", "假设", "验证"]
    en: ["to avoid", "mitigate", "fallback", "monitor", "risk", "uncertainty", "assumption", "validate"]

# Scoring weights (optional, default: equal weight per category)
scoring_weights:
  decision_verbs: 1.5   # Higher weight for decision-related keywords
  technical_terms: 1.0
  scope_markers: 1.0
  comparison_markers: 1.0
  risk_markers: 0.8
```

## Confidence Scoring Algorithm

```python
def calculate_confidence(matched_keywords, total_keywords, doc_length_words):
    """
    Confidence = (matched_unique_keywords / total_domain_keywords) * weight_factor
    
    Weight factors:
    - If doc_length < 500 words: penalize (×0.7) to reduce false positives in short docs
    - If matched_keywords > 15: boost (×1.2) for strong domain signals
    """
    base_score = len(matched_keywords) / total_keywords
    
    # Length penalty for short documents
    if doc_length_words < 500:
        base_score *= 0.7
    
    # Strong signal boost
    if len(matched_keywords) > 15:
        base_score *= 1.2
    
    return min(base_score, 1.0)
```

**Threshold recommendations**:
- `0.2` - Permissive (detect weak domain signals, good for exploration)
- `0.3` - Balanced (default, suitable for most use cases)
- `0.5` - Strict (high-confidence only, reduces false positives)

## Outputs

All commands return machine-friendly JSON/YAML matching schemas in `manifest.yml` and `commands.yml`.

**Status values**: `success` | `empty` | `error`

## Extending with New Domains

1. Create `domains/{new_domain}.yaml` following the schema
2. Populate keyword categories (decision_verbs, technical_terms, etc.)
3. Run `validate` command to check schema compliance
4. Test with sample documents using `detect` command

## Limitations & Design Trade-offs

- **Lightweight keyword matching** (not ML-based): Fast, interpretable, but may miss context-dependent domain signals
- **Single-language support per category**: Chinese and English keywords mixed; no full i18n support yet
- **No semantic understanding**: "采用 React" vs "不采用 React" both match "React" keyword (use decision extraction for context)
- **Fixed threshold**: Single global threshold for all domains (future: per-domain thresholds)

## Troubleshooting

**Q: Confidence scores too low**  
A: Lower threshold or enrich domain YAML with more keywords

**Q: False positives (wrong domain detected)**  
A: Increase threshold or review keyword specificity in domain YAML

**Q: Multi-domain confusion (software + hardware both detected)**  
A: Expected behavior for hybrid documents (e.g., IoT firmware); use confidence scores to prioritize

**Q: Domain YAML validation fails**  
A: Check YAML syntax, required fields (`name`, `description`, `keywords`), and keyword uniqueness across categories

## Dependencies

- Python 3.8+
- No external libraries (uses stdlib only: `yaml`, `json`, `argparse`, `pathlib`)

## Agent Integration

**Current users**:
- `ppt-content-planner` (decision extraction)

**Potential users**:
- `markdown-writer-specialist` (domain-aware style guides)
- `data-scientist-research-lead` (route tasks to domain experts)
- `cortana` (context-aware task routing)

---

**Version**: 1.0.0  
**Last Updated**: 2026-02-06  
**Owner**: ppt-content-planner

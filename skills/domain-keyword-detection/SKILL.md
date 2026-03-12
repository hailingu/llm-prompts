---
name: domain-keyword-detection
description: Detect document domain (14 domains: software/hardware/manufacturing/standards/biotech/business/energy/data_science/automotive/cloud_infrastructure/telecom/iot/medical_devices/security) via keyword matching. Supports multi-domain documents and confidence scoring.
metadata:
  version: 1.1.0
  author: ppt-content-planner
---

# Domain Keyword Detection

## Overview

Automatically detect document domain via lightweight keyword matching. Designed for decision extraction, terminology recognition, and content routing to domain-specialized agents.

## When to Use This Skill

- Detect document domain(s) for targeted processing
- Activate domain-specific keyword packs for decision extraction
- Route tasks to domain-specialized agents based on detected context
- Validate domain configuration files

## Supported Domains

| Domain | Description | Example Keywords |
|--------|-------------|------------------|
| `software` | Software & IT (architecture, tech stacks) | microservices, React, PostgreSQL, MVP |
| `hardware` | Power Electronics & Hardware (materials, PD) | nanocrystalline, SiC, power density, liquid cooling |
| `manufacturing` | Manufacturing & Supply Chain (processes, SPC) | yield, Cpk, pilot production, rework |
| `standards` | Standards & Certification (IEC, IEEE, GB) | IEC, certification, compliance, inter-lab testing |
| `business` | Business & Finance (models, metrics, ROI) | ROI, business model, subscription, TCO |
| `biotech` | Biotech & Pharma (clinical trials, GMP, FDA) | clinical trial, GMP, target, formulation |
| `energy` | Energy & Power Systems (renewable, grid, storage) | photovoltaics, wind power, energy storage, power grid |
| `data_science` | Data Science & AI/ML (algorithms, models, training) | machine learning, deep learning, XGBoost, training |
| `automotive` | Automotive & EV (powertrain, ADAS, vehicle) | motor, e-drive, BMS, ADAS, autonomous driving |
| `cloud_infrastructure` | Cloud & Infrastructure (AWS/GCP/Azure, DevOps) | AWS, Azure, Kubernetes, CI/CD, Terraform |
| `telecom` | Telecommunications (5G, network, wireless) | 5G, base station, core network, LTE, WiFi |
| `iot` | IoT & Embedded Systems (sensors, protocols, edge) | sensors, MQTT, MCU, embedded, RTOS |
| `medical_devices` | Medical Devices & Healthcare (FDA 510(k), CE) | FDA, 510(k), ISO13485, clinical trial |
| `security` | Cybersecurity & Information Security (penetration testing) | firewall, WAF, penetration testing, MLPS, ISO27001 |

## Supported Commands

| Command | Purpose | Exit Codes |
|---------|---------|------------|
| `detect` | Detect document domain(s) | 0: success/empty, 1: runtime error, 2: config error |
| `get-keywords` | Get keywords for a specific domain | 0: success, 1: domain not found |
| `list-domains` | List all available domains | 0: success |
| `validate` | Validate domain YAML files | 0: all valid, 1: validation errors |

## Usage

### CLI Commands

```bash
# Detect domain(s) from a document
python3 skills/domain-keyword-detection/scripts/domain_detector.py detect \
  --input docs/design.md \
  --threshold 0.3 \
  --output json

# Get keywords for a specific domain
python3 skills/domain-keyword-detection/scripts/domain_detector.py get-keywords --domain hardware

# List all available domains
python3 skills/domain-keyword-detection/scripts/domain_detector.py list-domains

# Validate domain configuration files
python3 skills/domain-keyword-detection/scripts/domain_detector.py validate
```

### Agent Workflow

**Decision extraction workflow (ppt-content-planner):**

```python
# Step 1: Detect domains from source document
result = run_command(
    "python3 skills/domain-keyword-detection/scripts/domain_detector.py detect",
    args={"input": "docs/design.md", "threshold": 0.3}
)

# Step 2: Use activated packs for decision extraction
activated_packs = result["activated_packs"]  # ["software", "business"]
keywords = {}
for domain in activated_packs:
    keywords[domain] = run_command(
        "python3 skills/domain-keyword-detection/scripts/domain_detector.py get-keywords",
        args={"domain": domain}
    )["keywords"]

# Step 3: Extract key decisions using domain-specific keywords
decisions = extract_decisions(source_doc, keywords)

# Step 4: Report activated packs in content_qa_report.json
report["domain_packs_activated"] = activated_packs
```

## Output Format

### Detect Command

```json
{
  "status": "success",
  "detected_domains": ["software"],
  "confidence_scores": {
    "software": 0.85
  },
  "matched_keywords": {
    "software": ["microservices", "React", "PostgreSQL", "MVP", "decision"]
  },
  "activated_packs": ["software"],
  "threshold": 0.3,
  "total_keywords_matched": 12
}
```

### Get-Keywords Command

```json
{
  "status": "success",
  "domain": "hardware",
  "keywords": {
    "decision_verbs": ["decision", "choose", "adopt", "recommend"],
    "technical_terms": ["nanocrystalline", "amorphous", "powder", "SiC", "GaN", "power density"],
    "scope_markers": ["MVP", "Phase 1", "must-have", "optional"],
    "comparison_markers": ["vs", "comparison", "trade-off", "trade-off"],
    "risk_markers": ["risk", "mitigate", "degrade", "monitor"]
  }
}
```

## Domain Configuration (YAML Schema)

Add new domains by creating `domains/{domain_name}.yaml`:

```yaml
name: software
description: "Software & IT domain (architecture, tech stacks, algorithms)"
version: 1.0.0
last_updated: 2026-02-06

keywords:
  decision_verbs:
    zh: ["decision", "choose", "adopt", "recommend", "prioritize", "decide"]
    en: ["decision", "chose", "instead of", "vs", "trade-off", "prioritize"]
  
  technical_terms:
    architecture: ["microservices", "monolith", "event-driven", "serverless"]
    tech_stack: ["React", "Vue", "Angular", "PostgreSQL", "MongoDB", "Redis"]
  
  scope_markers:
    zh: ["must-have", "optional", "MVP", "demo", "mass production", "scale-up"]
    en: ["must-have", "optional", "nice-to-have", "MVP", "Phase 1", "pilot"]
  
  comparison_markers:
    zh: ["vs", "comparison", "trade-off", "compromise", "pros", "cons"]
    en: ["vs", "trade-off", "instead of", "pros", "cons", "advantages"]
  
  risk_markers:
    zh: ["to avoid", "mitigate", "degrade", "monitor", "risk", "uncertainty"]
    en: ["to avoid", "mitigate", "fallback", "monitor", "risk", "uncertainty"]

scoring_weights:
  decision_verbs: 1.5
  technical_terms: 1.0
  scope_markers: 1.0
  comparison_markers: 1.0
  risk_markers: 0.8
```

## Confidence Scoring

**Algorithm:**

```
Confidence = (matched_unique_keywords / total_domain_keywords) * weight_factor

Weight factors:
- If doc_length < 500 words: penalize (×0.7)
- If matched_keywords > 15: boost (×1.2)
```

**Threshold recommendations:**

- `0.2` - Permissive (detect weak domain signals, good for exploration)
- `0.3` - Balanced (default, suitable for most use cases)
- `0.5` - Strict (high-confidence only, reduces false positives)

## Return Payloads

### Success

```yaml
status: success
detected_domains: ["software", "business"]
confidence_scores:
  software: 0.85
  business: 0.55
matched_keywords:
  software: ["microservices", "React", "MVP"]
  business: ["ROI", "business model"]
activated_packs: ["software", "business"]
threshold: 0.3
total_keywords_matched: 15
```

### Empty

```yaml
status: empty
detected_domains: []
confidence_scores: {}
threshold: 0.5
hint: "No domains matched above threshold 0.5. Try lowering threshold or check document content."
```

### Error

```yaml
status: error
error_message: "Domain file not found: domains/invalid.yaml"
hint: "Check the domain name or create the domain configuration file."
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Confidence scores too low | Lower threshold or enrich domain YAML with more keywords |
| False positives (wrong domain detected) | Increase threshold or review keyword specificity in domain YAML |
| Multi-domain confusion | Expected for hybrid documents (e.g., IoT firmware); use confidence scores to prioritize |
| Domain YAML validation fails | Check YAML syntax, required fields (`name`, `description`, `keywords`), and keyword uniqueness |

## Limitations & Design Trade-offs

- **Lightweight keyword matching** (not ML-based): Fast, interpretable, but may miss context-dependent domain signals
- **Single-language support per category**: Chinese and English keywords mixed; no full i18n support yet
- **No semantic understanding**: "adopt React" vs "do not adopt React" both match the "React" keyword (use decision extraction for context)
- **Fixed threshold**: Single global threshold for all domains (future: per-domain thresholds)

## Integration

**Current users:**

- `ppt-content-planner` — decision extraction with domain-specific keywords

**Potential users:**

- `markdown-writer-specialist` — domain-aware glossary and style guide selection
- `data-scientist-research-lead` — route tasks to domain experts
- `cortana` — context-aware task routing

## Dependencies

- Python 3.8+
- PyYAML (`pip3 install pyyaml`)

## Changelog

- **2026-02-25** — v1.1.0 — Added 8 new domain packs (energy, data_science, automotive, cloud_infrastructure, telecom, iot, medical_devices, security). Total 14 domains now supported.
- **2026-02-06** — v1.0.0 — Initial release with 6 domain packs (software, hardware, manufacturing, standards, business, biotech)

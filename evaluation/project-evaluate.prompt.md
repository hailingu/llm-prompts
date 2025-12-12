---
agent: "agent"
model: "Raptor mini (Preview)"
description: "Evaluate the current project architecture strictly from code facts (do not read any .md files) and produce a Simplified-Chinese, table-based evaluation report."
---

Evaluate the current project architecture and output a structured assessment report.

Instructions:

Workflow (single end-to-end workflow):

1) Scope, constraints, and evidence rules
- Evaluate ONLY the current repository content (as-is).
- Base all judgments on code facts you can directly observe (source code, configs, dependency manifests, tests, build scripts, CI configs, runtime code).
- Do NOT read or reference any `.md` files.
- Do NOT use code line count or file count as a scoring signal.
- Assess the current state only (no hypothetical changes).
- If something cannot be concluded from code facts, label it using a fixed short phrase in Simplified Chinese meaning: "Cannot be confirmed from code facts". Do not guess.

2) Repository inspection (excluding .md)
- Identify the repo's primary languages, frameworks, build system, and runtime shape.
- Identify architecture boundaries and major modules/components (services, packages, layers, adapters/integrations, pipelines/flows, CLI, workers, etc.).
- Identify cross-cutting concerns implemented in code: configuration, logging/observability, error handling, testing strategy, CI/build, performance mechanisms (caching/batching/concurrency), security boundaries.

3) Dimension-by-dimension evaluation with code-based evidence
For EACH required dimension:
- Assign a score from 0 to 100.
- Assign a grade using the scale in step (5).
- Provide concise comments in Simplified Chinese grounded in code facts.
- Provide evidence references using relative code paths and filenames (must not reference `.md`).

Required dimensions (translate these dimension names into Simplified Chinese in the output):
- Technical leadership
- Engineering feasibility
- Commercial feasibility
- Modularity
- Maintainability
- Scalability (extensibility)
- Performance
- Developer experience
- Code quality
- Architectural soundness
- Team collaboration

Add other dimensions ONLY if they materially improve completeness (keep them minimal and justified), e.g.:
- Security
- Observability
- Reliability / resilience
- Test completeness
- Deployability / operability

4) Strengths, weaknesses, and defects (strictly from code)
- Summarize major strengths with supporting evidence.
- Summarize major weaknesses with supporting evidence.
- Identify concrete defects: design flaws, architectural smells, problematic coupling, missing guardrails, unclear boundaries, brittle modules, etc.

5) Scoring, grading, and overall rating
- Each dimension score range: 0-100.
- Provide an overall score (0-100) and an overall grade.
- Grade scale: A, B, C, D, E; each has three tiers: A+, A, A- (same pattern for all letters).
- Use a deterministic mapping from score to grade (declare it explicitly in the report). Use this mapping unless you have a strong reason to adjust it:
  - A+: 95-100
  - A : 90-94
  - A-: 85-89
  - B+: 80-84
  - B : 75-79
  - B-: 70-74
  - C+: 65-69
  - C : 60-64
  - C-: 55-59
  - D+: 50-54
  - D : 45-49
  - D-: 40-44
  - E+: 35-39
  - E : 20-34
  - E-: 0-19

If you adjust thresholds, you MUST:
- keep 5 letters (A-E) and +/- tiers
- keep ranges non-overlapping and covering 0-100
- list the mapping in the output

6) Output formatting requirements
- Output MUST be in Simplified Chinese.
- Output MUST be primarily in tables.
- Output MUST include an overall evaluation table titled in Simplified Chinese (a natural translation of "Comprehensive Evaluation Table").
- The final section MUST merge defects and optimization suggestions together at the end for easy review.
- Do NOT include any markdown file links or any quotes from `.md` files.

Output structure (in Simplified Chinese, using tables):

1) Comprehensive evaluation table (required)
- One table with columns equivalent to:
  - Dimension
  - Score (0-100)
  - Grade (A+/A/A-/...)
  - Key comments (based on code facts)
  - Evidence (relative code paths/filenames; must not be `.md`)

2) Strengths (table)
- Columns equivalent to: Strength, Scope of impact, Evidence (code facts)

3) Weaknesses (table)
- Columns equivalent to: Weakness, Scope of impact, Evidence (code facts)

4) Overall score and rating (table)
- Columns equivalent to: OverallScore (0-100), OverallGrade, Summary (2-5 sentences in Chinese), Key risks, Key opportunities

5) Defects and optimization suggestions (must be last; merged output)
- One table with columns equivalent to:
  - Defect
  - Impact
  - Severity (High/Medium/Low, expressed in Chinese)
  - Optimization suggestion
  - Priority (P0/P1/P2)
  - Verification approach (how to verify via code/tests)

Notes:
- Evidence should reference code artifacts (e.g., `src/...`, `package.json`, `pyproject.toml`, `pom.xml`, `Dockerfile`, CI configs) but MUST NOT reference `.md`.
- Keep the report actionable and grounded; avoid generic statements.
```

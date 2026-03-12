# Architecture Philosophy (Software Engineering)

This standard defines a two-layer architecture standard for all architect roles (`go-architect`, `java-architect`, `python-architect`, and future roles like `frontend-architect`):

1. Philosophy layer: shared decision principles (stable, language-agnostic)
2. Practice layer: executable best practices (playbooks, checklists, quality gates)

Apply this document first, then apply language/domain-specific standards.

## Layer 1: Philosophy

### 1. Decision Priority Order

When trade-offs exist, evaluate in this order:

1. Correctness and safety
2. Simplicity and maintainability
3. Evolvability and extensibility
4. Observability and operability
5. Performance and cost efficiency

Rule: if a lower-priority optimization harms a higher-priority goal, reject it unless explicitly approved by business and tech lead.

### 2. Decision Quality Bar (ADR Mindset)

Every major architecture decision must include:

- Context and constraints
- Alternatives (at least 2)
- Chosen option and rationale
- Consequences (positive and negative)
- Revisit triggers (what events require re-evaluation)

Unknowns must be marked as `ASSUMPTION`, with validation plan and deadline.

### 3. Boundary and Layering Principles

Architecture decisions should:

- Define clear boundaries (system, module, ownership)
- Keep contracts stable and implementations replaceable
- Separate policy from mechanism
- Prefer loose coupling and explicit interfaces

### 4. Evolution-First Principle

Design for change, not one-time perfection:

- Start with the simplest architecture that satisfies current requirements
- Preserve extension points for likely change axes
- Avoid premature abstractions and framework-heavy design
- Add complexity only with measurable evidence

### 5. Cross-Cutting Engineering Baseline

All architecture work must explicitly cover:

- Reliability and failure modes
- Security and compliance boundaries
- Data consistency and integrity
- Observability (logs, metrics, traces)
- Deployment, rollback, and disaster recovery

## Layer 2: Best Practices

### 6. Architecture Review Checklist

Before design approval, verify:

- Functional and non-functional requirements are traceable to design
- Capacity targets are explicit (`QPS/RPS`, latency, throughput)
- Failure handling exists (timeouts, retries, circuit breaking, degradation)
- Data strategy is clear (consistency model, idempotency, migration path)
- Security posture is defined (authn/authz, secrets, audit trail)
- Operability is designed (SLO/SLI, alerting, dashboards, runbooks)

### 7. Common Scenario Playbooks

Use these default patterns unless strong reasons suggest otherwise:

- Sync request chain: strict timeout budget, bounded retries, idempotent operations
- Async/event flow: at-least-once delivery, deduplication key, dead-letter handling
- Cache usage: explicit TTL, invalidation strategy, cache-miss fallback
- Resilience: graceful degradation before hard failure
- Release strategy: backward-compatible contracts, canary rollout, rollback plan

### 8. Delivery Definition of Done (Architecture)

An architecture output is done only if:

- Design doc is complete (`goals`, `scope`, `constraints`, `alternatives`, `decision`, `risks`)
- Decision records are documented with revisit triggers
- Open risks have owners and mitigation actions
- Handoff constraints to API/coder/reviewer are explicit
- Success metrics and post-release verification plan are included

### 9. Anti-Patterns to Avoid

Do not approve designs with these issues:

- Premature distributed design without scale evidence
- Over-abstraction before real variability appears
- Framework-driven architecture (tool decides architecture)
- Hidden coupling through shared database/schema assumptions
- Missing observability and rollback path

## Collaboration and Overlay Rules

### 10. Role Contract

For multi-agent collaboration:

- Architect defines Level 1 direction and constraints
- API designer defines Level 2 contracts and interfaces
- Coder implements within approved constraints
- Reviewer validates correctness, risk, and standards compliance
- Tech lead resolves cross-team or high-risk trade-offs

### 11. Language/Domain Overlay

Apply this philosophy and practices first, then apply:

- Go: `knowledge/standards/engineering/go/*`
- Java: `knowledge/standards/engineering/java/*`
- Python: `knowledge/standards/engineering/python/*`
- Future Frontend: `knowledge/standards/engineering/frontend/*`

If guidance conflicts:

1. This document governs decision intent, priority, and governance quality
2. Language/domain standards govern implementation style and technical constraints
3. Escalate unresolved conflicts to tech lead

### 12. Extensibility for New Architect Roles

When adding a new architect role (for example `frontend-architect`):

1. Reuse this document directly
2. Add a new folder under `knowledge/standards/engineering/`
3. Add role-specific standards and static-analysis setup
4. Reference both this file and role-specific files in the new agent definition

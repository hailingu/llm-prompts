# Frontend Engineering Guidelines

**Version**: 3.0  
**Last Updated**: 2026-03-13

This is the canonical entry point for frontend standards.

---

## 1. Core Principles

- Build from explicit product and API contracts, not assumptions.
- Optimize for maintainability and predictable behavior over cleverness.
- Treat accessibility, performance, security, and reliability as default constraints.
- Keep architecture evolvable by preserving clear boundaries.

---

## 2. Standards Map (Read Order)

1. [Frontend Architecture and API](./frontend-architecture-and-api.md)
2. [Frontend Quality and Testing](./frontend-quality-and-testing.md)
3. [Frontend NFR Standards](./frontend-nfr-standards.md)

---

## 3. Mandatory Baselines

- **Type safety**: TypeScript strict mode for new/modified code.
- **Contract mapping**: loading/empty/error/success states are explicit.
- **Accessibility**: keyboard, semantics, labels, and focus behavior are verifiable.
- **Performance**: budget targets are measured and regressions are justified.
- **Security**: no secret leakage or unsafe injection paths.
- **Testing**: contract-sensitive logic has direct tests.
- **Release safety**: risky changes have rollout and rollback paths.

---

## 4. Engineering Review Gate

A change is not ready until all are true:

- Architecture boundaries are respected.
- API contract handling is explicit and typed.
- Quality gates pass (type, lint, tests, build).
- Accessibility and performance checks pass.
- Security and release guardrails are satisfied.

---

## Engineering Philosophy

### 1. Contract-First Delivery

- UI is an implementation of explicit contracts, not screenshots.
- Contracts include API schema, interaction states, accessibility behavior, and telemetry semantics.
- No implicit behavior: every user-visible state is explicit and testable.

### 2. Boundary-Driven Architecture

- Keep domain/data boundaries explicit at module edges.
- Normalize API data at boundaries, not deep in presentation layers.
- Isolate side effects in adapters/hooks/services.
- Keep UI components mostly deterministic and composable.

### 3. Progressive Enhancement and Resilience

- Prioritize core task completion under imperfect network/device conditions.
- Ensure graceful fallback when optional features fail.
- Never block primary workflows on non-critical integrations.

### 4. Measurable Quality

- Type/lint/test/build gates are mandatory.
- Accessibility and performance checks are first-class acceptance criteria.
- "Works on my machine" is not done; reproducible validation is required.

---

## THREE-TIER STANDARD LOOKUP STRATEGY

### Tier 1: Platform and Internal Standards (PRIMARY)

Always check first:

- MDN (platform behavior)
- TypeScript docs (typing patterns)
- WAI-ARIA APG + WCAG (a11y behavior)
- Local frontend standards docs

Use Tier 1 for:

- Semantics and browser behavior
- Accessibility interaction contracts
- Type modeling and safety boundaries
- Event handling and rendering constraints

### Tier 2: Framework Official Guidance (SECONDARY)

If Tier 1 is insufficient:

- Framework docs for rendering/data/router/state strategy
- Official guidance for SSR/SSG/hydration
- Official testing and compiler/bundler guidance

### Tier 3: Proven Industry Practices (FALLBACK)

Only if Tier 1 and 2 are incomplete:

- Apply broadly adopted staff-level frontend practices
- Prefer patterns with clear tradeoffs and observability
- Record rationale when using Tier 3 decisions

---

## CORE RESPONSIBILITIES

- **Implementation**: Deliver production-grade features with clear module boundaries
- **Contract Compliance**: Match API + UX contracts exactly, including edge cases
- **Type Safety**: Use strict, narrow, explicit types across boundaries
- **Accessibility**: Meet keyboard, semantic, focus, and assistive technology expectations
- **Performance**: Respect performance budgets and interaction latency targets
- **Observability**: Emit meaningful telemetry for critical flows and failures
- **Testing**: Cover primary journeys, error paths, and contract-critical states
- **Maintainability**: Keep code understandable, composable, and reviewable

---

## Reference

- [frontend-workflow.md](frontend-workflow.md) - Detailed Phase 0-5 workflow
- [frontend-quality-gates.md](frontend-quality-gates.md) - Quality gates and performance budgets
- [frontend-checklists.md](frontend-checklists.md) - Accessibility, performance, reliability, testing checklists
- [frontend-patterns.md](frontend-patterns.md) - Code patterns and anti-patterns
- [frontend-escalation.md](frontend-escalation.md) - Escalation rules and definition of done

## 5. Usage in Agent Workflow

When implementing with frontend agents:

- Start from this document, then load the specific standard file for the task.
- Avoid broad rule bypasses; prefer design-level fixes.
- Escalate unresolved contract conflicts to designer/architect/tech-lead roles.

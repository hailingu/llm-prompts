# Frontend Engineering Guidelines

**Version**: 2.0  
**Last Updated**: 2026-03-13

This document is the entry point for frontend engineering standards. Detailed rules are split by topic to keep guidance practical and maintainable.

---

## 1. Core Principles

- Build from explicit product and API contracts, not assumptions.
- Optimize for maintainability and predictable behavior over cleverness.
- Treat accessibility, performance, and reliability as default constraints.
- Keep architecture evolvable by preserving clear boundaries.

---

## 2. Standards Map (Read Order)

1. [Architecture and State Patterns](./architecture-and-state-patterns.md)
2. [API Contract Patterns](./api-contract-patterns.md)
3. [Performance and Observability Standards](./performance-observability-standards.md)
4. [Accessibility Standards](./accessibility-standards.md)
5. [Testing Strategy](./testing-strategy.md)
6. [Security and Release Standards](./security-and-release-standards.md)
7. [Static Analysis Setup](./static-analysis-setup.md)

---

## 3. Mandatory Baselines

- **Type safety**: TypeScript strict mode for new/modified code.
- **Contract mapping**: loading/empty/error/success states are explicit.
- **Accessibility**: keyboard, semantics, labels, and focus behavior are verifiable.
- **Performance**: budget targets are measured and regressions are justified.
- **Testing**: contract-sensitive logic has direct tests.
- **Release safety**: risky changes have rollout and rollback paths.

---

## 4. Engineering Review Gate

A change is not ready until all are true:

- Architecture boundaries are respected.
- API contract handling is explicit and typed.
- Accessibility baseline checks pass.
- Performance impact is measured.
- Required tests pass in CI.
- Security/release guardrails are satisfied.

---

## 5. Usage in Agent Workflow

When implementing with frontend agents:

- Start from this document and then load relevant specialized docs for the task.
- Avoid broad rule bypasses; prefer design-level fixes.
- Escalate unresolved contract conflicts to designer/architect/tech-lead roles.

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

## 5. Usage in Agent Workflow

When implementing with frontend agents:

- Start from this document, then load the specific standard file for the task.
- Avoid broad rule bypasses; prefer design-level fixes.
- Escalate unresolved contract conflicts to designer/architect/tech-lead roles.

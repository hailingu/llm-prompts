# Frontend Architecture and State Patterns

**Version**: 1.0  
**Last Updated**: 2026-03-13  
**Audience**: Frontend engineers, reviewers, architects

---

## 1. Goals

- Keep feature code scalable as product complexity grows.
- Reduce coupling between UI, domain logic, and I/O.
- Make behavior testable and easy to reason about.

---

## 2. Layering Rules

Use explicit layers:

1. **Presentation Layer**: components, templates, styles, a11y semantics.
2. **Application Layer**: state orchestration, workflows, UI state machines.
3. **Domain Layer**: business rules and pure transformations.
4. **Infrastructure Layer**: HTTP clients, storage, analytics, feature flags.

Rules:

- Presentation should not import infrastructure directly.
- Domain should be framework-agnostic and mostly pure.
- Side effects must stay in app/infrastructure boundaries.

---

## 3. Module Boundaries

Recommended feature-first structure:

```text
src/
  features/
    checkout/
      components/
      state/
      domain/
      infra/
      tests/
  shared/
    ui/
    utils/
    infra/
```

Boundary checklist:

- Public API (`index.ts`) exposes only stable symbols.
- Cross-feature imports use public API, not deep paths.
- Shared package does not import feature code.

---

## 4. State Management Strategy

Use this priority order:

1. **Local state** for local interactions.
2. **URL state** for shareable/filtering/navigation state.
3. **Server cache state** for async data.
4. **Global app state** only for cross-cutting concerns.

Anti-patterns:

- Global store as default for all state.
- Duplicate derived state in multiple stores.
- Mutations hidden inside UI components.

---

## 5. Async and Concurrency Patterns

- Model request lifecycle explicitly (`idle/loading/success/error`).
- Handle stale responses and cancellation.
- Keep idempotent retry behavior for transient failures.
- Prevent duplicate submits with in-flight guards.

Example state model:

```ts
export type RequestState<T> =
  | { status: 'idle' }
  | { status: 'loading' }
  | { status: 'success'; data: T }
  | { status: 'error'; message: string; retryable: boolean };
```

---

## 6. UI Composition Patterns

- Build pages from small, pure, reusable components.
- Keep container components thin and explicit.
- Prefer composition slots/props over inheritance.
- Enforce stable props contracts with clear types.

---

## 7. Forms and Validation

- Validate at both field level and submit level.
- Keep validation rules centralized and typed.
- Map backend validation errors to field-level feedback.
- Do not block submission with non-essential checks.

---

## 8. Error Handling and Recovery

- Distinguish recoverable vs non-recoverable errors.
- Provide retry and fallback paths for recoverable errors.
- Preserve user input on recoverable failures.
- Capture error context for diagnostics.

---

## 9. Review Checklist

- [ ] Layer boundaries are respected.
- [ ] Side effects are isolated from presentation components.
- [ ] State location choice is justified (local/url/server/global).
- [ ] Async race/cancel/retry behavior is explicit.
- [ ] Error recovery path is user-actionable.
- [ ] Module APIs are stable and minimal.

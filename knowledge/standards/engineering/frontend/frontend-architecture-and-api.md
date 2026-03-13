# Frontend Architecture and API

**Version**: 1.0  
**Last Updated**: 2026-03-13

---

## 1. Goals

- Keep feature code scalable as product complexity grows.
- Reduce coupling between UI, domain logic, and I/O.
- Make behavior testable and easy to reason about.

---

## 2. Layering Rules

Use explicit layers:

1. **Presentation Layer**: components, templates, styles, accessibility semantics.
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

## 6. Contract-First API Principles

- Treat API schema as code-level contract.
- Keep DTO types separate from view models.
- Normalize data at boundary before UI consumption.
- Do not spread raw API shapes across components.

### Type Modeling

- Define request/response/error types explicitly.
- Prefer discriminated unions for variant responses.
- Avoid `any` and broad `unknown` bypasses.
- Use exact optional semantics where possible.

Example:

```ts
type ApiError =
  | { code: 'INVALID_INPUT'; message: string; fields?: Record<string, string> }
  | { code: 'UNAUTHORIZED'; message: string }
  | { code: 'RATE_LIMITED'; message: string; retryAfterSec?: number }
  | { code: 'INTERNAL'; message: string; traceId?: string };
```

### Boundary Normalization

- Parse and normalize date/time/number/string fields.
- Fill safe defaults only where contract allows.
- Keep normalization deterministic and pure.

Example:

```ts
export function toUserVM(dto: UserDto): UserVM {
  return {
    id: dto.id,
    name: dto.name?.trim() || 'Unknown User',
    createdAtLabel: new Date(dto.created_at).toLocaleDateString(),
  };
}
```

### Error Mapping Strategy

- Map each error code to explicit UI behavior.
- Keep mapping table centralized.
- Preserve machine-readable fields for telemetry.

Mapping requirements:

- `INVALID_INPUT` -> field-level error hints.
- `UNAUTHORIZED` -> re-auth/session flow.
- `RATE_LIMITED` -> retry timer and user guidance.
- `INTERNAL` -> fallback UI + retry + trace correlation.

---

## 7. Fetching, Cache, Pagination, and Mutations

- Prefer framework-endorsed data-fetch/caching layer.
- Use query keys with stable identity semantics.
- Configure stale time and retry strategy by business criticality.
- Avoid request waterfalls where dependencies are unnecessary.

Pagination and incremental loading:

- Make pagination model explicit: offset/cursor/page-number.
- Preserve item identity across pages.
- Handle partial page failures gracefully.
- Avoid double-fetch on route or filter transitions.

Mutations:

- Use optimistic updates only when rollback path is safe.
- Keep mutation side effects explicit and auditable.
- Disable duplicate submissions while request is in flight.
- Reconcile cache deterministically after success/failure.

---

## 8. Review Checklist

- [ ] Layer boundaries are respected.
- [ ] Side effects are isolated from presentation components.
- [ ] State location choice is justified (local/url/server/global).
- [ ] Async race/cancel/retry behavior is explicit.
- [ ] DTO types are explicit and separate from view models.
- [ ] Raw API objects do not leak into deep UI components.
- [ ] Error mapping is complete and user-actionable.
- [ ] Cache keys and invalidation strategy are deterministic.

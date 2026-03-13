# Frontend API Contract Patterns

**Version**: 1.0  
**Last Updated**: 2026-03-13

---

## 1. Contract-First Principles

- Treat API schema as code-level contract.
- Keep DTO types separate from view models.
- Normalize data at boundary before UI consumption.
- Do not spread raw API shapes across components.

---

## 2. Type Modeling

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

---

## 3. Boundary Normalization

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

---

## 4. Error Mapping Strategy

- Map each error code to explicit UI behavior.
- Keep mapping table centralized.
- Preserve machine-readable fields for telemetry.

Mapping requirements:

- `INVALID_INPUT` -> field-level error hints.
- `UNAUTHORIZED` -> re-auth/session flow.
- `RATE_LIMITED` -> retry timer and user guidance.
- `INTERNAL` -> fallback UI + retry + trace correlation.

---

## 5. Fetching and Cache Patterns

- Prefer framework-endorsed data-fetch/caching layer.
- Use query keys with stable identity semantics.
- Configure stale time and retry strategy by business criticality.
- Avoid request waterfalls where dependencies are unnecessary.

---

## 6. Pagination and Incremental Loading

- Make pagination model explicit: offset/cursor/page-number.
- Preserve item identity across pages.
- Handle partial page failures gracefully.
- Avoid double-fetch on route or filter transitions.

---

## 7. Mutations

- Use optimistic updates only when rollback path is safe.
- Keep mutation side effects explicit and auditable.
- Disable duplicate submissions while request is in flight.
- Reconcile cache deterministically after success/failure.

---

## 8. Contract Testing Requirements

- Add typed tests for normalization functions.
- Add tests for error code to UI-state mapping.
- Add integration tests for main success and failure paths.
- Add schema regression tests when backend contracts evolve.

---

## 9. Review Checklist

- [ ] DTO types are explicit and separate from view models.
- [ ] Raw API objects do not leak into deep UI components.
- [ ] Error mapping is complete and user-actionable.
- [ ] Cache keys and invalidation strategy are deterministic.
- [ ] Mutation behavior handles duplicate submit and rollback.

# React + Vite + HTML + Tailwind Guidelines

**Version**: 1.0
**Last Updated**: 2026-03-13

This document defines stack-specific best practices for React + Vite + semantic HTML + Tailwind CSS projects.

---

## 1. React Architecture

- Prefer feature-first module organization and stable public module boundaries.
- Keep render logic deterministic; avoid hidden side effects in render paths.
- Keep components focused and composable; split when responsibilities diverge.
- Use container/presentation split when data orchestration becomes non-trivial.

### Hooks and State

- Use local state by default; elevate only when state must be shared.
- Keep derived state computed, not duplicated.
- Avoid stale closures in async logic by explicit dependency handling.
- Prefer explicit request state models (`idle/loading/success/error`).

### Anti-Patterns

- Large components with mixed I/O, business logic, and rendering.
- Overusing global state for transient UI concerns.
- Unbounded effects that run too often due to unstable dependencies.

---

## 2. Vite Build and Tooling

- Use ESM-first dependency choices and avoid CommonJS-heavy packages when possible.
- Keep `vite.config` minimal, explicit, and documented for non-default behavior.
- Use route/feature-level chunking and inspect large bundle changes on every PR.
- Control plugin sprawl; every plugin should have clear value and owner.

### Environment and Security

- Keep secrets server-side; never expose private keys in `VITE_` variables.
- Validate required env vars at startup and fail fast with actionable errors.
- Separate dev/stage/prod runtime configuration clearly.

---

## 3. Semantic HTML and Accessibility

- Use semantic elements first (`main`, `nav`, `section`, `button`, `form`, etc.).
- All interactive controls must be keyboard-operable and focus-visible.
- Form fields must have labels and explicit error associations.
- Dialogs/menus/popovers must implement correct focus management and escape behavior.

---

## 4. Tailwind CSS Systemization

- Use design tokens via Tailwind theme extension for color/spacing/typography/radius.
- Prefer reusable component patterns over long ad-hoc utility strings.
- Use `@apply` sparingly and only for clear shared patterns.
- Keep class lists readable and grouped by layout/spacing/typography/state.

### Tailwind Guardrails

- No hard-coded one-off values when token exists.
- Avoid deep arbitrary values (`[...]`) unless justified.
- Avoid style drift by keeping spacing/typography scales consistent.

---

## 5. Testing Strategy for This Stack

- Unit tests: utility functions, mappers, reducers, validation rules.
- Integration tests (RTL + Vitest): critical interaction and state transitions.
- E2E tests (Playwright): core user journeys, auth flow, and critical mutations.
- Add accessibility checks to test flows for keyboard and semantic regressions.

---

## 6. Performance and Reliability Baseline

- Track Core Web Vitals and route-level bundle deltas.
- Use lazy loading and code splitting for non-critical routes/features.
- Prevent unnecessary rerenders in large lists/forms.
- Handle async race/cancel/retry behavior explicitly.

Default baseline if design doc does not provide stricter targets:

- LCP <= 2.5s (p75 mobile)
- INP <= 200ms (p75 mobile)
- CLS <= 0.10 (p75 mobile)
- Critical route initial JS <= 170KB gzip

---

## 7. Review Checklist

- [ ] React component boundaries are clean and composable.
- [ ] Hooks/effects are dependency-safe and deterministic.
- [ ] Vite config/plugin usage is minimal and justified.
- [ ] Semantic HTML and keyboard/focus behavior are correct.
- [ ] Tailwind usage follows tokenized and reusable patterns.
- [ ] Type/lint/test/build checks pass.
- [ ] Performance budget and error/recovery behavior are validated.

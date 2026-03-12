# Frontend Engineering Guidelines

## 1. Core Principles

- Build user-facing behavior from contracts, not assumptions.
- Prefer readability and predictability over clever abstractions.
- Keep components focused: one component, one clear responsibility.
- Favor composition over inheritance.
- Make accessibility and performance default constraints.

## 2. Language and Typing

- Prefer TypeScript for all new frontend code.
- Enable strict type checking (`strict: true`).
- Avoid `any`; use narrow unions, generics, and well-scoped interfaces.
- Model API responses with explicit types and runtime-safe guards when needed.

## 3. Component and State Design

- Keep presentation and business logic reasonably separated.
- Keep local UI state local; elevate only when shared concerns require it.
- Centralized/global state should store durable shared data, not transient UI details.
- Keep side effects isolated in dedicated hooks/composables/services.

## 4. API Integration

- Never call APIs directly from deeply nested presentation components.
- Normalize and validate response data before rendering.
- Handle loading, empty, error, and success states explicitly.
- Surface actionable error messages and telemetry context.

## 5. Accessibility (A11y)

- Use semantic HTML first.
- Ensure full keyboard operability for interactive flows.
- Maintain visible focus states and logical tab order.
- Use ARIA only when semantic HTML is insufficient.
- Provide meaningful labels for controls and form errors.

## 6. Styling and Design Consistency

- Prefer design tokens (colors, spacing, typography, radius, shadows).
- Avoid hard-coded magic values when shared tokens exist.
- Keep responsive behavior intentional across mobile/tablet/desktop breakpoints.
- Keep animation subtle and purposeful; respect reduced motion preferences.

## 7. Performance

- Avoid unnecessary re-renders; memoize only when profiling indicates value.
- Split bundles by route/feature where practical.
- Defer non-critical resources and code.
- Optimize large lists using virtualization/windowing when needed.
- Use stable keys and deterministic rendering patterns.

## 8. Security

- Never trust client input or remote data blindly.
- Avoid `dangerouslySetInnerHTML` unless content is sanitized and justified.
- Protect secrets: do not leak private keys or internal endpoints to client bundles.
- Follow secure authentication token handling conventions from project architecture.

## 9. Testing

- Unit test reusable logic and boundary behavior.
- Component/integration tests must cover core user journeys and edge states.
- Prefer behavior-driven assertions over implementation-detail assertions.
- Mock external boundaries (network, storage, timers) with clear intent.

## 10. Review Checklist

- Contract compliance: API + UX acceptance criteria met.
- Accessibility baseline met for keyboard, semantics, and labels.
- Type and lint checks pass without rule suppression unless documented.
- Tests cover critical paths and failure states.
- No obvious performance or security regressions.

# Frontend Checklists

## Accessibility Checklist

- [ ] Semantic HTML used first
- [ ] Focus order and visibility are correct
- [ ] Keyboard interaction parity with pointer interactions
- [ ] Form labels/errors linked correctly
- [ ] Icon buttons have accessible names

## Performance Checklist

- [ ] Avoid unnecessary rerenders for frequent updates
- [ ] Defer non-critical code and assets
- [ ] Prevent avoidable request waterfalls
- [ ] Large lists virtualized where needed
- [ ] Bundle growth justified and measured

## Reliability and Security Checklist

- [ ] Timeout/retry/cancel behavior defined
- [ ] Error states recoverable by user action
- [ ] Sensitive data not exposed in logs/client artifacts
- [ ] Unsafe HTML rendering avoided or sanitized

## Testing Checklist

- [ ] Primary user journey covered
- [ ] Error and retry paths covered
- [ ] Contract-sensitive edge cases covered
- [ ] Assertions focus on behavior, not internals

---

## Reference

See [frontend-quality-gates.md](frontend-quality-gates.md) for performance budgets and build gates.

# Frontend Accessibility Standards

**Version**: 1.0  
**Last Updated**: 2026-03-13

Reference baseline: WCAG 2.2 AA and WAI-ARIA Authoring Practices.

---

## 1. Semantic First

- Use native semantic elements before ARIA roles.
- Headings must preserve logical hierarchy.
- Interactive controls must be real controls (`button`, `a`, input types).

---

## 2. Keyboard Accessibility

- All interactive actions are keyboard-operable.
- Focus order matches visual and task order.
- Visible focus indicator must not be removed.
- Escape behavior must be consistent for overlays/dialogs.

---

## 3. Forms

- Each input has associated label.
- Required fields and constraints are explicit.
- Errors are programmatically connected (`aria-describedby`).
- Error messages are specific and actionable.

---

## 4. Dynamic Content and Announcements

- Status updates use appropriate live regions.
- Loading indicators should include accessible text.
- Dialogs/trays must trap focus and restore focus on close.
- Toasts/alerts should not steal focus unless blocking action is required.

---

## 5. Visual Accessibility

- Contrast meets WCAG AA for text and UI controls.
- Content remains usable at 200% zoom and responsive reflow.
- Motion respects `prefers-reduced-motion`.
- Information must not rely only on color.

---

## 6. Media and Non-Text Content

- Informative images require meaningful alt text.
- Decorative images use empty alt.
- Video/audio content should include captions/transcripts where applicable.
- Icon-only controls need accessible names.

---

## 7. Testing and Validation

Minimum checks before merge:

- Keyboard-only walkthrough of critical journeys
- Automated a11y linting
- Screen-reader spot checks on high-impact flows
- Contrast and zoom checks on major pages

---

## 8. Common Anti-Patterns

- Clickable `div`/`span` instead of semantic controls
- Removing focus outlines without replacement
- Placeholder as the only input label
- Announcing non-critical updates too aggressively

---

## 9. Review Checklist

- [ ] Semantic structure is correct.
- [ ] Keyboard navigation is complete and logical.
- [ ] Focus behavior is visible and predictable.
- [ ] Form labels and error mappings are accessible.
- [ ] Contrast, zoom, and motion preferences are respected.

# Frontend Static Analysis Setup

This document defines a minimal, practical quality gate for frontend projects.

## 1. Required Tooling

- Type checking: `typescript` (`tsc --noEmit`)
- Linting: `eslint`
- Formatting: `prettier`
- Testing: `vitest` or `jest`
- Accessibility linting: `eslint-plugin-jsx-a11y` (or framework equivalent)

## 2. package.json Scripts (Minimum)

```json
{
  "scripts": {
    "typecheck": "tsc --noEmit",
    "lint": "eslint .",
    "format:check": "prettier --check .",
    "test": "vitest run"
  }
}
```

If using Jest, replace `vitest run` with project-appropriate Jest command.

## 3. TypeScript Baseline (`tsconfig.json`)

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "ESNext",
    "strict": true,
    "noUncheckedIndexedAccess": true,
    "exactOptionalPropertyTypes": true,
    "noFallthroughCasesInSwitch": true,
    "moduleResolution": "Bundler",
    "jsx": "react-jsx",
    "skipLibCheck": true
  }
}
```

Adjust `jsx` and module options to framework/bundler conventions.

## 4. ESLint Baseline (Conceptual)

- Enable TypeScript ruleset.
- Enable framework recommended rules.
- Enable accessibility rules (`jsx-a11y/recommended`) for JSX-based stacks.
- Disallow unused variables, implicit any, and unreachable code patterns.

## 5. Prettier Baseline

- Use project-shared Prettier config.
- Keep formatting fully automated and deterministic.
- Avoid style-only lint rules that conflict with Prettier.

## 6. CI Gate Recommendation

In CI, run in this order:

1. `npm run typecheck`
2. `npm run lint`
3. `npm run format:check`
4. `npm test`
5. `npm run build` (if defined)

Fail fast on any non-zero exit code.

For performance-sensitive applications, also include:

6. route-level bundle diff check
7. Core Web Vitals lab check (tooling chosen by project)

## 7. Agent Auto-Configuration Policy

When agent detects missing quality gates:

- Add missing scripts to `package.json`
- Add minimal config files (`tsconfig.json`, ESLint config, Prettier config) if absent
- Prefer non-breaking incremental setup
- Report all changes clearly to the user

Related standards:

- `knowledge/standards/engineering/frontend/performance-observability-standards.md`
- `knowledge/standards/engineering/frontend/testing-strategy.md`

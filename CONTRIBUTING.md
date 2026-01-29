# Contributing

Thank you for wanting to contribute! This project follows an open, collaborative workflow. Please read the sections below before opening issues or pull requests.

## How to contribute

1. Fork the repository and create a feature branch: `git checkout -b feat/scope-description` or `fix/area-description`.
2. Keep changes focused and atomic — one logical change per PR.
3. Write clear commit messages using the repository commit template `.gitmessage`. Commits should be in Chinese and follow the `type(scope): summary` format (e.g., `docs(standards): update README`).
4. Update `CHANGELOG.md` under `## Unreleased` when you make notable changes.

## Branching and PRs

- Branch name convention: `type/short-description` (e.g., `feat/add-cli`, `fix/readme-typo`).
- Target branch: `dev` for ongoing work, open a PR to `dev` when ready for review.
- Provide a descriptive PR body and reference related issues (e.g., `Issue: #123`).

## Tests and Quality

- Run linters / formatters as appropriate. For Markdown, run `markdownlint` locally if available.
- Add or update documentation when behavior changes.

## Commit Guidelines

- Use Conventional Commit-style messages. Summary line <= 50 chars; wrap body at 72 chars.
- Use types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`, `ci`, `build`, `revert`.

## Review Process

- PRs are reviewed by maintainers and contributors. Respond to review comments promptly.
- Merge via Squash or Merge when approvals are satisfied and CI passes.


Thank you for contributing — we appreciate your time and effort!
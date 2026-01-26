# llm-prompts

![banner](docs/banner.svg)

A curated collection of LLM prompts, agent definitions, and templates for building, evaluating, and operating agent-based workflows. Use these prompts and guidelines to standardize agent behaviour, commit workflows, and documentation across projects.

> Quick link: [CHANGELOG](./CHANGELOG.md) • [Contributing guide](./CONTRIBUTING.md) • [Security](./SECURITY.md) • [Code of Conduct](./CODE_OF_CONDUCT.md)

## Badges

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/hailingu/llm-prompts/actions)
[![Version](https://img.shields.io/badge/version-v0.0.1--beta.0-blue)](https://github.com/hailingu/llm-prompts/releases)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Contributors](https://img.shields.io/github/contributors/hailingu/llm-prompts)](https://github.com/hailingu/llm-prompts/graphs/contributors)

## Table of Contents

- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Quick Start](#quick-start)
- [Usage](#usage)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License & Support](#license--support)

## Features

- 🔧 Collection of reusable LLM prompts and agent roles (in `agents/`) for common workflows.
- 📚 Standards and templates for documentation (`standards/`, `templates/`, `java-standards/`).
- 🧪 Guidance for static analysis and CI-friendly documentation (Checkstyle, PMD, SpotBugs notes).
- 📝 Commit & PR helpers (`.gitmessage`, `.github/prompts`) for consistent contribution workflow.

## Getting Started

### Prerequisites

- Git >= 2.30
- Java >= 17 (for Checkstyle/Java guides)
- Maven >= 3.6 (if you run Java static-analysis locally)
- Node.js >= 18 (optional; for JS tooling)
- Python >= 3.12 (optional; for automation scripts)

### Installation

```bash
# clone
git clone https://github.com/hailingu/llm-prompts.git
cd llm-prompts

# (optional) enable the repo's commit template locally
git config --local commit.template .gitmessage
```

### Quick Start

```bash
# see available prompts
ls .github/prompts

# open the Alibaba Java guidelines (example)
less java-standards/alibaba-java-guidelines.md

# create a conventional commit (template provided)
git add -A
git commit
```

## Usage

- Prompts and agents live under `agents/` and `.github/prompts/`. Use them to generate PR descriptions, commit messages, design docs, and more.
- Standardization files are in `standards/` and `java-standards/` (Checkstyle config, guidelines, and templates).
- Follow the commit template `.gitmessage` and the `CONTRIBUTING.md` workflow for consistent contributions.

## Roadmap

- [ ] Add automated markdownlint and CI checks to run on push.
- [ ] Provide a CLI for applying prompts and generating PR descriptions.
- [ ] Expand example prompts and templates with runnable demos.

## Contributing

Please read `CONTRIBUTING.md` for contribution guidelines, commit message conventions, and PR process. We welcome issues and PRs — thanks for improving this project!

## License & Support

- License: [MIT](LICENSE)
- Bugs & feature requests: open an issue in this repository
- Security: see `SECURITY.md` for reporting vulnerabilities

---

> This README follows the project's documentation standards. See `CHANGELOG.md` for recent changes.

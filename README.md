# llm-prompts

<p>

![License](https://img.shields.io/github/license/hailingu/llm-prompts)
![Contributors](https://img.shields.io/github/contributors/hailingu/llm-prompts)
![Issues](https://img.shields.io/github/issues/hailingu/llm-prompts)
![Pull Requests](https://img.shields.io/github/issues-pr/hailingu/llm-prompts)
![Last Commit](https://img.shields.io/github/last-commit/hailingu/llm-prompts)

</p>

A curated collection of LLM prompts, agent roles, and templates to standardize agent-based workflows, documentation, and contributor experience across projects.

---

## About

This repository contains reusable prompts, agent definitions, and documentation templates designed for:

- 🤖 **LLM Agents** - Reusable prompts and role definitions for common workflows
- 📚 **Documentation Standards** - Templates and guidelines for Java, Python, Go, and more
- 🔧 **Developer Tools** - Commit helpers, PR templates, and CI-friendly documentation
- 📖 **Best Practices** - Industry-standard coding guidelines and design patterns

## Features

- 🤖 **Agent System** - 32+ specialized LLM agents for different workflows (in `agents/`)
  - **Language Agents**: Python, Java, Go coder/reviewer/architect/doc-writer
  - **Data Science Agents**: Algorithm designer, engineer, researcher, evaluator
  - **Utility Agents**: Git specialist, markdown writer, README/Changelog specialist
  - **PPT Specialist**: HTML slide generation with multi-brand support (KPMG/McKinsey/BCG/Bain/Deloitte)
- 🛠️ **Skills** - 13 reusable skill modules for specialized tasks
  - `memory-manager`: Context persistence across sessions
  - `stock-price-tracker`: Real-time stock data via Yahoo Finance
  - `news-search`: Web news search with DuckDuckGo
  - `domain-keyword-detection`: 14-domain keyword detection (software, biotech, automotive, etc.)
  - `markdown-formatter` / `md-table-fixer`: Markdown utilities
  - `ppt-brand-style-system`: Multi-brand-style design tokens
  - `ppt-chart-engine`: Chart selection and rendering
  - `ppt-component-library`: Reusable slide blocks and composition patterns
  - `ppt-map-storytelling`: Geographic narrative patterns for map-first slides
  - `ppt-slide-layout-library`: 14 layout types
  - `ppt-visual-qa`: Automated slide quality assurance (80+ gates)
  - `rss-reader`: Feed parsing and structured content extraction
- 📚 **Standards** - Cross-language and language-specific coding guidelines
  - `knowledge/standards/common/`: API patterns, design review, collaboration protocols
  - `knowledge/standards/engineering/java/`: Alibaba Java guidelines + Checkstyle
  - `knowledge/standards/engineering/python/`: Pythonic conventions
  - `knowledge/standards/engineering/go/`: Effective Go guidelines
  - `knowledge/standards/data-science/`: ML/DS best practices
- 📝 **Templates** - Design doc and module templates (Google-style)
- 🔧 **Developer Tools** - Commit helpers, PR templates, CI-friendly documentation

## Quick Start

```bash
# Clone the repository
git clone https://github.com/hailingu/llm-prompts.git
cd llm-prompts

# Install for current project (recommended)
bash scripts/setup.sh --plugin codex --scope project

# Validate installation
bash scripts/doctor.sh --plugin codex --scope project

# Optional: install only selected skills/agents in global scope
bash scripts/setup.sh --plugin codex --scope global --skills rss-reader,news-search --agents python-coder-specialist,python-code-reviewer
```

### Plugin Adapters

Install adapters by plugin:

```bash
# Codex / Cline / Copilot / Kimi
bash scripts/setup.sh --plugin codex --scope project
bash scripts/setup.sh --plugin cline --scope project
bash scripts/setup.sh --plugin copilot --scope project
bash scripts/setup.sh --plugin kimi --scope project

# Install all adapters at once
bash scripts/setup.sh --plugin all --scope project
```

Notes:
- `agents/` is source-of-truth.
- `.github/agents/` is optional compatibility mirror and is generated when using `--plugin copilot` or `--plugin all` in project scope.

Optional global scope:

```bash
bash scripts/setup.sh --plugin codex --scope global

# install no skills/agents (adapter only)
bash scripts/setup.sh --plugin codex --scope global --skills none --agents none
```

### Explore the Repository

```bash
# View available prompts and agents
ls agents/

# Check out the documentation standards
ls knowledge/standards/common/

# Open the Java guidelines (example)
cat knowledge/standards/engineering/java/alibaba-java-guidelines.md

# Task prompts (breakdown -> execute)
ls prompts/task/
# - task-breakdown.prompt.md
# - task-execute.prompt.md

# GitHub mirror is auto-generated when using Copilot adapter
bash scripts/setup.sh --plugin copilot --scope project

# Optional manual mirror sync
bash scripts/sync_agents_to_github.sh
```

## Repository Structure

```
llm-prompts/
├── adapters/           # Plugin adapters (codex/cline/copilot/kimi)
├── agents/             # Source-of-truth agent role definitions
├── .github/
│   ├── ISSUE_TEMPLATE/ # Issue templates (Bug, Feature, etc.)
│   ├── workflows/      # CI/CD workflows
│   └── copilot-instructions.md # Generated by setup for Copilot
├── config/             # Installation manifests (plugins/skills)
├── docs/               # Additional documentation
├── memory/             # Memory templates and context artifacts
├── prompts/
│   ├── analysis/
│   ├── git/
│   └── task/           # task-breakdown / task-execute prompts
├── scripts/            # setup/install/doctor scripts
├── skills/             # Reusable skill definitions
├── knowledge/
│   ├── standards/
│   │   ├── common/         # Cross-language standards
│   │   ├── engineering/    # language standards: go/java/python
│   │   └── data-science/   # DS/ML standards
│   └── templates/          # Reusable document templates
└── .gitmessage         # Git commit template
```

## Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) for:

- Code of conduct
- Pull request process
- Commit message conventions
- Development setup

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Security

For security vulnerabilities, please refer to our [Security Policy](SECURITY.md).

## Related Links

- [CHANGELOG](CHANGELOG.md)
- [Code of Conduct](CODE_OF_CONDUCT.md)
- [Discussions](https://github.com/hailingu/llm-prompts/discussions)

---

<p align="center">
  Built with ❤️ for the developer community
</p>

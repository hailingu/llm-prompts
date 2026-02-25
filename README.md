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

- 🤖 **Agent System** - 33+ specialized LLM agents for different workflows (in `agents/`)
  - **Language Agents**: Python, Java, Go coder/reviewer/architect/doc-writer
  - **Data Science Agents**: Algorithm designer, engineer, researcher, evaluator
  - **Utility Agents**: Git specialist, markdown writer, README/Changelog specialist
  - **PPT Specialist**: HTML slide generation with multi-brand support (KPMG/McKinsey/BCG/Bain/Deloitte)
- 🛠️ **Skills** - 10 reusable skill modules for specialized tasks
  - `memory-manager`: Context persistence across sessions
  - `stock-price-tracker`: Real-time stock data via Yahoo Finance
  - `news-search`: Web news search with DuckDuckGo
  - `domain-keyword-detection`: 14-domain keyword detection (software, biotech, automotive, etc.)
  - `markdown-formatter` / `md-table-fixer`: Markdown utilities
  - `ppt-brand-system`: Multi-brand design tokens
  - `ppt-chart-engine`: Chart selection and rendering
  - `ppt-slide-layout-library`: 14 layout types
  - `ppt-visual-qa`: Automated slide quality assurance (80+ gates)
- 📚 **Standards** - Cross-language and language-specific coding guidelines
  - `standards/`: API patterns, design review, collaboration protocols
  - `java-standards/`: Alibaba Java guidelines + Checkstyle
  - `python-standards/`: Pythonic conventions
  - `go-standards/`: Effective Go guidelines
  - `data-science-standards/`: ML/DS best practices
- 📝 **Templates** - Design doc and module templates (Google-style)
- 🔧 **Developer Tools** - Commit helpers, PR templates, CI-friendly documentation

## Quick Start

```bash
# Clone the repository
git clone https://github.com/hailingu/llm-prompts.git
cd llm-prompts

# (Optional) Enable the commit template locally
git config --local commit.template .gitmessage
```

### Explore the Repository

```bash
# View available prompts and agents
ls agents/

# Check out the documentation standards
ls standards/

# Open the Java guidelines (example)
cat java-standards/alibaba-java-guidelines.md
```

## Repository Structure

```
llm-prompts/
├── agents/              # LLM agent role definitions and prompts
├── .github/
│   ├── ISSUE_TEMPLATE/ # Issue templates (Bug, Feature, etc.)
│   └── prompts/        # GitHub-specific prompts
├── docs/               # Additional documentation
├── prompts/            # General prompts organized by category
├── skills/             # Reusable skill definitions
├── standards/          # Cross-language standards
├── java-standards/     # Java-specific guidelines
├── go-standards/      # Go-specific guidelines
├── python-standards/  # Python-specific guidelines
└── templates/          # Reusable document templates
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

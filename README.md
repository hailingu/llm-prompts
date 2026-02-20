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

- 🔧 Collection of reusable LLM prompts and agent roles (in `agents/`) for common workflows.
- 📚 Standards and templates for documentation (`standards/`, `templates/`, `java-standards/`).
- 🧪 Guidance for static analysis and CI-friendly documentation (Checkstyle, PMD, SpotBugs notes).
- 📝 Commit & PR helpers (`.gitmessage`, `.github/prompts`) for consistent contribution workflow.

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

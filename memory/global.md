# Global Memory

## System Context
- **Workspace**: llm-prompts project
- **Agent System**: Multi-agent framework with specialized agents for different domains
- **Last Updated**: 2026-02-25

## User Preferences & Patterns
*No user preferences recorded yet.*

## Key Decisions & Learnings
*No key decisions recorded yet.*

## Project Milestones
*No project milestones recorded yet.*

## Agent Usage Patterns
*No agent usage patterns recorded yet.*

## Skill Inventory
*Record skills created and their purposes here.*

---

## Update Log

### 2026-02-25
- **Memory system initialized**: Created MEMORY.md and memory-manager skill
- **Stock price tracker skill**: Created skills/stock-price-tracker/ for real-time stock price retrieval via Yahoo Finance
  - Purpose: Enable AI agents to access current stock prices without manual web browsing
  - Features: Multi-symbol support, JSON/CSV output, real-time data
  - Dependencies: yfinance, pandas
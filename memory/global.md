# Global Knowledge & Context Control
> **System Status**: Active Session - Geopolitics 2026 Simulation
> **Current Date (Simulated)**: March 3, 2026

## 1. Active Mission: Operation Epic Fury Strategy Deck
**Goal**: Create a 16-slide BCG-style HTML presentation analyzing the China Strategy post-Operation Epic Fury.
**Current Phase**: Execution / Content Expansion (Outline expanded to 34 slides; added Russia-Ukraine linkage).

### Key Constraints (Immutable)
- **Scenario Root**: "Operation Epic Fury" (US/Israel decapitation strike on Iran) occurred **Feb 28, 2026**.
- **Key Status**: Ali Khamenei (KIA), IRGC Command (Decapitated), Natanz (Damaged).
- **Core Dilemma**: China must secure energy/periphery without triggering direct US war ("Asymmetric Balancing").
- **Output Standard**: 
  - Format: HTML5 + Tailwind CSS (Full interactive capabilities).
  - Design System: BCG Standard (Green `#007645`, Sergeant/Georgia fonts).
  - File Structure: `docs/presentations/geopolitics_2026_v1/`.

## 2. Research Index (L2 Memory)
*Use these sources for factual grounding before semantic search.*
- [US-Israel-Iran Relations Report](research/US_Israel_Iran_Relations_Report.md) - *Baseline War Scenario*
- [China Strategy 2026](research/china_strategy_2026.md) - *Strategic Pillars Formulation*
- [China-Israel Relations](research/china_israel_relations_2026.md) - *Diplomatic channel analysis*

## 3. Decisions (Technical & Process)
### Decision: Memory Organization
- **Rule**: Use shared global memory as the single source of truth.
- **Structure**: Split by domain (coding/research) but index here.

### Decision: Presentation Architecture
- **Tech Stack**: Iframe-based player (`presentation.html`) loading individual slide files (`slide-N.html`).
- **Reasoning**: Allows modular editing of single slides without breaking the master container.

## 4. User Preferences
- **Refinement Mode**: When asked to "optimize", focus on visual storytelling (maps, diagrams) over text density.
- **Workflow**: Complex tasks must utilize the `memory/` system for context persistence.

## 5. Process Improvements
- **Pre-Computation**: Before generating HTML, always simpler "Thinking" files (markdown) to validate content logic.



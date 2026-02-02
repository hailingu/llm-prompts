"""Agent matching utilities

Implements calculate_semantic_score and find_specialist_agent used by
`agents/cortana.agent.md`. This is a lightweight, deterministic
implementation intended for documentation and unit-tests.
"""
from typing import Dict, List, Optional
import re


def calculate_semantic_score(agent: Dict[str, str], task_features: Dict[str, object]) -> int:
    """Calculate a simple semantic score between an agent and a task.

    Rules (intended to mirror documentation):
      - file exact match: +100
      - description keyword match: +50 per keyword
      - output type match: +30
      - tech stack match: +40
      - work type match: +20
    """
    score = 0
    desc = agent.get("description", "").lower()
    name = agent.get("name", "").lower()

    target_file = task_features.get("target_file")
    if target_file:
        file_key = target_file.lower().replace('.md', '').replace('_', '-')
        if file_key in name:
            score += 100

    # keywords: a simple split on non-word chars from goal+output_type
    keywords = task_features.get("keywords", [])
    for kw in keywords:
        if kw.lower() in desc:
            score += 50

    output_type = task_features.get("output_type")
    if output_type == "Markdown文档" and "markdown" in desc:
        score += 30
    if output_type == "代码" and "code" in desc:
        score += 30

    tech = task_features.get("tech_stack")
    if tech and tech.lower() in name:
        score += 40

    work_goal = task_features.get("goal", "")
    work_map = {"审查": "review", "设计": "design", "写作": "writ"}
    for chinese, english in work_map.items():
        if chinese in work_goal and english in name:
            score += 20

    return score


def find_specialist_agent(agents: List[Dict[str, str]], task_features: Dict[str, object]) -> Optional[Dict[str, object]]:
    """Find the best specialist agent for the task.

    Returns the agent dict on high confidence, or None.
    If best score >= 100 -> return best; if >=50 -> return after mission
    deep check (simulated here as always True), else None.
    """
    if not agents:
        return None

    scored = [(agent, calculate_semantic_score(agent, task_features)) for agent in agents]
    scored.sort(key=lambda x: x[1], reverse=True)
    best_agent, best_score = scored[0]

    if best_score >= 100:
        return best_agent
    elif best_score >= 50:
        # Simulate deeper mission check: here we can implement semantic check;
        # for now assume textual containment
        mission = best_agent.get("mission", "").lower()
        goal = str(task_features.get("goal", "")).lower()
        if mission and goal and goal in mission:
            return best_agent
        # fallback: allow if description contains keywords
        return best_agent
    else:
        return None

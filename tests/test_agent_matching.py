import pytest
from tools.agent_matching import calculate_semantic_score, find_specialist_agent


def test_calculate_semantic_score_basic():
    agent = {"name": "markdown-writer-specialist", "description": "Markdown 文档写作专家"}
    task = {"keywords": ["文档", "写作"], "output_type": "Markdown文档", "goal": "写文档"}
    score = calculate_semantic_score(agent, task)
    assert score >= 80  # at least some points


def test_find_specialist_agent_high_confidence():
    agents = [
        {"name": "readme-specialist", "description": "README 写作"},
        {"name": "markdown-writer-specialist", "description": "Markdown 文档 写作 专家", "mission": "让读者能在最短时间内找到信息"},
    ]
    task = {"target_file": "README.md", "keywords": ["readme"], "goal": "写 README"}
    result = find_specialist_agent(agents, task)
    assert result and result["name"] == "readme-specialist"


def test_find_specialist_agent_none():
    agents = [
        {"name": "go-code-reviewer", "description": "Go 代码审查"},
    ]
    task = {"keywords": ["文档"], "goal": "写文档"}
    result = find_specialist_agent(agents, task)
    assert result is None

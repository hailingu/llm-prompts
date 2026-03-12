{
  "inputs": [],
  "servers": {
    "llm_prompts": {
      "type": "stdio",
      "command": "bash",
      "args": ["-lc", "cd __REPO_ROOT__ && echo 'Use repo-local prompts/skills/agents'"],
      "env": {}
    }
  }
}

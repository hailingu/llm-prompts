# llm-prompts

My LLM prompts

## Git commit message template

This repo ships a commit message template at [.gitmessage](.gitmessage).

- Enable for this repo (recommended): `git config --local commit.template .gitmessage`
- Enable globally (optional): `git config --global commit.template ~/.gitmessage`
  - Then copy it once: `cp .gitmessage ~/.gitmessage`

Notes:

- If you use `git commit -m "..."`, Git will not open the template.

Commit message guidelines:

- Commit messages should be written in Chinese and follow the `.gitmessage` template.
- Use `type(scope): summary` in the commit header and fill in details in the body and footer if needed.

#!/usr/bin/env python3
from pathlib import Path
p=Path('standards/google-design-doc-standards.md')
s=p.read_text()

# This will merge lines where a table row/header is split across lines
# Pattern: a line that contains '|' followed by a newline and then spaces and a '|' -> merge
old = None
while True:
    import re
    m = re.search(r"\|[^\n]*\n[ \t]+\|", s)
    if not m:
        break
    # Replace the newline and following indentation with a single space
    s = s[:m.start()] + s[m.start():m.end()].replace('\n', ' ').replace('  ', ' ') + s[m.end():]

# Write back
p.write_text(s)
print('Merged table linebreaks')

#!/usr/bin/env python3
from pathlib import Path
p=Path('standards/google-design-doc-standards.md')
s=p.read_text().splitlines()

def find_tables(lines):
    tables=[]
    in_table=False
    cur=[]
    start=None
    for i,l in enumerate(lines):
        if l.strip().startswith('|'):
            if not in_table:
                in_table=True
                start=i
                cur=[l]
            else:
                cur.append(l)
        else:
            if in_table:
                if len(cur)>1:
                    tables.append((start, cur))
                in_table=False
                cur=[]
    if in_table and len(cur)>1:
        tables.append((start, cur))
    return tables


def pipe_positions(line):
    return [i for i,ch in enumerate(line) if ch=='|']


tables=find_tables(s)
print('Found', len(tables), 'tables')
problems=[]
for start, lines in tables:
    header=lines[0]
    header_pipes=pipe_positions(header)
    mis=[]
    for idx, row in enumerate(lines[1:], start=1):
        row_pipes=pipe_positions(row)
        if row_pipes!=header_pipes:
            mis.append((idx+start, row_pipes, row))
    if mis:
        problems.append((start+1, header, mis))

if not problems:
    print('No misaligned tables detected (pipe positions match header).')
else:
    for start_line, header, mis in problems:
        print('\nTable starting at line', start_line)
        print('Header:', header)
        for ln, pipes, row in mis:
            print('  Misaligned line', ln, 'pipes:', pipes)
            print('   ', row)

#!/usr/bin/env python3
from pathlib import Path
import re

p=Path('standards/google-design-doc-standards.md')
s=p.read_text()
lines=s.splitlines()

out_lines=[]
i=0
changed=False

while i < len(lines):
    if lines[i].lstrip().startswith('|'):
        # start of table block
        j=i
        block=[]
        while j < len(lines) and lines[j].lstrip().startswith('|'):
            block.append(lines[j])
            j+=1
        # Merge broken rows within block: accumulate until pipe count matches header
        merged_rows=[]
        # First, determine expected pipe count from first non-empty logical line in block
        # We'll accumulate lines into a buffer until buffer.count('|') >= 2 and buffer.endswith('|')
        k=0
        while k < len(block):
            buf=block[k]
            # if buf already seems complete (has multiple pipes and ends with '|'), take it
            if buf.count('|') >= 2 and buf.rstrip().endswith('|'):
                merged_rows.append(buf)
                k+=1
                continue
            # else accumulate subsequent lines until condition
            m=k+1
            while m < len(block):
                buf = buf + ' ' + block[m].strip()
                if buf.count('|') >= 2 and buf.rstrip().endswith('|'):
                    break
                m+=1
            merged_rows.append(buf)
            k = m+1
        # Now parse rows into cells
        rows=[ [c.strip() for c in re.split(r"\|", r)[1:-1]] for r in merged_rows if r.strip()!='']
        if len(rows) > 1:
            # compute max widths
            cols = max(len(r) for r in rows)
            widths = [0]*cols
            for r in rows:
                for idx,cell in enumerate(r):
                    widths[idx]=max(widths[idx], len(cell))
            # build new table lines
            new_block=[]
            for ridx, r in enumerate(rows):
                # pad cells
                cells=[ (r[cidx] if cidx < len(r) else '') .ljust(widths[cidx]) for cidx in range(cols)]
                new_block.append('| ' + ' | '.join(cells) + ' |')
                if ridx==0:
                    sep = '| ' + ' | '.join('-'*w for w in widths) + ' |'
                    new_block.append(sep)
            # replace block in output
            out_lines.extend(new_block)
            changed=True
        else:
            # not a real table, just copy as-is
            out_lines.extend(block)
        i=j
    else:
        out_lines.append(lines[i])
        i+=1

if changed:
    p.write_text('\n'.join(out_lines)+"\n")
    print('Rebuilt tables; file updated')
else:
    print('No table changes needed')

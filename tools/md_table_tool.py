#!/usr/bin/env python3
"""md_table_tool.py

CLI tool to detect and fix Markdown table alignment (MD060) using display
width-aware padding (handles CJK widths). Can operate on a single file or
recursively on a directory. Creates backups before modifying files.

Usage:
  python3 tools/md_table_tool.py detect <path>
  python3 tools/md_table_tool.py fix <path>
  python3 tools/md_table_tool.py --help
"""
from pathlib import Path
import sys
import argparse
import unicodedata
import shutil
from datetime import datetime
import re


def display_width(s: str) -> int:
    w = 0
    for ch in s:
        ea = unicodedata.east_asian_width(ch)
        if ea in ('W', 'F'):
            w += 2
        else:
            w += 1
    return w


def split_table_blocks(lines):
    tables = []
    in_table = False
    cur = []
    start = None
    for i, l in enumerate(lines):
        if l.strip().startswith('|'):
            if not in_table:
                in_table = True
                start = i
                cur = [l.rstrip('\n')]
            else:
                cur.append(l.rstrip('\n'))
        else:
            if in_table:
                if len(cur) > 1:
                    tables.append((start, cur))
                in_table = False
                cur = []
    if in_table and len(cur) > 1:
        tables.append((start, cur))
    return tables


def merge_broken_rows(block):
    # Merge rows that are split across lines (heuristic)
    merged = []
    i = 0
    while i < len(block):
        buf = block[i]
        if buf.count('|') >= 2 and buf.rstrip().endswith('|'):
            merged.append(buf)
            i += 1
            continue
        j = i + 1
        while j < len(block):
            buf = buf + ' ' + block[j].strip()
            if buf.count('|') >= 2 and buf.rstrip().endswith('|'):
                break
            j += 1
        merged.append(buf)
        i = j + 1
    return merged


def parse_row(row):
    # split on | and exclude leading/trailing empty segments
    parts = [c.strip() for c in re.split(r"\|", row)[1:-1]]
    return parts


def build_aligned_block(rows):
    parsed = [parse_row(r) for r in rows if r.strip()]
    if not parsed:
        return rows
    cols = max(len(r) for r in parsed)
    # normalize row lengths
    for r in parsed:
        if len(r) < cols:
            r.extend([''] * (cols - len(r)))
    # compute display widths
    widths = [0] * cols
    for r in parsed:
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], display_width(cell))
    # build lines
    out = []
    for idx, r in enumerate(parsed):
        cells = []
        for i, cell in enumerate(r):
            # pad using display width
            pad = widths[i] - display_width(cell)
            cells.append(cell + ' ' * pad)
        out.append('| ' + ' | '.join(cells) + ' |')
        if idx == 0:
            sep = '| ' + ' | '.join('-' * widths[i] for i in range(cols)) + ' |'
            out.append(sep)
    return out


def check_alignment(block):
    header_pipes = [i for i, ch in enumerate(block[0]) if ch == '|']
    mis = []
    for idx, r in enumerate(block[1:], start=1):
        rp = [i for i, ch in enumerate(r) if ch == '|']
        if rp != header_pipes:
            mis.append((idx + 1, rp, r))
    return mis


def detect_file(path: Path):
    s = path.read_text()
    lines = s.splitlines()
    tables = split_table_blocks(lines)
    problems = []
    for start, block in tables:
        merged = merge_broken_rows(block)
        mis = check_alignment(merged)
        if mis:
            problems.append((start + 1, block[0], mis))
    return problems


def fix_file(path: Path, make_backup=True):
    s = path.read_text()
    lines = s.splitlines()
    tables = split_table_blocks(lines)
    if not tables:
        return False, 0
    out_lines = []
    i = 0
    changed_count = 0
    while i < len(lines):
        if lines[i].strip().startswith('|'):
            # capture table block
            j = i
            block = []
            while j < len(lines) and lines[j].strip().startswith('|'):
                block.append(lines[j].rstrip('\n'))
                j += 1
            merged = merge_broken_rows(block)
            aligned = build_aligned_block(merged)
            # determine if changed
            if aligned != block and aligned:
                changed_count += 1
                out_lines.extend(aligned)
            else:
                out_lines.extend(block)
            i = j
        else:
            out_lines.append(lines[i])
            i += 1
    if changed_count > 0:
        if make_backup:
            bak = path.with_suffix(path.suffix + '.bak')
            shutil.copy2(path, bak)
        path.write_text('\n'.join(out_lines) + '\n')
        return True, changed_count
    return False, 0


def process_path(mode, path_str):
    p = Path(path_str)
    files = []
    if p.is_dir():
        files = list(p.rglob('*.md'))
    elif p.is_file():
        files = [p]
    else:
        print('Path not found:', path_str)
        return 2
    total_problems = 0
    total_changed = 0
    for f in files:
        if mode == 'detect':
            problems = detect_file(f)
            if problems:
                print(f'File: {f} - {len(problems)} misaligned table(s)')
                for start, header, mis in problems:
                    print('  Table starting at line', start)
                    print('   ', header)
                    for ln, pipes, row in mis:
                        print('    Misaligned line', ln)
                total_problems += len(problems)
        elif mode == 'fix':
            ok, changed = fix_file(f)
            if changed > 0:
                print(f'Fixed {changed} table(s) in {f} (backup created: {f}.bak)')
                total_changed += changed
    if mode == 'detect':
        if total_problems == 0:
            print('No misaligned tables detected (MD060 check OK).')
        else:
            print(f'Total misaligned tables: {total_problems}')
    elif mode == 'fix':
        if total_changed == 0:
            print('No changes made.')
        else:
            print(f'Total tables fixed: {total_changed}')
    return 0


def main():
    ap = argparse.ArgumentParser(description='Detect and fix Markdown table alignment (MD060)')
    ap.add_argument('command', choices=['detect', 'fix'], help='detect or fix')
    ap.add_argument('path', help='file or directory path (markdown files)')
    args = ap.parse_args()
    return process_path(args.command, args.path)

if __name__ == '__main__':
    sys.exit(main())

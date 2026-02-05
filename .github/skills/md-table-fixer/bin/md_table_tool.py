#!/usr/bin/env python3
"""
Wrapper script for md_table_tool.py so skills can reference a predictable path
This script delegates to the real implementation at tools/md_table_tool.py
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # repo root
real = ROOT / 'tools' / 'md_table_tool.py'

if not real.exists():
    print(f"Error: underlying tool not found at {real}", file=sys.stderr)
    sys.exit(2)

# Execute the real script with the same args
os.execv(sys.executable, [sys.executable, str(real)] + sys.argv[1:])
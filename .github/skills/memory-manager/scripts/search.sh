#!/bin/bash
# Simple memory search - no dependencies

if [ -z "$1" ]; then
    echo "Usage: search.sh <keyword> [theme]"
    exit 1
fi

KEYWORD="$1"
THEME="${2:-*}"

grep -r "$KEYWORD" memory/$THEME/ 2>/dev/null | head -20

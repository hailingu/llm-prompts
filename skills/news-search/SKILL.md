---
name: news-search
description: "Browser-first news retrieval with paywall-aware stop policy, combining DuckDuckGo Lite discovery and RSS feeds."
metadata:
  version: 3.0.0
  author: cortana
---

# News Search (Browser-First + Paywall Aware)

## Overview

This skill retrieves news with a browser-first workflow to reduce brittle crawler behavior.

Flow:

1. Discover candidate links from `duckduckgo` and/or `rss`
2. Open each candidate using browser retrieval (Playwright)
3. Detect paywall/subscription signals
4. Stop deep extraction for paywalled pages (default behavior)

## Why This Update

- Reduces dependence on regex-only page scraping
- Better matches real page rendering behavior
- Enforces a clear paywall policy: detect and stop
- Keeps fallback mode available when Playwright is unavailable

## CLI

```bash
# Browser-first (default), all sources
python3 skills/news-search/scripts/search.py "OpenAI latest model" --source all --max 5

# RSS-only discovery + browser retrieval
python3 skills/news-search/scripts/search.py "AI" --source rss --category ai --max 5

# HTTP fallback mode
python3 skills/news-search/scripts/search.py "global markets" --retrieval http --max 5

# Skip detail retrieval (discovery only)
python3 skills/news-search/scripts/search.py "chip industry" --retrieval none --max 8

# Continue even if paywall markers are found
python3 skills/news-search/scripts/search.py "financial times" --allow-paywalled
```

## Parameters

- `--source`: `duckduckgo | rss | all` (default: `all`)
- `--category`: `tech | finance | sports | entertainment | global | domestic | military | gaming | ai`
- `--max`: maximum results (default: `5`)
- `--retrieval`: `browser | http | none` (default: `browser`)
- `--allow-paywalled`: disable stop-on-paywall behavior
- `--timeout`: request timeout in seconds

## Output

The script returns JSON with:

- `results[*].retrieval_mode`: `browser | http | none`
- `results[*].access`: `open | paywalled | unknown`
- `results[*].paywall_reason`: matched signal when paywalled
- `metadata.paywalled_count`
- `metadata.warnings` (for fallback events like missing Playwright)

## Paywall Policy

Default policy is strict:

- if paywall markers are detected, deep extraction stops for that URL
- link is still returned with `access=paywalled`
- no attempt to bypass login/subscription walls

## Notes

- Browser mode requires Playwright Python runtime.
- If Playwright is unavailable, browser mode automatically falls back to `http` retrieval and reports a warning.

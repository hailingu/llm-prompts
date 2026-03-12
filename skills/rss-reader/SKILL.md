---
name: rss-reader
description: "RSS/Atom feed reader and subscription manager with health validation, fallback feed URLs, and default-feed refresh."
metadata:
  version: 1.1.0
  author: cortana
---

# RSS Reader & Subscription Manager

## Overview

This skill manages RSS subscriptions and fetches articles with cache and concurrency.

New enhancements:

- `fallback_urls` support in default feed config
- `validate` command to check feed availability and parseability
- `refresh-defaults` command to sync updated default feeds into user subscriptions

## Commands

```bash
# List subscriptions
python3 skills/rss-reader/scripts/rss_reader.py list

# Fetch articles
python3 skills/rss-reader/scripts/rss_reader.py fetch --limit 20

# Validate current subscriptions
python3 skills/rss-reader/scripts/rss_reader.py validate

# Validate default feed set
python3 skills/rss-reader/scripts/rss_reader.py validate --use-defaults

# Validate only one category
python3 skills/rss-reader/scripts/rss_reader.py validate --use-defaults --category ai

# Sync updated default feeds into subscriptions
python3 skills/rss-reader/scripts/rss_reader.py refresh-defaults

# Sync only one category
python3 skills/rss-reader/scripts/rss_reader.py refresh-defaults --category news
```

## Feed Config (`config/feeds.yaml`)

Each feed item supports:

- `name`: display name
- `url`: primary feed URL
- `fallback_urls` (optional): alternate URLs tried when primary fails

## Recommended Maintenance Workflow

1. Update `config/feeds.yaml` with new/alternative feed links.
2. Run `validate --use-defaults` to check health.
3. Run `refresh-defaults` to sync changes into subscriptions.
4. Re-run `validate` to confirm subscribed set is healthy.

## Notes

- `validate` returns non-zero exit code when any feed fails.
- `fetch` automatically tries `fallback_urls` when available.
- YAML default feeds require `PyYAML`; without it, defaults cannot be loaded.

---
name: rss-reader
description: "RSS/Atom feed reader and subscription manager - fetch, parse, and organize RSS feeds from multiple sources with offline reading support."
metadata:
  version: 1.0.0
  author: cortana
---

# RSS Reader & Subscription Manager

## Overview
This skill provides a comprehensive RSS/Atom feed reading and subscription management solution for AI agents. It enables users to subscribe to their favorite RSS/Atom feeds, organize them by category, and fetch the latest articles with full-text content extraction.

## Why this Skill?
- **Structured Content**: RSS feeds provide well-organized, machine-readable content
- **No Hallucination**: Direct feed parsing ensures accurate information retrieval
- **Multi-Source Aggregation**: Subscribe to multiple feeds and aggregate content by category
- **Offline Reading**: Cache articles for offline access and reference
- **Full-Text Extraction**: Extract full article content beyond just the summary

## Core Features

### 1. Subscription Management
- Add/remove RSS/Atom feed subscriptions
- Organize feeds by custom categories (Tech, Finance, News, etc.)
- Import/export OPML subscription files
- Validate feed URLs before adding

### 2. Feed Fetching & Parsing
- Support RSS 2.0, Atom 1.0, and JSON Feed formats
- Automatic feed type detection
- Handle feed redirects and encoding issues
- Extract metadata (author, published date, categories, tags)

### 3. Article Fetching
- Fetch full article content using readability algorithms
- Extract main content, images, and metadata
- Filter by date range, keyword, or category
- Sort by date, relevance, or source

### 4. Output Formats
- JSON: Structured data for AI consumption
- Markdown: Human-readable article summaries
- HTML: Formatted article view

## Usage

### Command Line Usage

```bash
# Add a new RSS feed subscription
python3 skills/rss-reader/scripts/rss_reader.py add "https://www.36kr.com/feed" --category tech

# List all subscriptions
python3 skills/rss-reader/scripts/rss_reader.py list

# List feeds by category
python3 skills/rss-reader/scripts/rss_reader.py list --category tech

# Fetch latest articles from all feeds
python3 skills/rss-reader/scripts/rss_reader.py fetch

# Fetch articles from specific category
python3 skills/rss-reader/scripts/rss_reader.py fetch --category tech

# Fetch articles from specific feed
python3 skills/rss-reader/scripts/rss_reader.py fetch --feed "36kr"

# Fetch and get full article content
python3 skills/rss-reader/scripts/rss_reader.py fetch --full-text --limit 10

# Search articles by keyword
python3 skills/rss-reader/scripts/rss_reader.py search "AI"

# Remove a feed
python3 skills/rss-reader/scripts/rss_reader.py remove "https://www.36kr.com/feed"

# Export subscriptions to OPML
python3 skills/rss-reader/scripts/rss_reader.py export --output subscriptions.opml

# Import subscriptions from OPML
python3 skills/rss-reader/scripts/rss_reader.py import --file subscriptions.opml
```

### Programmatic Usage

```python
from rss_reader import RSSReader, SubscriptionManager

# Initialize reader
reader = RSSReader()

# Add a subscription
manager = SubscriptionManager()
manager.add_feed("https://www.36kr.com/feed", category="tech")

# Fetch articles
articles = reader.fetch_feed("https://www.36kr.com/feed")
for article in articles:
    print(f"{article.title} - {article.published}")

# Get full content
full_article = reader.get_full_article(article.url)
```

## Pre-configured Feed Categories

The skill comes with **25+ high-quality RSS feeds** pre-configured across 7 categories:

| Category | Feeds |
|----------|-------|
| **Tech** (科技) | 36Kr, IT之家, 虎嗅, 极客公园, 爱范儿, 少数派 |
| **Finance** (财经) | 新浪财经, 华尔街见闻, 第一财经, 经济观察报 |
| **News** (新闻) | 环球网, 央视新闻, 新浪新闻, 参考消息, 财新网 |
| **AI** (人工智能) | MIT Tech Review, OpenAI Blog, Anthropic Blog, Hugging Face |
| **Dev** (开发) | Hacker News, Dev.to, 掘金, 开源中国, GitHub Blog |
| **Product** (产品) | Product Hunt, Smashing Magazine |
| **Business** (商业) | YC Blog, 36Kr创投 |

## Output Format

### JSON Output
```json
{
  "status": "success",
  "source": "36kr",
  "articles": [
    {
      "title": "AI领域最新突破：GPT-5发布",
      "url": "https://www.36kr.com/p/1234567890",
      "published": "2026-02-25T10:30:00Z",
      "author": "36Kr",
      "summary": "今日，OpenAI正式发布GPT-5...",
      "content": "完整文章内容...",
      "categories": ["AI", "大模型"]
    }
  ],
  "metadata": {
    "total_articles": 20,
    "fetch_time_ms": 1234
  }
}
```

### Markdown Output
```markdown
## AI领域最新突破：GPT-5发布

**来源**: 36Kr | **发布时间**: 2026-02-25

今日，OpenAI正式发布GPT-5...

[阅读原文](https://www.36kr.com/p/1234567890)

---
```

## Implementation Details

### Architecture
```
skills/rss-reader/
├── SKILL.md                 # This file
├── scripts/
│   └── rss_reader.py        # Main CLI and library
└── data/
    └── subscriptions.json  # Feed storage (created on first run)
```

### Core Components
1. **FeedParser**: Parse RSS 2.0, Atom 1.0, JSON Feed
2. **ContentExtractor**: Extract full article using readability
3. **SubscriptionManager**: CRUD operations for subscriptions
4. **CacheManager**: Local caching for offline access
5. **OPMLHandler**: Import/export OPML files

### Dependencies
- `feedparser`: RSS/Atom parsing (via `pip install feedparser`)
- `beautifulsoup4`: HTML parsing for content extraction
- `requests`: HTTP client (optional, falls back to urllib)

### Fallback Mode
If dependencies are not installed, the script falls back to:
- Basic XML parsing using Python's built-in `xml.etree.ElementTree`
- Simple regex-based content extraction

## Agent Integration Guidelines

When an agent uses this skill to retrieve information:
1. **Specify Category**: Always specify the relevant category for better results
2. **Limit Results**: Use `--limit` to avoid overwhelming the user
3. **Include Source**: Always cite the source feed and URL
4. **Date Filter**: Use `--since` to get recent articles only

## Error Handling
- **Invalid Feed URL**: Returns error with validation message
- **Network Timeout**: Returns partial results with warning
- **Parse Error**: Skips malformed entries, continues with valid ones
- **No Updates**: Returns empty list if no new articles since last fetch

## Best Practices
1. **Use Categories**: Organize feeds by topic for easier management
2. **Regular Updates**: Fetch feeds periodically to stay updated
3. **Limit Full-Text**: Only fetch full text when needed (slower)
4. **Cache Usage**: Use cached articles when available for faster response
5. **OPML Backup**: Export subscriptions regularly as backup

---
name: news-search
description: "Perform clean, structured web searches via DuckDuckGo HTML to retrieve the latest news, current events, and static factual information."
metadata:
  version: 1.0.0
  author: cortana
---

# News Search (Anti-Hallucination)

## Overview
This skill provides a clean, structured way for AI agents (like Cortana) to perform web searches using DuckDuckGo's HTML interface. It is specifically designed to retrieve **latest news, current events, and static factual information** (e.g., tech news, financial updates, historical facts) to prevent AI hallucinations.

## Why this Skill?
When an AI agent lacks real-time data or specific domain knowledge, it may confidently generate plausible but incorrect information (hallucination). By using this skill, the agent can:
1. **Bypass UI Noise**: Directly extract titles, URLs, and snippets without dealing with complex HTML DOMs or JavaScript.
2. **Enforce Transparency**: Provide real sources and URLs for the news retrieved.
3. **Minimize Token Usage**: Return only the top N most relevant news snippets in a clean JSON format, avoiding context window overflow.

> **Note**: This skill is **not suitable** for querying dynamic, database-driven information like real-time flight schedules, train tickets, or e-commerce prices, as search engine snippets do not reliably capture deep-linked dynamic data.

## Usage
The skill is implemented as a standalone Python script that requires no external dependencies (uses standard library `urllib` and `re`).

### Supported Categories
You can use the `--category` flag to narrow down the news search. The script automatically appends `site:` operators to restrict searches to high-quality, authoritative sources for each category. Supported categories are:
- `tech` (科技): 36Kr, IT之家, 虎嗅
- `finance` (财经): 新浪财经, 华尔街见闻, 第一财经
- `sports` (体育): 新浪体育, 虎扑, 直播吧
- `entertainment` (娱乐): 新浪娱乐, 网易娱乐
- `global` (国际): 环球网国际, 央视国际
- `domestic` (国内): 新浪国内, 央视国内
- `military` (军事): 环球网军事, 新浪军事
- `gaming` (游戏): 游民星空, 机核网, IGN中国

```bash
# Search for general news
python3 skills/news-search/scripts/search.py "2026年2月25日" --max 3

# Search for specific category news
python3 skills/news-search/scripts/search.py "2026年2月25日" --category finance --max 3
```

## Output Format
The script outputs a JSON object containing the news search results:

```json
{
  "status": "success",
  "query": "2026年2月25日 财经新闻",
  "results": [
    {
      "title": "新浪财经 - 2026年2月25日股市收盘播报",
      "url": "https://finance.sina.com.cn/...",
      "snippet": "今日A股三大指数集体收涨..."
    }
  ]
}
```

## Implementation Details

### Architecture
The skill is a lightweight Python script (`scripts/search.py`) that acts as a wrapper around DuckDuckGo's HTML search interface (`https://html.duckduckgo.com/html/`).

### Core Components
1. **HTTP Request**: Uses `urllib.request` to fetch the HTML page. A `User-Agent` header is required to prevent DuckDuckGo from blocking the request.
2. **HTML Parsing**: Uses regular expressions (`re.findall`) to extract the search results. This avoids the need for external dependencies like `BeautifulSoup` or `lxml`, making the skill highly portable.
3. **URL Decoding**: DuckDuckGo often wraps destination URLs in a redirect link (`//duckduckgo.com/l/?uddg=...`). The script parses and unquotes the actual destination URL.
4. **JSON Output**: The results are formatted as a JSON object, making it easy for AI agents to parse and use the data programmatically.

## Integration
This skill is primarily used by `cortana` and `data-scientist-research-lead` to validate assumptions before generating responses.

### Agent Response Guidelines
When an agent uses this skill to answer a user's query, the agent **MUST** adhere to the following formatting rules:
1. **Cite Sources**: Always append the source URL (and optionally the source name) to each news item presented to the user.
2. **Timestamp**: Always include the exact time the search was performed (precise to the second) to provide temporal context for the news.
3. **Transparency**: Do not present the news as the agent's own knowledge; explicitly state that it was retrieved via search.
4. **Format Example**:
   - "今日A股三大指数集体收涨... [信源: 新浪财经](https://finance.sina.com.cn/...) (检索时间: 2026-02-25 10:15:30)"

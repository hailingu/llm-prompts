---
name: news-search
description: "Perform clean, structured web searches via multiple sources (DuckDuckGo, Baidu, Bing, RSS) to retrieve the latest news, current events, and static factual information."
metadata:
  version: 2.0.0
  author: cortana
---

# News Search (Anti-Hallucination)

## Overview
This skill provides a clean, structured way for AI agents (like Cortana) to perform web searches using **multiple search engines and news sources**. It is specifically designed to retrieve **latest news, current events, and static factual information** to prevent AI hallucinations.

## Why this Skill?
When an AI agent lacks real-time data or specific domain knowledge, it may confidently generate plausible but incorrect information (hallucination). By using this skill, the agent can:
1. **Multi-Source Coverage**: Access news from DuckDuckGo, Baidu, Bing, and RSS feeds
2. **Bypass UI Noise**: Directly extract titles, URLs, and snippets without dealing with complex HTML DOMs
3. **Enforce Transparency**: Provide real sources and URLs for the news retrieved
4. **Minimize Token Usage**: Return only the top N most relevant news snippets in a clean JSON format

## Supported Search Sources

| Source | Best For | Language Support |
|--------|----------|------------------|
| `duckduckgo` | General news, international | Multi-language |
| `baidu` | Chinese news, local content | Chinese |
| `bing` | International news, tech | Multi-language |
| `rss` | Structured feeds, specific sources | Multi-language |

## Usage
The skill is implemented as a standalone Python script that requires no external dependencies (uses standard library only).

### Basic Usage

```bash
# Search using default (DuckDuckGo)
python3 skills/news-search/scripts/search.py "上海交通 2026年2月25日" --max 5

# Search using specific source
python3 skills/news-search/scripts/search.py "上海交通" --source baidu --max 5
python3 skills/news-search/scripts/search.py "Shanghai traffic" --source bing --max 5

# Search using RSS feeds
python3 skills/news-search/scripts/search.py "tech" --source rss --max 5
```

### Advanced Usage

```bash
# Search with category filter
python3 skills/news-search/scripts/search.py "2026年2月" --category finance --max 3

# Combined search (multiple sources)
python3 skills/news-search/scripts/search.py "AI news" --source all --max 3

# Search specific news site
python3 skills/news-search/scripts/search.py "site:36kr.com AI" --source bing --max 3
```

### Supported Categories
Use `--category` flag to narrow down news by topic:
- `tech` (科技): 36Kr, IT之家, 虎嗅
- `finance` (财经): 新浪财经, 华尔街见闻, 第一财经
- `sports` (体育): 新浪体育, 虎扑, 直播吧
- `entertainment` (娱乐): 新浪娱乐, 网易娱乐
- `global` (国际): 环球网国际, 央视国际
- `domestic` (国内): 新浪国内, 央视国内
- `military` (军事): 环球网军事, 新浪军事
- `gaming` (游戏): 游民星空, 机核网, IGN中国

### Output Format
The script outputs a JSON object containing the news search results:

```json
{
  "status": "success",
  "source": "baidu",
  "query": "上海交通 2026年2月25日",
  "results": [
    {
      "title": "上海地铁早高峰限流措施将继续实施",
      "url": "https://www.sohu.com/a/989590849_121117079",
      "snippet": "2月25日，上海地铁早高峰限流措施将继续实施，届时将有2座地铁站早高峰计划限流。",
      "source": "搜狐网"
    }
  ],
  "metadata": {
    "total_results": 5,
    "search_time_ms": 1234
  }
}
```

## Implementation Details

### Architecture
The skill is a lightweight Python script (`scripts/search.py`) that supports multiple search backends:

1. **DuckDuckGo HTML** (`https://html.duckduckgo.com/html/`)
   - Default, no API key required
   - Good for general news

2. **Baidu Search** (`https://www.baidu.com/s`)
   - Best for Chinese language content
   - Good for local/regional news

3. **Bing HTML** (`https://www.bing.com/news/search`)
   - Good for international news
   - High quality news sources

4. **RSS Feed Aggregator**
   - Pre-configured feeds for common news sources
   - Structured data, faster parsing

### Core Components
1. **HTTP Request**: Uses `urllib.request` with proper headers
2. **HTML Parsing**: Uses regular expressions (`re.findall`) for zero dependencies
3. **URL Decoding**: Handles various URL encoding schemes
4. **JSON Output**: Clean JSON format for AI consumption

### RSS Feed Sources (Pre-configured)
```python
rss_feeds = {
    'tech': [
        'https://www.36kr.com/feed',
        'https://www.ithome.com/rss',
    ],
    'finance': [
        'https://finance.sina.com.cn/rss/index.xml',
    ],
    'global': [
        'https://www.huanqiu.com/rss',
    ]
}
```

## Integration
This skill is primarily used by `cortana` and `data-scientist-research-lead` to validate assumptions before generating responses.

### Agent Response Guidelines
When an agent uses this skill to answer a user's query, the agent **MUST** adhere to the following formatting rules:
1. **Cite Sources**: Always append the source URL (and optionally the source name) to each news item
2. **Timestamp**: Always include the exact time the search was performed (precise to the second)
3. **Transparency**: Do not present the news as the agent's own knowledge; explicitly state that it was retrieved via search
4. **Format Example**:
   - "上海地铁早高峰限流措施将继续实施... [信源: 搜狐网](https://www.sohu.com/...) (检索时间: 2026-02-25 13:46:00)"

## Error Handling
The script handles common errors gracefully:
- **Timeout**: Returns error with timeout message
- **No Results**: Returns empty results array with status "success"
- **Network Error**: Returns error with specific exception message
- **Parse Error**: Returns partial results with warning

## Best Practices
1. **Use Baidu for Chinese news**: Better coverage for local Chinese content
2. **Use Bing for international news**: Higher quality English news sources
3. **Use DuckDuckGo for privacy-sensitive queries**: No tracking
4. **Use RSS for specific sources**: When you need news from a particular outlet
5. **Combine sources**: Use `--source all` to get diverse perspectives
</parameter>
<task_progress>
- [x] 阅读当前 news-search skill 实现
- [x] 分析现有能力与可扩展方式
- [x] 设计新的新闻获取方式（支持 baidu, bing, rss）
- [x] 更新 skill 文档
- [ ] 更新搜索脚本
- [ ] 测试新功能
</task_progress>
</invoke>
</minimax:tool_call>
"""Main RSS reader class with concurrent fetching and caching."""

from __future__ import annotations

import logging
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

from .cache import CacheManager
from .exceptions import FeedParseError
from .http_client import HTTPClient
from .models import Article, FeedResult
from .parsers import RSSParser, AtomParser, JSONFeedParser, FeedParser
from .subscriptions import SubscriptionManager, DEFAULT_FEEDS

logger = logging.getLogger('rss_reader')

# Constants
MAX_WORKERS = 5  # concurrent fetch threads
MAX_ARTICLE_CONTENT = 5000  # characters
MAX_FULL_TEXT_ARTICLES = 5

# Try to import optional dependencies
try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False
    logger.debug("BeautifulSoup not installed, using regex-based extraction")


class RSSReader:
    """Main RSS feed reader class with concurrent fetching and caching."""
    
    def __init__(self, cache_dir: Optional[Path] = None, use_cache: bool = True):
        self.http_client = HTTPClient()
        self.parsers: List[FeedParser] = [
            JSONFeedParser(),
            RSSParser(),
            AtomParser(),
        ]
        
        # Setup cache
        cache_path = cache_dir or (Path(__file__).parent.parent / "data" / "cache")
        self.cache = CacheManager(cache_path) if use_cache else None
        self.use_cache = use_cache
    
    def _make_request(self, url: str) -> str:
        """Make HTTP request to fetch feed content."""
        return self.http_client.fetch(url)
    
    def _parse_feed(self, content: str, feed_name: str, feed_url: str) -> List[Article]:
        """Auto-detect and parse feed type."""
        for parser in self.parsers:
            if parser.can_parse(content):
                logger.debug(f"Using {parser.__class__.__name__} for {feed_name}")
                return parser.parse(content, feed_name, feed_url)
        
        # Fallback: try each parser
        for parser in self.parsers:
            try:
                return parser.parse(content, feed_name, feed_url)
            except FeedParseError:
                continue
        
        raise FeedParseError("Unable to parse feed: unknown format")
    
    def fetch_feed(self, url: str, feed_name: Optional[str] = None) -> FeedResult:
        """Fetch and parse an RSS/Atom feed."""
        start_time = time.time()
        
        if feed_name is None:
            parsed = urllib.parse.urlparse(url)
            feed_name = parsed.netloc.replace("www.", "")
        
        # Check cache
        if self.use_cache and self.cache:
            cached = self.cache.get(url)
            if cached:
                articles = [Article(**a) for a in cached.get('articles', [])]
                return FeedResult(
                    feed_name=feed_name,
                    feed_url=url,
                    articles=articles,
                    fetch_time_ms=0
                )
        
        try:
            content = self._make_request(url)
            articles = self._parse_feed(content, feed_name, url)
            
            fetch_time_ms = int((time.time() - start_time) * 1000)
            
            # Cache the result
            if self.use_cache and self.cache:
                self.cache.set(url, {'articles': [a.to_dict() for a in articles]})
            
            return FeedResult(
                feed_name=feed_name,
                feed_url=url,
                articles=articles,
                fetch_time_ms=fetch_time_ms
            )
        
        except Exception as e:
            logger.error(f"Failed to fetch feed {feed_name}: {e}")
            return FeedResult(
                feed_name=feed_name,
                feed_url=url,
                error=str(e),
                fetch_time_ms=int((time.time() - start_time) * 1000)
            )

    def fetch_feed_with_fallback(self, feed: Dict[str, Any]) -> FeedResult:
        """Fetch feed using primary URL, then optional fallback URLs."""
        primary_url = feed.get("url", "")
        feed_name = feed.get("name")
        fallback_urls = feed.get("fallback_urls", []) or []
        candidate_urls = [primary_url] + [u for u in fallback_urls if u and u != primary_url]

        last_error = "no valid candidate url"
        for candidate in candidate_urls:
            result = self.fetch_feed(candidate, feed_name)
            if not result.error:
                # Keep logical feed_url stable as the configured primary URL.
                result.feed_url = primary_url or candidate
                return result
            last_error = result.error

        return FeedResult(
            feed_name=feed_name or "Unknown",
            feed_url=primary_url,
            error=f"all feed URLs failed: {last_error}",
        )
    
    def fetch_feeds_concurrent(self, feeds: List[Dict[str, Any]],
                                max_workers: int = MAX_WORKERS) -> List[FeedResult]:
        """Fetch multiple feeds concurrently."""
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_feed = {
                executor.submit(self.fetch_feed_with_fallback, f): f
                for f in feeds
            }
            
            for future in as_completed(future_to_feed):
                feed = future_to_feed[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error fetching {feed.get('name', feed.get('url'))}: {e}")
                    results.append(FeedResult(
                        feed_name=feed.get('name', 'Unknown'),
                        feed_url=feed.get('url', ''),
                        error=str(e)
                    ))
        
        return results
    
    def get_full_article(self, url: str) -> str:
        """Extract full article content from URL."""
        try:
            html_content = self._make_request(url)
            
            if HAS_BS4:
                return self._extract_with_beautifulsoup(html_content)
            else:
                return self._extract_with_regex(html_content)
        
        except Exception as e:
            logger.error(f"Failed to fetch article {url}: {e}")
            return f"Failed to fetch article: {str(e)}"
    
    def _extract_with_beautifulsoup(self, html_content: str) -> str:
        """Extract article content using BeautifulSoup."""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove unwanted elements
        for elem in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
            elem.decompose()
        
        # Try common article selectors
        selectors = [
            'article',
            '[class*="article"]',
            '[class*="content"]',
            '[class*="post"]',
            '[id*="content"]',
            '[id*="article"]',
            'main',
            '.post-content',
            '.entry-content',
            '.article-content',
            '.article-body',
        ]
        
        for selector in selectors:
            elem = soup.select_one(selector)
            if elem:
                text = elem.get_text(strip=True, separator='\n')
                if len(text) > 200:  # Ensure we got meaningful content
                    return text[:MAX_ARTICLE_CONTENT]
        
        # Fallback: get body text
        body = soup.find('body')
        if body:
            text = body.get_text(strip=True, separator='\n')
            return text[:MAX_ARTICLE_CONTENT]
        
        return ""
    
    def _extract_with_regex(self, html_content: str) -> str:
        """Extract article content using regex (fallback without BeautifulSoup)."""
        import re
        import html as html_module
        
        # Remove script and style tags
        cleaned = re.sub(r'<script[^>]*>.*?</script>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
        cleaned = re.sub(r'<style[^>]*>.*?</style>', '', cleaned, flags=re.DOTALL | re.IGNORECASE)
        cleaned = re.sub(r'<nav[^>]*>.*?</nav>', '', cleaned, flags=re.DOTALL | re.IGNORECASE)
        cleaned = re.sub(r'<header[^>]*>.*?</header>', '', cleaned, flags=re.DOTALL | re.IGNORECASE)
        cleaned = re.sub(r'<footer[^>]*>.*?</footer>', '', cleaned, flags=re.DOTALL | re.IGNORECASE)
        
        # Get body content
        body_match = re.search(r'<body[^>]*>(.*?)</body>', cleaned, re.DOTALL | re.IGNORECASE)
        if body_match:
            body = body_match.group(1)
            # Remove HTML tags
            text = re.sub(r'<[^>]+>', '', body)
            # Decode HTML entities
            text = html_module.unescape(text)
            # Clean whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            return text[:MAX_ARTICLE_CONTENT]
        
        return ""


# =============================================================================
# Utility Functions
# =============================================================================

def parse_date(date_str: Optional[str]):
    """Parse various date formats into datetime object."""
    if not date_str:
        return None
    
    from datetime import datetime
    
    # Common RSS/Atom date formats
    formats = [
        '%a, %d %b %Y %H:%M:%S %z',
        '%a, %d %b %Y %H:%M:%S GMT',
        '%Y-%m-%dT%H:%M:%S%z',
        '%Y-%m-%dT%H:%M:%SZ',
        '%Y-%m-%d %H:%M:%S',
        '%Y-%m-%d',
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    
    return None


def filter_by_date(articles: List[Dict], since: Optional[str] = None,
                   until: Optional[str] = None) -> List[Dict]:
    """Filter articles by date range."""
    if not since and not until:
        return articles
    
    filtered = []
    since_dt = parse_date(since) if since else None
    until_dt = parse_date(until) if until else None
    
    for article in articles:
        pub_date = parse_date(article.get('published'))
        
        if pub_date:
            # Normalize to naive for comparison
            p_date = pub_date.replace(tzinfo=None) if pub_date.tzinfo else pub_date
            s_date = since_dt.replace(tzinfo=None) if since_dt and since_dt.tzinfo else since_dt
            u_date = until_dt.replace(tzinfo=None) if until_dt and until_dt.tzinfo else until_dt

            if s_date and p_date < s_date:
                continue
            if u_date and p_date > u_date:
                continue
        
        filtered.append(article)
    
    return filtered


def output_markdown(articles: List[Dict], include_content: bool = False) -> str:
    """Format articles as markdown."""
    lines = []
    
    for article in articles:
        lines.append(f"## {article['title']}")
        lines.append("")
        
        meta = []
        if article.get('feed_name'):
            meta.append(f"**来源**: {article['feed_name']}")
        if article.get('published'):
            meta.append(f"**发布时间**: {article['published']}")
        if article.get('author'):
            meta.append(f"**作者**: {article['author']}")
        
        if meta:
            lines.append(" | ".join(meta))
            lines.append("")
        
        content = article.get('content') or article.get('summary', '')
        if content:
            # Truncate long content
            if len(content) > 500:
                content = content[:500] + "..."
            lines.append(content)
            lines.append("")
        
        lines.append(f"[阅读原文]({article['url']})")
        lines.append("")
        lines.append("---")
        lines.append("")
    
    return "\n".join(lines)


def get_feeds_to_fetch(manager: SubscriptionManager, args) -> List[Dict[str, str]]:
    """Determine which feeds to fetch based on arguments.
    
    Priority: --use-defaults > --feed > --category > user subscriptions
    
    If user has no subscriptions or specified category has no feeds,
    automatically fallback to default feeds.
    """
    # Priority: --use-defaults > --feed > --category > user subscriptions
    if getattr(args, 'use_defaults', False):
        # Use default feeds by category or all
        if getattr(args, 'category', None) and args.category in DEFAULT_FEEDS:
            return DEFAULT_FEEDS[args.category][:]  # Return a copy
        else:
            feeds = []
            for cat, cat_feeds in DEFAULT_FEEDS.items():
                feeds.extend(cat_feeds)
            return feeds
    
    if getattr(args, 'feed', None):
        return [f for f in manager.subscriptions.get("feeds", []) 
                if args.feed.lower() in f["name"].lower() or args.feed in f["url"]]
    
    # Get user subscription feeds
    user_feeds = manager.subscriptions.get("feeds", [])
    user_categories = manager.subscriptions.get("categories", {})
    
    if getattr(args, 'category', None):
        # First try user's category subscriptions
        cat_urls = user_categories.get(args.category, [])
        category_feeds = [f for f in user_feeds if f["url"] in cat_urls]
        
        # If no user feeds in this category, fallback to default feeds
        if not category_feeds and args.category in DEFAULT_FEEDS:
            return DEFAULT_FEEDS[args.category][:]
        
        return category_feeds
    
    # If no user subscriptions, fallback to all default feeds
    if not user_feeds:
        feeds = []
        for cat, cat_feeds in DEFAULT_FEEDS.items():
            feeds.extend(cat_feeds)
        return feeds
    
    return user_feeds

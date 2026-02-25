"""Data models for RSS reader."""

from __future__ import annotations

import html
import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, List, Dict, Any


@dataclass
class Article:
    """Represents a single article from an RSS feed."""
    title: str
    url: str
    published: Optional[str] = None
    author: Optional[str] = None
    summary: Optional[str] = None
    content: Optional[str] = None
    categories: List[str] = field(default_factory=list)
    feed_name: Optional[str] = None
    feed_url: Optional[str] = None
    fetched_at: Optional[str] = None
    
    def __post_init__(self):
        """Clean up article data after initialization."""
        if self.title:
            self.title = html.unescape(self.title)
        if self.summary:
            self.summary = html.unescape(self.summary)
        if self.fetched_at is None:
            self.fetched_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert article to dictionary."""
        return asdict(self)
    
    @property
    def content_hash(self) -> str:
        """Generate a unique hash for this article."""
        content = f"{self.title}:{self.url}"
        return hashlib.md5(content.encode()).hexdigest()


@dataclass
class FeedResult:
    """Result of fetching a single feed."""
    feed_name: str
    feed_url: str
    articles: List[Article] = field(default_factory=list)
    error: Optional[str] = None
    fetch_time_ms: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "feed_name": self.feed_name,
            "feed_url": self.feed_url,
            "article_count": len(self.articles),
            "error": self.error,
            "fetch_time_ms": self.fetch_time_ms,
            "articles": [a.to_dict() for a in self.articles]
        }

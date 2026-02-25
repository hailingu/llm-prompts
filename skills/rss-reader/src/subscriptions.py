"""Subscription manager for RSS feeds."""

from __future__ import annotations

import json
import logging
import os
import urllib.parse
from pathlib import Path
from typing import Dict, Optional, List, Any

logger = logging.getLogger('rss_reader.subscriptions')

# Try to import YAML support
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    logger.debug("PyYAML not installed, using built-in default feeds")


def _load_default_feeds_from_config() -> Dict[str, List[Dict[str, str]]]:
    """Load default feeds from YAML configuration file."""
    config_path = Path(__file__).parent.parent / "config" / "feeds.yaml"
    
    if not config_path.exists():
        logger.warning(f"Config file not found: {config_path}, using empty defaults")
        return {}
    
    if not HAS_YAML:
        logger.warning("PyYAML not installed, cannot load config file")
        return {}
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Validate and normalize config
        feeds = {}
        if isinstance(config, dict):
            for category, items in config.items():
                if isinstance(items, list):
                    feeds[category] = [
                        {"name": item.get("name", "Unknown"), "url": item["url"]}
                        for item in items
                        if item.get("url")  # Only include items with URL
                    ]
        
        logger.debug(f"Loaded {sum(len(v) for v in feeds.values())} feeds from config")
        return feeds
    
    except Exception as e:
        logger.warning(f"Failed to load config: {e}, using empty defaults")
        return {}


# Try to load from config file, fall back to empty dict
DEFAULT_FEEDS: Dict[str, List[Dict[str, str]]] = _load_default_feeds_from_config()


class SubscriptionManager:
    """Manages RSS feed subscriptions."""
    
    def __init__(self, data_file: Optional[Path] = None):
        self.data_file = data_file or (Path(__file__).parent.parent / "data" / "subscriptions.json")
        self.subscriptions: Dict[str, Any] = {"feeds": [], "categories": {}}
        self._ensure_data_dir()
        self._load()
    
    def _ensure_data_dir(self) -> None:
        """Create data directory if it doesn't exist."""
        self.data_file.parent.mkdir(parents=True, exist_ok=True)
    
    def _load(self) -> None:
        """Load subscriptions from file."""
        if self.data_file.exists():
            try:
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    self.subscriptions = json.load(f)
                logger.debug(f"Loaded {len(self.subscriptions.get('feeds', []))} subscriptions")
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse subscriptions file: {e}")
                self.subscriptions = {"feeds": [], "categories": {}}
    
    def _save(self) -> None:
        """Save subscriptions to file."""
        try:
            with open(self.data_file, 'w', encoding='utf-8') as f:
                json.dump(self.subscriptions, f, ensure_ascii=False, indent=2)
            logger.debug("Saved subscriptions")
        except IOError as e:
            logger.error(f"Failed to save subscriptions: {e}")
            from .exceptions import SubscriptionError
            raise SubscriptionError(f"Failed to save subscriptions: {e}")
    
    def add_feed(self, url: str, name: Optional[str] = None, 
                 category: str = "uncategorized") -> Dict[str, Any]:
        """Add a new feed subscription."""
        # Validate URL
        if not url.startswith(('http://', 'https://')):
            return {"status": "error", "message": "Invalid URL format. Must start with http:// or https://"}
        
        # Check for duplicates
        for feed in self.subscriptions.get("feeds", []):
            if feed["url"] == url:
                return {"status": "error", "message": f"Feed already exists: {feed.get('name', url)}"}
        
        # Extract name from URL if not provided
        if not name:
            parsed = urllib.parse.urlparse(url)
            name = parsed.netloc.replace("www.", "")
        
        feed = {
            "url": url,
            "name": name,
            "category": category,
            "added_at": self._get_current_time()
        }
        
        self.subscriptions.setdefault("feeds", []).append(feed)
        
        # Add to category
        if category not in self.subscriptions.get("categories", {}):
            self.subscriptions.setdefault("categories", {})[category] = []
        if url not in self.subscriptions["categories"][category]:
            self.subscriptions["categories"][category].append(url)
        
        self._save()
        logger.info(f"Added feed: {name} ({url})")
        return {"status": "success", "message": f"Added feed: {name}", "feed": feed}
    
    def _get_current_time(self) -> str:
        """Get current time in ISO format."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def remove_feed(self, url: str) -> Dict[str, Any]:
        """Remove a feed subscription."""
        feeds = self.subscriptions.get("feeds", [])
        original_count = len(feeds)
        
        # Find feed name before removing
        feed_name = next((f.get("name", url) for f in feeds if f["url"] == url), url)
        
        self.subscriptions["feeds"] = [f for f in feeds if f["url"] != url]
        
        # Remove from categories
        for cat_urls in self.subscriptions.get("categories", {}).values():
            if url in cat_urls:
                cat_urls.remove(url)
        
        if len(self.subscriptions["feeds"]) < original_count:
            self._save()
            logger.info(f"Removed feed: {feed_name}")
            return {"status": "success", "message": f"Removed feed: {feed_name}"}
        
        return {"status": "error", "message": f"Feed not found: {url}"}
    
    def list_feeds(self, category: Optional[str] = None) -> Dict[str, Any]:
        """List all feeds or feeds in a specific category."""
        if category:
            cat_feeds = self.subscriptions.get("categories", {}).get(category, [])
            feeds = [f for f in self.subscriptions.get("feeds", []) if f["url"] in cat_feeds]
        else:
            feeds = self.subscriptions.get("feeds", [])
        
        return {
            "status": "success", 
            "feeds": feeds, 
            "categories": list(self.subscriptions.get("categories", {}).keys()),
            "total_count": len(feeds)
        }
    
    def get_categories(self) -> List[str]:
        """Get all categories."""
        return list(self.subscriptions.get("categories", {}).keys())
    
    def get_feed_by_url(self, url: str) -> Optional[Dict[str, Any]]:
        """Get feed info by URL."""
        for feed in self.subscriptions.get("feeds", []):
            if feed["url"] == url:
                return feed
        return None


def get_default_feeds() -> Dict[str, List[Dict[str, str]]]:
    """Get the default feeds configuration."""
    return DEFAULT_FEEDS


def reload_default_feeds() -> None:
    """Reload default feeds from config file."""
    global DEFAULT_FEEDS
    DEFAULT_FEEDS = _load_default_feeds_from_config()
    logger.info(f"Reloaded {sum(len(v) for v in DEFAULT_FEEDS.values())} default feeds from config")

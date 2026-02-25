"""Cache manager for RSS reader."""

from __future__ import annotations

import json
import hashlib
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger('rss_reader.cache')

# Constants
DEFAULT_CACHE_TTL = 3600  # 1 hour


class CacheManager:
    """Manages local caching of feed data for offline access."""
    
    def __init__(self, cache_dir: Path, ttl: int = DEFAULT_CACHE_TTL):
        self.cache_dir = cache_dir
        self.ttl = ttl
        self._ensure_cache_dir()
    
    def _ensure_cache_dir(self) -> None:
        """Create cache directory if it doesn't exist."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_path(self, key: str) -> Path:
        """Get cache file path for a key."""
        safe_key = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{safe_key}.json"
    
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get cached data if not expired."""
        cache_path = self._get_cache_path(key)
        
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Check if cache is expired
            cached_at = datetime.fromisoformat(data.get('cached_at', '2000-01-01'))
            if datetime.now() - cached_at > timedelta(seconds=self.ttl):
                logger.debug(f"Cache expired for key: {key}")
                return None
            
            logger.debug(f"Cache hit for key: {key}")
            return data.get('content')
        
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Cache read error: {e}")
            return None
    
    def set(self, key: str, content: Dict[str, Any]) -> None:
        """Cache data with timestamp."""
        cache_path = self._get_cache_path(key)
        
        data = {
            'cached_at': datetime.now().isoformat(),
            'content': content
        }
        
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.debug(f"Cached data for key: {key}")
        except IOError as e:
            logger.warning(f"Cache write error: {e}")
    
    def clear(self) -> int:
        """Clear all cached data. Returns number of files deleted."""
        count = 0
        for cache_file in self.cache_dir.glob('*.json'):
            try:
                cache_file.unlink()
                count += 1
            except IOError as e:
                logger.warning(f"Failed to delete cache file {cache_file}: {e}")
        logger.info(f"Cleared {count} cache files")
        return count

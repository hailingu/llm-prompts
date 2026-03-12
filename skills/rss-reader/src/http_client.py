"""HTTP client with retry logic for fetching feeds."""

from __future__ import annotations

import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Dict, Optional

from .exceptions import FeedFetchError

logger = logging.getLogger('rss_reader.http')

# Constants
REQUEST_TIMEOUT = 15  # seconds
MAX_RETRIES = 3
RETRY_DELAY = 1  # second


class HTTPClient:
    """HTTP client with retry logic and proper headers."""
    
    DEFAULT_HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'application/rss+xml, application/atom+xml, application/feed+json, application/xml, text/xml, application/json, */*',
        'Accept-Language': 'en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7',
    }
    
    def __init__(self, timeout: int = REQUEST_TIMEOUT, max_retries: int = MAX_RETRIES):
        self.timeout = timeout
        self.max_retries = max_retries
    
    def fetch(self, url: str, headers: Optional[Dict[str, str]] = None) -> str:
        """Fetch URL content with retry logic."""
        request_headers = {**self.DEFAULT_HEADERS, **(headers or {})}
        
        last_error = None
        for attempt in range(self.max_retries):
            try:
                req = urllib.request.Request(url, headers=request_headers)
                response = urllib.request.urlopen(req, timeout=self.timeout)
                
                # Handle encoding
                charset = 'utf-8'
                content_type = response.headers.get('Content-Type', '')
                if 'charset=' in content_type:
                    charset = content_type.split('charset=')[-1].split(';')[0].strip()
                
                content = response.read()
                
                # Try specified charset first, fall back to utf-8
                try:
                    return content.decode(charset)
                except UnicodeDecodeError:
                    return content.decode('utf-8', errors='replace')
            
            except urllib.error.HTTPError as e:
                last_error = FeedFetchError(f"HTTP {e.code}: {e.reason}")
                if e.code >= 400 and e.code < 500:
                    # Client error, don't retry
                    break
            except urllib.error.URLError as e:
                last_error = FeedFetchError(f"URL error: {e.reason}")
            except Exception as e:
                last_error = FeedFetchError(f"Request failed: {e}")
            
            # Wait before retry
            if attempt < self.max_retries - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
        
        raise last_error or FeedFetchError("Unknown error")

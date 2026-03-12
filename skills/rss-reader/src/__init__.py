"""RSS Reader - A powerful RSS/Atom feed reader and subscription manager."""

__version__ = "1.0.0"

# Main exports
from .exceptions import (
    RSSError,
    FeedFetchError,
    FeedParseError,
    SubscriptionError,
    CacheError,
)

from .models import (
    Article,
    FeedResult,
)

from .cache import CacheManager

from .subscriptions import (
    SubscriptionManager,
    DEFAULT_FEEDS,
)

from .parsers import (
    FeedParser,
    RSSParser,
    AtomParser,
    JSONFeedParser,
)

from .http_client import HTTPClient

from .reader import RSSReader

__all__ = [
    # Exceptions
    'RSSError',
    'FeedFetchError',
    'FeedParseError',
    'SubscriptionError',
    'CacheError',
    # Models
    'Article',
    'FeedResult',
    # Managers
    'CacheManager',
    'SubscriptionManager',
    # Parsers
    'FeedParser',
    'RSSParser',
    'AtomParser',
    'JSONFeedParser',
    # HTTP
    'HTTPClient',
    # Reader
    'RSSReader',
    # Constants
    'DEFAULT_FEEDS',
]

"""Custom exceptions for RSS reader."""


class RSSError(Exception):
    """Base exception for RSS reader errors."""
    pass


class FeedFetchError(RSSError):
    """Raised when fetching a feed fails."""
    pass


class FeedParseError(RSSError):
    """Raised when parsing a feed fails."""
    pass


class SubscriptionError(RSSError):
    """Raised for subscription management errors."""
    pass


class CacheError(RSSError):
    """Raised for cache-related errors."""
    pass

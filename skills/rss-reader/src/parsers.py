"""Feed parsers for RSS, Atom, and JSON Feed formats."""

from __future__ import annotations

import json
import logging
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from typing import List, Optional

from .models import Article
from .exceptions import FeedParseError

logger = logging.getLogger('rss_reader.parsers')


class FeedParser(ABC):
    """Abstract base class for feed parsers."""
    
    @abstractmethod
    def can_parse(self, content: str) -> bool:
        """Check if this parser can handle the content."""
        pass
    
    @abstractmethod
    def parse(self, content: str, feed_name: str, feed_url: str) -> List[Article]:
        """Parse the feed content and return articles."""
        pass


class RSSParser(FeedParser):
    """Parser for RSS 2.0 feeds."""
    
    def can_parse(self, content: str) -> bool:
        content = content.strip()
        return content.startswith('<?xml') and '<rss' in content or content.startswith('<rss')
    
    def parse(self, content: str, feed_name: str, feed_url: str) -> List[Article]:
        """Parse RSS 2.0 feed."""
        articles = []
        
        try:
            root = ET.fromstring(content)
            channel = root.find('channel')
            
            for item in root.findall('.//item'):
                try:
                    article = self._parse_item(item, feed_name, feed_url)
                    articles.append(article)
                except Exception as e:
                    logger.warning(f"Failed to parse RSS item: {e}")
                    continue
        
        except ET.ParseError as e:
            raise FeedParseError(f"Failed to parse RSS feed: {e}")
        
        return articles
    
    def _parse_item(self, item: ET.Element, feed_name: str, feed_url: str) -> Article:
        """Parse a single RSS item."""
        title_elem = item.find('title')
        title = title_elem.text if title_elem is not None else "Untitled"
        
        link_elem = item.find('link')
        url = link_elem.text if link_elem is not None else ""
        
        pub_date = item.find('pubDate')
        published = pub_date.text if pub_date is not None else None
        
        # Try different author fields
        author_elem = item.find('author') or item.find('{http://purl.org/dc/elements/1.1/}creator')
        author = author_elem.text if author_elem is not None else None
        
        description = item.find('description')
        summary = description.text if description is not None else None
        
        # Get categories
        categories = [cat.text for cat in item.findall('category') if cat.text]
        
        return Article(
            title=title,
            url=url,
            published=published,
            author=author,
            summary=summary,
            categories=categories,
            feed_name=feed_name,
            feed_url=feed_url
        )


class AtomParser(FeedParser):
    """Parser for Atom 1.0 feeds."""
    
    NS = {'atom': 'http://www.w3.org/2005/Atom'}
    
    def can_parse(self, content: str) -> bool:
        content = content.strip()
        return '<feed' in content and ('xmlns="http://www.w3.org/2005/Atom"' in content or 'atom:' in content)
    
    def parse(self, content: str, feed_name: str, feed_url: str) -> List[Article]:
        """Parse Atom 1.0 feed."""
        articles = []
        
        try:
            root = ET.fromstring(content)
            
            for entry in self._find_entries(root):
                try:
                    article = self._parse_entry(entry, feed_name, feed_url)
                    articles.append(article)
                except Exception as e:
                    logger.warning(f"Failed to parse Atom entry: {e}")
                    continue
        
        except ET.ParseError as e:
            raise FeedParseError(f"Failed to parse Atom feed: {e}")
        
        return articles
    
    def _find_entries(self, root: ET.Element) -> List[ET.Element]:
        """Find all entry elements."""
        # Try with namespace first
        entries = root.findall('atom:entry', self.NS)
        if not entries:
            # Try without namespace
            entries = root.findall('entry')
        return entries
    
    def _parse_entry(self, entry: ET.Element, feed_name: str, feed_url: str) -> Article:
        """Parse a single Atom entry."""
        title_elem = self._find_element(entry, 'title')
        title = title_elem.text if title_elem is not None else "Untitled"
        
        # Get link
        url = self._get_link(entry)
        
        # Published date
        published_elem = self._find_element(entry, 'published') or self._find_element(entry, 'updated')
        published = published_elem.text if published_elem is not None else None
        
        # Author
        author_elem = self._find_element(entry, 'author')
        if author_elem is not None:
            name_elem = self._find_element(author_elem, 'name')
            author = name_elem.text if name_elem is not None else None
        else:
            author = None
        
        # Summary
        summary_elem = self._find_element(entry, 'summary') or self._find_element(entry, 'content')
        summary = summary_elem.text if summary_elem is not None else None
        
        # Categories
        categories = []
        for cat in entry.findall('atom:category', self.NS) + entry.findall('category'):
            term = cat.get('term') or cat.get('label')
            if term:
                categories.append(term)
        
        return Article(
            title=title,
            url=url,
            published=published,
            author=author,
            summary=summary,
            categories=categories,
            feed_name=feed_name,
            feed_url=feed_url
        )
    
    def _find_element(self, parent: ET.Element, tag: str) -> Optional[ET.Element]:
        """Find element with or without namespace."""
        elem = parent.find(f'atom:{tag}', self.NS)
        if elem is None:
            elem = parent.find(tag)
        return elem
    
    def _get_link(self, entry: ET.Element) -> str:
        """Get the best link from an entry."""
        # Try with namespace
        for link in entry.findall('atom:link', self.NS):
            rel = link.get('rel')
            if rel == 'alternate' or rel is None:
                return link.get('href', '')
        
        # Try without namespace
        for link in entry.findall('link'):
            rel = link.get('rel')
            if rel == 'alternate' or rel is None:
                return link.get('href', '')
        
        return ''


class JSONFeedParser(FeedParser):
    """Parser for JSON Feed format (https://jsonfeed.org/)."""
    
    def can_parse(self, content: str) -> bool:
        content = content.strip()
        if content.startswith('{'):
            try:
                data = json.loads(content)
                return data.get('version', '').startswith('https://jsonfeed.org/')
            except json.JSONDecodeError:
                pass
        return False
    
    def parse(self, content: str, feed_name: str, feed_url: str) -> List[Article]:
        """Parse JSON Feed."""
        articles = []
        
        try:
            data = json.loads(content)
            items = data.get('items', [])
            
            for item in items:
                article = Article(
                    title=item.get('title', 'Untitled'),
                    url=item.get('url', item.get('external_url', '')),
                    published=item.get('date_published'),
                    author=item.get('author', {}).get('name') if isinstance(item.get('author'), dict) else item.get('author'),
                    summary=item.get('summary'),
                    content=item.get('content_html') or item.get('content_text'),
                    categories=item.get('tags', []),
                    feed_name=feed_name,
                    feed_url=feed_url
                )
                articles.append(article)
        
        except json.JSONDecodeError as e:
            raise FeedParseError(f"Failed to parse JSON feed: {e}")
        
        return articles

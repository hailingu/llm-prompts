"""CLI commands for RSS reader."""

from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Callable, Any

from .cache import CacheManager
from .reader import RSSReader, output_markdown, get_feeds_to_fetch
from .subscriptions import SubscriptionManager, DEFAULT_FEEDS


def cmd_add(args) -> int:
    """Add a new feed subscription."""
    manager = SubscriptionManager()
    result = manager.add_feed(args.url, args.name, args.category)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "success" else 1


def cmd_list(args) -> int:
    """List feed subscriptions."""
    manager = SubscriptionManager()
    result = manager.list_feeds(args.category)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def cmd_remove(args) -> int:
    """Remove a feed subscription."""
    manager = SubscriptionManager()
    result = manager.remove_feed(args.url)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "success" else 1


def cmd_fetch(args) -> int:
    """Fetch articles from feeds."""
    manager = SubscriptionManager()
    reader = RSSReader(use_cache=not args.no_cache)
    
    # Determine which feeds to fetch
    feeds_to_fetch = get_feeds_to_fetch(manager, args)
    
    if not feeds_to_fetch:
        print(json.dumps({"status": "error", "message": "No feeds found"}))
        return 1
    
    # Fetch feeds concurrently
    results = reader.fetch_feeds_concurrent(feeds_to_fetch)
    
    # Aggregate articles
    all_articles = []
    for result in results:
        if result.error:
            print(f"Warning: {result.feed_name}: {result.error}", file=sys.stderr)
            continue
        for article in result.articles:
            article_dict = article.to_dict()
            article_dict["category"] = next(
                (f.get('category', 'unknown') for f in feeds_to_fetch if f['url'] == result.feed_url),
                'unknown'
            )
            all_articles.append(article_dict)
    
    # Sort by published date
    all_articles.sort(key=lambda x: x.get("published", ""), reverse=True)
    
    # Filter by date if specified
    from .reader import filter_by_date
    if args.since or args.until:
        all_articles = filter_by_date(all_articles, args.since, args.until)
    
    # Apply limit
    if args.limit:
        all_articles = all_articles[:args.limit]
    
    # Get full text if requested
    if args.full_text and all_articles:
        for article in all_articles[:5]:  # MAX_FULL_TEXT_ARTICLES
            try:
                article["content"] = reader.get_full_article(article["url"])
            except Exception as e:
                import logging
                logging.getLogger('rss_reader').warning(f"Failed to get full text for {article['url']}: {e}")
                article["content"] = article.get("summary", "")
    
    # Output results
    if args.markdown:
        print(output_markdown(all_articles, include_content=args.full_text))
    else:
        result = {
            "status": "success",
            "total_articles": len(all_articles),
            "feeds_fetched": len([r for r in results if not r.error]),
            "feeds_failed": len([r for r in results if r.error]),
            "articles": all_articles
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
    
    return 0


def cmd_search(args) -> int:
    """Search articles by keyword."""
    manager = SubscriptionManager()
    reader = RSSReader()
    
    all_articles = []
    
    # Fetch from all feeds
    for feed in manager.subscriptions.get("feeds", []):
        try:
            result = reader.fetch_feed(feed["url"], feed["name"])
            for article in result.articles:
                article_dict = article.to_dict()
                article_dict["category"] = feed.get("category", "unknown")
                all_articles.append(article_dict)
        except Exception:
            continue
    
    # Filter by keyword
    keyword = args.keyword.lower()
    filtered = [a for a in all_articles 
                if keyword in a.get("title", "").lower() 
                or keyword in a.get("summary", "").lower()
                or keyword in a.get("content", "").lower()]
    
    # Sort and limit
    filtered.sort(key=lambda x: x.get("published", ""), reverse=True)
    filtered = filtered[:args.limit]
    
    result = {
        "status": "success",
        "query": args.keyword,
        "total_results": len(filtered),
        "articles": filtered
    }
    
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def cmd_export(args) -> int:
    """Export subscriptions to OPML."""
    manager = SubscriptionManager()
    
    opml_lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<opml version="2.0">',
        '<head><title>RSS Subscriptions</title></head>',
        '<body>'
    ]
    
    for category, urls in manager.subscriptions.get("categories", {}).items():
        opml_lines.append(f'  <outline text="{category}" title="{category}">')
        for url in urls:
            feed = manager.get_feed_by_url(url)
            if feed:
                name = feed.get("name", url)
                opml_lines.append(f'    <outline type="rss" text="{name}" title="{name}" xmlUrl="{url}"/>')
        opml_lines.append('  </outline>')
    
    opml_lines.extend(['</body>', '</opml>'])
    
    output_file = args.output or "subscriptions.opml"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(opml_lines))
    
    print(json.dumps({"status": "success", "message": f"Exported to {output_file}"}))
    return 0


def cmd_import(args) -> int:
    """Import subscriptions from OPML."""
    manager = SubscriptionManager()
    
    try:
        with open(args.file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        root = ET.fromstring(content)
        imported_count = 0
        
        # Parse OPML - handle nested outlines
        def parse_outline(outline_elem: ET.Element, category: str = "uncategorized"):
            nonlocal imported_count
            
            # If this outline has xmlUrl, it's a feed
            url = outline_elem.get('xmlUrl')
            if url:
                name = outline_elem.get('title') or outline_elem.get('text', 'Unknown')
                result = manager.add_feed(url, name, category)
                if result["status"] == "success":
                    imported_count += 1
            else:
                # This is a category outline
                new_category = outline_elem.get('text') or outline_elem.get('title', category)
                for child in outline_elem.findall('outline'):
                    parse_outline(child, new_category)
        
        # Find all top-level outlines
        body = root.find('body')
        if body is not None:
            for outline in body.findall('outline'):
                parse_outline(outline)
        
        print(json.dumps({
            "status": "success", 
            "message": f"Imported {imported_count} feeds from {args.file}"
        }))
        return 0
    
    except FileNotFoundError:
        print(json.dumps({"status": "error", "message": f"File not found: {args.file}"}))
        return 1
    except ET.ParseError as e:
        print(json.dumps({"status": "error", "message": f"Failed to parse OPML: {e}"}))
        return 1
    except Exception as e:
        print(json.dumps({"status": "error", "message": f"Failed to import: {str(e)}"}))
        return 1


def cmd_cache(args) -> int:
    """Manage cache."""
    cache_dir = Path(__file__).parent.parent / "data" / "cache"
    
    if args.clear:
        cache = CacheManager(cache_dir)
        count = cache.clear()
        print(json.dumps({"status": "success", "message": f"Cleared {count} cache files"}))
        return 0
    
    # Show cache stats
    cache_files = list(cache_dir.glob('*.json')) if cache_dir.exists() else []
    total_size = sum(f.stat().st_size for f in cache_files) if cache_files else 0
    
    print(json.dumps({
        "status": "success",
        "cache_dir": str(cache_dir),
        "file_count": len(cache_files),
        "total_size_bytes": total_size,
        "total_size_mb": round(total_size / (1024 * 1024), 2)
    }))
    return 0


def cmd_categories(args) -> int:
    """List available categories."""
    manager = SubscriptionManager()
    categories = manager.get_categories()
    
    # Add default categories
    all_categories = set(categories) | set(DEFAULT_FEEDS.keys())
    
    result = {
        "status": "success",
        "user_categories": categories,
        "default_categories": list(DEFAULT_FEEDS.keys()),
        "all_categories": sorted(list(all_categories))
    }
    
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0

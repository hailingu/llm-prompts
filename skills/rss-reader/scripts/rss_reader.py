#!/usr/bin/env python3
"""
RSS/Atom Feed Reader and Subscription Manager
Main entry point for CLI.

Provides feed fetching, parsing, and subscription management capabilities.

Features:
- RSS 2.0, Atom 1.0, and JSON Feed support
- Subscription management with categories
- Full-text article extraction
- OPML import/export
- Concurrent feed fetching
- Local caching for offline access
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.cli import (
    cmd_add,
    cmd_list,
    cmd_remove,
    cmd_fetch,
    cmd_search,
    cmd_export,
    cmd_import,
    cmd_cache,
    cmd_categories,
)


def main() -> int:
    """Main entry point for RSS reader CLI."""
    parser = argparse.ArgumentParser(
        description='RSS/Atom Feed Reader and Subscription Manager',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Add a feed
  %(prog)s add "https://www.36kr.com/feed" --category tech

  # Fetch articles from all subscriptions
  %(prog)s fetch --limit 20

  # Fetch from specific category
  %(prog)s fetch --category tech --markdown

  # Use default feeds
  %(prog)s fetch --use-defaults --category ai

  # Search articles
  %(prog)s search "AI" --limit 10
"""
    )
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Add command
    parser_add = subparsers.add_parser('add', help='Add a new RSS feed subscription')
    parser_add.add_argument('url', type=str, help='RSS feed URL')
    parser_add.add_argument('--name', type=str, help='Feed name (optional)')
    parser_add.add_argument('--category', type=str, default='uncategorized', help='Category')
    
    # List command
    parser_list = subparsers.add_parser('list', help='List feed subscriptions')
    parser_list.add_argument('--category', type=str, help='Filter by category')
    
    # Remove command
    parser_remove = subparsers.add_parser('remove', help='Remove a feed subscription')
    parser_remove.add_argument('url', type=str, help='Feed URL to remove')
    
    # Fetch command
    parser_fetch = subparsers.add_parser('fetch', help='Fetch articles from feeds')
    parser_fetch.add_argument('--category', type=str, help='Filter by category')
    parser_fetch.add_argument('--feed', type=str, help='Filter by feed name/URL')
    parser_fetch.add_argument('--limit', type=int, default=20, help='Maximum articles to return')
    parser_fetch.add_argument('--full-text', action='store_true', help='Fetch full article content')
    parser_fetch.add_argument('--markdown', action='store_true', help='Output as markdown')
    parser_fetch.add_argument('--since', type=str, help='Fetch articles since date (YYYY-MM-DD)')
    parser_fetch.add_argument('--until', type=str, help='Fetch articles until date (YYYY-MM-DD)')
    parser_fetch.add_argument('--use-defaults', action='store_true', help='Use default feeds instead of subscriptions')
    parser_fetch.add_argument('--no-cache', action='store_true', help='Disable cache')
    
    # Search command
    parser_search = subparsers.add_parser('search', help='Search articles by keyword')
    parser_search.add_argument('keyword', type=str, help='Search keyword')
    parser_search.add_argument('--limit', type=int, default=10, help='Maximum results')
    
    # Export command
    parser_export = subparsers.add_parser('export', help='Export subscriptions to OPML')
    parser_export.add_argument('--output', type=str, help='Output file name')
    
    # Import command
    parser_import = subparsers.add_parser('import', help='Import subscriptions from OPML')
    parser_import.add_argument('--file', type=str, required=True, help='OPML file to import')
    
    # Cache command
    parser_cache = subparsers.add_parser('cache', help='Manage cache')
    parser_cache.add_argument('--clear', action='store_true', help='Clear all cache')
    
    # Categories command
    parser_categories = subparsers.add_parser('categories', help='List available categories')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Dispatch to command handler
    commands = {
        'add': cmd_add,
        'list': cmd_list,
        'remove': cmd_remove,
        'fetch': cmd_fetch,
        'search': cmd_search,
        'export': cmd_export,
        'import': cmd_import,
        'cache': cmd_cache,
        'categories': cmd_categories,
    }
    
    handler = commands.get(args.command)
    if handler:
        return handler(args)
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())

#!/usr/bin/env python3
"""
Memory Manager - Persistent memory management for AI agents.

Provides file-based memory persistence across sessions.
"""

import os
import json
import argparse
import datetime
from pathlib import Path
import sys

class MemoryManager:
    """File-based memory management system."""
    
    def __init__(self, workspace_root=None):
        self.workspace_root = workspace_root or os.getcwd()
        self.memory_dir = os.path.join(self.workspace_root, "memory")
        self.global_memory_file = os.path.join(self.workspace_root, "MEMORY.md")
        
        # Ensure directories exist
        os.makedirs(self.memory_dir, exist_ok=True)
    
    def get_theme_path(self, theme, timestamp=None):
        """Get path for theme-based memory file."""
        if timestamp is None:
            timestamp = datetime.datetime.now()
        
        # Format: YYYY-MM-DD_HH (hour granularity)
        time_str = timestamp.strftime("%Y-%m-%d_%H")
        theme_dir = os.path.join(self.memory_dir, theme)
        os.makedirs(theme_dir, exist_ok=True)
        
        return os.path.join(theme_dir, f"{time_str}.md")
    
    def read_global_memory(self):
        """Read global long-term memory."""
        if not os.path.exists(self.global_memory_file):
            return "# Global Memory\n\n*No global memory yet.*"
        
        with open(self.global_memory_file, 'r', encoding='utf-8') as f:
            return f.read()
    
    def write_global_memory(self, content, append=True):
        """Write to global long-term memory."""
        if append and os.path.exists(self.global_memory_file):
            with open(self.global_memory_file, 'a', encoding='utf-8') as f:
                f.write(f"\n\n{content}")
        else:
            with open(self.global_memory_file, 'w', encoding='utf-8') as f:
                f.write(content)
    
    def read_theme_memory(self, theme, hours_back=24):
        """Read recent theme-based memories."""
        theme_dir = os.path.join(self.memory_dir, theme)
        if not os.path.exists(theme_dir):
            return []
        
        files = sorted(os.listdir(theme_dir), reverse=True)
        memories = []
        
        # Get recent files (within specified hours)
        cutoff = datetime.datetime.now() - datetime.timedelta(hours=hours_back)
        
        for file in files:
            if not file.endswith('.md'):
                continue
            
            # Parse timestamp from filename
            try:
                time_str = file.replace('.md', '')
                file_time = datetime.datetime.strptime(time_str, "%Y-%m-%d_%H")
                
                if file_time >= cutoff:
                    file_path = os.path.join(theme_dir, file)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    memories.append({
                        'timestamp': file_time.isoformat(),
                        'filename': file,
                        'content': content
                    })
            except ValueError:
                continue
        
        return memories
    
    def write_theme_memory(self, theme, content, timestamp=None):
        """Write to theme-based memory."""
        file_path = self.get_theme_path(theme, timestamp)
        
        # Add timestamp header
        now = timestamp or datetime.datetime.now()
        timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")
        formatted_content = f"## {timestamp_str}\n\n{content}\n"
        
        # Append to existing file or create new
        if os.path.exists(file_path):
            with open(file_path, 'a', encoding='utf-8') as f:
                f.write(f"\n\n{formatted_content}")
        else:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(formatted_content)
        
        return file_path
    
    def list_themes(self):
        """List all available memory themes."""
        if not os.path.exists(self.memory_dir):
            return []
        
        themes = []
        for item in os.listdir(self.memory_dir):
            item_path = os.path.join(self.memory_dir, item)
            if os.path.isdir(item_path):
                themes.append(item)
        
        return sorted(themes)
    
    def search_memories(self, query, max_results=10):
        """Search across all memories for matching content."""
        results = []
        
        # Search global memory
        global_content = self.read_global_memory()
        if query.lower() in global_content.lower():
            results.append({
                'type': 'global',
                'match': 'MEMORY.md',
                'content': global_content[:200] + '...' if len(global_content) > 200 else global_content
            })
        
        # Search theme memories
        for theme in self.list_themes():
            memories = self.read_theme_memory(theme, hours_back=168)  # 1 week
            
            for memory in memories:
                if query.lower() in memory['content'].lower():
                    results.append({
                        'type': 'theme',
                        'theme': theme,
                        'timestamp': memory['timestamp'],
                        'match': f"{theme}/{memory['filename']}",
                        'content': memory['content'][:200] + '...' if len(memory['content']) > 200 else memory['content']
                    })
                    
                    if len(results) >= max_results:
                        break
            
            if len(results) >= max_results:
                break
        
        return results

def main():
    """CLI interface for memory manager."""
    parser = argparse.ArgumentParser(
        description='Memory Manager for AI Agents',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s read-global
  %(prog)s read-theme --theme agent-optimization
  %(prog)s write-theme --theme stock-tracker --content "Created stock price tracker skill"
  %(prog)s list-themes
  %(prog)s search --query "stock"
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Read global memory
    read_global_parser = subparsers.add_parser('read-global', help='Read global memory')
    
    # Write global memory
    write_global_parser = subparsers.add_parser('write-global', help='Write to global memory')
    write_global_parser.add_argument('--content', required=True, help='Content to write')
    write_global_parser.add_argument('--append', action='store_true', help='Append to existing content')
    
    # Read theme memory
    read_theme_parser = subparsers.add_parser('read-theme', help='Read theme memory')
    read_theme_parser.add_argument('--theme', required=True, help='Theme name')
    read_theme_parser.add_argument('--hours-back', type=int, default=24, help='Hours to look back')
    
    # Write theme memory
    write_theme_parser = subparsers.add_parser('write-theme', help='Write to theme memory')
    write_theme_parser.add_argument('--theme', required=True, help='Theme name')
    write_theme_parser.add_argument('--content', required=True, help='Content to write')
    
    # List themes
    list_parser = subparsers.add_parser('list-themes', help='List all themes')
    
    # Search memories
    search_parser = subparsers.add_parser('search', help='Search across memories')
    search_parser.add_argument('--query', required=True, help='Search query')
    search_parser.add_argument('--max-results', type=int, default=10, help='Maximum results')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    manager = MemoryManager()
    
    try:
        if args.command == 'read-global':
            result = manager.read_global_memory()
            print(json.dumps({'status': 'success', 'content': result}, ensure_ascii=False))
        
        elif args.command == 'write-global':
            manager.write_global_memory(args.content, args.append)
            print(json.dumps({'status': 'success', 'message': 'Global memory updated'}, ensure_ascii=False))
        
        elif args.command == 'read-theme':
            memories = manager.read_theme_memory(args.theme, args.hours_back)
            print(json.dumps({'status': 'success', 'theme': args.theme, 'memories': memories}, ensure_ascii=False))
        
        elif args.command == 'write-theme':
            file_path = manager.write_theme_memory(args.theme, args.content)
            print(json.dumps({'status': 'success', 'message': f'Themed memory written to {file_path}'}, ensure_ascii=False))
        
        elif args.command == 'list-themes':
            themes = manager.list_themes()
            print(json.dumps({'status': 'success', 'themes': themes}, ensure_ascii=False))
        
        elif args.command == 'search':
            results = manager.search_memories(args.query, args.max_results)
            print(json.dumps({'status': 'success', 'query': args.query, 'results': results}, ensure_ascii=False))
    
    except Exception as e:
        print(json.dumps({'status': 'error', 'message': str(e)}, ensure_ascii=False))
        sys.exit(1)

if __name__ == '__main__':
    main()
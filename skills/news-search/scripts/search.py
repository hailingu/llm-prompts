#!/usr/bin/env python3
import sys
import json
import urllib.request
import urllib.parse
import re
import argparse

def search_duckduckgo(query, category=None, max_results=3):
    actual_query = f"{query} {category}新闻" if category else query
    # Using DuckDuckGo Lite for better stability
    url = 'https://lite.duckduckgo.com/lite/?q=' + urllib.parse.quote(actual_query)
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
    
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            html = response.read().decode('utf-8')
        
        # In Lite version, results are in tables
        # Titles and URLs are in <a class="result-link" href="...">...</a>
        # Snippets are in <td class="result-snippet">...</td>
        
        links = re.findall(r'<a[^>]+class="result-link"[^>]+href="([^"]+)"[^>]*>([\s\S]*?)</a>', html)
        snippets_raw = re.findall(r'<td class="result-snippet">([\s\S]*?)</td>', html)
        
        results = []
        for (u, t), s in zip(links[:max_results], snippets_raw[:max_results]):
            # Clean up redirect URLs
            final_url = u
            if 'uddg=' in u:
                try:
                    encoded_part = u.split('uddg=')[1].split('&')[0]
                    final_url = urllib.parse.unquote(encoded_part)
                except Exception:
                    pass
            
            # Ensure URL is absolute
            if final_url.startswith('//'):
                final_url = 'https:' + final_url
            elif final_url.startswith('/'):
                final_url = 'https://duckduckgo.com' + final_url
                
            results.append({
                "title": re.sub(r'<[^>]+>', '', t).strip(),
                "url": final_url,
                "snippet": re.sub(r'<[^>]+>', '', s).strip()
            })
            
        return {"status": "success", "query": actual_query, "results": results}
        
    except Exception as e:
        return {"status": "error", "query": actual_query, "error_message": str(e)}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DuckDuckGo Lite Search for AI Agents')
    parser.add_argument('query', type=str, help='Search query (e.g., date or topic)')
    parser.add_argument('--category', type=str, choices=['tech', 'finance', 'sports', 'entertainment', 'global', 'domestic', 'military', 'gaming'], help='News category to append to the query')
    parser.add_argument('--max', type=int, default=3, help='Maximum number of results to return')
    
    args = parser.parse_args()
    
    category_map = {
        'tech': '科技',
        'finance': '财经',
        'sports': '体育',
        'entertainment': '娱乐',
        'global': '国际',
        'domestic': '国内',
        'military': '军事',
        'gaming': '游戏'
    }
    
    # We remove site_filter to keep it broader
    cat_keyword = category_map.get(args.category) if args.category else None
    
    result = search_duckduckgo(args.query, cat_keyword, args.max)
    print(json.dumps(result, ensure_ascii=False, indent=2))

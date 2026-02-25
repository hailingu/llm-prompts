#!/usr/bin/env python3
import sys
import json
import urllib.request
import urllib.parse
import re
import argparse

def search_duckduckgo(query, category=None, max_results=3):
    actual_query = f"{query} {category}新闻" if category else query
    url = 'https://html.duckduckgo.com/html/?q=' + urllib.parse.quote(actual_query)
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
    
    try:
        html = urllib.request.urlopen(req, timeout=10).read().decode('utf-8')
        
        # Extract titles, URLs, and snippets
        titles = re.findall(r'<a class="result__url" href="[^"]+">\s*([^<]+)\s*</a>', html)
        urls = re.findall(r'<a class="result__url" href="([^"]+)">', html)
        snippets = re.findall(r'<a class="result__snippet[^>]+>\s*([^<]+)\s*</a>', html)
        
        results = []
        for t, u, s in zip(titles[:max_results], urls[:max_results], snippets[:max_results]):
            # Clean up DuckDuckGo redirect URLs if present
            if u.startswith('//duckduckgo.com/l/?uddg='):
                u = urllib.parse.unquote(u.split('uddg=')[1].split('&')[0])
                
            results.append({
                "title": t.strip(),
                "url": u.strip(),
                "snippet": s.strip()
            })
            
        return {"status": "success", "query": actual_query, "results": results}
        
    except Exception as e:
        return {"status": "error", "query": actual_query, "error_message": str(e)}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DuckDuckGo HTML Search for AI Agents')
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
    
    site_map = {
        'tech': 'site:36kr.com OR site:ithome.com OR site:huxiu.com',
        'finance': 'site:finance.sina.com.cn OR site:wallstreetcn.com OR site:yicai.com',
        'sports': 'site:sports.sina.com.cn OR site:hupu.com OR site:zhibo8.cc',
        'entertainment': 'site:ent.sina.com.cn OR site:163.com/ent',
        'global': 'site:world.huanqiu.com OR site:news.cctv.com/world',
        'domestic': 'site:news.sina.com.cn/china OR site:news.cctv.com/china',
        'military': 'site:mil.huanqiu.com OR site:mil.news.sina.com.cn',
        'gaming': 'site:gamersky.com OR site:gcores.com OR site:ign.com.cn'
    }
    
    cat_keyword = category_map.get(args.category) if args.category else None
    site_filter = site_map.get(args.category) if args.category else None
    
    # Append site filter to query if category is specified
    final_query = f"{args.query} {site_filter}" if site_filter else args.query
    
    result = search_duckduckgo(final_query, cat_keyword, args.max)
    print(json.dumps(result, ensure_ascii=False, indent=2))

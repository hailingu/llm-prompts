#!/usr/bin/env python3
"""News search utility for AI agents.

Key capabilities:
- Multi-source discovery (DuckDuckGo Lite + RSS feeds)
- Optional browser-based retrieval (Playwright) for article extraction
- Paywall-aware stop policy: stop deep extraction once paywall is detected
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple


DEFAULT_TIMEOUT = 12
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

CATEGORY_MAP = {
    "tech": "科技",
    "finance": "财经",
    "sports": "体育",
    "entertainment": "娱乐",
    "global": "国际",
    "domestic": "国内",
    "military": "军事",
    "gaming": "游戏",
}

RSS_FEEDS: Dict[str, List[str]] = {
    "tech": [
        "https://www.36kr.com/feed",
        "https://www.ithome.com/rss/",
        "https://sspai.com/feed",
    ],
    "finance": [
        "https://finance.sina.com.cn/rss/index.xml",
        "https://www.yicai.com/rss",
        "https://www.cnbc.com/id/10001147/device/rss/rss.html",
    ],
    "global": [
        "https://feeds.bbci.co.uk/news/world/rss.xml",
        "https://feeds.npr.org/1004/rss.xml",
    ],
    "domestic": [
        "https://news.sina.com.cn/rss/china.xml",
        "https://www.zaobao.com.sg/rss.xml",
    ],
    "ai": [
        "https://openai.com/news/rss.xml",
        "https://www.anthropic.com/news/rss.xml",
        "https://huggingface.co/blog/feed.xml",
    ],
}

PAYWALL_PATTERNS = [
    r"subscribe to continue",
    r"subscription required",
    r"sign in to continue reading",
    r"paid\s+content",
    r"members?-only",
    r"登录后阅读",
    r"订阅后阅读",
    r"付费阅读",
    r"会员专享",
    r"剩余\d+%",
    r"请开通会员",
    r"试看结束",
]


@dataclass
class NewsItem:
    title: str
    url: str
    snippet: str = ""
    source: str = ""
    retrieval_mode: str = "none"
    access: str = "open"  # open | paywalled | unknown
    paywall_reason: str = ""


def _request(url: str, timeout: int = DEFAULT_TIMEOUT) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
        ctype = resp.headers.get("Content-Type", "")
        charset = "utf-8"
        if "charset=" in ctype:
            charset = ctype.split("charset=")[-1].split(";")[0].strip()
        try:
            return data.decode(charset, errors="replace")
        except LookupError:
            return data.decode("utf-8", errors="replace")


def _clean_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _ddg_search(query: str, max_results: int) -> List[NewsItem]:
    url = "https://lite.duckduckgo.com/lite/?q=" + urllib.parse.quote(query)
    html = _request(url)

    links = re.findall(r'<a[^>]+class="result-link"[^>]+href="([^"]+)"[^>]*>([\s\S]*?)</a>', html)
    snippets = re.findall(r'<td class="result-snippet">([\s\S]*?)</td>', html)

    results: List[NewsItem] = []
    for idx, link in enumerate(links[: max_results * 3]):
        raw_url, raw_title = link
        final_url = raw_url

        if "uddg=" in raw_url:
            try:
                encoded = raw_url.split("uddg=")[1].split("&")[0]
                final_url = urllib.parse.unquote(encoded)
            except Exception:
                final_url = raw_url

        if final_url.startswith("//"):
            final_url = "https:" + final_url
        elif final_url.startswith("/"):
            final_url = "https://duckduckgo.com" + final_url

        snippet = _clean_text(snippets[idx]) if idx < len(snippets) else ""
        title = _clean_text(raw_title)
        if not title or not final_url.startswith(("http://", "https://")):
            continue

        source = urllib.parse.urlparse(final_url).netloc.replace("www.", "")
        results.append(NewsItem(title=title, url=final_url, snippet=snippet, source=source))

    return results[:max_results]


def _rss_search(query: str, category: Optional[str], max_results: int) -> List[NewsItem]:
    keyword = query.lower()
    feeds: List[str] = []

    if category and category in RSS_FEEDS:
        feeds.extend(RSS_FEEDS[category])
    else:
        for group in RSS_FEEDS.values():
            feeds.extend(group)

    results: List[NewsItem] = []
    for feed_url in feeds:
        try:
            content = _request(feed_url)
            root = ET.fromstring(content)
        except Exception:
            continue

        # RSS 2.0
        for item in root.findall(".//item"):
            title = (item.findtext("title") or "").strip()
            link = (item.findtext("link") or "").strip()
            desc = (item.findtext("description") or "").strip()
            hay = f"{title} {desc}".lower()
            if keyword and keyword not in hay:
                continue
            if title and link:
                results.append(
                    NewsItem(
                        title=_clean_text(title),
                        url=link,
                        snippet=_clean_text(desc)[:220],
                        source=urllib.parse.urlparse(link).netloc.replace("www.", ""),
                    )
                )

        # Atom
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        for entry in root.findall(".//atom:entry", ns):
            title = (entry.findtext("atom:title", default="", namespaces=ns) or "").strip()
            summary = (entry.findtext("atom:summary", default="", namespaces=ns) or "").strip()
            link_elem = entry.find("atom:link", ns)
            href = ""
            if link_elem is not None:
                href = (link_elem.attrib.get("href") or "").strip()
            hay = f"{title} {summary}".lower()
            if keyword and keyword not in hay:
                continue
            if title and href:
                results.append(
                    NewsItem(
                        title=_clean_text(title),
                        url=href,
                        snippet=_clean_text(summary)[:220],
                        source=urllib.parse.urlparse(href).netloc.replace("www.", ""),
                    )
                )

        if len(results) >= max_results * 2:
            break

    return results[:max_results]


def _detect_paywall(text: str) -> Tuple[bool, str]:
    low = text.lower()
    for pattern in PAYWALL_PATTERNS:
        if re.search(pattern, low, flags=re.I):
            return True, pattern
    return False, ""


def _enrich_http(item: NewsItem) -> NewsItem:
    try:
        html = _request(item.url)
    except Exception as exc:
        item.retrieval_mode = "http"
        item.access = "unknown"
        item.paywall_reason = f"fetch_error:{exc}"
        return item

    text_probe = _clean_text(html)[:4000]
    is_paywalled, reason = _detect_paywall(text_probe)

    title_match = re.search(r"<title[^>]*>([\s\S]*?)</title>", html, flags=re.I)
    meta_desc = re.search(
        r'<meta[^>]+name=["\']description["\'][^>]+content=["\']([\s\S]*?)["\']',
        html,
        flags=re.I,
    )

    if title_match:
        item.title = _clean_text(title_match.group(1)) or item.title
    if meta_desc:
        item.snippet = _clean_text(meta_desc.group(1))[:240] or item.snippet

    item.retrieval_mode = "http"
    if is_paywalled:
        item.access = "paywalled"
        item.paywall_reason = reason
    else:
        item.access = "open"
    return item


def _enrich_browser(items: List[NewsItem], stop_on_paywall: bool, timeout: int) -> Tuple[List[NewsItem], Optional[str]]:
    try:
        from playwright.sync_api import sync_playwright
    except Exception as exc:
        return [_enrich_http(i) for i in items], f"playwright_unavailable:{exc}"

    enriched: List[NewsItem] = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        context = browser.new_context(user_agent=USER_AGENT)
        page = context.new_page()

        for item in items:
            try:
                page.goto(item.url, wait_until="domcontentloaded", timeout=timeout * 1000)
                page.wait_for_timeout(400)

                title = page.title() or item.title
                body_text = page.locator("body").inner_text(timeout=3000)
                snippet = re.sub(r"\s+", " ", body_text).strip()[:260]
                is_paywalled, reason = _detect_paywall(body_text[:5000])

                item.title = title.strip() if title else item.title
                if snippet:
                    item.snippet = snippet
                item.retrieval_mode = "browser"

                if is_paywalled:
                    item.access = "paywalled"
                    item.paywall_reason = reason
                    if stop_on_paywall:
                        # Stop deep extraction for this URL; continue next result.
                        enriched.append(item)
                        continue
                else:
                    item.access = "open"

            except Exception as exc:
                item.retrieval_mode = "browser"
                item.access = "unknown"
                item.paywall_reason = f"browser_error:{exc}"

            enriched.append(item)

        context.close()
        browser.close()

    return enriched, None


def _dedupe(items: List[NewsItem]) -> List[NewsItem]:
    seen = set()
    deduped: List[NewsItem] = []
    for item in items:
        key = item.url.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def search_news(
    query: str,
    source: str,
    category: Optional[str],
    max_results: int,
    retrieval: str,
    stop_on_paywall: bool,
    timeout: int,
) -> Dict[str, object]:
    started = time.time()

    source_results: Dict[str, List[NewsItem]] = {}
    if source in ("duckduckgo", "all"):
        actual_query = f"{query} {CATEGORY_MAP.get(category, '')}".strip()
        source_results["duckduckgo"] = _ddg_search(actual_query, max_results=max_results)

    if source in ("rss", "all"):
        source_results["rss"] = _rss_search(query, category=category, max_results=max_results)

    merged: List[NewsItem] = []
    for _, items in source_results.items():
        merged.extend(items)

    merged = _dedupe(merged)[:max_results]

    warnings: List[str] = []
    if retrieval == "browser":
        merged, warn = _enrich_browser(merged, stop_on_paywall=stop_on_paywall, timeout=timeout)
        if warn:
            warnings.append(warn)
    elif retrieval == "http":
        merged = [_enrich_http(i) for i in merged]
    else:
        for i in merged:
            i.retrieval_mode = "none"
            i.access = "unknown"

    paywalled_count = sum(1 for i in merged if i.access == "paywalled")

    return {
        "status": "success",
        "query": query,
        "source": source,
        "category": category,
        "retrieval": retrieval,
        "stop_on_paywall": stop_on_paywall,
        "results": [asdict(i) for i in merged],
        "metadata": {
            "total_results": len(merged),
            "paywalled_count": paywalled_count,
            "search_time_ms": int((time.time() - started) * 1000),
            "warnings": warnings,
            "searched_sources": list(source_results.keys()),
        },
    }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="News search with browser/paywall-aware retrieval")
    parser.add_argument("query", type=str, help="Search query")
    parser.add_argument(
        "--source",
        type=str,
        default="all",
        choices=["duckduckgo", "rss", "all"],
        help="Discovery source",
    )
    parser.add_argument(
        "--category",
        type=str,
        choices=["tech", "finance", "sports", "entertainment", "global", "domestic", "military", "gaming", "ai"],
        help="Optional category filter",
    )
    parser.add_argument("--max", type=int, default=5, help="Maximum number of results")
    parser.add_argument(
        "--retrieval",
        type=str,
        default="browser",
        choices=["browser", "http", "none"],
        help="How to retrieve article detail after discovery",
    )
    parser.add_argument(
        "--allow-paywalled",
        action="store_true",
        help="Allow deep retrieval even if paywall markers are detected",
    )
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="Request timeout in seconds")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    try:
        result = search_news(
            query=args.query,
            source=args.source,
            category=args.category,
            max_results=max(1, args.max),
            retrieval=args.retrieval,
            stop_on_paywall=not args.allow_paywalled,
            timeout=max(5, args.timeout),
        )
    except Exception as exc:
        result = {
            "status": "error",
            "query": args.query,
            "error_message": str(exc),
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 1

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("status") == "success" else 1


if __name__ == "__main__":
    sys.exit(main())

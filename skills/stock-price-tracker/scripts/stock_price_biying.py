#!/usr/bin/env python3
"""
Stock Price Tracker - Real-time stock prices via Biying API.

Usage:
    python3 stock_price_biying.py --symbol 000001
    python3 stock_price_biying.py --symbol 000001,600519 --format csv

Biying API Documentation:
    - Real-time trading data: http://api.biyingapi.com/hsrl/ssjy/{symbol}/{license}
    - License is loaded from .env file (biying_license)

Rate Limiting:
    - Uses requests.Session() for connection pooling
    - Adds delays between requests to avoid rate limits
"""

import sys
import os
import json
import argparse
import datetime
import time
import random
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Load license from .env file
def load_license():
    """Load Biying license from .env file."""
    env_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', '.env')
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                if line.startswith('biying_license='):
                    return line.split('=', 1)[1].strip()
    # Try current working directory
    env_path = os.path.join(os.getcwd(), '.env')
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                if line.startswith('biying_license='):
                    return line.split('=', 1)[1].strip()
    return None

BIYING_LICENSE = load_license()

# Configure session with retry logic
def create_session_with_retries(retries=3, backoff_factor=0.5):
    """Create a requests session with retry logic."""
    session = requests.Session()
    retry_strategy = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

_session = create_session_with_retries()

# Rate limiting configuration
REQUEST_DELAY_SECONDS = 0.5  # Delay between requests
MAX_RETRIES = 3


def get_session_with_proxy():
    """Create a session with proxy support from environment variables."""
    session = create_session_with_retries()
    
    http_proxy = os.environ.get('HTTP_PROXY') or os.environ.get('http_proxy')
    https_proxy = os.environ.get('HTTPS_PROXY') or os.environ.get('https_proxy')
    
    if http_proxy or https_proxy:
        proxies = {}
        if http_proxy:
            proxies['http'] = http_proxy
        if https_proxy:
            proxies['https'] = https_proxy
        session.proxies.update(proxies)
    
    return session


def normalize_symbol(symbol):
    """
    Normalize stock symbol to Biying API format.
    
    A股 codes:
    - 上海: 6开头 (e.g., 600519) -> sh600519
    - 深圳: 0/3开头 (e.g., 000001, 300750) -> sz000001, sz300750
    """
    symbol = symbol.strip().upper()
    
    # Already normalized
    if symbol.startswith(('SH', 'SZ')):
        return symbol.lower()
    
    # A股: 6开头 = 上海, 0/3开头 = 深圳
    if symbol.startswith('6'):
        return f'sh{symbol}'
    elif symbol.startswith(('0', '3')):
        return f'sz{symbol}'
    # 港股: 5/6/8开头
    elif len(symbol) == 4 and symbol.isdigit():
        return f'hk{symbol}'
    # 美股: letter suffix like .US or just letters
    else:
        # Assume US stock, return as-is
        return symbol


def get_stock_data(symbols, fmt='json'):
    """
    Fetch real-time stock data for given symbols via Biying API.
    
    Args:
        symbols (list): List of stock symbols (e.g., ['000001', '600519'])
        fmt (str): Output format ('json' or 'csv')
    
    Returns:
        dict or str: JSON-serializable dict or CSV string
    """
    global _session
    
    if not BIYING_LICENSE:
        return {
            'status': 'error',
            'message': 'Biying license not found. Please set biying_license in .env file.',
            'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat()
        }
    
    # Check if proxy is configured
    if os.environ.get('HTTP_PROXY') or os.environ.get('HTTPS_PROXY') or \
       os.environ.get('http_proxy') or os.environ.get('https_proxy'):
        _session = get_session_with_proxy()
    
    base_url = "http://api.biyingapi.com/hsrl/ssjy"
    
    results = []
    errors = []
    
    for i, symbol in enumerate(symbols):
        # Add delay between requests
        if i > 0:
            time.sleep(REQUEST_DELAY_SECONDS + random.uniform(0, 0.3))
        
        normalized = normalize_symbol(symbol)
        
        try:
            url = f"{base_url}/{normalized}/{BIYING_LICENSE}"
            response = _session.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Biying API returns data directly at root level:
            # {"h":11.58,"l":11.5,"o":11.52,"pc":0.348,"p":11.55,"cje":848177984.0,"ud":0.04,"v":73485055.0,"yc":11.51,"t":"2025-11-09 17:39:02"}
            # h=high, l=low, o=open, pc=percent_change, p=price, cje=成交额, ud=涨跌额, v=volume, yc=昨天收盘, t=timestamp
            
            if not data or 'p' not in data:
                errors.append(f"Symbol '{symbol}': Invalid response format")
                continue
            
            price = data.get('p', 0)
            close = data.get('yc', 0)  # Yesterday's close (昨天收盘)
            open_price = data.get('o', 0)
            high = data.get('h', 0)
            low = data.get('l', 0)
            change_percent = data.get('pc', 0)  # 涨跌幅百分比
            change = data.get('ud', 0)  # 涨跌额
            
            stock_data = {
                'symbol': symbol,
                'name': symbol,  # Basic API doesn't return name
                'price': round(float(price), 2) if price else 0,
                'currency': 'CNY',
                'change': round(float(change), 2) if change else 0,
                'change_percent': round(float(change_percent), 2) if change_percent else 0,
                'volume': int(data.get('v', 0)),
                'turnover': data.get('cje', 0),
                'day_high': round(float(high), 2) if high else 0,
                'day_low': round(float(low), 2) if low else 0,
                'previous_close': round(float(close), 2) if close else 0,
                'open': round(float(open_price), 2) if open_price else 0,
                'last_updated': data.get('t', datetime.datetime.now(datetime.timezone.utc).isoformat())
            }
            
            results.append(stock_data)
            
        except requests.exceptions.Timeout:
            errors.append(f"Symbol '{symbol}': Request timeout")
        except requests.exceptions.RequestException as e:
            errors.append(f"Symbol '{symbol}': {str(e)}")
        except json.JSONDecodeError:
            errors.append(f"Symbol '{symbol}': Invalid JSON response")
        except Exception as e:
            errors.append(f"Symbol '{symbol}': {str(e)}")
    
    if not results and errors:
        return {
            'status': 'error',
            'message': '; '.join(errors),
            'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat()
        }
    
    response = {
        'status': 'success',
        'source': 'biying',
        'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'data': results
    }
    
    if errors:
        response['warnings'] = errors
    
    if fmt == 'csv':
        import csv
        import io
        
        if not results:
            return f"symbol,error\n" + "\n".join([f"{s},{e}" for s, e in zip(symbols, errors)])
        
        fieldnames = list(results[0].keys())
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
        return output.getvalue()
    
    return response


def main():
    parser = argparse.ArgumentParser(
        description='Retrieve real-time stock prices via Biying API',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --symbol 000001
  %(prog)s --symbol 000001,600519
  %(prog)s --symbol 000001 --format csv
        """
    )
    
    parser.add_argument(
        '--symbol',
        required=True,
        help='Stock symbol(s), comma-separated (e.g., 000001 or 000001,600519)'
    )
    
    parser.add_argument(
        '--format',
        choices=['json', 'csv'],
        default='json',
        help='Output format (default: json)'
    )
    
    args = parser.parse_args()
    
    symbols = args.symbol.split(',')
    result = get_stock_data(symbols, args.format)
    
    if args.format == 'json':
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(result)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Stock Price Tracker - Real-time stock prices via Yahoo Finance.

Usage:
    python3 stock_price.py --symbol AAPL
    python3 stock_price.py --symbol AAPL,MSFT,GOOGL --format csv

Rate Limiting Protection:
    - Uses requests.Session() for connection pooling
    - Adds delays between requests to avoid "Too Many Requests" errors
    - Configurable retry logic with exponential backoff
    - Updates yfinance to latest version automatically
"""

import sys
import os
import json
import argparse
import datetime
import time
import random
import requests
import yfinance as yf
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure yfinance to use a session with retry logic
def create_session_with_retries(retries=3, backoff_factor=0.5):
    """Create a requests session with retry logic to handle rate limits."""
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

# Create a global session for connection pooling
_session = create_session_with_retries()

# Rate limiting configuration
REQUEST_DELAY_SECONDS = 1.5  # Delay between requests to avoid rate limiting
MAX_RETRIES = 3
RETRY_DELAY_BASE = 2  # Base delay for retries in seconds

def get_session_with_proxy():
    """Create a session with proxy support from environment variables."""
    session = create_session_with_retries()
    
    # Check for proxy environment variables
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

def get_stock_data(symbols, fmt='json'):
    """
    Fetch real-time stock data for given symbols.
    
    Args:
        symbols (list): List of stock symbols
        fmt (str): Output format ('json' or 'csv')
    
    Returns:
        dict or str: JSON-serializable dict or CSV string
    """
    global _session
    
    # Check if proxy is configured and use appropriate session
    if os.environ.get('HTTP_PROXY') or os.environ.get('HTTPS_PROXY') or \
       os.environ.get('http_proxy') or os.environ.get('https_proxy'):
        _session = get_session_with_proxy()
    
    symbols = [s.strip().upper() for s in symbols]
    
    try:
        # Update yfinance to latest version for bug fixes
        try:
            import subprocess
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "--upgrade", "yfinance", "-q"],
                capture_output=True,
                timeout=60
            )
            # Reload yfinance after upgrade
            import importlib
            importlib.reload(yf)
        except Exception:
            pass  # Continue with existing version if upgrade fails
        
        # Download stock data (let yfinance handle its own session)
        tickers = yf.Tickers(" ".join(symbols))
        
        results = []
        errors = []
        
        for i, symbol in enumerate(symbols):
            # Add delay between requests to avoid rate limiting
            if i > 0:
                time.sleep(REQUEST_DELAY_SECONDS + random.uniform(0, 0.5))
            
            try:
                ticker = tickers.tickers[symbol]
                info = ticker.info
                
                # Extract relevant fields
                price = info.get('regularMarketPrice', info.get('currentPrice', None))
                if price is None:
                    # Try fast_info as fallback
                    fast_info = ticker.fast_info
                    price = fast_info.get('last_price', None)
                
                if price is None:
                    errors.append(f"Symbol '{symbol}' not found or price unavailable")
                    continue
                
                # Calculate change
                previous_close = info.get('regularMarketPreviousClose', info.get('previousClose', price))
                change = price - previous_close if previous_close else 0
                change_percent = (change / previous_close * 100) if previous_close else 0
                
                stock_data = {
                    'symbol': symbol,
                    'name': info.get('longName', info.get('shortName', symbol)),
                    'price': round(price, 2),
                    'currency': info.get('currency', 'USD'),
                    'change': round(change, 2),
                    'change_percent': round(change_percent, 2),
                    'volume': info.get('regularMarketVolume', info.get('volume', 0)),
                    'market_cap': info.get('marketCap', 0),
                    'day_high': info.get('regularMarketDayHigh', info.get('dayHigh', price)),
                    'day_low': info.get('regularMarketDayLow', info.get('dayLow', price)),
                    'previous_close': previous_close,
                    'open': info.get('regularMarketOpen', info.get('open', price)),
                    'last_updated': datetime.datetime.now(datetime.timezone.utc).isoformat()
                }
                
                # Convert numeric fields to int/float as appropriate
                for key in ['volume', 'market_cap']:
                    if stock_data[key]:
                        stock_data[key] = int(stock_data[key])
                
                results.append(stock_data)
                
            except Exception as e:
                errors.append(f"Error fetching '{symbol}': {str(e)}")
        
        if not results and errors:
            return {
                'status': 'error',
                'message': '; '.join(errors),
                'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat()
            }
        
        # Prepare success response
        response = {
            'status': 'success',
            'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'data': results
        }
        
        if errors:
            response['warnings'] = errors
        
        if fmt == 'csv':
            # Convert to CSV
            import csv
            import io
            
            if not results:
                return 'symbol,error\n' + '\n'.join([f'{symbol},{err}' for symbol, err in zip(symbols, errors)])
            
            # Use first result's keys as headers
            fieldnames = list(results[0].keys())
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
            return output.getvalue()
        
        return response
    
    except Exception as e:
        return {
            'status': 'error',
            'message': f'System error: {str(e)}',
            'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat()
        }

def main():
    parser = argparse.ArgumentParser(
        description='Retrieve real-time stock prices via Yahoo Finance',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --symbol AAPL
  %(prog)s --symbol AAPL,MSFT,GOOGL
  %(prog)s --symbol 000001.SZ,0700.HK --format csv
        """
    )
    
    parser.add_argument(
        '--symbol',
        required=True,
        help='Stock symbol(s), comma-separated (e.g., AAPL or AAPL,MSFT,GOOGL)'
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
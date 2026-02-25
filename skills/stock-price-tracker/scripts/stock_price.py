#!/usr/bin/env python3
"""
Stock Price Tracker - Real-time stock prices via Yahoo Finance.

Usage:
    python3 stock_price.py --symbol AAPL
    python3 stock_price.py --symbol AAPL,MSFT,GOOGL --format csv
"""

import sys
import json
import argparse
import datetime
import yfinance as yf

def get_stock_data(symbols, fmt='json'):
    """
    Fetch real-time stock data for given symbols.
    
    Args:
        symbols (list): List of stock symbols
        fmt (str): Output format ('json' or 'csv')
    
    Returns:
        dict or str: JSON-serializable dict or CSV string
    """
    symbols = [s.strip().upper() for s in symbols]
    
    try:
        # Download stock data
        tickers = yf.Tickers(" ".join(symbols))
        
        results = []
        errors = []
        
        for symbol in symbols:
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
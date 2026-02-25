---
name: stock-price-tracker
description: "Retrieve real-time stock prices via Yahoo Finance API. Supports multiple stock symbols, currency conversion, and clean JSON output for agent processing."
metadata:
  version: 1.0.0
  author: cortana
---

# Stock Price Tracker

## Overview
This skill provides real-time stock price retrieval using Yahoo Finance's public API. It is designed for AI agents (like Cortana) to obtain accurate, up-to-date stock prices without manual web browsing. The output is clean JSON that can be directly consumed by other agents or tools.

## Why this Skill?
When an AI agent needs to discuss financial markets, investment strategies, or company valuations, it requires current stock prices. This skill:
1. **Provides Real-Time Data**: Fetches live prices, day change, volume, and market cap.
2. **Standardized Output**: Returns structured JSON with consistent fields.
3. **Multi-Symbol Support**: Query multiple stocks in a single request.
4. **No API Key Required**: Uses Yahoo Finance's free public endpoint.

> **Note**: This skill is for real-time price lookup only. For historical data or technical indicators, consider extending the skill or using dedicated financial data APIs.

## Usage
The skill is implemented as a Python script that uses the `yfinance` library. Ensure you have Python 3.7+ and the required dependencies installed.

### Installation
```bash
# Install required Python packages
pip install yfinance pandas
```

### CLI Commands
```bash
# Get real-time price for a single stock symbol
python3 skills/stock-price-tracker/scripts/stock_price.py --symbol AAPL

# Get prices for multiple symbols
python3 skills/stock-price-tracker/scripts/stock_price.py --symbol AAPL,MSFT,GOOGL

# Specify exchange suffix (e.g., Shanghai .SS, Shenzhen .SZ, Hong Kong .HK)
python3 skills/stock-price-tracker/scripts/stock_price.py --symbol 000001.SZ,0700.HK

# Output in JSON format (default)
python3 skills/stock-price-tracker/scripts/stock_price.py --symbol AAPL --format json

# Output in CSV format
python3 skills/stock-price-tracker/scripts/stock_price.py --symbol AAPL --format csv
```

## Output Format

### JSON Output Example
```json
{
  "status": "success",
  "timestamp": "2026-02-25T14:30:00Z",
  "data": [
    {
      "symbol": "AAPL",
      "name": "Apple Inc.",
      "price": 178.32,
      "currency": "USD",
      "change": 1.24,
      "change_percent": 0.70,
      "volume": 58201300,
      "market_cap": 2790000000000,
      "day_high": 179.50,
      "day_low": 177.10,
      "previous_close": 177.08,
      "open": 177.50,
      "last_updated": "2026-02-25T14:30:00Z"
    }
  ]
}
```

### CSV Output Example
```csv
symbol,name,price,currency,change,change_percent,volume,market_cap,day_high,day_low,previous_close,open,last_updated
AAPL,Apple Inc.,178.32,USD,1.24,0.70,58201300,2790000000000,179.50,177.10,177.08,177.50,2026-02-25T14:30:00Z
```

## Error Handling
If a symbol cannot be found or an error occurs, the skill returns:
```json
{
  "status": "error",
  "message": "Symbol 'INVALID' not found",
  "timestamp": "2026-02-25T14:30:00Z"
}
```

## Integration with Agents
Agents can call this skill via subprocess or Python's `submodule.run`. Example:

```python
import subprocess
import json

def get_stock_price(symbol):
    result = subprocess.run(
        ["python3", "skills/stock-price-tracker/scripts/stock_price.py", "--symbol", symbol],
        capture_output=True,
        text=True
    )
    return json.loads(result.stdout)

# Use in agent logic
price_data = get_stock_price("AAPL")
if price_data["status"] == "success":
    print(f"Current price: {price_data['data'][0]['price']}")
```

## Dependencies
- Python 3.7+
- `yfinance` >= 0.2.0
- `pandas` >= 1.3.0

## License
This skill is part of the llm-prompts project and follows the same licensing terms.
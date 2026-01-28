#!/usr/bin/env python3
"""
Download and manage delisted/suspended stock data for survivorship bias correction.

Survivorship bias occurs when analysis only includes currently successful companies.
This leads to overly optimistic ML models that don't account for failure cases.

Usage:
    python scripts/download_delisted_stocks.py --list
    python scripts/download_delisted_stocks.py --export
    python scripts/download_delisted_stocks.py --download --years 30
    python scripts/download_delisted_stocks.py --download --symbols RCOM,UNITECH --years 30
    python scripts/download_delisted_stocks.py --download --list-file data/delisted/delisted_list.csv
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
DELISTED_DIR = DATA_DIR / 'delisted'
DELISTED_PRICES_DIR = DELISTED_DIR / 'prices'
DELISTED_INFO_DIR = DELISTED_DIR / 'info'


class DelistedStockProvider:
    """Handles delisted/suspended stock data for survivorship bias correction."""
    
    KNOWN_DELISTED = {
        'RCOM': ('2022-12-12', 'bankruptcy', 1.85, 800),
        'UNITECH': ('2020-04-01', 'fraud_investigation', 3.50, 520),
        'JAYPEEINFRA': ('2021-01-01', 'bankruptcy', 0.95, 100),
        'RELCAPITAL': ('2021-12-01', 'bankruptcy', 8.50, 950),
        'DHFL': ('2020-11-01', 'fraud_bankruptcy', 15.0, 690),
        'VIDEOCON': ('2021-08-01', 'bankruptcy', 5.0, 400),
        'SATYAM': ('2009-04-01', 'fraud', 50.0, 550),
    }
    
    def __init__(self):
        DELISTED_DIR.mkdir(parents=True, exist_ok=True)
        DELISTED_PRICES_DIR.mkdir(parents=True, exist_ok=True)
        DELISTED_INFO_DIR.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, pd.DataFrame] = {}
    
    def get_delisted_stocks(self) -> List[str]:
        """Get list of all delisted/suspended stocks."""
        symbols = set(self.KNOWN_DELISTED.keys())
        
        for csv_file in DELISTED_DIR.glob('*.csv'):
            try:
                df = pd.read_csv(csv_file)
                if 'symbol' in df.columns:
                    symbols.update(df['symbol'].dropna().tolist())
            except Exception:
                pass
        
        return sorted(list(symbols))

    def get_symbols_from_file(self, list_file: Path) -> List[str]:
        """Read delisted symbols from a CSV/JSON list."""
        if not list_file.exists():
            return []
        try:
            if list_file.suffix.lower() == '.json':
                data = json.loads(list_file.read_text())
                if isinstance(data, dict) and 'symbols' in data:
                    return [s.strip().upper() for s in data['symbols'] if s]
                if isinstance(data, list):
                    return [s.strip().upper() for s in data if s]
                return []
            df = pd.read_csv(list_file)
            if 'symbol' in df.columns:
                return [s.strip().upper() for s in df['symbol'].dropna().astype(str)]
            return [s.strip().upper() for s in df.iloc[:, 0].dropna().astype(str)]
        except Exception as e:
            logger.warning(f"Could not read symbols from {list_file}: {e}")
            return []

    def download_symbol_list(self, list_url: str, output_file: Path) -> Optional[Path]:
        """Download a delisted symbol list from a URL to a local file."""
        try:
            import requests
        except Exception:
            logger.error("requests not installed. Run: pip install requests")
            return None

        try:
            resp = requests.get(list_url, timeout=30)
            if resp.status_code != 200:
                logger.warning(f"Failed to download list ({resp.status_code})")
                return None
            output_file.write_bytes(resp.content)
            logger.info(f"Downloaded delisted list to {output_file}")
            return output_file
        except Exception as e:
            logger.warning(f"Could not download list: {e}")
            return None
    
    def get_delisted_stock_info(self, symbol: str) -> Optional[Dict]:
        """Get information about a delisted stock."""
        if symbol in self.KNOWN_DELISTED:
            date, reason, last_price, peak_price = self.KNOWN_DELISTED[symbol]
            return {
                'symbol': symbol,
                'delisting_date': date,
                'reason': reason,
                'last_known_price': last_price,
                'peak_price': peak_price,
                'loss_from_peak': (1 - last_price / peak_price) * 100 if peak_price > 0 else 100
            }
        return None
    
    def get_training_weight(self, symbol: str) -> float:
        """Get sample weight for training."""
        if symbol in self.KNOWN_DELISTED:
            reason = self.KNOWN_DELISTED[symbol][1]
            if 'fraud' in reason:
                return 3.0
            elif reason == 'bankruptcy':
                return 2.0
            else:
                return 1.5
        return 1.0
    
    def mark_training_data_with_survival_label(self, df: pd.DataFrame, symbol_column: str = 'symbol') -> pd.DataFrame:
        """Add survival label to training data."""
        delisted_symbols = set(self.get_delisted_stocks())
        df = df.copy()
        df['survived'] = ~df[symbol_column].isin(delisted_symbols)
        return df
    
    def export_delisted_summary(self) -> pd.DataFrame:
        """Export summary of all delisted stocks."""
        records = []
        for symbol in self.get_delisted_stocks():
            info = self.get_delisted_stock_info(symbol)
            if info:
                records.append(info)
        
        df = pd.DataFrame(records)
        output_file = DELISTED_DIR / 'delisted_summary.csv'
        df.to_csv(output_file, index=False)
        logger.info(f"Exported summary to {output_file}")
        return df

    def download_price_history(
        self,
        symbol: str,
        years: int = 30,
        suffixes: Tuple[str, ...] = (".NS", ".BO", "")
    ) -> Optional[pd.DataFrame]:
        """Download historical prices for a symbol using yfinance."""
        try:
            import yfinance as yf
        except Exception:
            logger.error("yfinance not installed. Run: pip install yfinance")
            return None

        end_date = datetime.now()
        start_date = end_date - timedelta(days=years * 365)

        for suffix in suffixes:
            ticker = yf.Ticker(f"{symbol}{suffix}")
            try:
                df = ticker.history(start=start_date, end=end_date)
                if df is None or df.empty:
                    df = ticker.history(period=f"{years}y")
                if df is None or df.empty:
                    continue

                df = df.reset_index()
                df = df.rename(columns={
                    'Date': 'date',
                    'Open': 'open',
                    'High': 'high',
                    'Low': 'low',
                    'Close': 'close',
                    'Volume': 'volume'
                })
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                return df[['date', 'open', 'high', 'low', 'close', 'volume']]
            except Exception:
                continue

        return None

    def download_info(self, symbol: str, suffixes: Tuple[str, ...] = (".NS", ".BO", "")) -> Dict[str, Any]:
        """Download metadata for a delisted symbol using yfinance."""
        try:
            import yfinance as yf
        except Exception:
            return {'symbol': symbol}

        for suffix in suffixes:
            try:
                ticker = yf.Ticker(f"{symbol}{suffix}")
                info = ticker.info or {}
                if info:
                    return {
                        'symbol': symbol,
                        'name': info.get('longName', info.get('shortName', symbol)),
                        'sector': info.get('sector', 'Unknown'),
                        'industry': info.get('industry', 'Unknown'),
                        'market_cap': info.get('marketCap', 0),
                        'last_known_price': info.get('currentPrice', info.get('regularMarketPrice', 0)),
                        'pe_ratio': info.get('trailingPE'),
                        'pb_ratio': info.get('priceToBook'),
                        'dividend_yield': (info.get('dividendYield', 0) or 0) * 100,
                        'roe': (info.get('returnOnEquity') or 0) * 100 if info.get('returnOnEquity') else None,
                        'debt_to_equity': info.get('debtToEquity'),
                        'source': f"yfinance{suffix}",
                    }
            except Exception:
                continue

        return {'symbol': symbol}

    def download_delisted_prices(
        self,
        symbols: List[str],
        years: int = 30,
        throttle_seconds: float = 0.3
    ) -> None:
        """Download and store prices + info for all delisted symbols."""
        failures = []
        for i, symbol in enumerate(symbols, 1):
            logger.info(f"[{i}/{len(symbols)}] Downloading {symbol}")
            df = self.download_price_history(symbol, years=years)
            if df is None or df.empty:
                failures.append(symbol)
                continue

            price_path = DELISTED_PRICES_DIR / f"{symbol}.parquet"
            df.to_parquet(price_path, index=False)

            info = self.get_delisted_stock_info(symbol) or self.download_info(symbol)
            info_path = DELISTED_INFO_DIR / f"{symbol}.json"
            info_path.write_text(json.dumps(info, indent=2))

            time.sleep(throttle_seconds)

        if failures:
            failure_file = DELISTED_DIR / "download_failures.csv"
            pd.DataFrame({'symbol': failures}).to_csv(failure_file, index=False)
            logger.warning(f"Failed to download {len(failures)} symbols. See {failure_file}")


def main():
    parser = argparse.ArgumentParser(description='Manage delisted stock data')
    parser.add_argument('--list', action='store_true', help='List all known delisted stocks')
    parser.add_argument('--export', action='store_true', help='Export delisted stocks summary')
    parser.add_argument('--download', action='store_true', help='Download delisted prices and info')
    parser.add_argument('--years', type=int, default=30, help='Years of history to fetch (default: 30)')
    parser.add_argument('--symbols', type=str, help='Comma-separated symbols to download')
    parser.add_argument('--list-file', type=str, help='CSV/JSON file with delisted symbols')
    parser.add_argument('--list-url', type=str, help='URL to CSV/JSON list with delisted symbols')
    
    args = parser.parse_args()
    provider = DelistedStockProvider()
    
    if args.list:
        stocks = provider.get_delisted_stocks()
        print(f"\nKnown delisted stocks: {len(stocks)}")
        for symbol in stocks:
            info = provider.get_delisted_stock_info(symbol)
            if info:
                print(f"  {symbol}: {info['reason']} - Loss: {info['loss_from_peak']:.1f}%")
    
    elif args.export:
        df = provider.export_delisted_summary()
        print(f"\nExported {len(df)} stocks")

    elif args.download:
        default_list = DATA_DIR / 'delisted_list.csv'
        symbols = provider.get_delisted_stocks()
        list_file = Path(args.list_file) if args.list_file else (default_list if default_list.exists() else None)
        if args.list_url:
            list_file = provider.download_symbol_list(
                args.list_url,
                DELISTED_DIR / 'delisted_list.csv'
            )
        if list_file:
            symbols = provider.get_symbols_from_file(list_file)
        if args.symbols:
            symbols = [s.strip().upper() for s in args.symbols.split(',') if s.strip()]

        if not symbols:
            print("No delisted symbols found. Provide --list-file or --symbols.")
            return

        provider.download_delisted_prices(symbols, years=args.years)
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()

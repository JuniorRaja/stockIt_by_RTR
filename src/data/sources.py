"""Multi-source data ingestion for Stocron by RTR.

Priority:
1. Local database/parquet files (for offline operation)
2. Yahoo Finance API (for online refresh)
3. Jugaad Data (backup)
4. NSE Tools (for stock info only)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
import logging
import time
import json

from ..utils.config import get_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Data directories
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
PRICE_DIR = DATA_DIR / 'prices'
INFO_DIR = DATA_DIR / 'stock_info'
DB_PATH = DATA_DIR / 'db' / 'equity_intelligence.db'


class LocalDataSource:
    """Local data source - reads from parquet files and database."""
    
    def __init__(self):
        self._db_conn = None
        self._stock_info_cache = {}
        survivorship = get_config('survivorship_bias', {}) or {}
        self._include_delisted = survivorship.get('include_delisted', False)
        delisted_dir = survivorship.get('delisted_data_dir', 'data/delisted')
        self._delisted_dir = (PROJECT_ROOT / delisted_dir).resolve()

    def _get_delisted_price_file(self, symbol: str) -> Optional[Path]:
        if not self._include_delisted or not self._delisted_dir.exists():
            return None
        for subdir in ['prices', 'price_history', '']:
            base = self._delisted_dir / subdir if subdir else self._delisted_dir
            for ext in ['.parquet', '.csv']:
                candidate = base / f"{symbol}{ext}"
                if candidate.exists():
                    return candidate
        return None

    def _get_delisted_info_file(self, symbol: str) -> Optional[Path]:
        if not self._include_delisted or not self._delisted_dir.exists():
            return None
        for subdir in ['info', 'stock_info', '']:
            base = self._delisted_dir / subdir if subdir else self._delisted_dir
            candidate = base / f"{symbol}.json"
            if candidate.exists():
                return candidate
        return None
    
    def _get_db(self):
        """Get database connection."""
        if self._db_conn is None and DB_PATH.exists():
            try:
                import duckdb
                self._db_conn = duckdb.connect(str(DB_PATH), read_only=True)
            except Exception as e:
                logger.debug(f"Could not connect to database: {e}")
        return self._db_conn
    
    def get_stock_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get stock info from local storage."""
        # Check cache
        if symbol in self._stock_info_cache:
            return self._stock_info_cache[symbol]
        
        # Try JSON file first
        info_file = INFO_DIR / f"{symbol}.json"
        if info_file.exists():
            try:
                with open(info_file) as f:
                    info = json.load(f)
                    # Normalize field names
                    result = {
                        'symbol': symbol,
                        'name': info.get('longName', info.get('shortName', symbol)),
                        'sector': info.get('sector', 'Unknown'),
                        'industry': info.get('industry', 'Unknown'),
                        'market_cap': info.get('marketCap', 0),
                        'current_price': info.get('currentPrice', info.get('regularMarketPrice', 0)),
                        'pe_ratio': info.get('trailingPE'),
                        'pb_ratio': info.get('priceToBook'),
                        'dividend_yield': (info.get('dividendYield', 0) or 0) * 100,
                        'eps': info.get('trailingEps'),
                        'book_value': info.get('bookValue'),
                        'roe': (info.get('returnOnEquity') or 0) * 100 if info.get('returnOnEquity') else None,
                        'debt_to_equity': info.get('debtToEquity'),
                        'fifty_two_week_high': info.get('fiftyTwoWeekHigh'),
                        'fifty_two_week_low': info.get('fiftyTwoWeekLow'),
                        'beta': info.get('beta'),
                        'source': 'local_file'
                    }
                    self._stock_info_cache[symbol] = result
                    return result
            except Exception as e:
                logger.debug(f"Error reading info file for {symbol}: {e}")

        # Try delisted info
        delisted_info_file = self._get_delisted_info_file(symbol)
        if delisted_info_file:
            try:
                with open(delisted_info_file) as f:
                    info = json.load(f)
                    result = {
                        'symbol': symbol,
                        'name': info.get('name', symbol),
                        'sector': info.get('sector', 'Unknown'),
                        'industry': info.get('industry', 'Unknown'),
                        'market_cap': info.get('market_cap', 0),
                        'current_price': info.get('last_known_price', 0),
                        'pe_ratio': info.get('pe_ratio'),
                        'pb_ratio': info.get('pb_ratio'),
                        'dividend_yield': info.get('dividend_yield'),
                        'roe': info.get('roe'),
                        'debt_to_equity': info.get('debt_to_equity'),
                        'fifty_two_week_high': info.get('peak_price'),
                        'fifty_two_week_low': info.get('low_price'),
                        'delisted': True,
                        'source': 'delisted_file'
                    }
                    self._stock_info_cache[symbol] = result
                    return result
            except Exception as e:
                logger.debug(f"Error reading delisted info file for {symbol}: {e}")
        
        # Try database
        db = self._get_db()
        if db:
            try:
                row = db.execute(
                    "SELECT * FROM stocks WHERE symbol = ?", [symbol]
                ).fetchone()
                if row:
                    result = {
                        'symbol': row[0],
                        'name': row[1] or symbol,
                        'sector': row[2] or 'Unknown',
                        'industry': row[3] or 'Unknown',
                        'market_cap': row[4] or 0,
                        'current_price': row[5] or 0,
                        'pe_ratio': row[6],
                        'pb_ratio': row[7],
                        'dividend_yield': row[8],
                        'fifty_two_week_high': row[9],
                        'fifty_two_week_low': row[10],
                        'source': 'local_db'
                    }
                    self._stock_info_cache[symbol] = result
                    return result
            except Exception as e:
                logger.debug(f"Error querying database for {symbol}: {e}")
        
        return None
    
    def get_price_history(self, symbol: str, years: int = 10) -> Optional[pd.DataFrame]:
        """Get price history from local storage."""
        # Try parquet file first
        price_file = PRICE_DIR / f"{symbol}.parquet"
        if price_file.exists():
            try:
                df = pd.read_parquet(price_file)
                df['date'] = pd.to_datetime(df['date'])
                
                # Filter to requested years
                cutoff = datetime.now() - timedelta(days=years * 365)
                df = df[df['date'] >= cutoff]
                
                if not df.empty:
                    logger.info(f"Got {len(df)} local records for {symbol}")
                    return df.sort_values('date')
            except Exception as e:
                logger.debug(f"Error reading parquet for {symbol}: {e}")
        
        # Try database
        db = self._get_db()
        if db:
            try:
                cutoff = datetime.now() - timedelta(days=years * 365)
                df = db.execute("""
                    SELECT symbol, date, open, high, low, close, volume, source
                    FROM price_history 
                    WHERE symbol = ? AND date >= ?
                    ORDER BY date
                """, [symbol, cutoff.date()]).fetchdf()
                
                if not df.empty:
                    df['date'] = pd.to_datetime(df['date'])
                    logger.info(f"Got {len(df)} database records for {symbol}")
                    return df
            except Exception as e:
                logger.debug(f"Error querying price history for {symbol}: {e}")

        # Try delisted price history
        delisted_price_file = self._get_delisted_price_file(symbol)
        if delisted_price_file:
            try:
                if delisted_price_file.suffix == '.parquet':
                    df = pd.read_parquet(delisted_price_file)
                else:
                    df = pd.read_csv(delisted_price_file)
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                else:
                    df['date'] = pd.to_datetime(df.iloc[:, 0])
                cutoff = datetime.now() - timedelta(days=years * 365)
                df = df[df['date'] >= cutoff]
                if not df.empty:
                    logger.info(f"Got {len(df)} delisted records for {symbol}")
                    return df.sort_values('date')
            except Exception as e:
                logger.debug(f"Error reading delisted price history for {symbol}: {e}")
        
        return None
    
    def get_all_symbols(self) -> List[str]:
        """Get all available symbols from local storage."""
        symbols = set()
        
        # From parquet files
        if PRICE_DIR.exists():
            for f in PRICE_DIR.glob("*.parquet"):
                symbols.add(f.stem)
        
        # From info files
        if INFO_DIR.exists():
            for f in INFO_DIR.glob("*.json"):
                symbols.add(f.stem)
        
        # From database
        db = self._get_db()
        if db:
            try:
                rows = db.execute("SELECT DISTINCT symbol FROM stocks").fetchall()
                symbols.update(r[0] for r in rows)
            except:
                pass

        # From delisted data
        if self._include_delisted and self._delisted_dir.exists():
            for ext in ['*.parquet', '*.csv']:
                for f in self._delisted_dir.rglob(ext):
                    symbols.add(f.stem)
            for f in self._delisted_dir.rglob("*.json"):
                symbols.add(f.stem)
        
        return sorted(list(symbols))
    
    def has_data(self) -> bool:
        """Check if local data exists."""
        return (PRICE_DIR.exists() and any(PRICE_DIR.glob("*.parquet"))) or DB_PATH.exists()


class YahooFinanceSource:
    """Yahoo Finance data source (online only)."""
    
    def __init__(self):
        self._yf = None
    
    def _get_yf(self):
        if self._yf is None:
            try:
                import yfinance as yf
                self._yf = yf
            except ImportError:
                logger.warning("yfinance not installed")
        return self._yf
    
    def get_stock_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get stock info from Yahoo Finance."""
        yf = self._get_yf()
        if not yf:
            return None
        
        for suffix in ['.NS', '.BO']:
            try:
                ticker = yf.Ticker(f"{symbol}{suffix}")
                info = ticker.info
                
                if info and (info.get('regularMarketPrice') or info.get('currentPrice')):
                    return {
                        'symbol': symbol,
                        'name': info.get('longName', info.get('shortName', symbol)),
                        'sector': info.get('sector', 'Unknown'),
                        'industry': info.get('industry', 'Unknown'),
                        'market_cap': info.get('marketCap', 0),
                        'current_price': info.get('currentPrice', info.get('regularMarketPrice', 0)),
                        'pe_ratio': info.get('trailingPE'),
                        'pb_ratio': info.get('priceToBook'),
                        'dividend_yield': (info.get('dividendYield', 0) or 0) * 100,
                        'eps': info.get('trailingEps'),
                        'book_value': info.get('bookValue'),
                        'roe': (info.get('returnOnEquity') or 0) * 100 if info.get('returnOnEquity') else None,
                        'debt_to_equity': info.get('debtToEquity'),
                        'fifty_two_week_high': info.get('fiftyTwoWeekHigh'),
                        'fifty_two_week_low': info.get('fiftyTwoWeekLow'),
                        'beta': info.get('beta'),
                        'source': f'yahoo_finance{suffix}'
                    }
            except Exception as e:
                logger.debug(f"Yahoo info failed for {symbol}{suffix}: {e}")
                continue
        
        return None
    
    def get_price_history(self, symbol: str, years: int = 10) -> Optional[pd.DataFrame]:
        """Get price history from Yahoo Finance."""
        yf = self._get_yf()
        if not yf:
            return None
        
        for suffix in ['.NS', '.BO']:
            try:
                ticker = yf.Ticker(f"{symbol}{suffix}")
                df = ticker.history(period=f"{years}y")
                
                if df.empty:
                    df = ticker.history(period="max")
                
                if not df.empty:
                    df = df.reset_index()
                    df.columns = [c.lower().replace(' ', '_') for c in df.columns]
                    
                    if 'date' in df.columns:
                        try:
                            if hasattr(df['date'].dt, 'tz') and df['date'].dt.tz is not None:
                                df['date'] = df['date'].dt.tz_localize(None)
                        except:
                            pass
                        df['date'] = pd.to_datetime(df['date'])
                    
                    df['symbol'] = symbol
                    df['source'] = f'yahoo_finance{suffix}'
                    
                    required = ['date', 'open', 'high', 'low', 'close', 'volume', 'source']
                    if all(c in df.columns for c in required):
                        logger.info(f"Got {len(df)} Yahoo records for {symbol}")
                        return df[required]
                        
            except Exception as e:
                logger.debug(f"Yahoo price failed for {symbol}{suffix}: {e}")
                continue
        
        return None


class NSEToolsSource:
    """NSE Tools for real-time quotes and stock list."""
    
    def __init__(self):
        self._nse = None
    
    def _get_nse(self):
        if self._nse is None:
            try:
                from nsetools import Nse
                self._nse = Nse()
            except:
                pass
        return self._nse
    
    def get_stock_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        nse = self._get_nse()
        if not nse:
            return None
        
        try:
            quote = nse.get_quote(symbol)
            if quote:
                return {
                    'symbol': symbol,
                    'name': quote.get('companyName', symbol),
                    'current_price': quote.get('lastPrice', 0),
                    'pe_ratio': quote.get('pe'),
                    'fifty_two_week_high': quote.get('high52'),
                    'fifty_two_week_low': quote.get('low52'),
                    'source': 'nse_tools'
                }
        except Exception as e:
            logger.debug(f"NSE Tools failed for {symbol}: {e}")
        
        return None
    
    def get_all_stock_codes(self) -> Dict[str, str]:
        nse = self._get_nse()
        if nse:
            try:
                return nse.get_stock_codes()
            except:
                pass
        return {}


class DataSourceManager:
    """Manages data sources with local-first priority."""
    
    def __init__(self, offline_mode: bool = True):
        self.offline_mode = offline_mode
        self.local = LocalDataSource()
        self.yahoo = YahooFinanceSource()
        self.nse_tools = NSEToolsSource()
        
        self._info_cache: Dict[str, Dict[str, Any]] = {}
        self._price_cache: Dict[str, pd.DataFrame] = {}
        self._stock_list: Optional[pd.DataFrame] = None
        
        # Check if local data exists
        if self.local.has_data():
            logger.info("Local data available - running in offline mode")
        else:
            logger.warning("No local data found - online mode required")
            self.offline_mode = False
    
    def get_available_sources(self) -> List[str]:
        """Get list of available data sources."""
        sources = []
        
        if self.local.has_data():
            sources.append('local_data')
        
        if not self.offline_mode:
            try:
                import yfinance
                sources.append('yahoo_finance')
            except:
                pass
            
            try:
                from nsetools import Nse
                sources.append('nse_tools')
            except:
                pass
        
        return sources
    
    def get_stock_info(self, symbol: str, use_cache: bool = True) -> Optional[Dict[str, Any]]:
        """Get stock info - local first, then online."""
        cache_key = symbol.upper()
        
        if use_cache and cache_key in self._info_cache:
            return self._info_cache[cache_key]
        
        # Try local first
        info = self.local.get_stock_info(symbol)
        
        # Try online if no local data and not in offline mode
        if not info and not self.offline_mode:
            info = self.yahoo.get_stock_info(symbol)
            if not info:
                info = self.nse_tools.get_stock_info(symbol)
        
        if info:
            self._info_cache[cache_key] = info
            logger.info(f"Got info for {symbol} from {info.get('source', 'unknown')}")
        
        return info
    
    def get_price_history(self, symbol: str, years: int = 10, 
                          use_cache: bool = True) -> Optional[pd.DataFrame]:
        """Get price history - local first, then online."""
        cache_key = f"{symbol.upper()}_{years}y"
        
        if use_cache and cache_key in self._price_cache:
            cached = self._price_cache[cache_key]
            if cached is not None and not cached.empty:
                return cached
        
        # Try local first
        logger.info(f"Fetching price history for {symbol}...")
        df = self.local.get_price_history(symbol, years)
        
        # Try online if no local data and not in offline mode
        if (df is None or df.empty) and not self.offline_mode:
            logger.info(f"No local data, trying online for {symbol}...")
            df = self.yahoo.get_price_history(symbol, years)
        
        if df is not None and not df.empty:
            self._price_cache[cache_key] = df
            return df
        
        logger.warning(f"No price history found for {symbol}")
        return None
    
    def get_financials(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """Get financial statements."""
        # Try Yahoo Finance if online
        if not self.offline_mode:
            try:
                import yfinance as yf
                for suffix in ['.NS', '.BO']:
                    try:
                        ticker = yf.Ticker(f"{symbol}{suffix}")
                        return {
                            'income_statement': ticker.financials.T if ticker.financials is not None else pd.DataFrame(),
                            'balance_sheet': ticker.balance_sheet.T if ticker.balance_sheet is not None else pd.DataFrame(),
                            'cash_flow': ticker.cashflow.T if ticker.cashflow is not None else pd.DataFrame(),
                            'source': f'yahoo_finance{suffix}'
                        }
                    except:
                        continue
            except:
                pass
        
        return {
            'income_statement': pd.DataFrame(),
            'balance_sheet': pd.DataFrame(),
            'cash_flow': pd.DataFrame(),
            'source': 'none'
        }
    
    def get_all_stocks(self, force_refresh: bool = False) -> pd.DataFrame:
        """Get complete list of stocks."""
        if self._stock_list is not None and not force_refresh:
            return self._stock_list
        
        # Try local symbols first
        local_symbols = self.local.get_all_symbols()
        if local_symbols:
            self._stock_list = pd.DataFrame({
                'symbol': local_symbols,
                'name': local_symbols,  # Will be updated when info is fetched
                'exchange': 'NSE'
            })
            logger.info(f"Got {len(local_symbols)} symbols from local storage")
            return self._stock_list
        
        # Try NSE Tools
        if not self.offline_mode:
            codes = self.nse_tools.get_all_stock_codes()
            if codes:
                self._stock_list = pd.DataFrame([
                    {'symbol': k, 'name': v, 'exchange': 'NSE'}
                    for k, v in codes.items() if k != 'SYMBOL'
                ])
                logger.info(f"Got {len(self._stock_list)} symbols from NSE Tools")
                return self._stock_list
        
        return pd.DataFrame(columns=['symbol', 'name', 'exchange'])
    
    def search_stocks(self, query: str, limit: int = 20) -> pd.DataFrame:
        """Search stocks by symbol or name."""
        stocks = self.get_all_stocks()
        if stocks.empty:
            return pd.DataFrame()
        
        query = query.upper()
        symbol_match = stocks[stocks['symbol'].str.upper().str.startswith(query)]
        name_match = stocks[stocks['name'].str.upper().str.contains(query, na=False)]
        
        result = pd.concat([symbol_match, name_match]).drop_duplicates(subset=['symbol'])
        return result.head(limit)
    
    def refresh_data(self, symbol: str) -> Tuple[bool, str]:
        """Force refresh data from online sources."""
        if self.offline_mode:
            return False, "Running in offline mode. Run download_all_data.py to update."
        
        # Clear caches
        cache_key = symbol.upper()
        if cache_key in self._info_cache:
            del self._info_cache[cache_key]
        
        for key in list(self._price_cache.keys()):
            if key.startswith(cache_key):
                del self._price_cache[key]
        
        # Refetch
        info = self.get_stock_info(symbol, use_cache=False)
        prices = self.get_price_history(symbol, use_cache=False)
        
        if info and prices is not None and not prices.empty:
            return True, f"Data refreshed: {len(prices)} price records"
        elif info:
            return True, "Stock info refreshed, but no price history available"
        else:
            return False, "Failed to refresh data from online sources"
    
    def clear_cache(self):
        """Clear all caches."""
        self._info_cache.clear()
        self._price_cache.clear()

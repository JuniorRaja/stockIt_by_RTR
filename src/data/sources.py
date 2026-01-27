"""Multi-source data ingestion for Indian Equity Intelligence."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from abc import ABC, abstractmethod
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataSource(ABC):
    """Abstract base class for data sources."""
    
    @abstractmethod
    def get_stock_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        pass
    
    @abstractmethod
    def get_price_history(self, symbol: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        pass
    
    @abstractmethod
    def get_financials(self, symbol: str) -> Optional[Dict[str, pd.DataFrame]]:
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        pass


class YahooFinanceSource(DataSource):
    """Yahoo Finance data source using yfinance library."""
    
    def __init__(self, suffix: str = ".NS"):
        self.suffix = suffix
        self._available = None
    
    def _get_ticker(self, symbol: str):
        import yfinance as yf
        return yf.Ticker(f"{symbol}{self.suffix}")
    
    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            import yfinance as yf
            test = yf.Ticker("RELIANCE.NS")
            _ = test.info
            self._available = True
        except Exception:
            self._available = False
        return self._available
    
    def get_stock_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        try:
            ticker = self._get_ticker(symbol)
            info = ticker.info
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
                'source': 'yahoo_finance'
            }
        except Exception as e:
            logger.warning(f"Yahoo Finance failed for {symbol}: {e}")
            return None
    
    def get_price_history(self, symbol: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        try:
            ticker = self._get_ticker(symbol)
            df = ticker.history(start=start_date, end=end_date)
            if df.empty:
                return None
            df = df.reset_index()
            df.columns = [c.lower().replace(' ', '_') for c in df.columns]
            df['source'] = 'yahoo_finance'
            return df[['date', 'open', 'high', 'low', 'close', 'volume', 'source']]
        except Exception as e:
            logger.warning(f"Yahoo Finance price history failed for {symbol}: {e}")
            return None
    
    def get_financials(self, symbol: str) -> Optional[Dict[str, pd.DataFrame]]:
        try:
            ticker = self._get_ticker(symbol)
            return {
                'income_statement': ticker.financials.T if ticker.financials is not None else pd.DataFrame(),
                'balance_sheet': ticker.balance_sheet.T if ticker.balance_sheet is not None else pd.DataFrame(),
                'cash_flow': ticker.cashflow.T if ticker.cashflow is not None else pd.DataFrame(),
                'source': 'yahoo_finance'
            }
        except Exception as e:
            logger.warning(f"Yahoo Finance financials failed for {symbol}: {e}")
            return None


class NSEToolsSource(DataSource):
    """NSE Tools data source."""
    
    def __init__(self):
        self._available = None
        self._nse = None
    
    def _get_nse(self):
        if self._nse is None:
            from nsetools import Nse
            self._nse = Nse()
        return self._nse
    
    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            nse = self._get_nse()
            _ = nse.get_quote("RELIANCE")
            self._available = True
        except Exception:
            self._available = False
        return self._available
    
    def get_stock_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        try:
            nse = self._get_nse()
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
            logger.warning(f"NSE Tools failed for {symbol}: {e}")
        return None
    
    def get_price_history(self, symbol: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        return None  # NSE Tools doesn't provide historical data
    
    def get_financials(self, symbol: str) -> Optional[Dict[str, pd.DataFrame]]:
        return None
    
    def get_all_stock_codes(self) -> List[str]:
        try:
            nse = self._get_nse()
            return list(nse.get_stock_codes().keys())
        except Exception as e:
            logger.warning(f"Failed to get stock codes: {e}")
            return []


class DataSourceManager:
    """Manages multiple data sources with automatic fallback."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.sources: Dict[str, DataSource] = {
            'yahoo_finance': YahooFinanceSource(),
            'nse_tools': NSEToolsSource()
        }
        self.priority = self.config.get('priority', ['yahoo_finance', 'nse_tools'])
        self._info_cache: Dict[str, Dict[str, Any]] = {}
        self._price_cache: Dict[str, pd.DataFrame] = {}
    
    def get_available_sources(self) -> List[str]:
        return [name for name, source in self.sources.items() if source.is_available()]
    
    def get_stock_info(self, symbol: str, use_cache: bool = True) -> Optional[Dict[str, Any]]:
        cache_key = symbol.upper()
        if use_cache and cache_key in self._info_cache:
            return self._info_cache[cache_key]
        
        merged_info = {}
        for source_name in self.priority:
            source = self.sources.get(source_name)
            if source is None or not source.is_available():
                continue
            info = source.get_stock_info(symbol)
            if info:
                for key, value in info.items():
                    if key not in merged_info or merged_info[key] is None:
                        merged_info[key] = value
                logger.info(f"Got info for {symbol} from {source_name}")
        
        if merged_info:
            self._info_cache[cache_key] = merged_info
            return merged_info
        return None
    
    def get_price_history(self, symbol: str, years: int = 10, end_date: Optional[datetime] = None, use_cache: bool = True) -> Optional[pd.DataFrame]:
        end_date = end_date or datetime.now()
        start_date = end_date - timedelta(days=years * 365)
        cache_key = f"{symbol.upper()}_{years}y"
        
        if use_cache and cache_key in self._price_cache:
            return self._price_cache[cache_key]
        
        for source_name in self.priority:
            source = self.sources.get(source_name)
            if source is None or not source.is_available():
                continue
            df = source.get_price_history(symbol, start_date, end_date)
            if df is not None and not df.empty:
                logger.info(f"Got price history for {symbol} from {source_name}")
                self._price_cache[cache_key] = df
                return df
        return None
    
    def get_financials(self, symbol: str) -> Optional[Dict[str, pd.DataFrame]]:
        for source_name in self.priority:
            source = self.sources.get(source_name)
            if source is None or not source.is_available():
                continue
            financials = source.get_financials(symbol)
            if financials:
                logger.info(f"Got financials for {symbol} from {source_name}")
                return financials
        return None
    
    def get_all_nse_symbols(self) -> List[str]:
        nse_source = self.sources.get('nse_tools')
        if isinstance(nse_source, NSEToolsSource):
            return nse_source.get_all_stock_codes()
        return []
    
    def clear_cache(self):
        self._info_cache.clear()
        self._price_cache.clear()

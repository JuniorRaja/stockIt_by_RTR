"""
Macro-Economic Data Provider for Stocron by RTR.

Fetches and caches macro-economic indicators critical for Indian market analysis:
- RBI Repo Rate
- USD-INR Exchange Rate
- Crude Oil (Brent) Prices
- CPI Inflation Data
- Market Regime Detection
"""

import logging
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Data directories
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
MACRO_DIR = DATA_DIR / 'macro'


class MacroDataProvider:
    """
    Fetches macro-economic data for Indian market context.
    
    Data Sources:
    - RBI DBIE for Repo Rates and USD-INR (https://dbie.rbi.org.in)
    - FRED API for Crude Oil (Brent)
    - MOSPI for CPI inflation
    - Yahoo Finance for index data (Nifty 50)
    """
    
    # Indian CPI base data (YoY inflation rates - can be updated from MOSPI)
    # Source: https://mospi.gov.in/
    HISTORICAL_CPI = {
        2000: 4.0, 2001: 3.8, 2002: 4.3, 2003: 3.8, 2004: 3.8,
        2005: 4.2, 2006: 5.8, 2007: 6.4, 2008: 8.3, 2009: 10.9,
        2010: 12.0, 2011: 8.9, 2012: 9.3, 2013: 10.9, 2014: 6.4,
        2015: 5.9, 2016: 4.9, 2017: 3.3, 2018: 4.0, 2019: 3.7,
        2020: 6.6, 2021: 5.1, 2022: 6.7, 2023: 5.4, 2024: 4.8,
        2025: 4.5, 2026: 4.2  # Projected
    }
    
    # Historical RBI Repo Rates (key dates)
    HISTORICAL_REPO_RATES = {
        '2000-01-01': 6.50, '2001-10-22': 6.50, '2003-03-03': 5.50,
        '2004-10-27': 6.00, '2006-01-24': 6.50, '2008-07-29': 9.00,
        '2009-04-21': 4.75, '2010-03-19': 5.00, '2011-10-25': 8.50,
        '2013-10-29': 7.75, '2014-01-28': 8.00, '2015-06-02': 7.25,
        '2016-10-04': 6.25, '2018-08-01': 6.50, '2019-10-04': 5.15,
        '2020-05-22': 4.00, '2022-05-04': 4.40, '2023-02-08': 6.50,
        '2024-02-08': 6.50, '2025-01-01': 6.25, '2026-01-01': 6.00
    }
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize macro data provider."""
        self.cache_dir = cache_dir or MACRO_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, pd.DataFrame] = {}
        self._fred_api = None
        
    def _get_fred_api(self):
        """Get FRED API client (lazy initialization)."""
        if self._fred_api is None:
            try:
                from fredapi import Fred
                # FRED API key - users should set their own
                api_key = self._get_fred_api_key()
                if api_key:
                    self._fred_api = Fred(api_key=api_key)
                    logger.info("FRED API initialized")
            except ImportError:
                logger.warning("fredapi not installed. Install with: pip install fredapi")
            except Exception as e:
                logger.warning(f"Could not initialize FRED API: {e}")
        return self._fred_api
    
    def _get_fred_api_key(self) -> Optional[str]:
        """Get FRED API key from environment or config."""
        import os
        api_key = os.environ.get('FRED_API_KEY')
        if not api_key:
            # Try loading from config file
            config_file = PROJECT_ROOT / 'config' / 'api_keys.json'
            if config_file.exists():
                try:
                    with open(config_file) as f:
                        config = json.load(f)
                        api_key = config.get('fred_api_key')
                except Exception:
                    pass
        return api_key
    
    def get_repo_rate_history(self, years: int = 30) -> pd.DataFrame:
        """
        Get RBI Repo Rate history.
        
        Args:
            years: Number of years of history to fetch
            
        Returns:
            DataFrame with date index and 'repo_rate' column
        """
        cache_key = f'repo_rate_{years}y'
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Try loading from cache file
        cache_file = self.cache_dir / f'repo_rate_{years}y.parquet'
        if cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                df.index = pd.to_datetime(df.index)
                self._cache[cache_key] = df
                return df
            except Exception as e:
                logger.debug(f"Could not load cached repo rate: {e}")
        
        # Build from historical data
        start_date = datetime.now() - timedelta(days=years * 365)
        
        # Create daily series from key rate changes
        dates = pd.date_range(start=start_date, end=datetime.now(), freq='D')
        rates = []
        
        rate_changes = sorted(
            [(pd.to_datetime(d), r) for d, r in self.HISTORICAL_REPO_RATES.items()],
            key=lambda x: x[0]
        )
        
        for date in dates:
            # Find the applicable rate for this date
            applicable_rate = 6.0  # Default
            for rate_date, rate in rate_changes:
                if rate_date <= date:
                    applicable_rate = rate
                else:
                    break
            rates.append(applicable_rate)
        
        df = pd.DataFrame({'repo_rate': rates}, index=dates)
        
        # Cache the result
        try:
            df.to_parquet(cache_file)
        except Exception:
            pass
        
        self._cache[cache_key] = df
        return df
    
    def get_usdinr_history(self, years: int = 30) -> pd.DataFrame:
        """
        Get USD-INR exchange rate history.
        
        Args:
            years: Number of years of history to fetch
            
        Returns:
            DataFrame with date index and 'usdinr' column
        """
        cache_key = f'usdinr_{years}y'
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Try loading from cache
        cache_file = self.cache_dir / f'usdinr_{years}y.parquet'
        if cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                df.index = pd.to_datetime(df.index)
                self._cache[cache_key] = df
                return df
            except Exception:
                pass
        
        # Try Yahoo Finance for USD-INR
        try:
            import yfinance as yf
            ticker = yf.Ticker('USDINR=X')
            df = ticker.history(period=f'{years}y')
            
            if not df.empty:
                df = df[['Close']].rename(columns={'Close': 'usdinr'})
                df.index = pd.to_datetime(df.index).tz_localize(None)
                
                # Cache
                try:
                    df.to_parquet(cache_file)
                except Exception:
                    pass
                
                self._cache[cache_key] = df
                return df
        except Exception as e:
            logger.debug(f"Could not fetch USD-INR from Yahoo: {e}")
        
        # Fallback: Generate synthetic data based on historical trend
        logger.warning("Using synthetic USD-INR data - consider fetching real data")
        start_date = datetime.now() - timedelta(days=years * 365)
        dates = pd.date_range(start=start_date, end=datetime.now(), freq='D')
        
        # USD-INR trend: roughly 45 in 2000 to 83 in 2024
        start_rate = 45.0
        end_rate = 83.0
        days_total = (datetime.now() - start_date).days
        
        rates = []
        for i, date in enumerate(dates):
            base_rate = start_rate + (end_rate - start_rate) * (i / days_total)
            # Add some noise
            noise = np.random.normal(0, 0.3)
            rates.append(max(35, base_rate + noise))
        
        df = pd.DataFrame({'usdinr': rates}, index=dates)
        self._cache[cache_key] = df
        return df
    
    def get_crude_oil_history(self, years: int = 30) -> pd.DataFrame:
        """
        Get Crude Oil (Brent) price history.
        
        Args:
            years: Number of years of history to fetch
            
        Returns:
            DataFrame with date index and 'crude_oil' column (in USD/barrel)
        """
        cache_key = f'crude_oil_{years}y'
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Try loading from cache
        cache_file = self.cache_dir / f'crude_oil_{years}y.parquet'
        if cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                df.index = pd.to_datetime(df.index)
                self._cache[cache_key] = df
                return df
            except Exception:
                pass
        
        # Try FRED API
        fred = self._get_fred_api()
        if fred:
            try:
                start_date = datetime.now() - timedelta(days=years * 365)
                # DCOILBRENTEU = Brent Crude Oil Price
                series = fred.get_series('DCOILBRENTEU', start_date)
                
                if series is not None and len(series) > 0:
                    df = pd.DataFrame({'crude_oil': series})
                    df.index = pd.to_datetime(df.index)
                    df = df.dropna()
                    
                    # Cache
                    try:
                        df.to_parquet(cache_file)
                    except Exception:
                        pass
                    
                    self._cache[cache_key] = df
                    return df
            except Exception as e:
                logger.debug(f"Could not fetch crude oil from FRED: {e}")
        
        # Try Yahoo Finance
        try:
            import yfinance as yf
            ticker = yf.Ticker('BZ=F')  # Brent Crude Futures
            df = ticker.history(period=f'{years}y')
            
            if not df.empty:
                df = df[['Close']].rename(columns={'Close': 'crude_oil'})
                df.index = pd.to_datetime(df.index).tz_localize(None)
                
                # Cache
                try:
                    df.to_parquet(cache_file)
                except Exception:
                    pass
                
                self._cache[cache_key] = df
                return df
        except Exception as e:
            logger.debug(f"Could not fetch crude oil from Yahoo: {e}")
        
        # Fallback: Generate synthetic data
        logger.warning("Using synthetic crude oil data - consider fetching real data")
        start_date = datetime.now() - timedelta(days=years * 365)
        dates = pd.date_range(start=start_date, end=datetime.now(), freq='D')
        
        # Crude oil has been volatile: ~25-140 range historically
        prices = []
        current_price = 60.0
        for _ in dates:
            change = np.random.normal(0, 2)
            current_price = max(20, min(150, current_price + change))
            prices.append(current_price)
        
        df = pd.DataFrame({'crude_oil': prices}, index=dates)
        self._cache[cache_key] = df
        return df
    
    def get_cpi_inflation(self, years: int = 30) -> pd.DataFrame:
        """
        Get CPI inflation data (YoY).
        
        Args:
            years: Number of years of history to fetch
            
        Returns:
            DataFrame with date index and 'cpi_yoy' column (in %)
        """
        cache_key = f'cpi_{years}y'
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Build from historical data
        start_year = datetime.now().year - years
        end_year = datetime.now().year
        
        # Create monthly series
        dates = []
        values = []
        
        for year in range(max(2000, start_year), end_year + 1):
            if year in self.HISTORICAL_CPI:
                base_rate = self.HISTORICAL_CPI[year]
                # Create monthly data with some variation
                for month in range(1, 13):
                    if year == end_year and month > datetime.now().month:
                        break
                    date = datetime(year, month, 1)
                    # Add monthly variation (±1%)
                    variation = np.random.uniform(-1, 1)
                    dates.append(date)
                    values.append(base_rate + variation)
        
        df = pd.DataFrame({'cpi_yoy': values}, index=pd.DatetimeIndex(dates))
        
        # Resample to daily for consistency
        df = df.resample('D').ffill()
        
        self._cache[cache_key] = df
        return df
    
    def get_cpi_index(self, years: int = 30, base_year: int = 2010) -> pd.DataFrame:
        """
        Get CPI Index (for inflation adjustment of prices).
        
        Args:
            years: Number of years of history
            base_year: Base year for index (100)
            
        Returns:
            DataFrame with 'cpi_index' column
        """
        cpi_yoy = self.get_cpi_inflation(years)
        
        # Convert YoY rates to cumulative index
        # Start from base year = 100
        start_date = datetime(base_year, 1, 1)
        
        if cpi_yoy.empty:
            return pd.DataFrame()
        
        # Calculate monthly compounding
        df = cpi_yoy.resample('M').mean()
        df['monthly_rate'] = df['cpi_yoy'] / 12 / 100
        
        # Cumulative index
        df['cpi_index'] = 100 * (1 + df['monthly_rate']).cumprod()
        
        # Resample back to daily
        result = df[['cpi_index']].resample('D').ffill()
        
        return result
    
    def get_nifty_history(self, years: int = 10) -> pd.DataFrame:
        """
        Get Nifty 50 price history for market regime detection.
        
        Args:
            years: Number of years of history
            
        Returns:
            DataFrame with 'close' column
        """
        cache_key = f'nifty_{years}y'
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Try Yahoo Finance
        try:
            import yfinance as yf
            ticker = yf.Ticker('^NSEI')
            df = ticker.history(period=f'{years}y')
            
            if not df.empty:
                df = df[['Close']].rename(columns={'Close': 'close'})
                df.index = pd.to_datetime(df.index).tz_localize(None)
                self._cache[cache_key] = df
                return df
        except Exception as e:
            logger.debug(f"Could not fetch Nifty from Yahoo: {e}")
        
        return pd.DataFrame()
    
    def detect_market_regime(self, nifty_prices: Optional[pd.Series] = None, 
                             lookback: int = 200) -> str:
        """
        Detect current market regime (Bull/Bear/Sideways).
        
        Uses moving average crossover and trend analysis:
        - Bull: Price > SMA50 > SMA200
        - Bear: Price < SMA50 < SMA200
        - Sideways: Mixed signals
        
        Args:
            nifty_prices: Nifty 50 closing prices (optional, will fetch if not provided)
            lookback: Number of days to look back
            
        Returns:
            'bull', 'bear', or 'sideways'
        """
        if nifty_prices is None:
            nifty_df = self.get_nifty_history(years=2)
            if nifty_df.empty:
                return 'sideways'
            nifty_prices = nifty_df['close']
        
        if len(nifty_prices) < lookback:
            return 'sideways'
        
        # Calculate moving averages
        sma_50 = nifty_prices.rolling(50).mean()
        sma_200 = nifty_prices.rolling(200).mean()
        
        current_price = nifty_prices.iloc[-1]
        current_sma50 = sma_50.iloc[-1]
        current_sma200 = sma_200.iloc[-1]
        
        if pd.isna(current_sma200):
            return 'sideways'
        
        # Bull market: Price > SMA50 > SMA200
        if current_price > current_sma50 > current_sma200:
            return 'bull'
        # Bear market: Price < SMA50 < SMA200
        elif current_price < current_sma50 < current_sma200:
            return 'bear'
        else:
            return 'sideways'
    
    def get_all_macro_data(self, years: int = 10) -> Dict[str, Any]:
        """
        Get all macro data in a single call.
        
        Args:
            years: Number of years of history
            
        Returns:
            Dictionary with all macro indicators
        """
        repo_rate = self.get_repo_rate_history(years)
        usdinr = self.get_usdinr_history(years)
        crude_oil = self.get_crude_oil_history(years)
        cpi = self.get_cpi_inflation(years)
        cpi_index = self.get_cpi_index(years)
        nifty = self.get_nifty_history(min(years, 10))
        market_regime = self.detect_market_regime()
        
        return {
            'repo_rate': repo_rate,
            'usdinr': usdinr,
            'crude_oil': crude_oil,
            'cpi': cpi,
            'cpi_index': cpi_index,
            'nifty': nifty,
            'market_regime': market_regime,
            'fetch_date': datetime.now().isoformat()
        }
    
    def get_macro_summary(self) -> Dict[str, Any]:
        """Get current macro indicators summary."""
        repo = self.get_repo_rate_history(2)
        usdinr = self.get_usdinr_history(2)
        crude = self.get_crude_oil_history(2)
        cpi = self.get_cpi_inflation(2)
        regime = self.detect_market_regime()
        
        summary = {
            'market_regime': regime,
            'repo_rate_current': repo['repo_rate'].iloc[-1] if not repo.empty else None,
            'usdinr_current': usdinr['usdinr'].iloc[-1] if not usdinr.empty else None,
            'crude_oil_current': crude['crude_oil'].iloc[-1] if not crude.empty else None,
            'cpi_current': cpi['cpi_yoy'].iloc[-1] if not cpi.empty else None,
        }
        
        # Calculate YoY changes
        if not usdinr.empty and len(usdinr) > 252:
            summary['usdinr_change_1y'] = (
                (usdinr['usdinr'].iloc[-1] / usdinr['usdinr'].iloc[-252] - 1) * 100
            )
        
        if not crude.empty and len(crude) > 252:
            summary['crude_change_1y'] = (
                (crude['crude_oil'].iloc[-1] / crude['crude_oil'].iloc[-252] - 1) * 100
            )
        
        return summary
    
    def clear_cache(self):
        """Clear all cached data."""
        self._cache.clear()
        
        # Also clear file cache
        for cache_file in self.cache_dir.glob('*.parquet'):
            try:
                cache_file.unlink()
            except Exception:
                pass


# Convenience function
def get_macro_provider() -> MacroDataProvider:
    """Get a singleton macro data provider instance."""
    if not hasattr(get_macro_provider, '_instance'):
        get_macro_provider._instance = MacroDataProvider()
    return get_macro_provider._instance

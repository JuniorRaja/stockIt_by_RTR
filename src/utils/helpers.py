"""Helper utilities for Stocron by RTR."""

import numpy as np
import pandas as pd
from typing import Optional, Union
from datetime import datetime


def calculate_cagr(start_value: float, end_value: float, years: float) -> Optional[float]:
    """Calculate Compound Annual Growth Rate (CAGR)."""
    if start_value <= 0 or end_value <= 0 or years <= 0:
        return None
    try:
        return round(((end_value / start_value) ** (1 / years) - 1) * 100, 2)
    except (ZeroDivisionError, ValueError):
        return None


def calculate_volatility(returns: pd.Series, annualize: bool = True, trading_days: int = 252) -> Optional[float]:
    """Calculate volatility (standard deviation of returns)."""
    if len(returns) < 2:
        return None
    volatility = returns.std()
    if annualize:
        volatility = volatility * np.sqrt(trading_days)
    return round(volatility * 100, 2)


def calculate_max_drawdown(prices: pd.Series) -> tuple:
    """Calculate maximum drawdown from a price series."""
    if len(prices) < 2:
        return (0.0, None, None)
    running_max = prices.expanding().max()
    drawdown = (prices - running_max) / running_max
    max_dd_idx = drawdown.idxmin()
    max_dd = drawdown.loc[max_dd_idx]
    peak_idx = prices.loc[:max_dd_idx].idxmax()
    return (round(abs(max_dd) * 100, 2), peak_idx, max_dd_idx)


def format_indian_number(value: Union[int, float], precision: int = 2, prefix: str = "₹") -> str:
    """Format number in Indian notation (lakhs, crores)."""
    if value is None or pd.isna(value):
        return "N/A"
    abs_value = abs(value)
    sign = "-" if value < 0 else ""
    if abs_value >= 1e7:
        formatted = f"{abs_value / 1e7:.{precision}f} Cr"
    elif abs_value >= 1e5:
        formatted = f"{abs_value / 1e5:.{precision}f} L"
    elif abs_value >= 1e3:
        formatted = f"{abs_value / 1e3:.{precision}f} K"
    else:
        formatted = f"{abs_value:.{precision}f}"
    return f"{sign}{prefix}{formatted}"


def format_percentage(value: Optional[float], precision: int = 2) -> str:
    """Format value as percentage."""
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:+.{precision}f}%" if value > 0 else f"{value:.{precision}f}%"


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers."""
    if denominator == 0 or pd.isna(denominator):
        return default
    return numerator / denominator


def calculate_consistency_score(series: pd.Series, threshold: float = 0) -> float:
    """Calculate consistency score (% of periods meeting threshold)."""
    if len(series) == 0:
        return 0.0
    return round((series > threshold).sum() / len(series) * 100, 2)


def years_since(date: datetime) -> float:
    """Calculate years since a given date."""
    if date is None:
        return 0.0
    return (datetime.now() - date).days / 365.25


# ============================================================================
# Return Calculation Functions (Log Returns & Inflation-Adjusted)
# ============================================================================

def calculate_log_returns(prices: pd.Series) -> pd.Series:
    """
    Calculate log returns for better statistical properties.
    
    Log returns are preferred for:
    - Time-additive property (sum of log returns = total log return)
    - Better for long-term analysis
    - More normally distributed than simple returns
    - Essential for ML features and technical analysis
    
    Args:
        prices: Price series with DatetimeIndex
        
    Returns:
        Series of log returns
    """
    if len(prices) < 2:
        return pd.Series(dtype=float)
    
    log_returns = np.log(prices / prices.shift(1))
    return log_returns.dropna()


def calculate_simple_returns(prices: pd.Series) -> pd.Series:
    """
    Calculate simple percentage returns.
    
    Args:
        prices: Price series with DatetimeIndex
        
    Returns:
        Series of simple returns
    """
    if len(prices) < 2:
        return pd.Series(dtype=float)
    
    return prices.pct_change().dropna()


def calculate_real_returns(
    nominal_returns: pd.Series, 
    cpi_series: pd.Series,
    align_method: str = 'ffill'
) -> pd.Series:
    """
    Calculate inflation-adjusted (real) returns.
    
    For 30-year historical analysis, comparing prices in 1995 vs 2025
    requires inflation adjustment for accurate trend detection.
    
    Real Return = (1 + Nominal Return) / (1 + Inflation Rate) - 1
    
    Args:
        nominal_returns: Series of nominal returns (simple or log)
        cpi_series: CPI index or YoY inflation series
        align_method: How to align CPI to returns dates ('ffill' or 'interpolate')
        
    Returns:
        Series of inflation-adjusted returns
    """
    if nominal_returns.empty or cpi_series.empty:
        return nominal_returns
    
    # Align CPI to returns dates
    if align_method == 'interpolate':
        aligned_cpi = cpi_series.reindex(nominal_returns.index).interpolate(method='linear')
    else:
        aligned_cpi = cpi_series.reindex(nominal_returns.index).ffill()
    
    # Calculate inflation rate from CPI
    if 'cpi_index' in str(cpi_series.name) or cpi_series.max() > 50:
        # CPI is an index - calculate period-over-period change
        inflation_rate = aligned_cpi.pct_change()
    else:
        # CPI is already YoY rate - convert to period rate
        # Assuming daily data, divide annual rate by 252
        inflation_rate = aligned_cpi / 100 / 252
    
    # Calculate real returns
    real_returns = (1 + nominal_returns) / (1 + inflation_rate.fillna(0)) - 1
    
    return real_returns


def calculate_inflation_adjusted_prices(
    prices: pd.Series,
    cpi_index: pd.Series,
    target_date: Optional[datetime] = None
) -> pd.Series:
    """
    Convert historical prices to inflation-adjusted prices.
    
    Useful for comparing stock performance across decades.
    Adjusts all prices to the purchasing power of the target date.
    
    Args:
        prices: Historical price series
        cpi_index: CPI index series (not YoY rate)
        target_date: Date to adjust prices to (default: latest date)
        
    Returns:
        Inflation-adjusted price series
    """
    if prices.empty or cpi_index.empty:
        return prices
    
    # Align CPI to price dates
    aligned_cpi = cpi_index.reindex(prices.index).ffill().bfill()
    
    if aligned_cpi.empty:
        return prices
    
    # Get target CPI value
    if target_date is None:
        target_cpi = aligned_cpi.iloc[-1]
    else:
        target_cpi = aligned_cpi.loc[:target_date].iloc[-1]
    
    # Adjust prices: price * (target_cpi / historical_cpi)
    adjusted_prices = prices * (target_cpi / aligned_cpi)
    
    return adjusted_prices


def annualize_returns(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Annualize returns from a return series.
    
    Args:
        returns: Series of period returns
        periods_per_year: Number of periods per year (252 for daily, 12 for monthly)
        
    Returns:
        Annualized return as a decimal
    """
    if returns.empty:
        return 0.0
    
    # For log returns, sum and multiply
    if returns.abs().max() < 0.5:  # Likely log returns
        total_log_return = returns.sum()
        years = len(returns) / periods_per_year
        annualized = total_log_return / years if years > 0 else 0
        return np.exp(annualized) - 1  # Convert back to simple return
    else:
        # Simple returns - compound
        cumulative = (1 + returns).prod()
        years = len(returns) / periods_per_year
        if years > 0 and cumulative > 0:
            return cumulative ** (1 / years) - 1
        return 0.0


def calculate_rolling_real_returns(
    prices: pd.Series,
    cpi_index: pd.Series,
    window: int = 252
) -> pd.Series:
    """
    Calculate rolling real (inflation-adjusted) returns.
    
    Useful for understanding performance in real terms over time.
    
    Args:
        prices: Price series
        cpi_index: CPI index series
        window: Rolling window in periods
        
    Returns:
        Series of rolling real returns
    """
    # Get inflation-adjusted prices
    adjusted_prices = calculate_inflation_adjusted_prices(prices, cpi_index)
    
    if adjusted_prices.empty or len(adjusted_prices) < window:
        return pd.Series(dtype=float)
    
    # Calculate rolling returns
    rolling_returns = adjusted_prices.pct_change(window)
    
    return rolling_returns


def convert_log_to_simple(log_returns: pd.Series) -> pd.Series:
    """Convert log returns to simple returns."""
    return np.exp(log_returns) - 1


def convert_simple_to_log(simple_returns: pd.Series) -> pd.Series:
    """Convert simple returns to log returns."""
    return np.log(1 + simple_returns)

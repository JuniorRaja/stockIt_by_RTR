"""Helper utilities for Indian Equity Intelligence."""

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

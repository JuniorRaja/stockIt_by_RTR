"""Refresh local price data using Yahoo Finance."""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from .historic import load_historic_stock_prices, load_historic_index_prices


PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PRICE_DIR = DATA_DIR / "prices"
INFO_DIR = DATA_DIR / "stock_info"


def _normalize_yf_history(df: pd.DataFrame, symbol: str, source: str) -> pd.DataFrame:
    df = df.reset_index()
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    if "date" in df.columns:
        try:
            if hasattr(df["date"].dt, "tz") and df["date"].dt.tz is not None:
                df["date"] = df["date"].dt.tz_localize(None)
        except Exception:
            pass
        df["date"] = pd.to_datetime(df["date"])
    df["symbol"] = symbol
    df["source"] = source
    required = ["symbol", "date", "open", "high", "low", "close", "volume", "source"]
    return df[required]


def _load_existing_prices(symbol: str) -> Optional[pd.DataFrame]:
    price_file = PRICE_DIR / f"{symbol}.parquet"
    if price_file.exists():
        df = pd.read_parquet(price_file)
        df["date"] = pd.to_datetime(df["date"])
        return df

    df = load_historic_stock_prices(symbol)
    if df is not None and not df.empty:
        return df

    df = load_historic_index_prices(symbol)
    if df is not None and not df.empty:
        return df

    return None


def _save_prices(symbol: str, df: pd.DataFrame) -> None:
    PRICE_DIR.mkdir(parents=True, exist_ok=True)
    price_file = PRICE_DIR / f"{symbol}.parquet"
    df.to_parquet(price_file, index=False)


def _refresh_info(symbol: str) -> None:
    try:
        import yfinance as yf
        ticker = yf.Ticker(f"{symbol}.NS")
        info = ticker.info
        if info:
            clean_info = {
                k: v for k, v in info.items()
                if isinstance(v, (str, int, float, bool, type(None)))
            }
            clean_info["symbol"] = symbol
            clean_info["downloaded_at"] = datetime.now().isoformat()
            INFO_DIR.mkdir(parents=True, exist_ok=True)
            info_file = INFO_DIR / f"{symbol}.json"
            with open(info_file, "w") as f:
                json.dump(clean_info, f, indent=2)
    except Exception:
        return


def refresh_symbol_data(
    symbol: str,
    update_info: bool = True,
    max_years: int = 30,
    pause_range: Tuple[float, float] = (0.5, 1.5),
) -> dict:
    """Refresh a single symbol by filling the gap to today."""
    import random
    import yfinance as yf

    symbol = symbol.upper().strip()
    result = {
        "symbol": symbol,
        "updated": False,
        "new_records": 0,
        "total_records": 0,
        "error": None,
    }

    existing = _load_existing_prices(symbol)
    last_date = None
    if existing is not None and not existing.empty and "date" in existing.columns:
        last_date = pd.to_datetime(existing["date"]).max()

    end_date = datetime.now()
    if last_date is None:
        start_date = end_date - timedelta(days=max_years * 365)
    else:
        start_date = last_date + timedelta(days=1)

    if start_date.date() > end_date.date():
        result["updated"] = False
        result["total_records"] = len(existing) if existing is not None else 0
        return result

    time.sleep(random.uniform(*pause_range))

    for suffix in [".NS", ".BO"]:
        try:
            ticker = yf.Ticker(f"{symbol}{suffix}")
            df = ticker.history(start=start_date, end=end_date)
            if df is None or df.empty:
                continue
            df = _normalize_yf_history(df, symbol, f"yahoo{suffix}")
            break
        except Exception as e:
            if "Too Many Requests" in str(e) or "429" in str(e):
                time.sleep(5)
            df = None
            continue

    if df is None or df.empty:
        result["error"] = "No new data available"
        result["total_records"] = len(existing) if existing is not None else 0
        return result

    if existing is None or existing.empty:
        merged = df
    else:
        merged = pd.concat([existing, df], ignore_index=True)
        merged = merged.drop_duplicates(subset=["date"], keep="last")
        merged = merged.sort_values("date")

    _save_prices(symbol, merged)

    if update_info and not symbol.startswith("NIFTY"):
        _refresh_info(symbol)

    result["updated"] = True
    result["new_records"] = len(df)
    result["total_records"] = len(merged)
    return result

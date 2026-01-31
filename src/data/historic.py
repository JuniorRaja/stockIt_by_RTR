"""Helpers for local historic datasets."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
HISTORIC_DIR = DATA_DIR / "Historic_Analyzed"


def _normalize_name(name: str) -> str:
    return (
        name.upper()
        .replace("&", "AND")
        .replace("-", " ")
        .replace("_", " ")
        .replace("  ", " ")
        .strip()
    )


def _candidate_stock_dirs() -> List[Path]:
    return [
        HISTORIC_DIR / "nse_stock_data",
        HISTORIC_DIR / "NSE India Stock Data (1990 - 2021)" / "SCRIP",
        HISTORIC_DIR / "NSE India Stock Data (1990 - 2021)",
    ]


def resolve_historic_stock_dir() -> Optional[Path]:
    """Return the directory containing historic NSE stock CSVs."""
    for candidate in _candidate_stock_dirs():
        if candidate.exists() and any(candidate.glob("*.csv")):
            return candidate
    return None


def list_historic_stock_symbols() -> List[str]:
    """List symbols from historic NSE stock CSV files."""
    stock_dir = resolve_historic_stock_dir()
    if not stock_dir:
        return []
    return sorted({p.stem.upper() for p in stock_dir.glob("*.csv")})


def _candidate_index_dirs() -> List[Path]:
    return [
        HISTORIC_DIR / "NIFTY Sectoral & Thematic Indices [1990-2021]",
        HISTORIC_DIR / "Stock Market Index Data India (1990 - 2022)",
    ]


def _iter_index_files() -> List[Path]:
    files: List[Path] = []
    for directory in _candidate_index_dirs():
        if directory.exists():
            files.extend(directory.rglob("*.csv"))
    return files


def list_historic_index_names() -> List[str]:
    """List index names available in historic datasets."""
    names = set()
    for path in _iter_index_files():
        name = path.stem
        if name:
            names.add(name)
    return sorted(names)


def _build_index_file_map() -> Dict[str, Path]:
    """Map normalized index names to CSV paths."""
    mapping: Dict[str, Path] = {}
    for path in _iter_index_files():
        name = path.stem
        normalized = _normalize_name(name)
        mapping[normalized] = path
        mapping[normalized.replace(" ", "_")] = path
        if normalized.startswith("NIFTY "):
            mapping[normalized.replace("NIFTY ", "NIFTY_")] = path
    return mapping


_INDEX_FILE_MAP = None


def _get_index_file_map() -> Dict[str, Path]:
    global _INDEX_FILE_MAP
    if _INDEX_FILE_MAP is None:
        _INDEX_FILE_MAP = _build_index_file_map()
    return _INDEX_FILE_MAP


def resolve_index_csv(symbol: str) -> Optional[Path]:
    """Resolve index symbol to a historic CSV file if available."""
    symbol = symbol.strip().upper()
    if symbol == "NIFTY":
        symbol = "NIFTY 50"
    normalized = _normalize_name(symbol)
    mapping = _get_index_file_map()
    return mapping.get(normalized) or mapping.get(normalized.replace(" ", "_"))


def load_historic_stock_prices(symbol: str) -> Optional[pd.DataFrame]:
    """Load historic NSE stock prices into a standard schema."""
    stock_dir = resolve_historic_stock_dir()
    if not stock_dir:
        return None
    csv_path = stock_dir / f"{symbol}.csv"
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path)
    if df.empty:
        return None

    df.columns = [c.strip().lower().replace(" ", "_").replace("%", "percent") for c in df.columns]
    if "date" not in df.columns:
        return None

    df["date"] = pd.to_datetime(df["date"])
    df["symbol"] = symbol.upper()
    df["source"] = "historic_nse_csv"

    required = {
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
    }
    for key, column in required.items():
        if column not in df.columns:
            df[key] = None
        else:
            df[key] = pd.to_numeric(df[column], errors="coerce")

    return df[["symbol", "date", "open", "high", "low", "close", "volume", "source"]].dropna(
        subset=["date", "close"]
    )


def load_historic_index_prices(symbol: str) -> Optional[pd.DataFrame]:
    """Load historic index prices into a standard schema."""
    csv_path = resolve_index_csv(symbol)
    if not csv_path or not csv_path.exists():
        return None

    df = pd.read_csv(csv_path)
    if df.empty:
        return None

    df.columns = [c.strip().lower().replace(" ", "_").replace("%", "percent") for c in df.columns]
    if "date" not in df.columns:
        return None

    df["date"] = pd.to_datetime(df["date"])
    df["symbol"] = symbol.upper()
    df["source"] = "historic_nifty_csv"

    required = {
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
    }
    for key, column in required.items():
        if column not in df.columns:
            df[key] = None
        else:
            df[key] = pd.to_numeric(df[column], errors="coerce")

    df["volume"] = None
    return df[["symbol", "date", "open", "high", "low", "close", "volume", "source"]].dropna(
        subset=["date", "close"]
    )

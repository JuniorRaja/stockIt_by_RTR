"""Parquet-based caching for Stocron by RTR."""

import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import hashlib
import json
import logging

logger = logging.getLogger(__name__)


class CacheManager:
    """Manages Parquet-based caching for fast data access."""
    
    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.price_dir = self.cache_dir / "prices"
        self.financials_dir = self.cache_dir / "financials"
        self.analysis_dir = self.cache_dir / "analysis"
        self.metadata_dir = self.cache_dir / "metadata"
        
        for dir_path in [self.price_dir, self.financials_dir, self.analysis_dir, self.metadata_dir]:
            dir_path.mkdir(exist_ok=True)
        
        self._cache_metadata: Dict[str, Dict] = {}
        self._load_metadata()
    
    def _load_metadata(self):
        metadata_file = self.metadata_dir / "cache_index.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    self._cache_metadata = json.load(f)
            except Exception:
                self._cache_metadata = {}
    
    def _save_metadata(self):
        metadata_file = self.metadata_dir / "cache_index.json"
        try:
            with open(metadata_file, 'w') as f:
                json.dump(self._cache_metadata, f)
        except Exception as e:
            logger.warning(f"Failed to save cache metadata: {e}")
    
    def _is_cache_valid(self, cache_key: str, max_age_hours: int = 24) -> bool:
        if cache_key not in self._cache_metadata:
            return False
        metadata = self._cache_metadata[cache_key]
        cached_time = datetime.fromisoformat(metadata.get('timestamp', '2000-01-01'))
        return datetime.now() - cached_time < timedelta(hours=max_age_hours)
    
    def cache_price_history(self, symbol: str, df: pd.DataFrame):
        if df.empty:
            return
        cache_key = f"{symbol.upper()}_prices"
        file_path = self.price_dir / f"{symbol.upper()}.parquet"
        try:
            df.to_parquet(file_path, index=False)
            self._cache_metadata[cache_key] = {'timestamp': datetime.now().isoformat(), 'rows': len(df)}
            self._save_metadata()
        except Exception as e:
            logger.error(f"Failed to cache prices for {symbol}: {e}")
    
    def get_cached_price_history(self, symbol: str, max_age_hours: int = 24) -> Optional[pd.DataFrame]:
        cache_key = f"{symbol.upper()}_prices"
        if not self._is_cache_valid(cache_key, max_age_hours):
            return None
        file_path = self.price_dir / f"{symbol.upper()}.parquet"
        if not file_path.exists():
            return None
        try:
            return pd.read_parquet(file_path)
        except Exception:
            return None
    
    def cache_stock_list(self, symbols: List[str]):
        file_path = self.metadata_dir / "nse_symbols.json"
        try:
            with open(file_path, 'w') as f:
                json.dump({'symbols': symbols, 'count': len(symbols), 'timestamp': datetime.now().isoformat()}, f)
        except Exception as e:
            logger.error(f"Failed to cache stock list: {e}")
    
    def get_cached_stock_list(self, max_age_hours: int = 24) -> Optional[List[str]]:
        file_path = self.metadata_dir / "nse_symbols.json"
        if not file_path.exists():
            return None
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            cached_time = datetime.fromisoformat(data['timestamp'])
            if datetime.now() - cached_time > timedelta(hours=max_age_hours):
                return None
            return data.get('symbols', [])
        except Exception:
            return None
    
    def get_cache_stats(self) -> Dict[str, Any]:
        total_size = sum(f.stat().st_size for d in [self.price_dir, self.financials_dir, self.analysis_dir] for f in d.rglob("*") if f.is_file())
        return {
            'total_entries': len(self._cache_metadata),
            'price_files': len(list(self.price_dir.glob("*.parquet"))),
            'total_size_mb': round(total_size / (1024 * 1024), 2)
        }
    
    def clear_cache(self, data_type: Optional[str] = None):
        if data_type is None or data_type == 'prices':
            for f in self.price_dir.glob("*.parquet"):
                f.unlink()
        if data_type is None:
            self._cache_metadata = {}
        self._save_metadata()
        logger.info(f"Cleared cache: {data_type or 'all'}")

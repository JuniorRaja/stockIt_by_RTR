#!/usr/bin/env python3
"""
Download and manage delisted/suspended stock data for survivorship bias correction.

Survivorship bias occurs when analysis only includes currently successful companies.
This leads to overly optimistic ML models that don't account for failure cases.

Usage:
    python scripts/download_delisted_stocks.py --list
    python scripts/download_delisted_stocks.py --export
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
DELISTED_DIR = DATA_DIR / 'delisted'


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


def main():
    parser = argparse.ArgumentParser(description='Manage delisted stock data')
    parser.add_argument('--list', action='store_true', help='List all known delisted stocks')
    parser.add_argument('--export', action='store_true', help='Export delisted stocks summary')
    
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
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()

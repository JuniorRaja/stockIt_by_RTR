#!/usr/bin/env python3
"""
Indian Equity Intelligence - First Run Setup Script
Downloads initial data and sets up the local database.

Run with: python setup.py
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def check_dependencies():
    """Check if required dependencies are installed."""
    required = ['streamlit', 'yfinance', 'pandas', 'numpy', 'duckdb', 'plotly', 'sklearn', 'yaml']
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    
    if missing:
        print(f"Missing: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False
    print("✓ Dependencies installed")
    return True


def create_directories():
    """Create required directories."""
    dirs = ['data/cache/prices', 'data/cache/financials', 'data/cache/metadata', 'data/db', 'models']
    for d in dirs:
        (PROJECT_ROOT / d).mkdir(parents=True, exist_ok=True)
    print("✓ Directories created")


def get_popular_stocks():
    """Return list of popular NSE stocks."""
    return ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK', 'HINDUNILVR', 'SBIN', 'BHARTIARTL',
            'ITC', 'KOTAKBANK', 'LT', 'AXISBANK', 'ASIANPAINT', 'MARUTI', 'HCLTECH', 'WIPRO',
            'TITAN', 'BAJFINANCE', 'SUNPHARMA', 'NESTLEIND', 'TATAMOTORS', 'M&M', 'DRREDDY',
            'TECHM', 'BRITANNIA', 'PIDILITIND', 'HAVELLS', 'GODREJCP', 'DABUR', 'MARICO']


def download_data(data_manager, symbols, years=10, delay=0.5):
    """Download historical data for stocks."""
    print(f"\nDownloading data for {len(symbols)} stocks...")
    print(f"Estimated time: {len(symbols) * delay / 60:.1f} minutes\n")
    
    successful, failed = 0, []
    for symbol in tqdm(symbols, desc="Downloading"):
        try:
            info = data_manager.get_stock_info(symbol)
            prices = data_manager.get_price_history(symbol, years=years)
            if info and prices is not None and not prices.empty:
                successful += 1
            else:
                failed.append(symbol)
        except Exception:
            failed.append(symbol)
        time.sleep(delay)
    
    print(f"\n✓ Downloaded data for {successful}/{len(symbols)} stocks")
    if failed:
        print(f"Failed: {', '.join(failed[:10])}" + (f" +{len(failed)-10} more" if len(failed) > 10 else ""))
    return successful, failed


def build_database(db_manager, data_manager, symbols):
    """Build local database."""
    print("\nBuilding database...")
    for symbol in tqdm(symbols, desc="Building DB"):
        try:
            info = data_manager.get_stock_info(symbol, use_cache=True)
            if info:
                db_manager.upsert_stock(info)
            prices = data_manager.get_price_history(symbol, use_cache=True)
            if prices is not None and not prices.empty:
                db_manager.upsert_price_history(symbol, prices)
        except Exception:
            pass
    print(f"✓ Database built with {db_manager.get_stock_count()} stocks")


def verify_setup():
    """Verify setup completed."""
    print("\n" + "=" * 50)
    print("VERIFICATION")
    print("=" * 50)
    
    checks = [
        ('Directories', all((PROJECT_ROOT / d).exists() for d in ['data/db', 'data/cache', 'config'])),
        ('Config file', (PROJECT_ROOT / 'config/settings.yaml').exists()),
        ('Main app', (PROJECT_ROOT / 'app.py').exists())
    ]
    
    all_pass = True
    for name, passed in checks:
        print(f"  {'✓' if passed else '✗'} {name}")
        if not passed:
            all_pass = False
    return all_pass


def main():
    parser = argparse.ArgumentParser(description="Indian Equity Intelligence - Setup")
    parser.add_argument('--subset', type=int, help='Number of stocks to download')
    parser.add_argument('--years', type=int, default=10, help='Years of history')
    parser.add_argument('--skip-download', action='store_true', help='Skip data download')
    args = parser.parse_args()
    
    print("=" * 50)
    print("INDIAN EQUITY INTELLIGENCE - SETUP")
    print("=" * 50)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print("Step 1/5: Checking dependencies...")
    if not check_dependencies():
        sys.exit(1)
    
    print("\nStep 2/5: Creating directories...")
    create_directories()
    
    print("\nStep 3/5: Initializing managers...")
    from src.data.sources import DataSourceManager
    from src.data.database import DatabaseManager
    data_manager = DataSourceManager()
    db_manager = DatabaseManager(str(PROJECT_ROOT / 'data/db/equity_intelligence.db'))
    print("✓ Managers initialized")
    
    if not args.skip_download:
        print("\nStep 4/5: Getting stock list...")
        symbols = get_popular_stocks()
        if args.subset:
            symbols = symbols[:args.subset]
        print(f"Will download: {len(symbols)} stocks")
        
        print("\nStep 5/5: Downloading data...")
        download_data(data_manager, symbols, years=args.years)
        build_database(db_manager, data_manager, symbols)
    else:
        print("\nSkipping data download (--skip-download)")
    
    success = verify_setup()
    
    print("\n" + "=" * 50)
    if success:
        print("SETUP COMPLETE!")
        print("=" * 50)
        print("\nTo start: streamlit run app.py")
        print("Open: http://localhost:8501")
    else:
        print("SETUP INCOMPLETE")
        print("=" * 50)
        sys.exit(1)


if __name__ == "__main__":
    main()

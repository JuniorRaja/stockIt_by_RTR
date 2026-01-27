#!/usr/bin/env python3
"""
Indian Equity Intelligence - First Run Setup Script
Downloads initial data and sets up the local database.

Run with: python setup.py
Options:
  --full          Download all stocks (2000+, takes ~2-3 hours)
  --subset N      Download only N stocks (default: 50 popular stocks)
  --skip-download Skip data download, only setup directories
  --years N       Years of historical data (default: 10)
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import time
import logging
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def check_dependencies():
    """Check if required dependencies are installed."""
    # Show Python version for debugging
    print(f"Python: {sys.executable}")
    print(f"Version: {sys.version.split()[0]}")
    print()
    
    required = {
        'streamlit': 'streamlit',
        'yfinance': 'yfinance',
        'pandas': 'pandas',
        'numpy': 'numpy',
        'duckdb': 'duckdb',
        'plotly': 'plotly',
        'sklearn': 'scikit-learn',
        'yaml': 'pyyaml',
        'nsetools': 'nsetools'
    }
    
    missing = []
    for module, package in required.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"Missing packages: {', '.join(missing)}")
        print()
        print(f"Install with: {sys.executable} -m pip install -r requirements.txt")
        print()
        print("NOTE: Make sure you're using the same Python version for pip and running scripts!")
        return False
    
    # Check optional
    optional = {'jugaad_data': 'jugaad-data', 'feedparser': 'feedparser'}
    for module, package in optional.items():
        try:
            __import__(module)
            print(f"✓ {package} (optional)")
        except ImportError:
            print(f"○ {package} (optional, not installed)")
    
    print("✓ All required dependencies installed")
    return True


def create_directories():
    """Create required directories."""
    dirs = [
        'data/cache/prices', 'data/cache/financials', 
        'data/cache/analysis', 'data/cache/metadata',
        'data/db', 'data/stock_lists', 'models', 'logs'
    ]
    for d in dirs:
        (PROJECT_ROOT / d).mkdir(parents=True, exist_ok=True)
    print("✓ Directories created")


def get_popular_stocks():
    """Return list of popular NSE stocks for quick setup."""
    return [
        # Large Cap - Top 30
        'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK', 'HINDUNILVR', 
        'SBIN', 'BHARTIARTL', 'ITC', 'KOTAKBANK', 'LT', 'AXISBANK',
        'ASIANPAINT', 'MARUTI', 'HCLTECH', 'WIPRO', 'TITAN', 'BAJFINANCE',
        'SUNPHARMA', 'NESTLEIND', 'DRREDDY', 'TECHM', 'BRITANNIA', 'PIDILITIND',
        'HAVELLS', 'GODREJCP', 'DABUR', 'MARICO', 'HEROMOTOCO', 'EICHERMOT',
        # Mid Cap - Top 20
        'TATAPOWER', 'INDIGO', 'VEDL', 'GAIL', 'BANKBARODA', 'PNB',
        'IDFC', 'IDFCFIRSTB', 'FEDERALBNK', 'INDUSINDBK', 'LICHSGFIN',
        'MUTHOOTFIN', 'CHOLAFIN', 'SBILIFE', 'HDFCLIFE', 'ICICIGI',
        'BAJAJFINSV', 'TATACONSUM', 'VOLTAS', 'PAGEIND'
    ]


def fetch_all_stock_list(data_manager):
    """Fetch and save complete stock list."""
    print("\nFetching complete stock list from NSE and BSE...")
    
    stocks_df = data_manager.get_all_stocks(force_refresh=True)
    
    if not stocks_df.empty:
        # Save to JSON for quick loading
        stock_list_path = PROJECT_ROOT / 'data' / 'stock_lists' / 'all_stocks.json'
        stocks_dict = stocks_df.to_dict('records')
        with open(stock_list_path, 'w') as f:
            json.dump({
                'stocks': stocks_dict,
                'count': len(stocks_dict),
                'updated_at': datetime.now().isoformat()
            }, f)
        print(f"✓ Saved {len(stocks_dict)} stocks to stock list")
        return stocks_df
    else:
        print("⚠ Could not fetch stock list")
        return None


def download_stock_data(data_manager, symbols: list, years: int = 10, delay: float = 1.0):
    """Download historical data for stocks."""
    print(f"\nDownloading data for {len(symbols)} stocks...")
    print(f"This may take {len(symbols) * delay / 60:.1f} - {len(symbols) * delay * 2 / 60:.1f} minutes")
    print("(Yahoo Finance rate limits may cause some failures - this is normal)\n")
    
    successful = 0
    failed = []
    partial = []  # Got info but no price history
    
    for symbol in tqdm(symbols, desc="Downloading"):
        try:
            # Get stock info first
            info = data_manager.get_stock_info(symbol, use_cache=False)
            
            # Get price history
            prices = data_manager.get_price_history(symbol, years=years, use_cache=False)
            
            if info and prices is not None and not prices.empty:
                successful += 1
            elif info:
                partial.append(symbol)
            else:
                failed.append(symbol)
            
            # Rate limiting
            time.sleep(delay)
            
        except KeyboardInterrupt:
            print("\n⚠ Download interrupted by user")
            break
        except Exception as e:
            logger.debug(f"Error downloading {symbol}: {e}")
            failed.append(symbol)
    
    print(f"\n✓ Successfully downloaded: {successful}/{len(symbols)} stocks")
    if partial:
        print(f"○ Partial (info only): {len(partial)} stocks")
    if failed:
        print(f"✗ Failed: {len(failed)} stocks")
        if len(failed) <= 10:
            print(f"  Failed symbols: {', '.join(failed)}")
    
    return successful, partial, failed


def build_database(db_manager, data_manager, symbols: list):
    """Build local database with downloaded data."""
    print("\nBuilding local database...")
    
    count = 0
    for symbol in tqdm(symbols, desc="Building DB"):
        try:
            info = data_manager.get_stock_info(symbol, use_cache=True)
            if info:
                db_manager.upsert_stock(info)
                count += 1
            
            prices = data_manager.get_price_history(symbol, use_cache=True)
            if prices is not None and not prices.empty:
                db_manager.upsert_price_history(symbol, prices)
        except Exception as e:
            logger.debug(f"DB error for {symbol}: {e}")
    
    print(f"✓ Database built with {count} stocks")
    return count


def verify_setup():
    """Verify setup completed correctly."""
    print("\n" + "=" * 50)
    print("VERIFICATION")
    print("=" * 50)
    
    checks = [
        ('Data directories', (PROJECT_ROOT / 'data' / 'db').exists()),
        ('Cache directories', (PROJECT_ROOT / 'data' / 'cache').exists()),
        ('Config file', (PROJECT_ROOT / 'config' / 'settings.yaml').exists()),
        ('Main app', (PROJECT_ROOT / 'app.py').exists()),
        ('Stock list', (PROJECT_ROOT / 'data' / 'stock_lists' / 'all_stocks.json').exists()),
    ]
    
    all_pass = True
    for name, passed in checks:
        status = '✓' if passed else '✗'
        print(f"  {status} {name}")
        if not passed:
            all_pass = False
    
    return all_pass


def main():
    parser = argparse.ArgumentParser(description="Indian Equity Intelligence - Setup")
    parser.add_argument('--full', action='store_true', help='Download all stocks (2000+)')
    parser.add_argument('--subset', type=int, default=50, help='Number of stocks to download (default: 50)')
    parser.add_argument('--years', type=int, default=10, help='Years of historical data (default: 10)')
    parser.add_argument('--skip-download', action='store_true', help='Skip data download')
    parser.add_argument('--delay', type=float, default=1.0, help='Delay between requests (default: 1.0s)')
    args = parser.parse_args()
    
    print("=" * 50)
    print("INDIAN EQUITY INTELLIGENCE - SETUP")
    print("=" * 50)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Step 1: Check dependencies
    print("Step 1/5: Checking dependencies...")
    if not check_dependencies():
        sys.exit(1)
    
    # Step 2: Create directories
    print("\nStep 2/5: Creating directories...")
    create_directories()
    
    # Step 3: Initialize managers
    print("\nStep 3/5: Initializing data managers...")
    from src.data.sources import DataSourceManager
    from src.data.database import DatabaseManager
    
    data_manager = DataSourceManager()
    db_manager = DatabaseManager(str(PROJECT_ROOT / 'data' / 'db' / 'equity_intelligence.db'))
    
    # Show available sources
    available = data_manager.get_available_sources()
    print(f"Available data sources: {', '.join(available) if available else 'None detected yet'}")
    
    # Step 4: Fetch stock list and download data
    if not args.skip_download:
        print("\nStep 4/5: Fetching stock list...")
        stocks_df = fetch_all_stock_list(data_manager)
        
        # Determine which stocks to download
        if args.full and stocks_df is not None:
            symbols = stocks_df['symbol'].tolist()
            print(f"Full download mode: {len(symbols)} stocks")
        else:
            symbols = get_popular_stocks()[:args.subset]
            print(f"Downloading {len(symbols)} popular stocks")
        
        print("\nStep 5/5: Downloading historical data...")
        successful, partial, failed = download_stock_data(
            data_manager, symbols, years=args.years, delay=args.delay
        )
        
        # Build database
        build_database(db_manager, data_manager, symbols)
    else:
        print("\nStep 4/5: Skipping data download (--skip-download)")
        print("Step 5/5: Fetching stock list only...")
        fetch_all_stock_list(data_manager)
    
    # Verify
    success = verify_setup()
    
    print("\n" + "=" * 50)
    if success:
        print("SETUP COMPLETE!")
        print("=" * 50)
        print("\nTo start the application:")
        print("  streamlit run app.py")
        print("\nOpen in browser:")
        print("  http://localhost:8501")
        print("\nTo download more stocks later:")
        print("  python setup.py --full")
    else:
        print("SETUP INCOMPLETE - Some steps failed")
        print("=" * 50)
        print("Try running again or check the logs.")
        sys.exit(1)


if __name__ == "__main__":
    main()

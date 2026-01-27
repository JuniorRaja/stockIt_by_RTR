#!/usr/bin/env python3
"""
Indian Equity Intelligence - Complete Data Download Script

Downloads historical data for ALL NSE stocks and stores locally.
Run this ONCE with internet connection, then the app works offline.

Usage:
    python download_all_data.py                  # Download all stocks
    python download_all_data.py --workers 4      # Use 4 parallel workers
    python download_all_data.py --resume         # Resume interrupted download
    python download_all_data.py --symbols TCS,INFY  # Download specific stocks
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# Check dependencies first
def check_deps():
    missing = []
    for mod in ['pandas', 'yfinance', 'nsetools']:
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        print(f"ERROR: Missing packages: {', '.join(missing)}")
        print(f"\nPython being used: {sys.executable} (v{sys.version.split()[0]})")
        print(f"\nFix: {sys.executable} -m pip install -r requirements.txt")
        sys.exit(1)

check_deps()

import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('download.log')
    ]
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / 'data'
PRICE_DIR = DATA_DIR / 'prices'
INFO_DIR = DATA_DIR / 'stock_info'
PROGRESS_FILE = DATA_DIR / 'download_progress.json'


def setup_directories():
    """Create data directories."""
    for dir_path in [DATA_DIR, PRICE_DIR, INFO_DIR, DATA_DIR / 'stock_lists']:
        dir_path.mkdir(parents=True, exist_ok=True)


def get_all_nse_symbols():
    """Get list of all NSE symbols."""
    symbols = []
    
    # Try nsetools first
    try:
        from nsetools import Nse
        nse = Nse()
        codes = nse.get_stock_codes()
        # Handle both dict and list formats
        if isinstance(codes, dict):
            symbols = [k for k in codes.keys() if k != 'SYMBOL']
        elif isinstance(codes, list):
            # New format: list of dicts or tuples
            for item in codes:
                if isinstance(item, dict):
                    symbols.append(item.get('symbol', item.get('SYMBOL', '')))
                elif isinstance(item, (list, tuple)) and len(item) >= 1:
                    symbols.append(str(item[0]))
            symbols = [s for s in symbols if s and s != 'SYMBOL']
        logger.info(f"Got {len(symbols)} symbols from nsetools")
    except Exception as e:
        logger.warning(f"nsetools failed: {e}")
    
    # Fallback to saved list
    if not symbols:
        stock_list_file = DATA_DIR / 'stock_lists' / 'all_stocks.json'
        if stock_list_file.exists():
            try:
                with open(stock_list_file) as f:
                    data = json.load(f)
                    symbols = [s['symbol'] for s in data.get('stocks', [])]
                    logger.info(f"Got {len(symbols)} symbols from saved list")
            except:
                pass
    
    # Final fallback: NIFTY 500 stocks (most liquid)
    if not symbols:
        logger.info("Using built-in NIFTY 500 stock list")
        symbols = get_nifty500_stocks()
    
    return symbols


def get_nifty500_stocks():
    """Built-in list of NIFTY 500 stocks."""
    # Top 500 NSE stocks by market cap (as of 2024)
    return [
        "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "HINDUNILVR", "SBIN",
        "BHARTIARTL", "ITC", "KOTAKBANK", "LT", "AXISBANK", "ASIANPAINT", "MARUTI",
        "HCLTECH", "WIPRO", "TITAN", "BAJFINANCE", "SUNPHARMA", "NESTLEIND",
        "TATAMOTORS", "DRREDDY", "TECHM", "BRITANNIA", "PIDILITIND", "HAVELLS",
        "GODREJCP", "DABUR", "MARICO", "HEROMOTOCO", "EICHERMOT", "BAJAJ-AUTO",
        "M&M", "ULTRACEMCO", "GRASIM", "ADANIENT", "ADANIPORTS", "POWERGRID",
        "NTPC", "ONGC", "COALINDIA", "IOC", "BPCL", "GAIL", "JSWSTEEL",
        "TATASTEEL", "HINDALCO", "VEDL", "TATAPOWER", "INDIGO", "DIVISLAB",
        "CIPLA", "APOLLOHOSP", "SBILIFE", "HDFCLIFE", "ICICIGI", "BAJAJFINSV",
        "TATACONSUM", "VOLTAS", "PAGEIND", "INDUSINDBK", "BANKBARODA", "PNB",
        "FEDERALBNK", "IDFCFIRSTB", "LICHSGFIN", "MUTHOOTFIN", "CHOLAFIN",
        "SHREECEM", "AMBUJACEM", "ACC", "DMART", "NAUKRI", "MPHASIS",
        "LTIM", "PERSISTENT", "COFORGE", "TATAELXSI", "LTTS", "PIIND",
        "AARTIIND", "SRF", "ASTRAL", "POLYCAB", "KEI", "CROMPTON",
        "BLUEDART", "CONCOR", "MOTHERSON", "BOSCHLTD", "MRF", "BALKRISIND",
        "APOLLOTYRE", "CEAT", "EXIDEIND", "AMARAJABAT", "TVSMOTOR", "ESCORTS",
        "BHARATFORG", "SCHAEFFLER", "SKFINDIA", "TIMKEN", "CUMMINSIND", "THERMAX",
        "AIAENG", "GRINDWELL", "CARBORUNIV", "SUPREMEIND", "FINOLEX", "APLAPOLLO",
        "JINDALSAW", "RATNAMANI", "MAHSEAMLES", "WELCORP", "MANAKSIA", "TIINDIA",
        "HINDPETRO", "MRPL", "CHENNPETRO", "CASTROLIND", "GSPL", "GUJGASLTD",
        "MAHANAGAR", "IGL", "ATGL", "PETRONET", "RVNL", "IRFC", "IRCTC",
        "RAILTEL", "RITES", "BEL", "HAL", "BHEL", "GRSE", "COCHINSHIP",
        "MAZAGON", "NBCC", "NCC", "ASHOKA", "PNC", "CAPACITE", "JKCEMENT",
        "RAMCOCEM", "HEIDELBERG", "PRISMCEM", "ORIENTCEM", "DALBHARAT", "JKLAKSHMI",
        "STARCEMENT", "KARURVYSYA", "DCBBANK", "BANDHANBNK", "RBLBANK", "UJJIVANSFB",
        "CUB", "EQUITASBNK", "SURYAROSNI", "ORIENTELEC", "JYOTHYLAB", "VSTIND",
        "GODFRYPHLP", "RADICO", "GLOBUSSPR", "UNITDSPR", "TIPSINDLTD", "SAREGAMA",
        "PVRINOX", "ZEEL", "SUNTV", "TV18BRDCST", "NETWORK18", "NAZARA", "DELTACORP",
        "TRIDENT", "WELSPUNIND", "HIMATSEIDE", "RAYMOND", "ARVIND", "KPITTECH",
        "CYIENT", "MASTEK", "SONATSOFTW", "HAPPSTMNDS", "ROUTE", "NEWGEN",
        "LATENTVIEW", "TATACOMM", "STLTECH", "HFCL", "TEJAS", "VODAFONE",
        "IDEA", "TTML", "ITI", "KAYNES", "DIXON", "AMBER", "AETHER",
        "CLEAN", "PPLPHARMA", "GRANULES", "LAURUSLABS", "NATCOPHARM", "AUROPHARMA",
        "ALKEM", "TORNTPHARM", "GLENMARK", "BIOCON", "ZYDUSLIFE", "LUPIN",
        "IPCALAB", "ABBOTINDIA", "PFIZER", "GLAXO", "SANOFI", "JBCHEPHARM",
        "AJANTPHARM", "LALPATHLAB", "METROPOLIS", "THYROCARE", "MAXHEALTH", "FORTIS",
        "MEDANTA", "ASTER", "KIMS", "NH", "YATHARTH", "RAINBOW", "SYNGENE",
        "GLAND", "SHILPAMED", "JUBLINGREA", "FLUOROCHEM", "NAVINFLUOR", "DEEPAKFERT",
        "GNFC", "GSFC", "COROMANDEL", "UPL", "RALLIS", "BAYER", "FMC",
        "SUMICHEM", "DHANUKA", "GODREJAGRO", "INSECTICIDE", "SHRIRAMPPS", "KRBL",
        "LTFOODS", "AVANTIFEED", "WATERBASE", "APEX", "CERA", "SOMANY",
        "HINDWARE", "KAJARIA", "ORIENTBELL", "JSPL", "NMDC", "VEDL", "MOIL",
        "GMRAIRPORT", "GMRINFRA", "ADANIGREEN", "ADANITRANS", "TATAPOWER", "NHPC",
        "SJVN", "TORNTPOWER", "CESC", "JSW", "JSWENERGY", "JPPOWER", "RPOWER",
        "NESCO", "SOBHA", "BRIGADE", "PRESTIGE", "GODREJPROP", "DLF", "OBEROIRLTY",
        "PHOENIXLTD", "MAHLIFE", "LODHA", "SUNTECK", "KOLTEPATIL", "ASHIANA",
        "PGHH", "COLPAL", "GILLETTE", "KANSAINER", "BERGEPAINT", "AKZOINDIA",
        "CENTURYPLY", "GREENPLY", "GREENPANEL", "RUSHIL", "BAJAJHIND", "DWARIKESH",
        "DHAMPURSUG", "BALRAMCHIN", "SHARDACROP", "DCW", "TATACHEM", "ATUL",
        "GALAXYSURF", "FINEORG", "CLEAN", "ANUPAM", "INOXWIND", "SUZLON",
        "SIEMENS", "ABB", "CGPOWER", "BHEL", "POWERMECH", "KALPATPOWR", "KEC",
        "LXCHEM", "IONEXCHANG", "SAFARI", "VIP", "BATAINDIA", "RELAXO",
        "CAMPUS", "METROBRAND", "MANYAVAR", "SHOPERSTOP", "TRENT", "VMART",
        "DMART", "ABFRL", "LUXIND", "DOLLAR", "RUPA", "PGEL", "ORIENTCEM",
        "JKPAPER", "SATIA", "WSTCSTPAPR", "TNPL", "ANDHRAPAP", "CENTURYTEX",
        "SOMANYCERA", "KAJARIA", "CERA", "HSIL", "BOROSIL", "LAOPALA",
        "INDHOTEL", "LEMONTRE", "CHALET", "EIH", "TAJGVK", "MAHINDCIE",
        "AUTOAXLES", "SUNDARMFIN", "SUNDARMHLD", "MSTC", "MMTC", "STC", "HUDCO",
        "PFC", "RECLTD", "CANFINHOME", "HOMEFIRST", "AAVAS", "APTUS", "REPCO",
        "SHRIRAMFIN", "MANAPPURAM", "IIFL", "POONAWALLA", "CREDITACC", "FUSION",
        "SPANDANA", "UGROCAP", "SATIN", "CAMS", "CDSL", "MCX", "BSE", "IEX",
        "ANGELONE", "ICICISEC", "MOTILALOFS", "360ONE", "HDFCAMC", "NIPPONIND",
        "UTIAMC", "NAM-INDIA", "MFSL", "SBICARD", "BAJAJHFL", "CANFINHOME",
        "LICHSGFIN", "PNBHOUSING", "INDIASHLTR", "IBREALEST", "RUSTOMJEE",
        "MAHLOG", "ALLCARGO", "TCIEXP", "GATI", "BLUEDARTES", "DELHIVERY",
        "AEGISCHEM", "GRSE", "EIDPARRY", "JKIL", "ZOMATO", "PAYTM", "NYKAA",
        "POLICYBZR", "CARTRADE", "RATEGAIN", "EASEMYTRIP", "IXIGO", "YATRA",
        "THOMASCOOK", "SOTL", "MAHSCOOTER", "VESUVIUS", "CARYSIL", "CELLO",
        "WINDLAS", "LAURUS", "MEDPLUS", "MANKIND", "ERIS", "SOLARA", "CAPLIPOINT"
    ]


def download_stock_data(symbol: str, years: int = 15) -> dict:
    """Download all data for a single stock."""
    result = {
        'symbol': symbol,
        'success': False,
        'price_records': 0,
        'info': False,
        'error': None
    }
    
    try:
        import yfinance as yf
        
        # Try NSE first, then BSE
        for suffix in ['.NS', '.BO']:
            try:
                ticker = yf.Ticker(f"{symbol}{suffix}")
                
                # Get price history
                df = ticker.history(period="max")
                
                if df.empty:
                    df = ticker.history(period=f"{years}y")
                
                if not df.empty:
                    # Process and save price data
                    df = df.reset_index()
                    df.columns = [c.lower().replace(' ', '_') for c in df.columns]
                    
                    # Handle timezone
                    if 'date' in df.columns:
                        try:
                            if hasattr(df['date'].dt, 'tz') and df['date'].dt.tz is not None:
                                df['date'] = df['date'].dt.tz_localize(None)
                        except:
                            pass
                        df['date'] = pd.to_datetime(df['date'])
                    
                    df['symbol'] = symbol
                    df['source'] = f'yahoo{suffix}'
                    
                    # Save to parquet
                    price_file = PRICE_DIR / f"{symbol}.parquet"
                    df.to_parquet(price_file, index=False)
                    result['price_records'] = len(df)
                    
                    # Get stock info
                    info = ticker.info
                    if info:
                        info_file = INFO_DIR / f"{symbol}.json"
                        # Filter serializable fields
                        clean_info = {k: v for k, v in info.items() 
                                     if isinstance(v, (str, int, float, bool, type(None)))}
                        clean_info['symbol'] = symbol
                        clean_info['downloaded_at'] = datetime.now().isoformat()
                        
                        with open(info_file, 'w') as f:
                            json.dump(clean_info, f)
                        result['info'] = True
                    
                    result['success'] = True
                    return result
                    
            except Exception as e:
                continue
        
        result['error'] = "No data from any source"
        
    except Exception as e:
        result['error'] = str(e)
    
    return result


def load_progress():
    """Load download progress."""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {'completed': [], 'failed': [], 'last_update': None}


def save_progress(progress):
    """Save download progress."""
    progress['last_update'] = datetime.now().isoformat()
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f)


def download_all(symbols: list, workers: int = 2, delay: float = 1.0, resume: bool = False):
    """Download data for all symbols with parallel workers."""
    progress = load_progress() if resume else {'completed': [], 'failed': [], 'last_update': None}
    
    # Filter out already completed
    if resume:
        symbols = [s for s in symbols if s not in progress['completed']]
        logger.info(f"Resuming: {len(progress['completed'])} already done, {len(symbols)} remaining")
    
    total = len(symbols)
    completed = 0
    failed = 0
    
    logger.info(f"Starting download of {total} stocks with {workers} workers")
    start_time = time.time()
    
    def download_with_delay(symbol):
        result = download_stock_data(symbol)
        time.sleep(delay)  # Rate limiting
        return result
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(download_with_delay, s): s for s in symbols}
        
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                result = future.result()
                
                if result['success']:
                    progress['completed'].append(symbol)
                    completed += 1
                    logger.info(f"✓ {symbol}: {result['price_records']} records")
                else:
                    progress['failed'].append({'symbol': symbol, 'error': result['error']})
                    failed += 1
                    logger.warning(f"✗ {symbol}: {result['error']}")
                
                # Progress update
                done = completed + failed
                elapsed = time.time() - start_time
                rate = done / elapsed if elapsed > 0 else 0
                eta = (total - done) / rate if rate > 0 else 0
                
                if done % 10 == 0:
                    logger.info(f"Progress: {done}/{total} ({done/total*100:.1f}%) - "
                               f"Success: {completed}, Failed: {failed} - "
                               f"ETA: {eta/60:.1f} min")
                    save_progress(progress)
                    
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                progress['failed'].append({'symbol': symbol, 'error': str(e)})
                failed += 1
    
    save_progress(progress)
    
    elapsed = time.time() - start_time
    logger.info(f"\n{'='*50}")
    logger.info(f"DOWNLOAD COMPLETE")
    logger.info(f"{'='*50}")
    logger.info(f"Total: {total}")
    logger.info(f"Success: {completed} ({completed/total*100:.1f}%)")
    logger.info(f"Failed: {failed}")
    logger.info(f"Time: {elapsed/60:.1f} minutes")
    logger.info(f"Data saved to: {DATA_DIR}")
    
    return completed, failed


def build_database():
    """Build DuckDB database from downloaded parquet files."""
    logger.info("Building database from downloaded files...")
    
    import duckdb
    
    db_path = DATA_DIR / 'db' / 'equity_intelligence.db'
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    conn = duckdb.connect(str(db_path))
    
    # Create tables
    conn.execute("""
        CREATE TABLE IF NOT EXISTS stocks (
            symbol VARCHAR PRIMARY KEY,
            name VARCHAR,
            sector VARCHAR,
            industry VARCHAR,
            market_cap DOUBLE,
            current_price DOUBLE,
            pe_ratio DOUBLE,
            pb_ratio DOUBLE,
            dividend_yield DOUBLE,
            fifty_two_week_high DOUBLE,
            fifty_two_week_low DOUBLE,
            updated_at TIMESTAMP
        )
    """)
    
    conn.execute("""
        CREATE TABLE IF NOT EXISTS price_history (
            symbol VARCHAR,
            date DATE,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            volume BIGINT,
            source VARCHAR,
            PRIMARY KEY (symbol, date)
        )
    """)
    
    # Load stock info
    info_files = list(INFO_DIR.glob("*.json"))
    logger.info(f"Loading {len(info_files)} stock info files...")
    
    for info_file in info_files:
        try:
            with open(info_file) as f:
                info = json.load(f)
            
            conn.execute("""
                INSERT OR REPLACE INTO stocks 
                (symbol, name, sector, industry, market_cap, current_price, 
                 pe_ratio, pb_ratio, dividend_yield, fifty_two_week_high, 
                 fifty_two_week_low, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, [
                info.get('symbol'),
                info.get('longName', info.get('shortName', info.get('symbol'))),
                info.get('sector'),
                info.get('industry'),
                info.get('marketCap'),
                info.get('currentPrice', info.get('regularMarketPrice')),
                info.get('trailingPE'),
                info.get('priceToBook'),
                info.get('dividendYield'),
                info.get('fiftyTwoWeekHigh'),
                info.get('fiftyTwoWeekLow')
            ])
        except Exception as e:
            logger.debug(f"Error loading {info_file}: {e}")
    
    # Load price history from parquet files
    price_files = list(PRICE_DIR.glob("*.parquet"))
    logger.info(f"Loading {len(price_files)} price history files...")
    
    for price_file in price_files:
        try:
            symbol = price_file.stem
            conn.execute(f"""
                INSERT OR REPLACE INTO price_history 
                SELECT symbol, date, open, high, low, close, 
                       CAST(volume AS BIGINT) as volume, source
                FROM read_parquet('{price_file}')
            """)
        except Exception as e:
            logger.debug(f"Error loading {price_file}: {e}")
    
    # Create indexes
    conn.execute("CREATE INDEX IF NOT EXISTS idx_price_symbol ON price_history(symbol)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_price_date ON price_history(date)")
    
    # Get stats
    stock_count = conn.execute("SELECT COUNT(*) FROM stocks").fetchone()[0]
    price_count = conn.execute("SELECT COUNT(*) FROM price_history").fetchone()[0]
    
    conn.close()
    
    logger.info(f"Database built: {stock_count} stocks, {price_count} price records")
    logger.info(f"Database path: {db_path}")


def main():
    parser = argparse.ArgumentParser(description="Download all Indian stock market data")
    parser.add_argument('--workers', type=int, default=2, help='Parallel workers (default: 2)')
    parser.add_argument('--delay', type=float, default=1.5, help='Delay between requests (default: 1.5s)')
    parser.add_argument('--resume', action='store_true', help='Resume interrupted download')
    parser.add_argument('--symbols', type=str, help='Comma-separated symbols to download')
    parser.add_argument('--build-db-only', action='store_true', help='Only build database from existing files')
    parser.add_argument('--years', type=int, default=15, help='Years of history (default: 15)')
    args = parser.parse_args()
    
    print("=" * 60)
    print("INDIAN EQUITY INTELLIGENCE - COMPLETE DATA DOWNLOAD")
    print("=" * 60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    setup_directories()
    
    if args.build_db_only:
        build_database()
        return
    
    # Get symbols
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(',')]
    else:
        symbols = get_all_nse_symbols()
    
    if not symbols:
        print("ERROR: No symbols found. Please run setup.py first or provide --symbols")
        sys.exit(1)
    
    print(f"Symbols to download: {len(symbols)}")
    print(f"Workers: {args.workers}")
    print(f"Delay: {args.delay}s")
    print(f"Estimated time: {len(symbols) * args.delay / args.workers / 60:.0f} minutes")
    print()
    
    # Confirm
    if len(symbols) > 100 and not args.resume:
        response = input("This will download a lot of data. Continue? [y/N]: ")
        if response.lower() != 'y':
            print("Aborted.")
            return
    
    # Download
    completed, failed = download_all(
        symbols, 
        workers=args.workers, 
        delay=args.delay, 
        resume=args.resume
    )
    
    # Build database
    if completed > 0:
        print("\nBuilding database...")
        build_database()
    
    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE!")
    print("=" * 60)
    print(f"\nThe app will now work OFFLINE with {completed} stocks.")
    print("To start: docker-compose up")


if __name__ == "__main__":
    main()

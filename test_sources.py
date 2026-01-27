#!/usr/bin/env python3
"""Test script to debug data source issues."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

def test_yfinance():
    """Test Yahoo Finance directly."""
    print("\n=== Testing Yahoo Finance ===")
    try:
        import yfinance as yf
        print("✓ yfinance imported")
        
        # Test TCS
        for suffix in ['.NS', '.BO']:
            ticker = yf.Ticker(f'TCS{suffix}')
            print(f"\nTrying TCS{suffix}...")
            
            # Test info
            try:
                info = ticker.info
                price = info.get('currentPrice') or info.get('regularMarketPrice')
                print(f"  Price: {price}")
            except Exception as e:
                print(f"  Info error: {e}")
            
            # Test history
            try:
                hist = ticker.history(period='5d')
                print(f"  History (5d): {len(hist)} rows")
                if not hist.empty:
                    print(f"  Last close: {hist['Close'].iloc[-1]:.2f}")
                    return True
            except Exception as e:
                print(f"  History error: {e}")
        
    except ImportError:
        print("✗ yfinance not installed")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    return False


def test_jugaad():
    """Test Jugaad Data."""
    print("\n=== Testing Jugaad Data ===")
    try:
        from jugaad_data.nse import stock_df
        from datetime import datetime, timedelta
        print("✓ jugaad_data imported")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        try:
            df = stock_df(symbol='TCS', from_date=start_date.date(), to_date=end_date.date())
            print(f"  Got {len(df)} rows for TCS")
            if not df.empty:
                return True
        except Exception as e:
            print(f"  Error fetching: {e}")
        
    except ImportError:
        print("○ jugaad_data not installed (optional)")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    return False


def test_nsetools():
    """Test NSE Tools."""
    print("\n=== Testing NSE Tools ===")
    try:
        from nsetools import Nse
        print("✓ nsetools imported")
        
        nse = Nse()
        
        # Test stock codes
        try:
            codes = nse.get_stock_codes()
            print(f"  Stock codes: {len(codes)}")
        except Exception as e:
            print(f"  Stock codes error: {e}")
        
        # Test quote
        try:
            quote = nse.get_quote('TCS')
            if quote:
                print(f"  TCS price: {quote.get('lastPrice')}")
                return True
        except Exception as e:
            print(f"  Quote error: {e}")
        
    except ImportError:
        print("✗ nsetools not installed")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    return False


def test_data_manager():
    """Test the DataSourceManager."""
    print("\n=== Testing DataSourceManager ===")
    try:
        from src.data.sources import DataSourceManager
        
        dm = DataSourceManager()
        
        # Available sources
        sources = dm.get_available_sources()
        print(f"Available sources: {sources}")
        
        # Test stock info
        print("\nGetting TCS info...")
        info = dm.get_stock_info('TCS')
        if info:
            print(f"  Name: {info.get('name')}")
            print(f"  Price: {info.get('current_price')}")
            print(f"  Source: {info.get('source')}")
        else:
            print("  ✗ No info returned")
        
        # Test price history
        print("\nGetting TCS price history...")
        prices = dm.get_price_history('TCS', years=1)
        if prices is not None and not prices.empty:
            print(f"  Got {len(prices)} price records")
            print(f"  Source: {prices['source'].iloc[0]}")
            print(f"  Last date: {prices['date'].iloc[-1]}")
            print(f"  Last close: {prices['close'].iloc[-1]:.2f}")
            return True
        else:
            print("  ✗ No price history returned")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    return False


def main():
    print("=" * 50)
    print("DATA SOURCE DIAGNOSTIC TEST")
    print("=" * 50)
    
    results = {
        'yfinance': test_yfinance(),
        'jugaad': test_jugaad(),
        'nsetools': test_nsetools(),
        'data_manager': test_data_manager()
    }
    
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    for name, passed in results.items():
        status = '✓ PASS' if passed else '✗ FAIL'
        print(f"  {name}: {status}")
    
    if results['data_manager']:
        print("\n✓ Data sources working correctly!")
    else:
        print("\n⚠ Data source issues detected.")
        print("\nTroubleshooting:")
        print("1. Check internet connection")
        print("2. Try: pip install --upgrade yfinance")
        print("3. Yahoo Finance may be rate limiting - wait a few minutes")
        print("4. Try installing jugaad-data: pip install jugaad-data --no-deps")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Quick download - Downloads top 100 stocks for immediate use.
Run this first, then run download_all_data.py for complete data.
"""

import subprocess
import sys

# Top 100 most traded NSE stocks
TOP_100_STOCKS = [
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
    "APOLLOTYRE", "CEAT", "EXIDEIND", "AMARAJABAT", "TVSMOTOR", "ESCORTS"
]

if __name__ == "__main__":
    print("=" * 60)
    print("QUICK DOWNLOAD - Top 100 Stocks")
    print("=" * 60)
    print()
    print("This will download data for the top 100 most traded stocks.")
    print("For complete market data, run: python download_all_data.py")
    print()
    
    symbols = ",".join(TOP_100_STOCKS)
    
    subprocess.run([
        sys.executable, "download_all_data.py",
        "--symbols", symbols,
        "--workers", "3"
    ])
    
    print()
    print("Quick download complete!")
    print("You can now run: docker-compose up --build")
    print()
    print("For complete data (2000+ stocks), run:")
    print("  python download_all_data.py")

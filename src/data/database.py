"""Database management for Indian Equity Intelligence."""

import duckdb
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages local DuckDB database for storing stock data."""
    
    def __init__(self, db_path: str = "data/db/equity_intelligence.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = duckdb.connect(str(self.db_path))
        self._initialize_schema()
    
    def _initialize_schema(self):
        """Create database schema if not exists."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS stocks (
                symbol VARCHAR PRIMARY KEY, name VARCHAR, sector VARCHAR,
                industry VARCHAR, market_cap DOUBLE, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS price_history (
                symbol VARCHAR, date DATE, open DOUBLE, high DOUBLE, low DOUBLE,
                close DOUBLE, volume BIGINT, source VARCHAR, PRIMARY KEY (symbol, date)
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS financials (
                symbol VARCHAR, fiscal_year VARCHAR, metric_name VARCHAR,
                metric_value DOUBLE, report_type VARCHAR, source VARCHAR,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (symbol, fiscal_year, metric_name, report_type)
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS shareholding (
                symbol VARCHAR, date DATE, promoter_holding DOUBLE, promoter_pledge DOUBLE,
                fii_holding DOUBLE, dii_holding DOUBLE, public_holding DOUBLE,
                source VARCHAR, PRIMARY KEY (symbol, date)
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS red_flags (
                id INTEGER PRIMARY KEY, symbol VARCHAR, flag_type VARCHAR,
                severity VARCHAR, description VARCHAR, detected_date DATE,
                data_date DATE, is_active BOOLEAN DEFAULT TRUE
            )
        """)
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_price_symbol_date ON price_history(symbol, date)")
        logger.info("Database schema initialized")
    
    def upsert_stock(self, stock_data: Dict[str, Any]):
        self.conn.execute("""
            INSERT OR REPLACE INTO stocks (symbol, name, sector, industry, market_cap, updated_at)
            VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, [stock_data.get('symbol'), stock_data.get('name'), stock_data.get('sector'),
              stock_data.get('industry'), stock_data.get('market_cap')])
    
    def upsert_price_history(self, symbol: str, df: pd.DataFrame):
        if df.empty:
            return
        df = df.copy()
        df['symbol'] = symbol
        columns = ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume', 'source']
        df = df[columns]
        self.conn.execute("INSERT OR REPLACE INTO price_history SELECT * FROM df")
        logger.info(f"Inserted {len(df)} price records for {symbol}")
    
    def get_price_history(self, symbol: str, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> pd.DataFrame:
        query = "SELECT * FROM price_history WHERE symbol = ?"
        params = [symbol]
        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)
        query += " ORDER BY date"
        return self.conn.execute(query, params).fetchdf()
    
    def get_all_symbols(self) -> List[str]:
        result = self.conn.execute("SELECT symbol FROM stocks ORDER BY symbol").fetchall()
        return [row[0] for row in result]
    
    def search_stocks(self, query: str, limit: int = 20) -> pd.DataFrame:
        search_term = f"%{query}%"
        return self.conn.execute("""
            SELECT symbol, name, sector, industry, market_cap FROM stocks
            WHERE symbol LIKE ? OR name LIKE ? ORDER BY market_cap DESC NULLS LAST LIMIT ?
        """, [search_term, search_term, limit]).fetchdf()
    
    def get_stock_count(self) -> int:
        result = self.conn.execute("SELECT COUNT(*) FROM stocks").fetchone()
        return result[0] if result else 0
    
    def add_red_flag(self, symbol: str, flag_type: str, severity: str, description: str, data_date: datetime):
        self.conn.execute("""
            INSERT INTO red_flags (symbol, flag_type, severity, description, detected_date, data_date, is_active)
            VALUES (?, ?, ?, ?, CURRENT_DATE, ?, TRUE)
        """, [symbol, flag_type, severity, description, data_date])
    
    def get_red_flags(self, symbol: str, active_only: bool = True) -> pd.DataFrame:
        query = "SELECT * FROM red_flags WHERE symbol = ?"
        if active_only:
            query += " AND is_active = TRUE"
        return self.conn.execute(query + " ORDER BY detected_date DESC", [symbol]).fetchdf()
    
    def close(self):
        self.conn.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

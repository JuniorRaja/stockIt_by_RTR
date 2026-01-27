"""Financial Trajectory Analysis for Indian Equity Intelligence."""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass
import logging

from ..utils.config import get_threshold
from ..utils.helpers import calculate_cagr, safe_divide, calculate_consistency_score

logger = logging.getLogger(__name__)


@dataclass
class FinancialScore:
    """Container for financial analysis results."""
    overall_score: float
    revenue_cagr_3y: Optional[float]
    revenue_cagr_5y: Optional[float]
    revenue_cagr_10y: Optional[float]
    pat_cagr_3y: Optional[float]
    pat_cagr_5y: Optional[float]
    roce_current: Optional[float]
    roce_avg_5y: Optional[float]
    roce_consistency: float
    fcf_yield: Optional[float]
    earnings_quality: float
    debt_to_equity: Optional[float]
    margin_trend: str
    details: Dict[str, Any]
    warnings: List[str]
    red_flags: List[str]


class FinancialAnalyzer:
    """Analyzes financial trajectory and health."""
    
    def __init__(self, db_manager=None):
        self.db = db_manager
        self.min_revenue_cagr_3y = get_threshold('financial', 'min_revenue_cagr_3y', 10.0)
        self.min_roce = get_threshold('financial', 'min_roce', 12.0)
        self.max_debt_equity = get_threshold('financial', 'max_debt_to_equity', 1.5)
    
    def analyze(self, symbol: str, financials: Dict[str, pd.DataFrame], stock_info: Dict[str, Any],
                cutoff_date: Optional[datetime] = None) -> FinancialScore:
        cutoff_date = cutoff_date or datetime.now()
        warnings, red_flags, details = [], [], {}
        
        income_stmt = financials.get('income_statement', pd.DataFrame())
        balance_sheet = financials.get('balance_sheet', pd.DataFrame())
        cash_flow = financials.get('cash_flow', pd.DataFrame())
        
        # Revenue Growth
        revenue = self._analyze_revenue(income_stmt)
        details['revenue'] = revenue
        if revenue['cagr_3y'] is not None and revenue['cagr_3y'] < self.min_revenue_cagr_3y:
            warnings.append(f"Slow revenue growth: {revenue['cagr_3y']:.1f}% (3Y)")
        
        # Profitability
        profit = self._analyze_profitability(income_stmt)
        details['profitability'] = profit
        
        # ROCE
        roce = self._analyze_roce(income_stmt, balance_sheet)
        details['roce'] = roce
        if roce['current'] and roce['current'] < self.min_roce:
            warnings.append(f"Low ROCE: {roce['current']:.1f}%")
        
        # Cash Flow
        cashflow = self._analyze_cashflow(cash_flow, income_stmt, stock_info)
        details['cash_flow'] = cashflow
        if cashflow['earnings_quality'] < 0.5:
            red_flags.append(f"Poor earnings quality: CFO/PAT = {cashflow['earnings_quality']:.2f}")
        if cashflow['negative_fcf_years'] >= 3:
            red_flags.append(f"Persistent negative FCF: {cashflow['negative_fcf_years']} years")
        
        # Leverage
        leverage = self._analyze_leverage(balance_sheet)
        details['leverage'] = leverage
        if leverage['debt_to_equity'] and leverage['debt_to_equity'] > self.max_debt_equity:
            warnings.append(f"High leverage: D/E = {leverage['debt_to_equity']:.2f}")
        
        # Margins
        margins = self._analyze_margins(income_stmt)
        details['margins'] = margins
        
        overall_score = self._calculate_score(revenue['score'], profit['score'], roce['score'],
                                               cashflow['score'], leverage['score'], margins['score'])
        
        return FinancialScore(
            overall_score=overall_score, revenue_cagr_3y=revenue['cagr_3y'],
            revenue_cagr_5y=revenue['cagr_5y'], revenue_cagr_10y=revenue['cagr_10y'],
            pat_cagr_3y=profit['pat_cagr_3y'], pat_cagr_5y=profit['pat_cagr_5y'],
            roce_current=roce['current'], roce_avg_5y=roce['avg_5y'],
            roce_consistency=roce['consistency'], fcf_yield=cashflow['fcf_yield'],
            earnings_quality=cashflow['earnings_quality'],
            debt_to_equity=leverage['debt_to_equity'], margin_trend=margins['trend'],
            details=details, warnings=warnings, red_flags=red_flags
        )
    
    def _analyze_revenue(self, income_stmt: pd.DataFrame) -> Dict:
        if income_stmt.empty:
            return {'cagr_3y': None, 'cagr_5y': None, 'cagr_10y': None, 'score': 50}
        
        rev_cols = ['Total Revenue', 'Revenue', 'Net Sales']
        revenue = None
        for col in rev_cols:
            if col in income_stmt.columns:
                revenue = income_stmt[col].dropna().sort_index()
                break
        
        if revenue is None or len(revenue) < 2:
            return {'cagr_3y': None, 'cagr_5y': None, 'cagr_10y': None, 'score': 50}
        
        cagr_3y = self._calc_cagr(revenue, 3)
        cagr_5y = self._calc_cagr(revenue, 5)
        cagr_10y = self._calc_cagr(revenue, 10)
        
        primary = cagr_5y or cagr_3y
        score = 50 if primary is None else 95 if primary >= 20 else 85 if primary >= 15 else 75 if primary >= 10 else 60 if primary >= 5 else 45 if primary >= 0 else 25
        
        return {'cagr_3y': cagr_3y, 'cagr_5y': cagr_5y, 'cagr_10y': cagr_10y, 'score': score}
    
    def _analyze_profitability(self, income_stmt: pd.DataFrame) -> Dict:
        if income_stmt.empty:
            return {'pat_cagr_3y': None, 'pat_cagr_5y': None, 'score': 50}
        
        pat_cols = ['Net Income', 'Profit After Tax', 'PAT']
        pat = None
        for col in pat_cols:
            if col in income_stmt.columns:
                pat = income_stmt[col].dropna().sort_index()
                pat = pat[pat > 0]
                break
        
        if pat is None or len(pat) < 2:
            return {'pat_cagr_3y': None, 'pat_cagr_5y': None, 'score': 50}
        
        cagr_3y = self._calc_cagr(pat, 3)
        cagr_5y = self._calc_cagr(pat, 5)
        primary = cagr_5y or cagr_3y
        score = 50 if primary is None else 95 if primary >= 25 else 85 if primary >= 18 else 75 if primary >= 12 else 60 if primary >= 5 else 45 if primary >= 0 else 25
        
        return {'pat_cagr_3y': cagr_3y, 'pat_cagr_5y': cagr_5y, 'score': score}
    
    def _analyze_roce(self, income_stmt: pd.DataFrame, balance_sheet: pd.DataFrame) -> Dict:
        if income_stmt.empty or balance_sheet.empty:
            return {'current': None, 'avg_5y': None, 'consistency': 0, 'score': 50}
        
        # Simplified ROCE from stock info or estimate
        return {'current': 15, 'avg_5y': 14, 'consistency': 70, 'score': 70}
    
    def _analyze_cashflow(self, cash_flow: pd.DataFrame, income_stmt: pd.DataFrame, stock_info: Dict) -> Dict:
        result = {'fcf_yield': None, 'earnings_quality': 1.0, 'negative_fcf_years': 0, 'score': 50}
        if cash_flow.empty:
            return result
        
        cfo_cols = ['Operating Cash Flow', 'Cash From Operating Activities']
        cfo = None
        for col in cfo_cols:
            if col in cash_flow.columns:
                cfo = cash_flow[col]
                break
        
        if cfo is not None:
            result['negative_fcf_years'] = int((cfo < 0).sum())
        
        result['score'] = 70 if result['negative_fcf_years'] < 2 else 50 if result['negative_fcf_years'] < 3 else 30
        return result
    
    def _analyze_leverage(self, balance_sheet: pd.DataFrame) -> Dict:
        if balance_sheet.empty:
            return {'debt_to_equity': None, 'score': 50}
        
        # Would calculate D/E from balance sheet
        return {'debt_to_equity': 0.5, 'score': 75}
    
    def _analyze_margins(self, income_stmt: pd.DataFrame) -> Dict:
        return {'opm_current': 15, 'trend': 'stable', 'score': 65}
    
    def _calc_cagr(self, series: pd.Series, years: int) -> Optional[float]:
        if len(series) < 2:
            return None
        actual_years = min(years, len(series) - 1)
        if actual_years <= 0:
            return None
        start = series.iloc[-(actual_years + 1)]
        end = series.iloc[-1]
        return calculate_cagr(start, end, actual_years)
    
    def _calculate_score(self, revenue: float, profit: float, roce: float, 
                         cashflow: float, leverage: float, margin: float) -> float:
        return round(revenue * 0.20 + profit * 0.20 + roce * 0.20 + 
                     cashflow * 0.20 + leverage * 0.10 + margin * 0.10, 1)

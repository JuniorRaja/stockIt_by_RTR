"""Legacy & Governance Analysis for Indian Equity Intelligence."""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging

from ..utils.config import get_threshold

logger = logging.getLogger(__name__)


@dataclass
class GovernanceScore:
    """Container for governance analysis results."""
    overall_score: float
    years_listed: int
    promoter_holding: float
    promoter_holding_trend: str
    pledge_ratio: float
    pledge_trend: str
    dividend_consistency: float
    auditor_stability: str
    related_party_risk: str
    details: Dict[str, Any]
    warnings: List[str]
    red_flags: List[str]


class GovernanceAnalyzer:
    """Analyzes corporate governance and legacy factors."""
    
    def __init__(self, db_manager=None, cache_manager=None):
        self.db = db_manager
        self.cache = cache_manager
        self.min_years_listed = get_threshold('governance', 'min_years_listed', 5)
        self.min_promoter_holding = get_threshold('governance', 'min_promoter_holding', 25.0)
        self.max_pledge_ratio = get_threshold('governance', 'max_pledge_ratio', 20.0)
    
    def analyze(self, symbol: str, stock_info: Dict[str, Any], shareholding_history: pd.DataFrame,
                dividend_history: pd.DataFrame, corporate_actions: Optional[pd.DataFrame] = None,
                cutoff_date: Optional[datetime] = None) -> GovernanceScore:
        cutoff_date = cutoff_date or datetime.now()
        warnings, red_flags, details = [], [], {}
        
        # Years Listed Analysis
        years_listed = self._estimate_years_listed(stock_info)
        listing_score = min(100, years_listed * 5) if years_listed >= 5 else years_listed * 10
        details['years_listed'] = years_listed
        if years_listed < self.min_years_listed:
            warnings.append(f"Company listed for only {years_listed} years (minimum: {self.min_years_listed})")
        
        # Promoter Holding Analysis
        promoter_analysis = self._analyze_promoter_holding(shareholding_history, cutoff_date)
        details['promoter_analysis'] = promoter_analysis
        if promoter_analysis['current_holding'] < self.min_promoter_holding:
            warnings.append(f"Low promoter holding: {promoter_analysis['current_holding']:.1f}%")
        if promoter_analysis['trend'] == 'decreasing' and promoter_analysis['change_5y'] < -10:
            red_flags.append(f"Significant promoter stake reduction: {promoter_analysis['change_5y']:.1f}%")
        
        # Pledge Analysis
        pledge_analysis = self._analyze_pledge(shareholding_history, cutoff_date)
        details['pledge_analysis'] = pledge_analysis
        if pledge_analysis['current_pledge'] > self.max_pledge_ratio:
            red_flags.append(f"High promoter pledge: {pledge_analysis['current_pledge']:.1f}%")
        
        # Dividend Analysis
        dividend_analysis = self._analyze_dividends(dividend_history, cutoff_date)
        details['dividend_analysis'] = dividend_analysis
        
        # Auditor Analysis
        auditor_analysis = self._analyze_auditor(corporate_actions, cutoff_date)
        details['auditor_analysis'] = auditor_analysis
        if auditor_analysis['status'] == 'red_flag':
            red_flags.append(auditor_analysis['reason'])
        
        # Calculate overall score
        overall_score = self._calculate_score(listing_score, promoter_analysis['score'],
                                               pledge_analysis['score'], dividend_analysis['score'],
                                               auditor_analysis['score'])
        
        return GovernanceScore(
            overall_score=overall_score, years_listed=years_listed,
            promoter_holding=promoter_analysis['current_holding'],
            promoter_holding_trend=promoter_analysis['trend'],
            pledge_ratio=pledge_analysis['current_pledge'],
            pledge_trend=pledge_analysis['trend'],
            dividend_consistency=dividend_analysis['consistency'],
            auditor_stability=auditor_analysis['status'],
            related_party_risk='medium', details=details,
            warnings=warnings, red_flags=red_flags
        )
    
    def _estimate_years_listed(self, stock_info: Dict) -> int:
        # Would need listing date from data source; estimate for now
        return 10  # Default assumption
    
    def _analyze_promoter_holding(self, shareholding: pd.DataFrame, cutoff_date: datetime) -> Dict:
        if shareholding.empty or 'promoter_holding' not in shareholding.columns:
            return {'current_holding': 50, 'trend': 'stable', 'change_5y': 0, 'score': 70}
        
        df = shareholding[shareholding['date'] <= cutoff_date].sort_values('date')
        if df.empty:
            return {'current_holding': 50, 'trend': 'stable', 'change_5y': 0, 'score': 70}
        
        current = df['promoter_holding'].iloc[-1]
        five_years_ago = cutoff_date - timedelta(days=5*365)
        df_5y = df[df['date'] <= five_years_ago]
        holding_5y = df_5y['promoter_holding'].iloc[-1] if not df_5y.empty else current
        change_5y = current - holding_5y
        trend = 'increasing' if change_5y > 5 else 'decreasing' if change_5y < -5 else 'stable'
        score = min(100, (current / self.min_promoter_holding) * 50)
        if trend == 'decreasing':
            score = max(0, score - 15)
        
        return {'current_holding': current, 'trend': trend, 'change_5y': change_5y, 'score': score}
    
    def _analyze_pledge(self, shareholding: pd.DataFrame, cutoff_date: datetime) -> Dict:
        if shareholding.empty or 'promoter_pledge' not in shareholding.columns:
            return {'current_pledge': 0, 'trend': 'stable', 'score': 100}
        
        df = shareholding[shareholding['date'] <= cutoff_date].sort_values('date')
        if df.empty:
            return {'current_pledge': 0, 'trend': 'stable', 'score': 100}
        
        current = df['promoter_pledge'].iloc[-1] or 0
        score = 100 if current == 0 else 80 if current <= 10 else 60 if current <= 20 else 30 if current <= 50 else 10
        return {'current_pledge': current, 'trend': 'stable', 'score': score}
    
    def _analyze_dividends(self, dividend_history: pd.DataFrame, cutoff_date: datetime) -> Dict:
        if dividend_history.empty:
            return {'consistency': 0, 'score': 50}
        consistency = 60  # Default moderate
        score = 60
        return {'consistency': consistency, 'score': score}
    
    def _analyze_auditor(self, corporate_actions: Optional[pd.DataFrame], cutoff_date: datetime) -> Dict:
        return {'status': 'stable', 'reason': None, 'score': 80}
    
    def _calculate_score(self, listing: float, promoter: float, pledge: float, dividend: float, auditor: float) -> float:
        return round(listing * 0.15 + promoter * 0.30 + pledge * 0.25 + dividend * 0.15 + auditor * 0.15, 1)

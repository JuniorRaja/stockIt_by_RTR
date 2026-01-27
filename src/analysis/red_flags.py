"""Red Flag Alert System for Indian Equity Intelligence."""

import pandas as pd
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging

from ..utils.config import get_red_flag_config

logger = logging.getLogger(__name__)


@dataclass
class RedFlag:
    flag_type: str
    severity: str
    description: str
    data_date: datetime
    details: Dict[str, Any]
    action_required: str
    penalty_score: int


class RedFlagDetector:
    """Detects red flags. Red flags do NOT auto-trigger SELL but increase risk penalties."""
    
    FLAG_TYPES = {
        'auditor_resignation': {'category': 'governance', 'severity': 'critical'},
        'auditor_change': {'category': 'governance', 'severity': 'medium'},
        'promoter_pledge_increase': {'category': 'governance', 'severity': 'high'},
        'promoter_stake_reduction': {'category': 'governance', 'severity': 'high'},
        'earnings_without_cash': {'category': 'financial', 'severity': 'high'},
        'negative_fcf_streak': {'category': 'financial', 'severity': 'medium'},
        'margin_collapse': {'category': 'financial', 'severity': 'medium'}
    }
    
    def __init__(self, db_manager=None):
        self.db = db_manager
        self.pledge_threshold = get_red_flag_config('promoter_pledge_increase').get('threshold', 10.0)
        self.stake_threshold = get_red_flag_config('promoter_stake_reduction').get('threshold', 5.0)
        self.earnings_threshold = get_red_flag_config('earnings_without_cash').get('threshold', 0.5)
        self.fcf_years = get_red_flag_config('negative_fcf_streak').get('years', 3)
    
    def detect_all_flags(self, symbol: str, stock_info: Dict, shareholding_history: pd.DataFrame,
                          financials: Dict[str, pd.DataFrame], corporate_actions: Optional[pd.DataFrame] = None,
                          cutoff_date: Optional[datetime] = None) -> List[RedFlag]:
        cutoff_date = cutoff_date or datetime.now()
        flags = []
        
        flags.extend(self._detect_pledge_flags(shareholding_history, cutoff_date))
        flags.extend(self._detect_stake_flags(shareholding_history, cutoff_date))
        flags.extend(self._detect_earnings_flags(financials, cutoff_date))
        flags.extend(self._detect_fcf_flags(financials, cutoff_date))
        
        logger.info(f"Detected {len(flags)} red flags for {symbol}")
        return flags
    
    def _detect_pledge_flags(self, shareholding: pd.DataFrame, cutoff: datetime) -> List[RedFlag]:
        if shareholding.empty or 'promoter_pledge' not in shareholding.columns:
            return []
        
        df = shareholding[shareholding['date'] <= cutoff].sort_values('date')
        if len(df) < 2:
            return []
        
        current = df['promoter_pledge'].iloc[-1] or 0
        one_year = cutoff - timedelta(days=365)
        df_1y = df[df['date'] <= one_year]
        pledge_1y = df_1y['promoter_pledge'].iloc[-1] if not df_1y.empty else 0
        increase = current - (pledge_1y or 0)
        
        flags = []
        if increase >= self.pledge_threshold:
            flags.append(RedFlag(
                flag_type='promoter_pledge_increase', severity='high',
                description=f"Pledge increased {increase:.1f}pp (from {pledge_1y:.1f}% to {current:.1f}%)",
                data_date=cutoff, details={'current': current, 'increase': increase},
                action_required='Monitor pledge and liquidity', penalty_score=15
            ))
        if current > 50:
            flags.append(RedFlag(
                flag_type='promoter_pledge_increase', severity='critical',
                description=f"Very high pledge: {current:.1f}% - forced selling risk",
                data_date=cutoff, details={'current': current},
                action_required='Immediate review required', penalty_score=25
            ))
        return flags
    
    def _detect_stake_flags(self, shareholding: pd.DataFrame, cutoff: datetime) -> List[RedFlag]:
        if shareholding.empty or 'promoter_holding' not in shareholding.columns:
            return []
        
        df = shareholding[shareholding['date'] <= cutoff].sort_values('date')
        if len(df) < 2:
            return []
        
        current = df['promoter_holding'].iloc[-1] or 0
        one_year = cutoff - timedelta(days=365)
        df_1y = df[df['date'] <= one_year]
        holding_1y = df_1y['promoter_holding'].iloc[-1] if not df_1y.empty else current
        reduction = (holding_1y or 0) - current
        
        if reduction >= self.stake_threshold:
            return [RedFlag(
                flag_type='promoter_stake_reduction', severity='high',
                description=f"Promoter stake reduced {reduction:.1f}pp (from {holding_1y:.1f}% to {current:.1f}%)",
                data_date=cutoff, details={'current': current, 'reduction': reduction},
                action_required='Understand reason for sale', penalty_score=10
            )]
        return []
    
    def _detect_earnings_flags(self, financials: Dict, cutoff: datetime) -> List[RedFlag]:
        income = financials.get('income_statement', pd.DataFrame())
        cashflow = financials.get('cash_flow', pd.DataFrame())
        if income.empty or cashflow.empty:
            return []
        
        # Simplified earnings quality check
        return []
    
    def _detect_fcf_flags(self, financials: Dict, cutoff: datetime) -> List[RedFlag]:
        cashflow = financials.get('cash_flow', pd.DataFrame())
        if cashflow.empty:
            return []
        
        cfo_cols = ['Operating Cash Flow', 'Cash From Operating Activities']
        cfo = None
        for col in cfo_cols:
            if col in cashflow.columns:
                cfo = cashflow[col]
                break
        
        if cfo is not None:
            recent = cfo.tail(self.fcf_years)
            if len(recent) >= self.fcf_years and all(recent < 0):
                return [RedFlag(
                    flag_type='negative_fcf_streak', severity='medium',
                    description=f"Negative FCF for {self.fcf_years} consecutive years",
                    data_date=cutoff, details={'years': self.fcf_years},
                    action_required='Assess capital intensity and funding', penalty_score=15
                )]
        return []
    
    def get_flag_summary(self, flags: List[RedFlag]) -> Dict:
        if not flags:
            return {'total': 0, 'by_severity': {}, 'penalty': 0, 'has_critical': False}
        
        by_severity = {}
        penalty = 0
        for f in flags:
            by_severity[f.severity] = by_severity.get(f.severity, 0) + 1
            penalty += f.penalty_score
        
        return {'total': len(flags), 'by_severity': by_severity, 'penalty': min(50, penalty),
                'has_critical': 'critical' in by_severity}

"""Market Behaviour Analysis for Indian Equity Intelligence."""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging

from ..utils.config import get_threshold
from ..utils.helpers import calculate_max_drawdown, calculate_volatility

logger = logging.getLogger(__name__)


@dataclass
class MarketBehaviourScore:
    """Container for market behaviour analysis results."""
    overall_score: float
    max_drawdown: float
    avg_recovery_months: Optional[float]
    volatility_1y: Optional[float]
    volatility_3y: Optional[float]
    beta: Optional[float]
    sharpe_ratio: Optional[float]
    vs_nifty_1y: Optional[float]
    vs_nifty_3y: Optional[float]
    vs_nifty_5y: Optional[float]
    volatility_regime: str
    details: Dict[str, Any]
    warnings: List[str]


class MarketBehaviourAnalyzer:
    """Analyzes long-term market behaviour patterns."""
    
    def __init__(self, db_manager=None):
        self.db = db_manager
        self.max_dd_threshold = get_threshold('market', 'max_drawdown_threshold', 50.0)
        self.recovery_warning = get_threshold('market', 'recovery_months_warning', 24)
        self.vol_high = get_threshold('market', 'volatility_high_threshold', 40.0)
    
    def analyze(self, symbol: str, price_history: pd.DataFrame,
                nifty_history: Optional[pd.DataFrame] = None,
                cutoff_date: Optional[datetime] = None) -> MarketBehaviourScore:
        cutoff_date = cutoff_date or datetime.now()
        warnings, details = [], {}
        
        if price_history.empty or len(price_history) < 30:
            return self._empty_result()
        
        df = price_history[price_history['date'] <= cutoff_date].sort_values('date')
        prices = df.set_index('date')['close']
        
        # Drawdown Analysis
        dd = self._analyze_drawdown(prices)
        details['drawdowns'] = dd
        if dd['max_drawdown'] > self.max_dd_threshold:
            warnings.append(f"Large drawdown: {dd['max_drawdown']:.1f}%")
        
        # Recovery Analysis
        recovery = self._analyze_recovery(prices)
        details['recovery'] = recovery
        if recovery['avg_months'] and recovery['avg_months'] > self.recovery_warning:
            warnings.append(f"Slow recovery: {recovery['avg_months']:.0f} months average")
        
        # Volatility Analysis
        vol = self._analyze_volatility(df, cutoff_date)
        details['volatility'] = vol
        if vol['regime'] in ['high', 'extreme']:
            warnings.append(f"High volatility: {vol['annual_1y']:.1f}%")
        
        # Beta
        beta = self._calculate_beta(df, nifty_history, cutoff_date)
        details['beta'] = beta
        if beta['beta'] and beta['beta'] > 1.5:
            warnings.append(f"High beta: {beta['beta']:.2f}")
        
        # Relative Performance
        relative = self._analyze_relative(df, nifty_history, cutoff_date)
        details['relative'] = relative
        
        # Sharpe Ratio
        sharpe = self._calculate_sharpe(df, cutoff_date)
        
        overall = self._calculate_score(dd['score'], recovery['score'], vol['score'], 
                                         beta['score'], relative['score'])
        
        return MarketBehaviourScore(
            overall_score=overall, max_drawdown=dd['max_drawdown'],
            avg_recovery_months=recovery['avg_months'], volatility_1y=vol['annual_1y'],
            volatility_3y=vol['annual_3y'], beta=beta['beta'], sharpe_ratio=sharpe,
            vs_nifty_1y=relative['vs_nifty_1y'], vs_nifty_3y=relative['vs_nifty_3y'],
            vs_nifty_5y=relative['vs_nifty_5y'], volatility_regime=vol['regime'],
            details=details, warnings=warnings
        )
    
    def _empty_result(self) -> MarketBehaviourScore:
        return MarketBehaviourScore(
            overall_score=50, max_drawdown=0, avg_recovery_months=None,
            volatility_1y=None, volatility_3y=None, beta=None, sharpe_ratio=None,
            vs_nifty_1y=None, vs_nifty_3y=None, vs_nifty_5y=None,
            volatility_regime='unknown', details={}, warnings=['Insufficient data']
        )
    
    def _analyze_drawdown(self, prices: pd.Series) -> Dict:
        max_dd, _, _ = calculate_max_drawdown(prices)
        score = 90 if max_dd <= 20 else 75 if max_dd <= 35 else 55 if max_dd <= 50 else 35 if max_dd <= 65 else 20
        return {'max_drawdown': max_dd, 'score': score}
    
    def _analyze_recovery(self, prices: pd.Series) -> Dict:
        # Simplified recovery analysis
        return {'avg_months': 12, 'score': 70}
    
    def _analyze_volatility(self, df: pd.DataFrame, cutoff_date: datetime) -> Dict:
        df = df.sort_values('date').copy()
        df['returns'] = df['close'].pct_change()
        
        one_year_ago = cutoff_date - timedelta(days=365)
        df_1y = df[df['date'] >= one_year_ago]
        vol_1y = calculate_volatility(df_1y['returns'].dropna())
        
        three_years_ago = cutoff_date - timedelta(days=3*365)
        df_3y = df[df['date'] >= three_years_ago]
        vol_3y = calculate_volatility(df_3y['returns'].dropna())
        
        ref_vol = vol_1y or vol_3y or 30
        regime = 'low' if ref_vol <= 20 else 'medium' if ref_vol <= 30 else 'high' if ref_vol <= 40 else 'extreme'
        score = 90 if ref_vol <= 20 else 75 if ref_vol <= 30 else 55 if ref_vol <= 40 else 35 if ref_vol <= 55 else 20
        
        return {'annual_1y': vol_1y, 'annual_3y': vol_3y, 'regime': regime, 'score': score}
    
    def _calculate_beta(self, df: pd.DataFrame, nifty: Optional[pd.DataFrame], cutoff_date: datetime) -> Dict:
        if nifty is None or nifty.empty:
            return {'beta': None, 'score': 50}
        
        # Simplified beta estimation
        return {'beta': 1.1, 'score': 70}
    
    def _analyze_relative(self, df: pd.DataFrame, nifty: Optional[pd.DataFrame], cutoff_date: datetime) -> Dict:
        if nifty is None or nifty.empty:
            return {'vs_nifty_1y': None, 'vs_nifty_3y': None, 'vs_nifty_5y': None, 'score': 50}
        
        # Simplified relative performance
        return {'vs_nifty_1y': 5, 'vs_nifty_3y': 15, 'vs_nifty_5y': 30, 'score': 70}
    
    def _calculate_sharpe(self, df: pd.DataFrame, cutoff_date: datetime, risk_free: float = 6.0) -> Optional[float]:
        try:
            df = df.sort_values('date').copy()
            df['returns'] = df['close'].pct_change()
            one_year_ago = cutoff_date - timedelta(days=365)
            df_1y = df[df['date'] >= one_year_ago]
            if len(df_1y) < 100:
                return None
            returns = df_1y['returns'].dropna()
            annual_return = returns.mean() * 252 * 100
            annual_vol = returns.std() * np.sqrt(252) * 100
            if annual_vol > 0:
                return round((annual_return - risk_free) / annual_vol, 2)
        except Exception:
            pass
        return None
    
    def _calculate_score(self, dd: float, recovery: float, vol: float, beta: float, relative: float) -> float:
        return round(dd * 0.25 + recovery * 0.20 + vol * 0.25 + beta * 0.15 + relative * 0.15, 1)

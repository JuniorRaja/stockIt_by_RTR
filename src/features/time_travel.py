"""Time Travel Mode for Indian Equity Intelligence."""

import pandas as pd
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging

from ..utils.config import get_config
from ..engine.governance import GovernanceAnalyzer
from ..engine.financial import FinancialAnalyzer
from ..engine.valuation import ValuationAnalyzer
from ..engine.market import MarketBehaviourAnalyzer
from ..analysis.signal_generator import SignalGenerator, UserProfile
from ..analysis.red_flags import RedFlagDetector

logger = logging.getLogger(__name__)


@dataclass
class TimeTravelResult:
    cutoff_date: datetime
    signal_at_cutoff: Any
    governance_at_cutoff: Any
    financial_at_cutoff: Any
    valuation_at_cutoff: Any
    market_at_cutoff: Any
    red_flags_at_cutoff: List[Dict]
    actual_outcome: Optional[Dict[str, Any]]
    hindsight_analysis: Optional[str]


class TimeTravelEngine:
    """Time Travel: Re-analyze with historical data only. No future leakage."""
    
    # Default cutoffs - but will dynamically generate based on stock data
    DEFAULT_CUTOFFS = [2000, 2005, 2008, 2010, 2013, 2015, 2018, 2020, 2022, 2024]
    
    def __init__(self, db_manager=None):
        self.db = db_manager
        self.min_year = get_config('time_travel.min_year', 1996)  # Support older data
        self.governance = GovernanceAnalyzer(db_manager)
        self.financial = FinancialAnalyzer(db_manager)
        self.valuation = ValuationAnalyzer(db_manager)
        self.market = MarketBehaviourAnalyzer(db_manager)
        self.signal_gen = SignalGenerator()
        self.red_flag = RedFlagDetector(db_manager)
    
    def analyze_at_cutoff(self, symbol: str, cutoff_year: int, user_profile: UserProfile,
                          stock_info: Dict, price_history: pd.DataFrame, financials: Dict,
                          shareholding: pd.DataFrame, dividends: pd.DataFrame,
                          nifty_history: Optional[pd.DataFrame] = None,
                          include_outcome: bool = True) -> TimeTravelResult:
        cutoff = datetime(cutoff_year, 12, 31)
        
        # Filter data to cutoff
        prices_cut = self._filter(price_history, cutoff)
        fin_cut = self._filter_financials(financials, cutoff)
        share_cut = self._filter(shareholding, cutoff)
        div_cut = self._filter(dividends, cutoff)
        nifty_cut = self._filter(nifty_history, cutoff) if nifty_history is not None else None
        
        # Run analysis
        gov = self.governance.analyze(symbol, stock_info, share_cut, div_cut, cutoff_date=cutoff)
        fin = self.financial.analyze(symbol, fin_cut, stock_info, cutoff_date=cutoff)
        val = self.valuation.analyze(symbol, stock_info, prices_cut, fin_cut,
                                      earnings_growth=fin.pat_cagr_5y, cutoff_date=cutoff)
        mkt = self.market.analyze(symbol, prices_cut, nifty_cut, cutoff_date=cutoff)
        red_flags = self.red_flag.detect_all_flags(symbol, stock_info, share_cut, fin_cut, cutoff_date=cutoff)
        
        signal = self.signal_gen.generate_signal(
            user_profile, gov, fin, val, mkt,
            red_flags=[{'severity': r.severity, 'description': r.description} for r in red_flags]
        )
        
        outcome = None
        hindsight = None
        if include_outcome:
            outcome = self._calc_outcome(price_history, cutoff)
            hindsight = self._generate_hindsight(signal, outcome)
        
        return TimeTravelResult(
            cutoff_date=cutoff, signal_at_cutoff=signal, governance_at_cutoff=gov,
            financial_at_cutoff=fin, valuation_at_cutoff=val, market_at_cutoff=mkt,
            red_flags_at_cutoff=[{'type': r.flag_type, 'severity': r.severity, 'description': r.description} for r in red_flags],
            actual_outcome=outcome, hindsight_analysis=hindsight
        )
    
    def _filter(self, df: pd.DataFrame, cutoff: datetime) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        if 'date' in df.columns:
            df = df.copy()
            df['date'] = pd.to_datetime(df['date'])
            return df[df['date'] <= cutoff]
        return df
    
    def _filter_financials(self, financials: Dict, cutoff: datetime) -> Dict:
        filtered = {}
        for key, df in financials.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                try:
                    if hasattr(df.index, 'year'):
                        filtered[key] = df[df.index <= cutoff]
                    else:
                        years = [int(str(i)[:4]) for i in df.index]
                        mask = [y <= cutoff.year for y in years]
                        filtered[key] = df[mask]
                except Exception:
                    filtered[key] = df
            else:
                filtered[key] = df
        return filtered
    
    def _calc_outcome(self, prices: pd.DataFrame, cutoff: datetime) -> Optional[Dict]:
        if prices.empty:
            return None
        df = prices.copy()
        df['date'] = pd.to_datetime(df['date'])
        pre = df[df['date'] <= cutoff]
        post = df[df['date'] > cutoff]
        if pre.empty or post.empty:
            return None
        
        price_at = pre['close'].iloc[-1]
        outcome = {'price_at_cutoff': price_at, 'cutoff': cutoff.strftime('%Y-%m-%d')}
        
        for name, days in [('1y', 365), ('3y', 1095), ('5y', 1825)]:
            target = cutoff + timedelta(days=days)
            period = post[post['date'] <= target]
            if not period.empty:
                end = period['close'].iloc[-1]
                ret = ((end / price_at) - 1) * 100
                outcome[name] = {'return': round(ret, 2)}
        
        return outcome
    
    def _generate_hindsight(self, signal, outcome: Optional[Dict]) -> str:
        if not outcome:
            return "Insufficient data for hindsight."
        three_y = outcome.get('3y', {}).get('return', 0)
        sig = signal.signal
        
        if sig == "BUY":
            if three_y > 50:
                return f"CORRECT: BUY validated. Stock returned {three_y:.1f}% over 3 years."
            elif three_y > 0:
                return f"PARTIAL: BUY had modest {three_y:.1f}% return over 3 years."
            else:
                return f"INCORRECT: BUY didn't work. Stock returned {three_y:.1f}%."
        elif sig == "AVOID BUYING":
            if three_y < 0:
                return f"CORRECT: AVOID validated. Stock lost {abs(three_y):.1f}%."
            elif three_y < 20:
                return f"PARTIAL: AVOID reasonable. Stock returned only {three_y:.1f}%."
            else:
                return f"INCORRECT: AVOID was wrong. Stock returned {three_y:.1f}%."
        return f"Signal: {sig}. 3Y return: {three_y:.1f}%"
    
    def get_available_cutoffs(self, prices: pd.DataFrame) -> List[int]:
        """
        Generate available cutoff years based on the stock's actual data range.
        Shows ALL years from the stock's starting year to present.
        """
        if prices.empty:
            return []
        
        df = prices.copy()
        df['date'] = pd.to_datetime(df['date'])
        min_year = df['date'].min().year
        max_year = df['date'].max().year
        current_year = datetime.now().year
        
        # Need at least 1 year of post-cutoff data for meaningful analysis
        latest_cutoff = min(max_year - 1, current_year - 1)
        
        # Need at least 1 year of pre-cutoff data
        earliest_cutoff = min_year + 1
        
        if earliest_cutoff > latest_cutoff:
            return []
        
        # Generate ALL years from stock's start to latest valid cutoff
        cutoffs = list(range(earliest_cutoff, latest_cutoff + 1))
        
        return cutoffs

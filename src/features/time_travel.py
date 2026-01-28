"""Time Travel Mode for Stocron by RTR."""

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
    hindsight_signal: Optional[str] = None  # What signal SHOULD have been based on actual returns
    signal_accuracy: Optional[str] = None  # CORRECT, PARTIAL, INCORRECT


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
        hindsight_signal = None
        signal_accuracy = None
        if include_outcome:
            outcome = self._calc_outcome(price_history, cutoff)
            hindsight, hindsight_signal, signal_accuracy = self._generate_hindsight(signal, outcome)
        
        return TimeTravelResult(
            cutoff_date=cutoff, signal_at_cutoff=signal, governance_at_cutoff=gov,
            financial_at_cutoff=fin, valuation_at_cutoff=val, market_at_cutoff=mkt,
            red_flags_at_cutoff=[{'type': r.flag_type, 'severity': r.severity, 'description': r.description} for r in red_flags],
            actual_outcome=outcome, hindsight_analysis=hindsight,
            hindsight_signal=hindsight_signal, signal_accuracy=signal_accuracy
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
    
    def _generate_hindsight(self, signal, outcome: Optional[Dict]) -> tuple:
        """
        Generate hindsight analysis based on actual returns.
        
        Returns:
            tuple: (analysis_text, hindsight_signal, accuracy)
        
        Hindsight Signal Logic (based on 3Y returns, or 1Y if 3Y not available):
        - BUY: Return > 50% (3Y) or > 15% (1Y) - significant wealth creation
        - HOLD: Return 0-50% (3Y) or 0-15% (1Y) - modest returns
        - AVOID: Return < 0% - loss
        - SELL: Return < -30% - significant loss
        """
        if not outcome:
            return "Insufficient data for hindsight.", None, None
        
        # Use 3Y return if available, else 1Y
        three_y = outcome.get('3y', {}).get('return')
        one_y = outcome.get('1y', {}).get('return')
        five_y = outcome.get('5y', {}).get('return')
        
        # Determine best available return for analysis
        if three_y is not None:
            primary_return = three_y
            period = "3Y"
        elif one_y is not None:
            primary_return = one_y
            period = "1Y"
        else:
            return "Insufficient data for hindsight.", None, None
        
        # Determine what the signal SHOULD have been based on actual returns
        if period == "3Y":
            # 3-year thresholds (cumulative)
            if primary_return > 100:  # > 100% in 3 years (~26% CAGR)
                hindsight_signal = "STRONG BUY"
            elif primary_return > 50:  # > 50% in 3 years (~14% CAGR)
                hindsight_signal = "BUY"
            elif primary_return > 20:  # > 20% in 3 years (~6% CAGR)
                hindsight_signal = "HOLD"
            elif primary_return > 0:
                hindsight_signal = "WEAK HOLD"
            elif primary_return > -20:
                hindsight_signal = "AVOID"
            else:
                hindsight_signal = "SELL"
        else:
            # 1-year thresholds
            if primary_return > 30:
                hindsight_signal = "STRONG BUY"
            elif primary_return > 15:
                hindsight_signal = "BUY"
            elif primary_return > 5:
                hindsight_signal = "HOLD"
            elif primary_return > 0:
                hindsight_signal = "WEAK HOLD"
            elif primary_return > -15:
                hindsight_signal = "AVOID"
            else:
                hindsight_signal = "SELL"
        
        original_sig = signal.signal
        
        # Determine accuracy
        # Map signals to numeric scale for comparison
        signal_rank = {"STRONG BUY": 5, "BUY": 4, "HOLD": 3, "WEAK HOLD": 2.5, "AVOID BUYING": 2, "AVOID": 2, "SELL / EXIT": 1, "SELL": 1}
        
        original_rank = signal_rank.get(original_sig, 3)
        hindsight_rank = signal_rank.get(hindsight_signal, 3)
        
        diff = abs(original_rank - hindsight_rank)
        
        if diff <= 0.5:
            accuracy = "CORRECT"
        elif diff <= 1.5:
            accuracy = "PARTIAL"
        else:
            accuracy = "INCORRECT"
        
        # Generate analysis text
        returns_summary = []
        if one_y is not None:
            returns_summary.append(f"1Y: {one_y:+.1f}%")
        if three_y is not None:
            returns_summary.append(f"3Y: {three_y:+.1f}%")
        if five_y is not None:
            returns_summary.append(f"5Y: {five_y:+.1f}%")
        
        returns_text = ", ".join(returns_summary)
        
        if accuracy == "CORRECT":
            analysis = f"✓ Signal was {accuracy}. Original: {original_sig}, Optimal: {hindsight_signal}. Returns: {returns_text}"
        elif accuracy == "PARTIAL":
            analysis = f"◐ Signal was {accuracy}. Original: {original_sig}, Optimal: {hindsight_signal}. Returns: {returns_text}"
        else:
            analysis = f"✗ Signal was {accuracy}. Original: {original_sig}, but should have been {hindsight_signal}. Returns: {returns_text}"
        
        return analysis, hindsight_signal, accuracy
    
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

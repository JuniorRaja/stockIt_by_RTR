"""Valuation Context Analysis for Indian Equity Intelligence."""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass
import logging

from ..utils.config import get_threshold
from ..utils.helpers import safe_divide

logger = logging.getLogger(__name__)


@dataclass
class ValuationScore:
    """Container for valuation analysis results."""
    overall_score: float
    current_pe: Optional[float]
    pe_percentile_own: Optional[float]
    pe_percentile_sector: Optional[float]
    current_pb: Optional[float]
    pb_percentile_own: Optional[float]
    ev_ebitda: Optional[float]
    peg_ratio: Optional[float]
    stress_level: str
    valuation_zone: str
    details: Dict[str, Any]
    warnings: List[str]


class ValuationAnalyzer:
    """Analyzes valuation in context."""
    
    def __init__(self, db_manager=None):
        self.db = db_manager
        self.pe_expensive = get_threshold('valuation', 'pe_percentile_expensive', 80)
        self.pe_cheap = get_threshold('valuation', 'pe_percentile_cheap', 20)
    
    def analyze(self, symbol: str, stock_info: Dict[str, Any], price_history: pd.DataFrame,
                financials: Dict[str, pd.DataFrame], sector_valuations: Optional[Dict] = None,
                earnings_growth: Optional[float] = None, cutoff_date: Optional[datetime] = None) -> ValuationScore:
        cutoff_date = cutoff_date or datetime.now()
        warnings, details = [], {}
        
        current_pe = stock_info.get('pe_ratio')
        current_pb = stock_info.get('pb_ratio')
        
        # PE Analysis
        pe_analysis = self._analyze_pe(current_pe, price_history, financials)
        details['pe'] = pe_analysis
        if pe_analysis['percentile_own'] and pe_analysis['percentile_own'] > self.pe_expensive:
            warnings.append(f"PE at {pe_analysis['percentile_own']:.0f}th percentile (expensive)")
        
        # PB Analysis
        pb_analysis = self._analyze_pb(current_pb)
        details['pb'] = pb_analysis
        
        # EV/EBITDA
        ev_ebitda_analysis = self._analyze_ev_ebitda(stock_info, financials)
        details['ev_ebitda'] = ev_ebitda_analysis
        
        # PEG Ratio
        peg_analysis = self._analyze_peg(current_pe, earnings_growth)
        details['peg'] = peg_analysis
        if peg_analysis['ratio'] and peg_analysis['ratio'] > 2:
            warnings.append(f"High PEG ratio: {peg_analysis['ratio']:.2f}")
        
        valuation_zone = self._determine_zone(pe_analysis, pb_analysis)
        stress_level = self._calculate_stress(pe_analysis, pb_analysis, ev_ebitda_analysis)
        
        if stress_level in ['high', 'extreme']:
            warnings.append(f"Valuation stress: {stress_level}")
        
        overall_score = self._calculate_score(pe_analysis['score'], pb_analysis['score'],
                                               ev_ebitda_analysis['score'], peg_analysis['score'], stress_level)
        
        return ValuationScore(
            overall_score=overall_score, current_pe=current_pe,
            pe_percentile_own=pe_analysis['percentile_own'],
            pe_percentile_sector=pe_analysis.get('percentile_sector'),
            current_pb=current_pb, pb_percentile_own=pb_analysis['percentile_own'],
            ev_ebitda=ev_ebitda_analysis['current'], peg_ratio=peg_analysis['ratio'],
            stress_level=stress_level, valuation_zone=valuation_zone,
            details=details, warnings=warnings
        )
    
    def _analyze_pe(self, current_pe: Optional[float], price_history: pd.DataFrame, financials: Dict) -> Dict:
        if current_pe is None:
            return {'current': None, 'percentile_own': None, 'score': 50}
        
        # Estimate percentile (would calculate from historical PE)
        percentile = 50 if current_pe < 20 else 70 if current_pe < 35 else 85
        score = 90 if percentile <= 20 else 75 if percentile <= 40 else 60 if percentile <= 60 else 45 if percentile <= 80 else 25
        
        if current_pe > 50:
            score = max(0, score - 15)
        elif current_pe < 10:
            score = min(100, score + 10)
        
        return {'current': current_pe, 'percentile_own': percentile, 'score': score}
    
    def _analyze_pb(self, current_pb: Optional[float]) -> Dict:
        if current_pb is None:
            return {'current': None, 'percentile_own': None, 'score': 50}
        
        percentile = 30 if current_pb < 2 else 50 if current_pb < 3 else 70 if current_pb < 5 else 85
        score = 85 if percentile <= 20 else 70 if percentile <= 40 else 55 if percentile <= 60 else 40 if percentile <= 80 else 25
        
        return {'current': current_pb, 'percentile_own': percentile, 'score': score}
    
    def _analyze_ev_ebitda(self, stock_info: Dict, financials: Dict) -> Dict:
        # Simplified - would calculate from market cap, debt, cash, and EBITDA
        return {'current': 12, 'score': 65}
    
    def _analyze_peg(self, pe: Optional[float], growth: Optional[float]) -> Dict:
        if pe is None or growth is None or growth <= 0:
            return {'ratio': None, 'score': 50}
        
        peg = pe / growth
        score = 95 if peg < 0.5 else 85 if peg < 1 else 70 if peg < 1.5 else 55 if peg < 2 else 40 if peg < 3 else 25
        
        return {'ratio': round(peg, 2), 'score': score}
    
    def _determine_zone(self, pe: Dict, pb: Dict) -> str:
        scores = [v for v in [pe.get('percentile_own'), pb.get('percentile_own')] if v]
        if not scores:
            return 'fair'
        avg = np.mean(scores)
        return 'undervalued' if avg <= 25 else 'fair' if avg <= 50 else 'overvalued' if avg <= 75 else 'expensive'
    
    def _calculate_stress(self, pe: Dict, pb: Dict, ev_ebitda: Dict) -> str:
        indicators = 0
        if pe.get('percentile_own', 0) > 80:
            indicators += 2
        if pe.get('current', 0) and pe['current'] > 50:
            indicators += 2
        if pb.get('percentile_own', 0) > 80:
            indicators += 1
        
        return 'extreme' if indicators >= 4 else 'high' if indicators >= 2 else 'neutral' if indicators >= 1 else 'low'
    
    def _calculate_score(self, pe: float, pb: float, ev_ebitda: float, peg: float, stress: str) -> float:
        score = pe * 0.35 + pb * 0.20 + ev_ebitda * 0.25 + peg * 0.20
        if stress == 'extreme':
            score *= 0.7
        elif stress == 'high':
            score *= 0.85
        return round(score, 1)

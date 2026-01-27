"""Signal Generation for Indian Equity Intelligence."""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from dataclasses import dataclass
import logging

from ..utils.config import get_config, get_signal_weights, get_risk_profile

logger = logging.getLogger(__name__)


class Signal:
    BUY = "BUY"
    HOLD = "HOLD"
    AVOID = "AVOID BUYING"
    SELL = "SELL / EXIT"


@dataclass
class UserProfile:
    expected_return: float
    risk_appetite: str
    holding_tenure: int
    
    def to_dict(self) -> Dict:
        return {'expected_return': self.expected_return, 'risk_appetite': self.risk_appetite,
                'holding_tenure': self.holding_tenure}


@dataclass
class SignalResult:
    signal: str
    confidence: float
    composite_score: float
    dimension_scores: Dict[str, float]
    signal_reasoning: List[str]
    key_positives: List[str]
    key_negatives: List[str]
    user_profile_match: Dict[str, Any]
    red_flags_count: int
    warnings_count: int


class SignalGenerator:
    """Generates investment signals based on multi-dimensional analysis."""
    
    def __init__(self):
        self.weights = get_signal_weights()
        self.buy_threshold = get_config('signals.buy_threshold', 70)
        self.hold_threshold = get_config('signals.hold_threshold', 50)
        self.avoid_threshold = get_config('signals.avoid_threshold', 30)
    
    def generate_signal(self, user_profile: UserProfile, governance_score, financial_score,
                        valuation_score, market_score, ml_context=None, red_flags: List[Dict] = None) -> SignalResult:
        red_flags = red_flags or []
        
        dimension_scores = {
            'governance': governance_score.overall_score,
            'financial': financial_score.overall_score,
            'valuation': valuation_score.overall_score,
            'market_behaviour': market_score.overall_score
        }
        
        composite = self._calculate_composite(dimension_scores)
        adjusted, profile_match = self._apply_profile(composite, user_profile, financial_score, 
                                                       valuation_score, market_score)
        penalty = self._calculate_penalty(red_flags)
        final_score = max(0, adjusted - penalty)
        
        signal, confidence = self._determine_signal(final_score, red_flags, governance_score, financial_score)
        reasoning = self._generate_reasoning(signal, user_profile, dimension_scores, profile_match, red_flags)
        positives = self._extract_positives(governance_score, financial_score, valuation_score, market_score)
        negatives = self._extract_negatives(governance_score, financial_score, valuation_score, market_score, red_flags)
        
        return SignalResult(
            signal=signal, confidence=confidence, composite_score=round(final_score, 1),
            dimension_scores=dimension_scores, signal_reasoning=reasoning,
            key_positives=positives, key_negatives=negatives,
            user_profile_match=profile_match, red_flags_count=len(red_flags),
            warnings_count=self._count_warnings(governance_score, financial_score, valuation_score, market_score)
        )
    
    def _calculate_composite(self, scores: Dict[str, float]) -> float:
        return sum(scores[d] * self.weights.get(d, 0.25) for d in scores)
    
    def _apply_profile(self, score: float, profile: UserProfile, fin, val, mkt) -> Tuple[float, Dict]:
        adjusted = score
        match = {'return_expectation': {}, 'risk_match': {}, 'tenure_fit': {}}
        
        growth = fin.revenue_cagr_5y or 0
        pe_pct = val.pe_percentile_own or 50
        potential = growth + (5 if pe_pct < 30 else -5 if pe_pct > 70 else 0)
        gap = profile.expected_return - potential
        
        match['return_expectation'] = {'expected': profile.expected_return, 'estimated': round(potential, 1),
                                        'gap': round(gap, 1), 'meets': gap <= 0}
        if gap > 10:
            adjusted -= 15
        elif gap > 5:
            adjusted -= 8
        elif gap <= 0:
            adjusted += 5
        
        vol = mkt.volatility_1y or 30
        risk_cfg = get_risk_profile(profile.risk_appetite)
        max_vol = risk_cfg.get('max_volatility', 35)
        
        match['risk_match'] = {'user_risk': profile.risk_appetite, 'stock_vol': vol,
                               'max_acceptable': max_vol, 'within_tolerance': vol <= max_vol}
        if vol > max_vol * 1.5:
            adjusted -= 20
        elif vol > max_vol:
            adjusted -= 10
        elif vol < max_vol * 0.7:
            adjusted += 5
        
        match['tenure_fit'] = {'tenure': profile.holding_tenure, 'suitable': True, 'reason': 'Suitable'}
        
        return adjusted, match
    
    def _calculate_penalty(self, red_flags: List[Dict]) -> float:
        return min(50, sum(25 if f.get('severity') == 'critical' else 15 if f.get('severity') == 'high' else 8 
                          for f in red_flags))
    
    def _determine_signal(self, score: float, red_flags: List, gov, fin) -> Tuple[str, float]:
        if any(f.get('severity') == 'critical' for f in red_flags):
            return Signal.AVOID, 90.0
        if gov.overall_score < 30 and fin.overall_score < 30:
            return Signal.SELL, 80.0
        if score >= self.buy_threshold:
            return Signal.BUY, min(95, 70 + score - self.buy_threshold)
        if score >= self.hold_threshold:
            return Signal.HOLD, 60 + (score - self.hold_threshold) * 0.5
        if score >= self.avoid_threshold:
            return Signal.AVOID, 60 + (self.hold_threshold - score)
        return Signal.SELL, 70 + (self.avoid_threshold - score)
    
    def _generate_reasoning(self, signal: str, profile: UserProfile, scores: Dict, 
                            match: Dict, red_flags: List) -> List[str]:
        reasons = []
        if signal == Signal.BUY:
            reasons.append(f"Stock aligns with your profile ({profile.expected_return}% CAGR, {profile.risk_appetite} risk).")
        elif signal == Signal.HOLD:
            reasons.append("Mixed characteristics. Consider monitoring if already holding.")
        elif signal == Signal.AVOID:
            reasons.append("Stock does not align well with your investment profile.")
        else:
            reasons.append("Significant concerns detected. Review any existing position.")
        
        strongest = max(scores, key=scores.get)
        weakest = min(scores, key=scores.get)
        reasons.append(f"Strongest: {strongest.replace('_', ' ').title()} ({scores[strongest]:.0f}/100)")
        reasons.append(f"Weakest: {weakest.replace('_', ' ').title()} ({scores[weakest]:.0f}/100)")
        
        if not match['return_expectation'].get('meets', True):
            reasons.append(f"Return gap: Expected {match['return_expectation']['expected']}%, estimated {match['return_expectation']['estimated']}%")
        if not match['risk_match'].get('within_tolerance', True):
            reasons.append(f"Risk mismatch: Volatility {match['risk_match']['stock_vol']:.0f}% exceeds {match['risk_match']['max_acceptable']:.0f}%")
        if red_flags:
            reasons.append(f"Red flags detected: {len(red_flags)}")
        
        return reasons
    
    def _extract_positives(self, gov, fin, val, mkt) -> List[str]:
        pos = []
        if gov.promoter_holding > 50:
            pos.append(f"Strong promoter holding: {gov.promoter_holding:.1f}%")
        if gov.pledge_ratio == 0:
            pos.append("No promoter pledge")
        if fin.roce_current and fin.roce_current > 18:
            pos.append(f"High ROCE: {fin.roce_current:.1f}%")
        if fin.revenue_cagr_5y and fin.revenue_cagr_5y > 15:
            pos.append(f"Strong growth: {fin.revenue_cagr_5y:.1f}% revenue CAGR")
        if val.valuation_zone == 'undervalued':
            pos.append("Trading at attractive valuations")
        if mkt.volatility_regime == 'low':
            pos.append("Low volatility stock")
        return pos[:6]
    
    def _extract_negatives(self, gov, fin, val, mkt, red_flags: List) -> List[str]:
        neg = []
        neg.extend(gov.warnings[:2])
        neg.extend(gov.red_flags)
        neg.extend(fin.warnings[:2])
        neg.extend(fin.red_flags)
        neg.extend(val.warnings[:2])
        neg.extend(mkt.warnings[:2])
        for f in red_flags[:3]:
            neg.append(f.get('description', 'Red flag'))
        return neg[:8]
    
    def _count_warnings(self, gov, fin, val, mkt) -> int:
        return len(gov.warnings) + len(fin.warnings) + len(val.warnings) + len(mkt.warnings)

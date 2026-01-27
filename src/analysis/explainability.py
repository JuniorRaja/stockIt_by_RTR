"""Explainability Engine for Indian Equity Intelligence."""

from typing import Dict, Any, List
from dataclasses import dataclass

from .signal_generator import SignalResult, UserProfile, Signal


@dataclass
class ExplainabilityReport:
    summary: str
    signal_explanation: str
    why_not_buy: List[str]
    confidence_factors: Dict[str, Any]
    dimension_breakdown: Dict[str, Dict[str, Any]]
    risk_factors: List[str]
    thesis_invalidators: List[str]
    user_profile_analysis: Dict[str, Any]
    data_quality_notes: List[str]


class ExplainabilityEngine:
    """Generates comprehensive explanations. Every analysis MUST include 'Why NOT to buy'."""
    
    def generate_explanation(self, signal_result: SignalResult, user_profile: UserProfile,
                             governance_score, financial_score, valuation_score, market_score,
                             stock_info: Dict[str, Any], red_flags: List[Dict] = None) -> ExplainabilityReport:
        red_flags = red_flags or []
        
        summary = self._generate_summary(signal_result, stock_info, user_profile)
        signal_explanation = self._explain_signal(signal_result)
        why_not_buy = self._generate_why_not_buy(user_profile, governance_score, financial_score,
                                                   valuation_score, market_score, signal_result, red_flags)
        confidence = self._analyze_confidence(signal_result)
        breakdown = self._create_breakdown(governance_score, financial_score, valuation_score, market_score)
        risks = self._compile_risks(governance_score, financial_score, valuation_score, market_score, red_flags)
        invalidators = self._identify_invalidators(signal_result, financial_score)
        profile_analysis = {'profile': user_profile.to_dict(), 'match': signal_result.user_profile_match}
        data_notes = self._assess_data_quality()
        
        return ExplainabilityReport(
            summary=summary, signal_explanation=signal_explanation, why_not_buy=why_not_buy,
            confidence_factors=confidence, dimension_breakdown=breakdown, risk_factors=risks,
            thesis_invalidators=invalidators, user_profile_analysis=profile_analysis,
            data_quality_notes=data_notes
        )
    
    def _generate_summary(self, result: SignalResult, info: Dict, profile: UserProfile) -> str:
        name = info.get('name', info.get('symbol', 'This stock'))
        if result.signal == Signal.BUY:
            return f"{name} appears suitable for your profile. Score: {result.composite_score:.0f}/100. Review 'Why NOT to buy' section."
        elif result.signal == Signal.HOLD:
            return f"{name} shows mixed characteristics. Score: {result.composite_score:.0f}/100."
        elif result.signal == Signal.AVOID:
            return f"{name} does not align with your profile ({profile.expected_return}% CAGR, {profile.risk_appetite} risk)."
        return f"{name} shows significant concerns. Score: {result.composite_score:.0f}/100."
    
    def _explain_signal(self, result: SignalResult) -> str:
        lines = [f"Signal: {result.signal}", "", "Dimension Scores:"]
        for dim, score in result.dimension_scores.items():
            bar = "█" * int(score / 10) + "░" * (10 - int(score / 10))
            lines.append(f"  {dim.replace('_', ' ').title()}: {bar} {score:.0f}")
        lines.append(f"\nComposite: {result.composite_score:.0f}/100")
        return "\n".join(lines)
    
    def _generate_why_not_buy(self, profile: UserProfile, gov, fin, val, mkt, result: SignalResult, red_flags: List) -> List[str]:
        """MANDATORY: Generate 'Why NOT to buy' - shown even for BUY signals."""
        reasons = []
        
        # Return gap
        ret = result.user_profile_match.get('return_expectation', {})
        if not ret.get('meets', True):
            reasons.append(f"RETURN GAP: Expected {ret['expected']}%, estimated {ret['estimated']}%. Gap: {ret['gap']}%")
        
        # Risk mismatch
        risk = result.user_profile_match.get('risk_match', {})
        if not risk.get('within_tolerance', True):
            reasons.append(f"RISK MISMATCH: Volatility {risk['stock_vol']:.0f}% exceeds your tolerance ({risk['max_acceptable']:.0f}%)")
        
        # Governance
        for rf in gov.red_flags:
            reasons.append(f"GOVERNANCE: {rf}")
        if gov.promoter_holding_trend == 'decreasing':
            reasons.append(f"PROMOTER: Holding declining ({gov.details.get('promoter_analysis', {}).get('change_5y', 0):.1f}% over 5Y)")
        if gov.pledge_ratio > 10:
            reasons.append(f"PLEDGE: {gov.pledge_ratio:.1f}% shares pledged - forced selling risk")
        
        # Financial
        for rf in fin.red_flags:
            reasons.append(f"FINANCIAL: {rf}")
        if fin.roce_current and fin.roce_current < 12:
            reasons.append(f"CAPITAL: ROCE {fin.roce_current:.1f}% below cost of capital")
        if fin.debt_to_equity and fin.debt_to_equity > 1:
            reasons.append(f"LEVERAGE: D/E {fin.debt_to_equity:.2f}x creates financial risk")
        
        # Valuation
        if val.stress_level in ['high', 'extreme']:
            reasons.append(f"VALUATION: Stress level '{val.stress_level}' - high downside risk")
        if val.pe_percentile_own and val.pe_percentile_own > 75:
            reasons.append(f"EXPENSIVE: PE at {val.pe_percentile_own:.0f}th percentile of history")
        if val.peg_ratio and val.peg_ratio > 2:
            reasons.append(f"PEG: {val.peg_ratio:.2f} - paying premium for growth")
        
        # Market
        if mkt.max_drawdown > 40:
            reasons.append(f"DRAWDOWN: Stock fell {mkt.max_drawdown:.0f}% historically")
        if mkt.volatility_regime in ['high', 'extreme']:
            reasons.append(f"VOLATILITY: {mkt.volatility_regime} regime ({mkt.volatility_1y:.0f}%)")
        if mkt.beta and mkt.beta > 1.3:
            reasons.append(f"BETA: {mkt.beta:.2f} - amplifies market moves")
        
        # Red flags
        for f in red_flags:
            reasons.append(f"RED FLAG ({f.get('severity', 'medium').upper()}): {f.get('description', 'Issue detected')}")
        
        # Always include
        reasons.append("TIMING: Even good stocks can have poor timing. Markets can stay irrational.")
        reasons.append("DATA LIMITS: Analysis uses available data which may be incomplete or delayed.")
        
        if result.signal == Signal.BUY and len(reasons) < 4:
            reasons.append("EVEN FOR BUYS: No investment is risk-free. Position sizing and diversification matter.")
        
        return reasons
    
    def _analyze_confidence(self, result: SignalResult) -> Dict:
        return {
            'overall': result.confidence,
            'strengthening': [f for f in ['No red flags'] if result.red_flags_count == 0],
            'weakening': [f"{result.red_flags_count} red flags" for _ in [1] if result.red_flags_count > 0]
        }
    
    def _create_breakdown(self, gov, fin, val, mkt) -> Dict:
        return {
            'governance': {'score': gov.overall_score, 'warnings': gov.warnings, 'red_flags': gov.red_flags},
            'financial': {'score': fin.overall_score, 'warnings': fin.warnings, 'red_flags': fin.red_flags},
            'valuation': {'score': val.overall_score, 'warnings': val.warnings},
            'market': {'score': mkt.overall_score, 'warnings': mkt.warnings}
        }
    
    def _compile_risks(self, gov, fin, val, mkt, red_flags: List) -> List[str]:
        risks = list(set(gov.red_flags + fin.red_flags + [f.get('description', '') for f in red_flags]))
        if val.stress_level in ['high', 'extreme']:
            risks.append(f"Valuation stress: {val.stress_level}")
        if mkt.max_drawdown > 50:
            risks.append(f"Historical drawdown: {mkt.max_drawdown:.1f}%")
        return risks
    
    def _identify_invalidators(self, result: SignalResult, fin) -> List[str]:
        return [
            "Revenue growth slowing below historical average for 2+ quarters",
            "ROCE falling below cost of capital (10-12%)",
            "Promoter selling shares or increasing pledge significantly",
            "Auditor change or qualification in audit report",
            "Broader market correction or sector rotation",
            "Regulatory changes affecting business model"
        ]
    
    def _assess_data_quality(self) -> List[str]:
        return ["Data quality appears adequate for analysis"]

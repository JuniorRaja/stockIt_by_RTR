"""Scenario Simulator for Stocron by RTR."""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import logging

from ..utils.helpers import safe_divide

logger = logging.getLogger(__name__)


@dataclass
class ScenarioDefinition:
    name: str
    scenario_type: str
    description: str
    parameters: Dict[str, float]
    duration_years: int


@dataclass
class ScenarioResult:
    scenario: ScenarioDefinition
    base_case: Dict[str, Any]
    stressed_case: Dict[str, Any]
    impact_summary: Dict[str, float]
    signal_change: Optional[str]
    resilience_rating: str
    key_findings: List[str]


class ScenarioSimulator:
    """Stress-tests investment thesis under various scenarios."""
    
    PRESETS = {
        'revenue_stagnation': ScenarioDefinition(
            'Revenue Stagnation', 'revenue_stagnation',
            'Revenue growth drops to 0% for 3 years', {'revenue_growth': 0.0}, 3
        ),
        'margin_compression': ScenarioDefinition(
            'Margin Compression', 'margin_compression',
            'Operating margins compressed by 500bps', {'margin_reduction_bps': 500}, 2
        ),
        'rising_rates': ScenarioDefinition(
            'Rising Interest Rates', 'rate_increase',
            'Rates +200bps, valuation multiples compress', {'rate_increase_bps': 200, 'pe_compression_pct': 15}, 2
        ),
        'demand_slowdown': ScenarioDefinition(
            'Demand Slowdown', 'demand_slowdown',
            'Volume growth negative, prices stall', {'volume_growth': -5.0, 'price_growth': 0.0}, 2
        ),
        'cost_inflation': ScenarioDefinition(
            'Cost Inflation', 'cost_inflation',
            'Raw material costs increase significantly', {'rm_cost_increase_pct': 20, 'pass_through': 0.5}, 2
        ),
        'currency_shock': ScenarioDefinition(
            'Currency Depreciation', 'currency',
            'INR depreciates 15%', {'depreciation_pct': 15, 'import_dependency': 30}, 1
        )
    }
    
    def simulate(self, scenario_name: str, stock_info: Dict, financial_metrics: Dict,
                 valuation: Dict, user_profile: Dict, custom_params: Optional[Dict] = None) -> ScenarioResult:
        scenario = self.PRESETS.get(scenario_name)
        if scenario_name == 'custom' and custom_params:
            scenario = ScenarioDefinition('Custom', 'custom', 'User-defined', custom_params, 
                                           custom_params.get('duration', 2))
        if scenario is None:
            raise ValueError(f"Unknown scenario: {scenario_name}")
        
        base = self._calc_base(financial_metrics, valuation, user_profile)
        stressed = self._apply_stress(scenario, financial_metrics, valuation, user_profile)
        impact = self._calc_impact(base, stressed)
        signal_change = self._assess_signal_change(base, stressed, impact)
        resilience = self._rate_resilience(impact)
        findings = self._generate_findings(scenario, base, stressed, impact)
        
        return ScenarioResult(scenario=scenario, base_case=base, stressed_case=stressed,
                              impact_summary=impact, signal_change=signal_change,
                              resilience_rating=resilience, key_findings=findings)
    
    def _calc_base(self, fin: Dict, val: Dict, profile: Dict) -> Dict:
        rev_growth = fin.get('revenue_cagr_5y', 10)
        opm = fin.get('operating_margin', 15)
        pe = val.get('pe_ratio', 20)
        price = val.get('current_price', 100)
        tenure = profile.get('holding_tenure', 5)
        
        eps_growth = rev_growth * 0.8 + 2
        terminal_pe = pe * 0.9
        expected_ret = eps_growth + safe_divide((terminal_pe - pe), pe * tenure) * 100
        
        return {'rev_growth': rev_growth, 'opm': opm, 'pe': pe, 'terminal_pe': terminal_pe,
                'eps_growth': eps_growth, 'expected_return': expected_ret, 'price': price,
                'target_price': price * (1 + expected_ret/100) ** tenure}
    
    def _apply_stress(self, scenario: ScenarioDefinition, fin: Dict, val: Dict, profile: Dict) -> Dict:
        params = scenario.parameters
        tenure = profile.get('holding_tenure', 5)
        
        rev_growth = fin.get('revenue_cagr_5y', 10)
        opm = fin.get('operating_margin', 15)
        pe = val.get('pe_ratio', 20)
        price = val.get('current_price', 100)
        
        stressed_rev = rev_growth
        stressed_opm = opm
        stressed_pe = pe
        
        if scenario.scenario_type == 'revenue_stagnation':
            stressed_rev = params.get('revenue_growth', 0)
        elif scenario.scenario_type == 'margin_compression':
            stressed_opm = max(0, opm - params.get('margin_reduction_bps', 500) / 100)
        elif scenario.scenario_type == 'rate_increase':
            stressed_pe = pe * (1 - params.get('pe_compression_pct', 15) / 100)
        elif scenario.scenario_type == 'demand_slowdown':
            stressed_rev = params.get('volume_growth', -5) + params.get('price_growth', 0)
        elif scenario.scenario_type == 'cost_inflation':
            rm_inc = params.get('rm_cost_increase_pct', 20)
            pass_through = params.get('pass_through', 0.5)
            margin_impact = rm_inc * 0.6 * (1 - pass_through) * (100 - opm) / 100
            stressed_opm = max(0, opm - margin_impact)
        elif scenario.scenario_type == 'currency':
            dep = params.get('depreciation_pct', 15)
            imp_dep = params.get('import_dependency', 30)
            stressed_opm = max(0, opm - dep * imp_dep / 100)
        
        stressed_eps = stressed_rev
        if stressed_opm < opm:
            margin_drag = (opm - stressed_opm) / opm * 100
            stressed_eps -= margin_drag * 0.3
        
        stressed_terminal = stressed_pe * 0.85
        stressed_ret = stressed_eps + safe_divide((stressed_terminal - pe), pe * tenure) * 100
        
        return {'rev_growth': stressed_rev, 'opm': stressed_opm, 'pe': pe, 'terminal_pe': stressed_terminal,
                'eps_growth': stressed_eps, 'expected_return': stressed_ret, 'price': price,
                'target_price': price * (1 + stressed_ret/100) ** tenure, 'duration': scenario.duration_years}
    
    def _calc_impact(self, base: Dict, stressed: Dict) -> Dict:
        return {
            'rev_impact': stressed['rev_growth'] - base['rev_growth'],
            'margin_impact': stressed['opm'] - base['opm'],
            'eps_impact': stressed['eps_growth'] - base['eps_growth'],
            'pe_impact': stressed['terminal_pe'] - base['terminal_pe'],
            'return_impact': stressed['expected_return'] - base['expected_return'],
            'target_impact_pct': (stressed['target_price'] - base['target_price']) / base['target_price'] * 100
        }
    
    def _assess_signal_change(self, base: Dict, stressed: Dict, impact: Dict) -> Optional[str]:
        base_ret, stressed_ret = base['expected_return'], stressed['expected_return']
        if base_ret > 15 and stressed_ret < 10:
            return "BUY → HOLD: Returns below threshold under stress"
        if base_ret > 10 and stressed_ret < 5:
            return "HOLD → AVOID: Unattractive under stress"
        if base_ret > 5 and stressed_ret < 0:
            return "EXIT consideration: Negative returns under stress"
        if abs(impact['return_impact']) > 10:
            return "Signal sensitivity: High impact"
        return None
    
    def _rate_resilience(self, impact: Dict) -> str:
        composite = (abs(impact['return_impact']) + abs(impact['target_impact_pct'])) / 2
        if composite < 5:
            return 'robust'
        elif composite < 10:
            return 'resilient'
        elif composite < 20:
            return 'moderate'
        return 'fragile'
    
    def _generate_findings(self, scenario: ScenarioDefinition, base: Dict, stressed: Dict, impact: Dict) -> List[str]:
        findings = [f"Under {scenario.name} ({scenario.duration_years}Y):"]
        ret_impact = impact['return_impact']
        if ret_impact < -10:
            findings.append(f"Return drops {abs(ret_impact):.1f}pp (from {base['expected_return']:.1f}% to {stressed['expected_return']:.1f}%)")
        elif ret_impact < 0:
            findings.append(f"Moderate return impact: {abs(ret_impact):.1f}pp")
        else:
            findings.append("Returns relatively protected")
        
        if impact['margin_impact'] < -3:
            findings.append(f"Margins compressed {abs(impact['margin_impact']):.1f}pp")
        if impact['target_impact_pct'] < -20:
            findings.append(f"Target price drops {abs(impact['target_impact_pct']):.0f}%")
        if stressed['expected_return'] < 0:
            findings.append("WARNING: Negative returns under stress")
        
        return findings
    
    def run_all(self, stock_info: Dict, fin: Dict, val: Dict, profile: Dict) -> Dict[str, ScenarioResult]:
        results = {}
        for name in self.PRESETS:
            try:
                results[name] = self.simulate(name, stock_info, fin, val, profile)
            except Exception as e:
                logger.warning(f"Scenario {name} failed: {e}")
        return results
    
    def get_resilience_summary(self, results: Dict[str, ScenarioResult]) -> Dict:
        if not results:
            return {'overall': 'unknown'}
        ratings = [r.resilience_rating for r in results.values()]
        counts = {r: ratings.count(r) for r in ['robust', 'resilient', 'moderate', 'fragile']}
        
        if counts.get('fragile', 0) >= 2:
            overall = 'fragile'
        elif counts.get('fragile', 0) >= 1 or counts.get('moderate', 0) >= 3:
            overall = 'moderate'
        elif counts.get('robust', 0) >= 3:
            overall = 'robust'
        else:
            overall = 'resilient'
        
        return {'overall': overall, 'counts': counts}
    
    def list_scenarios(self) -> List[Dict]:
        return [{'key': k, 'name': v.name, 'description': v.description} for k, v in self.PRESETS.items()]

"""ML-Assisted Context Analysis for Indian Equity Intelligence."""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class MLContextScore:
    """Container for ML context analysis results."""
    archetype: str
    archetype_confidence: float
    archetype_description: str
    similar_companies: List[Dict[str, Any]]
    trajectory_cluster: int
    features_used: List[str]
    model_info: Dict[str, Any]
    disclaimer: str


class MLContextAnalyzer:
    """
    Provides ML-assisted context (NON-DECISIVE).
    Used only for context, never for final decision.
    """
    
    ARCHETYPES = {
        0: {'name': 'Steady Compounder', 'description': 'Consistent growth, high ROCE, low volatility.'},
        1: {'name': 'High Growth', 'description': 'Rapid revenue/profit growth, may have higher valuations.'},
        2: {'name': 'Cyclical', 'description': 'Performance tied to economic cycles.'},
        3: {'name': 'Turnaround', 'description': 'Previously stressed, showing recovery signs.'},
        4: {'name': 'Value Trap Risk', 'description': 'Optically cheap but deteriorating fundamentals.'},
        5: {'name': 'Defensive', 'description': 'Lower growth but stable earnings.'},
        6: {'name': 'Asset Heavy', 'description': 'Capital intensive, moderate returns.'},
        7: {'name': 'Speculative', 'description': 'High volatility, unclear fundamentals.'}
    }
    
    DISCLAIMER = ("ML analysis is for contextual understanding only. "
                  "It does NOT influence the final signal and should NOT be the sole basis for decisions.")
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.default_features = ['revenue_cagr_5y', 'pat_cagr_5y', 'roce_avg_5y', 'debt_to_equity',
                                  'promoter_holding', 'volatility_1y', 'pe_ratio', 'market_cap_log']
    
    def analyze(self, symbol: str, features: Dict[str, float],
                all_stocks_features: Optional[pd.DataFrame] = None) -> MLContextScore:
        feature_vector = self._prepare_features(features)
        archetype, confidence = self._classify_archetype(feature_vector)
        archetype_info = self.ARCHETYPES.get(archetype, self.ARCHETYPES[7])
        
        similar = []
        if all_stocks_features is not None and not all_stocks_features.empty:
            similar = self._find_similar(symbol, feature_vector, all_stocks_features)
        
        return MLContextScore(
            archetype=archetype_info['name'], archetype_confidence=confidence,
            archetype_description=archetype_info['description'],
            similar_companies=similar, trajectory_cluster=archetype,
            features_used=self.default_features,
            model_info={'model_loaded': self.model is not None, 'method': 'rule_based'},
            disclaimer=self.DISCLAIMER
        )
    
    def _prepare_features(self, features: Dict[str, float]) -> np.ndarray:
        vector = []
        for feat in self.default_features:
            value = features.get(feat, 0)
            if value is None or (isinstance(value, float) and np.isnan(value)):
                value = 0
            vector.append(value)
        return np.array(vector).reshape(1, -1)
    
    def _classify_archetype(self, features: np.ndarray) -> Tuple[int, float]:
        # Rule-based classification
        f = features[0]
        revenue_cagr, pat_cagr, roce, de, promoter, vol, pe, mcap = f
        
        scores = [0] * 8
        
        if roce > 15 and revenue_cagr > 8 and vol < 35:
            scores[0] += 3  # Steady Compounder
        if revenue_cagr > 20 or pat_cagr > 25:
            scores[1] += 3  # High Growth
        if de > 1:
            scores[6] += 2  # Asset Heavy
            scores[2] += 1  # Cyclical
        if revenue_cagr < 5 and pat_cagr < 5:
            scores[5] += 2 if roce > 12 else 0  # Defensive
            scores[4] += 2 if roce <= 12 else 0  # Value Trap
        if vol > 50:
            scores[7] += 3  # Speculative
        if promoter < 30:
            scores[7] += 1
        
        max_score = max(scores)
        archetype = scores.index(max_score)
        sorted_scores = sorted(scores, reverse=True)
        confidence = min(0.9, 0.5 + (sorted_scores[0] - sorted_scores[1]) * 0.1) if max_score > 0 else 0.3
        
        return archetype, confidence
    
    def _find_similar(self, symbol: str, features: np.ndarray, all_stocks: pd.DataFrame, top_n: int = 5) -> List[Dict]:
        try:
            all_stocks = all_stocks[all_stocks['symbol'] != symbol].copy()
            if all_stocks.empty:
                return []
            
            feature_cols = [c for c in self.default_features if c in all_stocks.columns]
            if not feature_cols:
                return []
            
            X = all_stocks[feature_cols].fillna(0).values
            distances = np.sqrt(np.sum((X - features[:, :len(feature_cols)]) ** 2, axis=1))
            closest = np.argsort(distances)[:top_n]
            
            return [{'symbol': all_stocks.iloc[i]['symbol'], 
                     'similarity': round(1 / (1 + distances[i]), 3)} for i in closest]
        except Exception:
            return []
    
    def list_archetypes(self) -> List[Dict]:
        return [{'id': k, **v} for k, v in self.ARCHETYPES.items()]

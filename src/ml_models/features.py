"""
Feature Engineering Module

Extracts and prepares features from stock data for ML models.
Combines outputs from all analysis engines into ML-ready feature vectors.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ..utils.config import get_macro_rsi_thresholds

logger = logging.getLogger(__name__)


@dataclass
class FeatureSet:
    """Container for extracted features."""
    symbol: str
    features: Dict[str, float]
    feature_names: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array in consistent order."""
        return np.array([self.features.get(name, 0.0) for name in self.feature_names])
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to single-row DataFrame."""
        return pd.DataFrame([self.features], columns=self.feature_names)


class FeatureEngineer:
    """
    Extracts features from stock data for ML models.
    
    Features are organized into categories:
    - Price features: Returns, volatility, momentum
    - Technical features: RSI, MACD, Bollinger bands
    - Fundamental features: From analysis engine outputs
    - Forecast features: From time-series forecaster
    """
    
    # Standard feature list
    PRICE_FEATURES = [
        "return_1d", "return_5d", "return_20d", "return_60d", "return_252d",
        "log_return_252d", "real_return_252d",
        "volatility_20d", "volatility_60d",
        "momentum_10d", "momentum_30d",
        "sma_ratio_20", "sma_ratio_50", "sma_ratio_200",
        "high_52w_pct", "low_52w_pct",
    ]
    
    TECHNICAL_FEATURES = [
        "rsi_14", "rsi_28",
        "macd", "macd_signal", "macd_histogram",
        "bb_position",  # Position within Bollinger Bands
        "atr_14",  # Average True Range
        "adx_14",  # Average Directional Index
    ]
    
    FUNDAMENTAL_FEATURES = [
        # From Governance Engine
        "years_listed", "promoter_holding", "pledge_ratio",
        "dividend_consistency", "auditor_stability_score",
        "governance_score",
        
        # From Financial Engine
        "revenue_cagr_3y", "revenue_cagr_5y",
        "pat_cagr_3y", "pat_cagr_5y",
        "roce", "fcf_yield", "debt_to_equity",
        "operating_margin", "earnings_quality",
        "financial_score",
        
        # From Valuation Engine
        "pe_ratio", "pe_percentile",
        "pb_ratio", "pb_percentile",
        "ev_ebitda", "peg_ratio",
        "valuation_score",
        
        # From Market Engine
        "max_drawdown", "avg_recovery_months",
        "volatility_1y", "volatility_3y",
        "beta", "sharpe_ratio",
        "relative_performance_1y",
        "market_score",
    ]
    
    FORECAST_FEATURES = [
        "forecast_trend_bullish",  # 1 if bullish, 0 otherwise
        "forecast_trend_strength",
        "forecast_return_5d",
        "forecast_return_30d",
        "forecast_confidence",
        "forecast_uncertainty",
    ]
    
    MACRO_FEATURES = [
        "repo_rate_current",
        "repo_rate_change_1y",
        "usdinr_level",
        "usdinr_change_1y",
        "crude_oil_level",
        "crude_oil_change_1y",
        "market_regime_bull",      # One-hot encoded
        "market_regime_bear",
        "cpi_inflation_yoy",
        "rsi_regime_adjusted",     # RSI adjusted for market regime
    ]
    
    def __init__(self, include_technical: bool = True, include_forecast: bool = True,
                 include_macro: bool = True):
        """
        Initialize feature engineer.
        
        Args:
            include_technical: Include technical analysis features
            include_forecast: Include forecast-based features
            include_macro: Include macro-economic features
        """
        self.include_technical = include_technical
        self.include_forecast = include_forecast
        self.include_macro = include_macro
        self._market_regime = 'sideways'  # Default regime
        
        # Build feature list
        self.feature_names = (
            self.PRICE_FEATURES +
            (self.TECHNICAL_FEATURES if include_technical else []) +
            self.FUNDAMENTAL_FEATURES +
            (self.FORECAST_FEATURES if include_forecast else []) +
            (self.MACRO_FEATURES if include_macro else [])
        )
    
    def extract_features(
        self,
        symbol: str,
        prices: pd.Series,
        governance_result: Optional[Any] = None,
        financial_result: Optional[Any] = None,
        valuation_result: Optional[Any] = None,
        market_result: Optional[Any] = None,
        forecast_result: Optional[Any] = None,
        macro_data: Optional[Dict[str, Any]] = None,
    ) -> FeatureSet:
        """
        Extract all features for a stock.
        
        Args:
            symbol: Stock symbol
            prices: Historical closing prices (DatetimeIndex)
            governance_result: Output from GovernanceAnalyzer
            financial_result: Output from FinancialAnalyzer
            valuation_result: Output from ValuationAnalyzer
            market_result: Output from MarketBehaviourAnalyzer
            forecast_result: Output from TimeSeriesForecaster
            macro_data: Macro-economic data from MacroDataProvider
            
        Returns:
            FeatureSet with all extracted features
        """
        features = {}
        
        # Price features
        cpi_index = None
        if macro_data is not None:
            cpi_index = macro_data.get('cpi_index')
        price_features = self._extract_price_features(prices, cpi_index=cpi_index)
        features.update(price_features)
        
        # Technical features
        if self.include_technical:
            technical_features = self._extract_technical_features(prices)
            features.update(technical_features)
        
        # Fundamental features from analysis engines
        fundamental_features = self._extract_fundamental_features(
            governance_result, financial_result, valuation_result, market_result
        )
        features.update(fundamental_features)
        
        # Forecast features
        if self.include_forecast and forecast_result is not None:
            forecast_features = self._extract_forecast_features(forecast_result, prices)
            features.update(forecast_features)
        
        # Macro features
        if self.include_macro:
            macro_features = self._extract_macro_features(macro_data, prices)
            features.update(macro_features)
        
        features = self._sanitize_features(features)
        return FeatureSet(
            symbol=symbol,
            features=features,
            feature_names=self.feature_names,
            metadata={
                "price_data_points": len(prices),
                "latest_price": float(prices.iloc[-1]) if len(prices) > 0 else 0.0,
                "market_regime": self._market_regime,
            }
        )

    def _sanitize_features(self, features: Dict[str, Any]) -> Dict[str, float]:
        """Coerce all feature values to floats, defaulting non-numerics to 0."""
        sanitized: Dict[str, float] = {}
        for name in self.feature_names:
            value = features.get(name, 0.0)
            try:
                if value is None or (isinstance(value, float) and np.isnan(value)):
                    sanitized[name] = 0.0
                else:
                    sanitized[name] = float(value)
            except Exception:
                sanitized[name] = 0.0
        return sanitized
    
    def _extract_price_features(self, prices: pd.Series, cpi_index: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """Extract price-based features."""
        features = {}
        
        if len(prices) < 2:
            return {name: 0.0 for name in self.PRICE_FEATURES}
        
        current = prices.iloc[-1]
        
        # Returns
        for period, name in [(1, "1d"), (5, "5d"), (20, "20d"), (60, "60d"), (252, "252d")]:
            if len(prices) > period:
                past = prices.iloc[-period-1]
                features[f"return_{name}"] = (current - past) / past if past != 0 else 0.0
            else:
                features[f"return_{name}"] = 0.0

        # Log returns (long-horizon stability)
        if len(prices) > 252:
            past = prices.iloc[-253]
            if past > 0 and current > 0:
                features["log_return_252d"] = float(np.log(current / past))
            else:
                features["log_return_252d"] = 0.0
        else:
            features["log_return_252d"] = 0.0

        # Inflation-adjusted return (if CPI index available)
        features["real_return_252d"] = 0.0
        if cpi_index is not None and not isinstance(cpi_index, (int, float)):
            cpi_series = cpi_index
            if isinstance(cpi_index, pd.DataFrame):
                cpi_series = cpi_index['cpi_index'] if 'cpi_index' in cpi_index.columns else cpi_index.iloc[:, 0]
            if isinstance(cpi_series, pd.Series) and not cpi_series.empty:
                cpi_series = cpi_series.copy()
                cpi_series.index = pd.to_datetime(cpi_series.index)
                price_series = prices.copy()
                price_series.index = pd.to_datetime(price_series.index)
                aligned_cpi = cpi_series.reindex(price_series.index, method='ffill')
                real_prices = price_series / (aligned_cpi / 100)
                if len(real_prices.dropna()) > 252:
                    real_current = real_prices.iloc[-1]
                    real_past = real_prices.iloc[-253]
                    if real_past and real_past != 0:
                        features["real_return_252d"] = float((real_current - real_past) / real_past)
        
        # Volatility (annualized)
        returns = prices.pct_change(fill_method=None).dropna()
        for period, name in [(20, "20d"), (60, "60d")]:
            if len(returns) >= period:
                features[f"volatility_{name}"] = returns.tail(period).std() * np.sqrt(252)
            else:
                features[f"volatility_{name}"] = 0.0
        
        # Momentum (rate of change)
        for period, name in [(10, "10d"), (30, "30d")]:
            if len(prices) > period:
                features[f"momentum_{name}"] = (current / prices.iloc[-period-1] - 1) if prices.iloc[-period-1] != 0 else 0.0
            else:
                features[f"momentum_{name}"] = 0.0
        
        # SMA ratios
        for period in [20, 50, 200]:
            if len(prices) >= period:
                sma = prices.tail(period).mean()
                features[f"sma_ratio_{period}"] = current / sma if sma != 0 else 1.0
            else:
                features[f"sma_ratio_{period}"] = 1.0
        
        # 52-week high/low position
        if len(prices) >= 252:
            high_52w = prices.tail(252).max()
            low_52w = prices.tail(252).min()
            range_52w = high_52w - low_52w
            
            features["high_52w_pct"] = (high_52w - current) / high_52w if high_52w != 0 else 0.0
            features["low_52w_pct"] = (current - low_52w) / low_52w if low_52w != 0 else 0.0
        else:
            features["high_52w_pct"] = 0.0
            features["low_52w_pct"] = 0.0
        
        return features
    
    def _extract_technical_features(self, prices: pd.Series) -> Dict[str, float]:
        """Extract technical indicator features."""
        features = {}
        
        if len(prices) < 30:
            return {name: 0.0 for name in self.TECHNICAL_FEATURES}
        
        # RSI
        for period in [14, 28]:
            features[f"rsi_{period}"] = self._calculate_rsi(prices, period)
        
        # MACD
        macd, signal, histogram = self._calculate_macd(prices)
        features["macd"] = macd
        features["macd_signal"] = signal
        features["macd_histogram"] = histogram
        
        # Bollinger Bands position
        features["bb_position"] = self._calculate_bb_position(prices)
        
        # ATR (using high=low=close for simplicity with just close prices)
        features["atr_14"] = self._calculate_atr(prices, 14)
        
        # ADX (simplified)
        features["adx_14"] = self._calculate_adx(prices, 14)
        
        return features
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain.iloc[-1] / loss.iloc[-1] if loss.iloc[-1] != 0 else 0
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi) if not np.isnan(rsi) else 50.0
    
    def _calculate_macd(
        self, 
        prices: pd.Series, 
        fast: int = 12, 
        slow: int = 26, 
        signal: int = 9
    ) -> Tuple[float, float, float]:
        """Calculate MACD indicator."""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        
        # Normalize by price for comparability
        current_price = prices.iloc[-1]
        if current_price != 0:
            return (
                float(macd_line.iloc[-1] / current_price * 100),
                float(signal_line.iloc[-1] / current_price * 100),
                float(histogram.iloc[-1] / current_price * 100)
            )
        return 0.0, 0.0, 0.0
    
    def _calculate_bb_position(self, prices: pd.Series, period: int = 20) -> float:
        """Calculate position within Bollinger Bands (0 = lower, 1 = upper)."""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        
        upper = sma + 2 * std
        lower = sma - 2 * std
        
        current = prices.iloc[-1]
        bb_range = upper.iloc[-1] - lower.iloc[-1]
        
        if bb_range != 0:
            position = (current - lower.iloc[-1]) / bb_range
            return float(np.clip(position, 0, 1))
        return 0.5
    
    def _calculate_atr(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate Average True Range (simplified with just close prices)."""
        # Using price range as proxy for true range
        high_low = prices.rolling(window=2).apply(lambda x: max(x) - min(x))
        atr = high_low.rolling(window=period).mean()
        
        # Normalize by current price
        current = prices.iloc[-1]
        if current != 0:
            return float(atr.iloc[-1] / current * 100)
        return 0.0
    
    def _calculate_adx(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate ADX (simplified directional movement indicator)."""
        # Simplified: Use absolute momentum as proxy
        returns = prices.pct_change(fill_method=None)
        abs_returns = returns.abs()
        
        # Directional movement
        pos_dm = returns.where(returns > 0, 0).rolling(window=period).sum()
        neg_dm = (-returns.where(returns < 0, 0)).rolling(window=period).sum()
        
        total_dm = pos_dm + neg_dm
        if total_dm.iloc[-1] != 0:
            dx = abs(pos_dm.iloc[-1] - neg_dm.iloc[-1]) / total_dm.iloc[-1] * 100
            return float(dx)
        return 0.0
    
    def _extract_fundamental_features(
        self,
        governance_result: Optional[Any],
        financial_result: Optional[Any],
        valuation_result: Optional[Any],
        market_result: Optional[Any]
    ) -> Dict[str, float]:
        """Extract features from analysis engine results."""
        features = {}
        
        # Governance features
        if governance_result:
            gov = governance_result
            features["years_listed"] = getattr(gov, 'years_listed', 0)
            features["promoter_holding"] = getattr(gov, 'promoter_holding', 0)
            features["pledge_ratio"] = getattr(gov, 'pledge_ratio', 0)
            features["dividend_consistency"] = getattr(gov, 'dividend_consistency_years', 0)
            features["auditor_stability_score"] = getattr(gov, 'auditor_stability', 0)
            features["governance_score"] = getattr(gov, 'overall_score', 50)
        else:
            for name in ["years_listed", "promoter_holding", "pledge_ratio", 
                        "dividend_consistency", "auditor_stability_score"]:
                features[name] = 0
            features["governance_score"] = 50
        
        # Financial features
        if financial_result:
            fin = financial_result
            features["revenue_cagr_3y"] = getattr(fin, 'revenue_cagr_3y', 0) or 0
            features["revenue_cagr_5y"] = getattr(fin, 'revenue_cagr_5y', 0) or 0
            features["pat_cagr_3y"] = getattr(fin, 'pat_cagr_3y', 0) or 0
            features["pat_cagr_5y"] = getattr(fin, 'pat_cagr_5y', 0) or 0
            features["roce"] = getattr(fin, 'roce_current', 0) or 0
            features["fcf_yield"] = getattr(fin, 'fcf_yield', 0) or 0
            features["debt_to_equity"] = getattr(fin, 'debt_to_equity', 0) or 0
            features["operating_margin"] = (
                fin.details.get('margins', {}).get('opm_current')
                if getattr(fin, 'details', None) else 0
            ) or 0
            features["earnings_quality"] = getattr(fin, 'earnings_quality', 0) or 0
            features["financial_score"] = getattr(fin, 'overall_score', 50)
        else:
            for name in ["revenue_cagr_3y", "revenue_cagr_5y", "pat_cagr_3y", "pat_cagr_5y",
                        "roce", "fcf_yield", "debt_to_equity", "operating_margin", "earnings_quality"]:
                features[name] = 0
            features["financial_score"] = 50
        
        # Valuation features
        if valuation_result:
            val = valuation_result
            features["pe_ratio"] = getattr(val, 'current_pe', 0) or 0
            features["pe_percentile"] = getattr(val, 'pe_percentile_own', 50) or 50
            features["pb_ratio"] = getattr(val, 'current_pb', 0) or 0
            features["pb_percentile"] = getattr(val, 'pb_percentile_own', 50) or 50
            features["ev_ebitda"] = getattr(val, 'ev_ebitda', 0) or 0
            features["peg_ratio"] = getattr(val, 'peg_ratio', 0) or 0
            features["valuation_score"] = getattr(val, 'overall_score', 50)
        else:
            for name in ["pe_ratio", "pb_ratio", "ev_ebitda", "peg_ratio"]:
                features[name] = 0
            for name in ["pe_percentile", "pb_percentile"]:
                features[name] = 50
            features["valuation_score"] = 50
        
        # Market features
        if market_result:
            mkt = market_result
            features["max_drawdown"] = abs(getattr(mkt, 'max_drawdown', 0) or 0)
            features["avg_recovery_months"] = getattr(mkt, 'avg_recovery_months', 0) or 0
            features["volatility_1y"] = getattr(mkt, 'volatility_1y', 0) or 0
            features["volatility_3y"] = getattr(mkt, 'volatility_3y', 0) or 0
            features["beta"] = getattr(mkt, 'beta', 1) or 1
            features["sharpe_ratio"] = getattr(mkt, 'sharpe_ratio', 0) or 0
            features["relative_performance_1y"] = getattr(mkt, 'vs_nifty_1y', 0) or 0
            features["market_score"] = getattr(mkt, 'overall_score', 50)
        else:
            for name in ["max_drawdown", "avg_recovery_months", "volatility_1y", 
                        "volatility_3y", "sharpe_ratio", "relative_performance_1y"]:
                features[name] = 0
            features["beta"] = 1
            features["market_score"] = 50
        
        return features
    
    def _extract_forecast_features(
        self, 
        forecast_result: Any,
        prices: pd.Series
    ) -> Dict[str, float]:
        """Extract features from forecast results."""
        features = {}
        
        if forecast_result is None or not getattr(forecast_result, 'success', False):
            return {name: 0.0 for name in self.FORECAST_FEATURES}
        
        # Trend features
        trend = getattr(forecast_result, 'trend_direction', 'neutral')
        features["forecast_trend_bullish"] = 1.0 if trend == "bullish" else 0.0
        features["forecast_trend_strength"] = getattr(forecast_result, 'trend_strength', 0.0)
        
        # Return predictions
        predictions = getattr(forecast_result, 'predictions', None)
        current_price = float(prices.iloc[-1]) if len(prices) > 0 else 1.0
        
        if predictions is not None and len(predictions) > 0:
            # 5-day return
            if len(predictions) >= 5:
                pred_5d = predictions[4]
                features["forecast_return_5d"] = (pred_5d - current_price) / current_price
            else:
                features["forecast_return_5d"] = 0.0
            
            # 30-day return
            if len(predictions) >= 30:
                pred_30d = predictions[29]
                features["forecast_return_30d"] = (pred_30d - current_price) / current_price
            else:
                pred_30d = predictions[-1]
                features["forecast_return_30d"] = (pred_30d - current_price) / current_price
        else:
            features["forecast_return_5d"] = 0.0
            features["forecast_return_30d"] = 0.0
        
        # Confidence and uncertainty
        confidence = getattr(forecast_result, 'confidence_scores', None)
        features["forecast_confidence"] = float(np.mean(confidence)) if confidence is not None else 0.5
        
        intervals = getattr(forecast_result, 'prediction_intervals', None)
        if intervals and 'upper_90' in intervals and 'lower_10' in intervals:
            spread = np.mean(intervals['upper_90'] - intervals['lower_10'])
            features["forecast_uncertainty"] = spread / current_price if current_price > 0 else 0.0
        else:
            features["forecast_uncertainty"] = 0.0
        
        return features
    
    def _extract_macro_features(
        self, 
        macro_data: Optional[Dict[str, Any]],
        prices: pd.Series
    ) -> Dict[str, float]:
        """
        Extract macro-economic features.
        
        Macro factors like Repo Rates, USD-INR, and Crude Oil are often
        better predictors of long-term cycles in the Indian market
        than technical indicators alone.
        """
        features = {name: 0.0 for name in self.MACRO_FEATURES}
        
        if macro_data is None:
            return features
        
        # Repo Rate
        repo = macro_data.get('repo_rate')
        if repo is not None:
            if isinstance(repo, pd.DataFrame):
                repo = repo['repo_rate'] if 'repo_rate' in repo.columns else repo.iloc[:, 0]
            if isinstance(repo, pd.Series) and not repo.empty:
                features['repo_rate_current'] = float(repo.iloc[-1])
                if len(repo) > 252:
                    features['repo_rate_change_1y'] = float(repo.iloc[-1] - repo.iloc[-252])
        
        # USD-INR
        usdinr = macro_data.get('usdinr')
        if usdinr is not None:
            if isinstance(usdinr, pd.DataFrame):
                usdinr = usdinr['usdinr'] if 'usdinr' in usdinr.columns else usdinr.iloc[:, 0]
            if isinstance(usdinr, pd.Series) and not usdinr.empty:
                features['usdinr_level'] = float(usdinr.iloc[-1])
                if len(usdinr) > 252:
                    features['usdinr_change_1y'] = (
                        (usdinr.iloc[-1] / usdinr.iloc[-252] - 1) * 100
                    )
        
        # Crude Oil
        crude = macro_data.get('crude_oil')
        if crude is not None:
            if isinstance(crude, pd.DataFrame):
                crude = crude['crude_oil'] if 'crude_oil' in crude.columns else crude.iloc[:, 0]
            if isinstance(crude, pd.Series) and not crude.empty:
                features['crude_oil_level'] = float(crude.iloc[-1])
                if len(crude) > 252:
                    features['crude_oil_change_1y'] = (
                        (crude.iloc[-1] / crude.iloc[-252] - 1) * 100
                    )
        
        # Market Regime (one-hot encoded)
        regime = macro_data.get('market_regime', 'sideways')
        self._market_regime = regime
        features['market_regime_bull'] = 1.0 if regime == 'bull' else 0.0
        features['market_regime_bear'] = 1.0 if regime == 'bear' else 0.0
        
        # CPI Inflation
        cpi = macro_data.get('cpi')
        if cpi is not None:
            if isinstance(cpi, pd.DataFrame):
                cpi = cpi['cpi_yoy'] if 'cpi_yoy' in cpi.columns else cpi.iloc[:, 0]
            if isinstance(cpi, pd.Series) and not cpi.empty:
                features['cpi_inflation_yoy'] = float(cpi.iloc[-1])
        
        # RSI adjusted for market regime
        if len(prices) >= 14:
            raw_rsi = self._calculate_rsi(prices, 14)
            features['rsi_regime_adjusted'] = self._adjust_rsi_for_regime(raw_rsi, regime)
        
        return features
    
    def _adjust_rsi_for_regime(self, rsi: float, regime: str) -> float:
        """
        Adjust RSI based on market regime.
        
        Returns a normalized score (-1 to 1) where:
        - Negative values indicate oversold
        - Positive values indicate overbought
        - Magnitude indicates strength of signal
        """
        thresholds = self.get_regime_rsi_thresholds(regime)
        oversold = thresholds['oversold']
        overbought = thresholds['overbought']
        midpoint = (oversold + overbought) / 2
        
        if rsi < oversold:
            # Oversold: normalize to -1 to -0.5
            return -1.0 + 0.5 * (rsi / oversold)
        elif rsi > overbought:
            # Overbought: normalize to 0.5 to 1
            return 0.5 + 0.5 * ((rsi - overbought) / (100 - overbought))
        else:
            # Neutral zone: normalize to -0.5 to 0.5
            return (rsi - midpoint) / (overbought - oversold)
    
    @staticmethod
    def get_regime_rsi_thresholds(regime: str) -> Dict[str, int]:
        """
        Get dynamic RSI thresholds based on market regime.
        
        In bull markets, RSI can stay overbought longer.
        In bear markets, oversold conditions can persist.
        """
        thresholds = get_macro_rsi_thresholds()
        return thresholds.get(regime, thresholds['sideways'])
    
    def set_market_regime(self, regime: str):
        """Set the current market regime for feature extraction."""
        self._market_regime = regime
    
    def create_training_dataset(
        self,
        stock_data: List[Dict[str, Any]],
        labels: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Create a training dataset from multiple stocks.
        
        Args:
            stock_data: List of dicts with 'symbol', 'prices', and engine results
            labels: Series mapping symbol to signal label (BUY/HOLD/AVOID/SELL)
            
        Returns:
            (X, y) tuple of features DataFrame and labels Series
        """
        all_features = []
        all_labels = []
        
        for data in stock_data:
            symbol = data['symbol']
            if symbol not in labels.index:
                continue
            
            feature_set = self.extract_features(
                symbol=symbol,
                prices=data.get('prices', pd.Series()),
                governance_result=data.get('governance'),
                financial_result=data.get('financial'),
                valuation_result=data.get('valuation'),
                market_result=data.get('market'),
                forecast_result=data.get('forecast'),
            )
            
            all_features.append(feature_set.features)
            all_labels.append(labels[symbol])
        
        X = pd.DataFrame(all_features, columns=self.feature_names)
        y = pd.Series(all_labels, name='signal')
        
        # Handle missing values
        X = X.fillna(0)
        
        return X, y

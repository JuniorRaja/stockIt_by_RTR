"""
ML Models Module for Stocron by RTR

This module provides a modular ML architecture with three layers:
- Layer 1 (Forecaster): Time-series prediction using Chronos-T5 or Lag-Llama
- Layer 2 (Classifier): Signal classification using LightGBM or CatBoost  
- Layer 3 (Explainer): Natural language explanations using Qwen2.5-3B

All models are optional and can be configured based on user's hardware.
"""

from .base import ModelConfig, MLLayerBase, ModelStatus
from .model_manager import ModelManager
from .forecaster import TimeSeriesForecaster, ChronosForecaster, LagLlamaForecaster
from .classifier import SignalClassifier, LightGBMClassifier, CatBoostClassifier
from .explainer import AnalysisExplainer, QwenExplainer
from .features import FeatureEngineer
from .ensemble import MLEnsemble, MLPrediction

__all__ = [
    # Base classes
    'ModelConfig',
    'MLLayerBase', 
    'ModelStatus',
    
    # Model Manager
    'ModelManager',
    
    # Layer 1: Forecasters
    'TimeSeriesForecaster',
    'ChronosForecaster',
    'LagLlamaForecaster',
    
    # Layer 2: Classifiers
    'SignalClassifier',
    'LightGBMClassifier',
    'CatBoostClassifier',
    
    # Layer 3: Explainers
    'AnalysisExplainer',
    'QwenExplainer',
    
    # Feature Engineering
    'FeatureEngineer',
    
    # Ensemble
    'MLEnsemble',
    'MLPrediction',
]

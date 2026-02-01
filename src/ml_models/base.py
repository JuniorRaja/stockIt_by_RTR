"""
Base classes for ML models module.

Provides abstract interfaces and common utilities for all ML layers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ModelStatus(Enum):
    """Status of a model."""
    NOT_DOWNLOADED = "not_downloaded"
    DOWNLOADING = "downloading"
    DOWNLOADED = "downloaded"
    LOADING = "loading"
    READY = "ready"
    ERROR = "error"


class ModelType(Enum):
    """Types of models available in each layer."""
    # Layer 1: Forecasters
    CHRONOS_T5_TINY = "chronos-t5-tiny"
    CHRONOS_T5_SMALL = "chronos-t5-small"
    CHRONOS_T5_BASE = "chronos-t5-base"
    LAG_LLAMA = "lag-llama"
    
    # Layer 2: Classifiers
    LIGHTGBM = "lightgbm"
    CATBOOST = "catboost"
    XGBOOST = "xgboost"
    
    # Layer 3: Explainers
    QWEN_3B = "qwen2.5-3b"
    QWEN_7B = "qwen2.5-7b"
    PHI3_MINI = "phi-3-mini"


@dataclass
class ModelConfig:
    """Configuration for an ML model."""
    model_type: ModelType
    model_path: Path
    device: str = "cpu"  # "cpu", "cuda", "mps"
    quantization: Optional[str] = None  # "int4", "int8", None
    max_memory_gb: float = 4.0
    batch_size: int = 1
    context_length: int = 512
    
    # Model-specific settings
    extra_config: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_downloaded(self) -> bool:
        """Check if model files exist."""
        if not self.model_path.exists():
            return False
        # Check for common model file patterns
        patterns = ['*.bin', '*.safetensors', '*.gguf', '*.pt', '*.pth', 'config.json']
        for pattern in patterns:
            if list(self.model_path.glob(pattern)):
                return True
        return False


@dataclass
class LayerOutput:
    """Base output structure for all ML layers."""
    success: bool
    layer_name: str
    model_used: str
    inference_time_ms: float
    device_used: str
    raw_output: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class ForecastOutput(LayerOutput):
    """Output from Layer 1: Time-series forecaster."""
    predictions: Optional[np.ndarray] = None  # Shape: (horizon,)
    prediction_intervals: Optional[Dict[str, np.ndarray]] = None  # {'lower_50': ..., 'upper_50': ..., etc.}
    horizon_days: int = 30
    confidence_scores: Optional[np.ndarray] = None
    trend_direction: str = "neutral"  # "bullish", "bearish", "neutral"
    trend_strength: float = 0.0  # 0-1


@dataclass
class ClassifierOutput(LayerOutput):
    """Output from Layer 2: Signal classifier."""
    signal: str = "HOLD"  # "BUY", "HOLD", "AVOID", "SELL"
    signal_probabilities: Optional[Dict[str, float]] = None  # {'BUY': 0.3, 'HOLD': 0.5, ...}
    confidence: float = 0.0  # 0-1
    feature_importance: Optional[Dict[str, float]] = None
    decision_path: Optional[List[str]] = None


@dataclass
class ExplainerOutput(LayerOutput):
    """Output from Layer 3: LLM explainer."""
    summary: str = ""
    detailed_analysis: str = ""
    key_points: List[str] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)
    opportunities: List[str] = field(default_factory=list)
    recommendation_rationale: str = ""


class MLLayerBase(ABC):
    """Abstract base class for all ML layers."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self._model = None
        self._status = ModelStatus.NOT_DOWNLOADED
        self._logger = logging.getLogger(self.__class__.__name__)
    
    @property
    def status(self) -> ModelStatus:
        return self._status
    
    @property
    def is_ready(self) -> bool:
        return self._status == ModelStatus.READY
    
    @property
    @abstractmethod
    def layer_name(self) -> str:
        """Name of this layer (e.g., 'forecaster', 'classifier', 'explainer')."""
        pass
    
    @abstractmethod
    def load_model(self) -> bool:
        """Load the model into memory. Returns True if successful."""
        pass
    
    @abstractmethod
    def unload_model(self) -> None:
        """Unload the model from memory to free resources."""
        pass
    
    @abstractmethod
    def predict(self, *args, **kwargs) -> LayerOutput:
        """Run inference. Specific signature depends on layer type."""
        pass
    
    def check_availability(self) -> Tuple[bool, str]:
        """
        Check if this model can run on current hardware.
        Returns (can_run, message).
        """
        import torch
        
        available_memory = self._get_available_memory()
        required_memory = self.config.max_memory_gb
        
        if available_memory < required_memory:
            return False, f"Insufficient memory. Need {required_memory}GB, have {available_memory:.1f}GB"
        
        # Check device availability
        if self.config.device == "cuda" and not torch.cuda.is_available():
            return False, "CUDA requested but not available. Set device='cpu' or 'mps'"
        
        if self.config.device == "mps" and not torch.backends.mps.is_available():
            return False, "MPS (Apple Silicon) requested but not available"
        
        return True, "Model can run on this hardware"
    
    def _get_available_memory(self) -> float:
        """Get available memory in GB."""
        import torch
        
        if self.config.device == "cuda" and torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return props.total_memory / (1024**3)
        elif self.config.device == "mps" and torch.backends.mps.is_available():
            # MPS doesn't have a direct memory query, estimate based on system
            try:
                import psutil
                return psutil.virtual_memory().available / (1024**3)
            except ImportError:
                return 8.0  # Conservative estimate
        else:
            # CPU - use system memory
            try:
                import psutil
                return psutil.virtual_memory().available / (1024**3)
            except ImportError:
                return 8.0
    
    def _detect_best_device(self) -> str:
        """Auto-detect the best available device."""
        import torch
        
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        return "cpu"


class ModelRegistry:
    """Registry of available models and their configurations."""
    
    MODELS = {
        # Layer 1: Time-series Forecasters
        ModelType.CHRONOS_T5_TINY: {
            "name": "Chronos-T5-Tiny",
            "hf_repo": "amazon/chronos-t5-tiny",
            "size_gb": 0.1,
            "min_memory_gb": 1.0,
            "description": "Smallest Chronos model, fast inference, basic accuracy",
            "layer": "forecaster",
        },
        ModelType.CHRONOS_T5_SMALL: {
            "name": "Chronos-T5-Small", 
            "hf_repo": "amazon/chronos-t5-small",
            "size_gb": 0.2,
            "min_memory_gb": 2.0,
            "description": "Good balance of speed and accuracy",
            "layer": "forecaster",
        },
        ModelType.CHRONOS_T5_BASE: {
            "name": "Chronos-T5-Base",
            "hf_repo": "amazon/chronos-t5-base",
            "size_gb": 0.5,
            "min_memory_gb": 4.0,
            "description": "Best Chronos accuracy, moderate resource usage",
            "layer": "forecaster",
        },
        ModelType.LAG_LLAMA: {
            "name": "Lag-Llama",
            "hf_repo": "time-series-foundation-models/Lag-Llama",
            "size_gb": 1.0,
            "min_memory_gb": 4.0,
            "description": "Probabilistic forecasting with uncertainty quantification",
            "layer": "forecaster",
        },
        
        # Layer 2: Classifiers (trained locally, no download needed)
        ModelType.LIGHTGBM: {
            "name": "LightGBM",
            "hf_repo": None,  # Trained locally
            "size_gb": 0.05,
            "min_memory_gb": 0.5,
            "description": "Fast gradient boosting, low memory usage",
            "layer": "classifier",
        },
        ModelType.CATBOOST: {
            "name": "CatBoost",
            "hf_repo": None,  # Trained locally
            "size_gb": 0.1,
            "min_memory_gb": 1.0,
            "description": "Best with categorical features, robust to overfitting",
            "layer": "classifier",
        },
        ModelType.XGBOOST: {
            "name": "XGBoost",
            "hf_repo": None,  # Trained locally
            "size_gb": 0.05,
            "min_memory_gb": 0.5,
            "description": "Battle-tested gradient boosting",
            "layer": "classifier",
        },
        
        # Layer 3: LLM Explainers
        ModelType.QWEN_3B: {
            "name": "Qwen2.5-3B-Instruct",
            "hf_repo": "Qwen/Qwen2.5-3B-Instruct",
            "gguf_repo": "Qwen/Qwen2.5-3B-Instruct-GGUF",
            "size_gb": 2.0,  # Quantized
            "min_memory_gb": 4.0,
            "description": "Good quality explanations, runs on most hardware",
            "layer": "explainer",
        },
        ModelType.QWEN_7B: {
            "name": "Qwen2.5-7B-Instruct",
            "hf_repo": "Qwen/Qwen2.5-7B-Instruct",
            "gguf_repo": "Qwen/Qwen2.5-7B-Instruct-GGUF",
            "size_gb": 4.5,  # Quantized
            "min_memory_gb": 8.0,
            "description": "Better explanations, needs more memory",
            "layer": "explainer",
        },
        ModelType.PHI3_MINI: {
            "name": "Phi-3-Mini",
            "hf_repo": "microsoft/Phi-3-mini-4k-instruct",
            "gguf_repo": "microsoft/Phi-3-mini-4k-instruct-gguf",
            "size_gb": 2.3,  # Quantized
            "min_memory_gb": 4.0,
            "description": "Microsoft's compact but capable model",
            "layer": "explainer",
        },
    }
    
    @classmethod
    def get_model_info(cls, model_type: ModelType) -> Dict[str, Any]:
        """Get information about a model."""
        return cls.MODELS.get(model_type, {})
    
    @classmethod
    def get_models_for_layer(cls, layer: str) -> List[ModelType]:
        """Get all model types available for a specific layer."""
        return [mt for mt, info in cls.MODELS.items() if info.get("layer") == layer]
    
    @classmethod
    def get_recommended_model(cls, layer: str, available_memory_gb: float) -> Optional[ModelType]:
        """Get the best model for a layer given available memory."""
        candidates = []
        for mt in cls.get_models_for_layer(layer):
            info = cls.MODELS[mt]
            if info.get("min_memory_gb", 0) <= available_memory_gb:
                candidates.append((mt, info.get("min_memory_gb", 0)))
        
        if not candidates:
            return None
        
        # Return the one requiring most memory that still fits
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]

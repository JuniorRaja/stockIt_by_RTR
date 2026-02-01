"""
Layer 1: Time-Series Forecaster

Provides price forecasting using foundation models:
- Chronos-T5 (Amazon): Fast, efficient, good for general forecasting
- Lag-Llama: Probabilistic forecasting with uncertainty quantification

Both models predict future price movements and provide confidence intervals.
"""

import logging
import time
from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .base import (
    ForecastOutput,
    MLLayerBase,
    ModelConfig,
    ModelStatus,
    ModelType,
)

logger = logging.getLogger(__name__)


class TimeSeriesForecaster(MLLayerBase):
    """Base class for time-series forecasting models."""
    
    @property
    def layer_name(self) -> str:
        return "forecaster"
    
    @abstractmethod
    def forecast(
        self,
        price_history: pd.Series,
        horizon: int = 30,
        num_samples: int = 100
    ) -> ForecastOutput:
        """
        Generate price forecasts.
        
        Args:
            price_history: Historical closing prices (DatetimeIndex)
            horizon: Number of days to forecast
            num_samples: Number of Monte Carlo samples for uncertainty
            
        Returns:
            ForecastOutput with predictions and confidence intervals
        """
        pass
    
    def predict(self, *args, **kwargs) -> ForecastOutput:
        """Wrapper for forecast method."""
        return self.forecast(*args, **kwargs)
    
    def _calculate_trend(
        self, 
        current_price: float, 
        predictions: np.ndarray
    ) -> Tuple[str, float]:
        """
        Calculate trend direction and strength from predictions.
        
        Returns:
            (direction: "bullish"/"bearish"/"neutral", strength: 0-1)
        """
        if predictions is None or len(predictions) == 0:
            return "neutral", 0.0
        
        # Calculate expected return
        mean_future_price = np.mean(predictions[-5:])  # Last 5 days average
        expected_return = (mean_future_price - current_price) / current_price
        
        # Determine direction
        if expected_return > 0.02:  # >2% expected gain
            direction = "bullish"
        elif expected_return < -0.02:  # >2% expected loss
            direction = "bearish"
        else:
            direction = "neutral"
        
        # Strength is absolute return, capped at 1
        strength = min(abs(expected_return) * 5, 1.0)  # 20% return = max strength
        
        return direction, strength


class ChronosForecaster(TimeSeriesForecaster):
    """
    Chronos-T5 based time-series forecaster.
    
    Amazon's Chronos is a family of pretrained time series forecasting models
    based on T5. It tokenizes time series values and trains on a large corpus
    of time series data.
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self._pipeline = None
    
    def load_model(self) -> bool:
        """Load Chronos model into memory."""
        if self._model is not None:
            return True
        
        try:
            self._status = ModelStatus.LOADING
            
            # Check if model path exists
            if not self.config.model_path.exists():
                # Try to use HuggingFace repo directly
                model_id = self._get_model_id()
            else:
                model_id = str(self.config.model_path)
            
            # Import chronos
            try:
                from chronos import ChronosPipeline
            except ImportError:
                self._logger.error(
                    "chronos-forecasting not installed. "
                    "Run: pip install chronos-forecasting"
                )
                self._status = ModelStatus.ERROR
                return False
            
            import torch
            
            # Determine device
            device = self.config.device
            if device == "auto":
                device = self._detect_best_device()
            
            # Handle special device types
            actual_device = device
            if device.startswith("openvino") or device == "directml":
                # These don't work directly with Chronos, fallback to CPU
                self._logger.warning(f"{device} not supported by Chronos, falling back to CPU")
                actual_device = "cpu"
            
            # Set dtype based on device for optimal performance
            if actual_device == "cuda":
                # Use bfloat16 for CUDA (faster inference)
                dtype = torch.bfloat16
                # Verify CUDA is actually working
                if not torch.cuda.is_available():
                    self._logger.warning("CUDA requested but not available, falling back to CPU")
                    actual_device = "cpu"
                    dtype = torch.float32
                else:
                    # Log GPU info for debugging
                    gpu_name = torch.cuda.get_device_name(0)
                    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    self._logger.info(f"Using CUDA GPU: {gpu_name} ({gpu_mem:.1f} GB VRAM)")
            elif actual_device == "mps":
                # MPS works best with float32
                dtype = torch.float32
                if not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
                    self._logger.warning("MPS requested but not available, falling back to CPU")
                    actual_device = "cpu"
            else:
                dtype = torch.float32
            
            self._logger.info(f"Loading Chronos model from {model_id} on {actual_device} with {dtype}")
            
            self._pipeline = ChronosPipeline.from_pretrained(
                model_id,
                device_map=actual_device,
                dtype=dtype,
            )
            
            # Store actual device used for reporting
            self._actual_device = actual_device
            self._model = self._pipeline
            self._status = ModelStatus.READY
            self._logger.info(f"Chronos model loaded successfully on {actual_device}")
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to load Chronos model: {e}")
            self._status = ModelStatus.ERROR
            return False
    
    def _get_model_id(self) -> str:
        """Get the HuggingFace model ID based on model type."""
        model_map = {
            ModelType.CHRONOS_T5_TINY: "amazon/chronos-t5-tiny",
            ModelType.CHRONOS_T5_SMALL: "amazon/chronos-t5-small",
            ModelType.CHRONOS_T5_BASE: "amazon/chronos-t5-base",
        }
        return model_map.get(self.config.model_type, "amazon/chronos-t5-base")
    
    def unload_model(self) -> None:
        """Unload model from memory."""
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None
        if self._model is not None:
            del self._model
            self._model = None
        
        # Force garbage collection
        import gc
        gc.collect()
        
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        
        self._status = ModelStatus.DOWNLOADED
    
    def forecast(
        self,
        price_history: pd.Series,
        horizon: int = 30,
        num_samples: int = 100
    ) -> ForecastOutput:
        """
        Generate price forecasts using Chronos.
        
        Args:
            price_history: Historical closing prices (should be at least 60 days)
            horizon: Days to forecast (1-64 recommended)
            num_samples: Monte Carlo samples for uncertainty
        """
        start_time = time.time()
        
        # Ensure model is loaded
        if not self.is_ready:
            if not self.load_model():
                return ForecastOutput(
                    success=False,
                    layer_name=self.layer_name,
                    model_used=self.config.model_type.value,
                    inference_time_ms=0,
                    device_used=self.config.device,
                    error="Failed to load model"
                )
        
        try:
            import torch
            
            # Prepare context (use log returns for stationarity)
            prices = price_history.values.astype(np.float32)
            
            # Use price series directly (Chronos handles normalization)
            context = torch.tensor(prices).unsqueeze(0)  # Shape: (1, seq_len)
            
            # Generate forecasts
            with torch.no_grad():
                forecast_samples = self._pipeline.predict(
                    context,
                    prediction_length=horizon,
                    num_samples=num_samples,
                )  # Shape: (1, num_samples, horizon)
            
            # Extract predictions
            samples = forecast_samples[0].cpu().numpy()  # (num_samples, horizon)
            
            # Calculate statistics
            mean_forecast = np.median(samples, axis=0)  # Median is more robust
            
            # Prediction intervals
            intervals = {
                "lower_10": np.percentile(samples, 10, axis=0),
                "lower_25": np.percentile(samples, 25, axis=0),
                "lower_50": np.percentile(samples, 50, axis=0),
                "upper_50": np.percentile(samples, 50, axis=0),
                "upper_75": np.percentile(samples, 75, axis=0),
                "upper_90": np.percentile(samples, 90, axis=0),
            }
            
            # Calculate confidence (inverse of relative spread)
            spread = (intervals["upper_90"] - intervals["lower_10"]) / mean_forecast
            confidence = 1.0 - np.clip(np.mean(spread), 0, 1)
            
            # Trend analysis
            current_price = prices[-1]
            trend_direction, trend_strength = self._calculate_trend(
                current_price, mean_forecast
            )
            
            inference_time = (time.time() - start_time) * 1000
            
            return ForecastOutput(
                success=True,
                layer_name=self.layer_name,
                model_used=self.config.model_type.value,
                inference_time_ms=inference_time,
                device_used=self.config.device,
                predictions=mean_forecast,
                prediction_intervals=intervals,
                horizon_days=horizon,
                confidence_scores=np.array([confidence] * horizon),
                trend_direction=trend_direction,
                trend_strength=trend_strength,
                raw_output=samples,
                metadata={
                    "num_samples": num_samples,
                    "context_length": len(prices),
                    "current_price": float(current_price),
                    "expected_price_30d": float(mean_forecast[-1]) if horizon >= 30 else float(mean_forecast[-1]),
                }
            )
            
        except Exception as e:
            self._logger.error(f"Forecast failed: {e}")
            return ForecastOutput(
                success=False,
                layer_name=self.layer_name,
                model_used=self.config.model_type.value,
                inference_time_ms=(time.time() - start_time) * 1000,
                device_used=self.config.device,
                error=str(e)
            )


class LagLlamaForecaster(TimeSeriesForecaster):
    """
    Lag-Llama based time-series forecaster.
    
    Lag-Llama is a foundation model for probabilistic time series forecasting
    based on a decoder-only transformer. It excels at uncertainty quantification.
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self._model = None
        self._tokenizer = None
    
    def load_model(self) -> bool:
        """Load Lag-Llama model into memory."""
        if self._model is not None:
            return True
        
        try:
            self._status = ModelStatus.LOADING
            
            import torch
            
            # Try importing lag-llama
            try:
                from lag_llama.gluon.estimator import LagLlamaEstimator
                from gluonts.dataset.pandas import PandasDataset
            except ImportError:
                self._logger.error(
                    "lag-llama not installed. "
                    "Run: pip install lag-llama"
                )
                self._status = ModelStatus.ERROR
                return False
            
            # Check model path
            model_path = self.config.model_path
            if model_path.exists():
                ckpt_path = model_path / "lag-llama.ckpt"
                if not ckpt_path.exists():
                    # Look for any .ckpt file
                    ckpts = list(model_path.glob("*.ckpt"))
                    if ckpts:
                        ckpt_path = ckpts[0]
                    else:
                        ckpt_path = None
            else:
                ckpt_path = None
            
            # Determine device
            device = self.config.device
            if device == "auto":
                device = self._detect_best_device()
            
            # Create estimator
            self._estimator = LagLlamaEstimator(
                ckpt_path=str(ckpt_path) if ckpt_path else None,
                prediction_length=30,
                context_length=512,
                num_samples=100,
                device=torch.device(device if device != "mps" else "cpu"),  # MPS not fully supported
                batch_size=1,
            )
            
            # Get predictor
            self._model = self._estimator.create_predictor(self._estimator.create_transformation())
            self._status = ModelStatus.READY
            self._logger.info("Lag-Llama model loaded successfully")
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to load Lag-Llama model: {e}")
            self._status = ModelStatus.ERROR
            return False
    
    def unload_model(self) -> None:
        """Unload model from memory."""
        if self._model is not None:
            del self._model
            self._model = None
        if hasattr(self, '_estimator') and self._estimator is not None:
            del self._estimator
            self._estimator = None
        
        import gc
        gc.collect()
        
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        
        self._status = ModelStatus.DOWNLOADED
    
    def forecast(
        self,
        price_history: pd.Series,
        horizon: int = 30,
        num_samples: int = 100
    ) -> ForecastOutput:
        """
        Generate probabilistic price forecasts using Lag-Llama.
        """
        start_time = time.time()
        
        # Ensure model is loaded
        if not self.is_ready:
            if not self.load_model():
                return ForecastOutput(
                    success=False,
                    layer_name=self.layer_name,
                    model_used=self.config.model_type.value,
                    inference_time_ms=0,
                    device_used=self.config.device,
                    error="Failed to load model"
                )
        
        try:
            from gluonts.dataset.pandas import PandasDataset
            
            # Prepare data in GluonTS format
            df = pd.DataFrame({
                "target": price_history.values,
                "start": price_history.index[0]
            })
            
            dataset = PandasDataset.from_long_dataframe(
                df.reset_index(),
                item_id="symbol" if "symbol" not in df.columns else None,
                target="target",
                timestamp="index"
            )
            
            # Generate forecasts
            forecasts = list(self._model.predict(dataset))
            
            if not forecasts:
                raise ValueError("No forecasts generated")
            
            forecast = forecasts[0]
            
            # Extract samples and statistics
            samples = forecast.samples  # (num_samples, horizon)
            mean_forecast = forecast.mean
            
            # Prediction intervals
            intervals = {
                "lower_10": forecast.quantile(0.1),
                "lower_25": forecast.quantile(0.25),
                "lower_50": forecast.quantile(0.5),
                "upper_50": forecast.quantile(0.5),
                "upper_75": forecast.quantile(0.75),
                "upper_90": forecast.quantile(0.9),
            }
            
            # Confidence from forecast uncertainty
            spread = (intervals["upper_90"] - intervals["lower_10"]) / np.abs(mean_forecast)
            confidence = 1.0 - np.clip(np.mean(spread), 0, 1)
            
            # Trend
            current_price = price_history.values[-1]
            trend_direction, trend_strength = self._calculate_trend(
                current_price, mean_forecast
            )
            
            inference_time = (time.time() - start_time) * 1000
            
            return ForecastOutput(
                success=True,
                layer_name=self.layer_name,
                model_used=self.config.model_type.value,
                inference_time_ms=inference_time,
                device_used=self.config.device,
                predictions=mean_forecast,
                prediction_intervals=intervals,
                horizon_days=horizon,
                confidence_scores=np.array([confidence] * len(mean_forecast)),
                trend_direction=trend_direction,
                trend_strength=trend_strength,
                raw_output=samples,
                metadata={
                    "num_samples": num_samples,
                    "context_length": len(price_history),
                    "current_price": float(current_price),
                }
            )
            
        except Exception as e:
            self._logger.error(f"Lag-Llama forecast failed: {e}")
            return ForecastOutput(
                success=False,
                layer_name=self.layer_name,
                model_used=self.config.model_type.value,
                inference_time_ms=(time.time() - start_time) * 1000,
                device_used=self.config.device,
                error=str(e)
            )


def create_forecaster(
    model_type: ModelType,
    model_path: Path,
    device: str = "auto"
) -> TimeSeriesForecaster:
    """
    Factory function to create the appropriate forecaster.
    
    Args:
        model_type: CHRONOS_T5_TINY, CHRONOS_T5_SMALL, CHRONOS_T5_BASE, or LAG_LLAMA
        model_path: Path to model files
        device: "auto", "cpu", "cuda", or "mps"
    """
    config = ModelConfig(
        model_type=model_type,
        model_path=model_path,
        device=device,
    )
    
    if model_type in [ModelType.CHRONOS_T5_TINY, ModelType.CHRONOS_T5_SMALL, ModelType.CHRONOS_T5_BASE]:
        return ChronosForecaster(config)
    elif model_type == ModelType.LAG_LLAMA:
        return LagLlamaForecaster(config)
    else:
        raise ValueError(f"Unknown forecaster model type: {model_type}")

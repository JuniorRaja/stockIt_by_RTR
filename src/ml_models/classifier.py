"""
Layer 2: Signal Classifier

Gradient boosting classifiers for BUY/HOLD/AVOID/SELL signal generation.
Supports:
- LightGBM: Fast training, low memory, handles missing values
- CatBoost: Best with categorical features, robust to overfitting
- XGBoost: Battle-tested, highly optimized

These models are trained locally on historical data and user feedback.
"""

import json
import logging
import pickle
import time
from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .base import (
    ClassifierOutput,
    MLLayerBase,
    ModelConfig,
    ModelStatus,
    ModelType,
)

logger = logging.getLogger(__name__)


# Signal labels
SIGNAL_LABELS = ["BUY", "HOLD", "AVOID", "SELL"]
SIGNAL_TO_IDX = {s: i for i, s in enumerate(SIGNAL_LABELS)}
IDX_TO_SIGNAL = {i: s for i, s in enumerate(SIGNAL_LABELS)}


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    n_estimators: int = 500
    learning_rate: float = 0.05
    max_depth: int = 6
    min_samples_leaf: int = 20
    early_stopping_rounds: int = 50
    validation_fraction: float = 0.2
    random_state: int = 42
    class_weight: str = "balanced"  # Handle imbalanced classes
    
    # Feature selection
    feature_importance_threshold: float = 0.01
    
    # Cross-validation
    n_cv_folds: int = 5


@dataclass
class TrainingResult:
    """Result of model training."""
    success: bool
    model_path: Path
    metrics: Dict[str, float] = field(default_factory=dict)
    feature_importance: Dict[str, float] = field(default_factory=dict)
    training_samples: int = 0
    validation_samples: int = 0
    error: Optional[str] = None


class SignalClassifier(MLLayerBase):
    """Base class for signal classification models."""
    
    def __init__(self, config: ModelConfig, training_config: Optional[TrainingConfig] = None):
        super().__init__(config)
        self.training_config = training_config or TrainingConfig()
        self._feature_names: List[str] = []
        self._label_encoder = None
    
    @property
    def layer_name(self) -> str:
        return "classifier"
    
    @abstractmethod
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: Optional[Tuple[pd.DataFrame, pd.Series]] = None
    ) -> TrainingResult:
        """
        Train the classifier on labeled data.
        
        Args:
            X: Features DataFrame
            y: Signal labels (BUY/HOLD/AVOID/SELL)
            eval_set: Optional validation set
        """
        pass
    
    @abstractmethod
    def classify(
        self,
        features: Dict[str, float]
    ) -> ClassifierOutput:
        """
        Classify a stock based on features.
        
        Args:
            features: Dictionary of feature name -> value
            
        Returns:
            ClassifierOutput with signal and probabilities
        """
        pass
    
    def predict(self, *args, **kwargs) -> ClassifierOutput:
        """Wrapper for classify method."""
        return self.classify(*args, **kwargs)
    
    def save_model(self, path: Optional[Path] = None) -> bool:
        """Save trained model to disk."""
        if self._model is None:
            self._logger.warning("No model to save")
            return False
        
        save_path = path or self.config.model_path
        save_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save model
            model_file = save_path / "model.pkl"
            with open(model_file, 'wb') as f:
                pickle.dump(self._model, f)
            
            # Save metadata
            metadata = {
                "model_type": self.config.model_type.value,
                "feature_names": self._feature_names,
                "training_config": {
                    "n_estimators": self.training_config.n_estimators,
                    "learning_rate": self.training_config.learning_rate,
                    "max_depth": self.training_config.max_depth,
                }
            }
            with open(save_path / "metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self._logger.info(f"Model saved to {save_path}")
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to save model: {e}")
            return False
    
    def _prepare_features(self, features: Dict[str, float]) -> np.ndarray:
        """Prepare feature vector for prediction."""
        if not self._feature_names:
            raise ValueError("Model not trained - no feature names")
        
        # Create feature array in correct order
        X = np.zeros(len(self._feature_names))
        for i, name in enumerate(self._feature_names):
            X[i] = features.get(name, 0.0)
        
        return X.reshape(1, -1)


class LightGBMClassifier(SignalClassifier):
    """
    LightGBM-based signal classifier.
    
    Advantages:
    - Very fast training and inference
    - Low memory usage
    - Handles missing values natively
    - Good with large feature sets
    """
    
    def load_model(self) -> bool:
        """Load trained model from disk."""
        if self._model is not None:
            return True
        
        model_path = self.config.model_path / "model.pkl"
        
        if not model_path.exists():
            self._logger.warning(f"No model found at {model_path}")
            self._status = ModelStatus.NOT_DOWNLOADED
            return False
        
        try:
            self._status = ModelStatus.LOADING
            
            with open(model_path, 'rb') as f:
                loaded = pickle.load(f)
            
            # Handle both formats: dict (from train_classifier.py) or model directly
            if isinstance(loaded, dict):
                self._model = loaded.get('model')
                self._feature_names = loaded.get('feature_names', [])
                self._logger.info(f"Loaded classifier from training script (type: {loaded.get('model_type', 'unknown')})")
            else:
                self._model = loaded
                # Load metadata from separate file
                metadata_path = self.config.model_path / "metadata.json"
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        self._feature_names = metadata.get("feature_names", [])
            
            self._status = ModelStatus.READY
            self._logger.info("Classifier model loaded successfully")
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to load model: {e}")
            self._status = ModelStatus.ERROR
            return False
    
    def unload_model(self) -> None:
        """Unload model from memory."""
        self._model = None
        self._status = ModelStatus.DOWNLOADED
    
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: Optional[Tuple[pd.DataFrame, pd.Series]] = None
    ) -> TrainingResult:
        """Train LightGBM classifier."""
        start_time = time.time()
        
        try:
            import lightgbm as lgb
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score, f1_score, classification_report
        except ImportError:
            return TrainingResult(
                success=False,
                model_path=self.config.model_path,
                error="lightgbm not installed. Run: pip install lightgbm"
            )
        
        try:
            # Store feature names
            self._feature_names = list(X.columns)
            
            # Encode labels
            y_encoded = y.map(SIGNAL_TO_IDX)
            
            # Split if no eval set provided
            if eval_set is None:
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y_encoded,
                    test_size=self.training_config.validation_fraction,
                    stratify=y_encoded,
                    random_state=self.training_config.random_state
                )
            else:
                X_train, y_train = X, y_encoded
                X_val, y_val = eval_set[0], eval_set[1].map(SIGNAL_TO_IDX)
            
            # Calculate class weights
            class_counts = y_train.value_counts()
            total = len(y_train)
            class_weights = {i: total / (len(class_counts) * count) 
                           for i, count in class_counts.items()}
            sample_weights = y_train.map(class_weights)
            
            # Create dataset
            train_data = lgb.Dataset(
                X_train, 
                label=y_train,
                weight=sample_weights,
                feature_name=self._feature_names
            )
            val_data = lgb.Dataset(
                X_val, 
                label=y_val,
                feature_name=self._feature_names,
                reference=train_data
            )
            
            # Training parameters
            params = {
                "objective": "multiclass",
                "num_class": len(SIGNAL_LABELS),
                "metric": "multi_logloss",
                "learning_rate": self.training_config.learning_rate,
                "max_depth": self.training_config.max_depth,
                "num_leaves": 2 ** self.training_config.max_depth - 1,
                "min_data_in_leaf": self.training_config.min_samples_leaf,
                "feature_fraction": 0.8,
                "bagging_fraction": 0.8,
                "bagging_freq": 5,
                "verbosity": -1,
                "seed": self.training_config.random_state,
            }
            
            # Train
            callbacks = [
                lgb.early_stopping(self.training_config.early_stopping_rounds),
                lgb.log_evaluation(period=100)
            ]
            
            self._model = lgb.train(
                params,
                train_data,
                num_boost_round=self.training_config.n_estimators,
                valid_sets=[train_data, val_data],
                valid_names=['train', 'valid'],
                callbacks=callbacks
            )
            
            # Evaluate
            y_pred = np.argmax(self._model.predict(X_val), axis=1)
            accuracy = accuracy_score(y_val, y_pred)
            f1 = f1_score(y_val, y_pred, average='weighted')
            
            # Feature importance
            importance = self._model.feature_importance(importance_type='gain')
            feature_importance = {
                name: float(imp) for name, imp in 
                zip(self._feature_names, importance)
            }
            
            # Save model
            self.save_model()
            
            self._status = ModelStatus.READY
            
            training_time = time.time() - start_time
            
            return TrainingResult(
                success=True,
                model_path=self.config.model_path,
                metrics={
                    "accuracy": accuracy,
                    "f1_weighted": f1,
                    "training_time_seconds": training_time,
                    "best_iteration": self._model.best_iteration,
                },
                feature_importance=feature_importance,
                training_samples=len(X_train),
                validation_samples=len(X_val),
            )
            
        except Exception as e:
            self._logger.error(f"Training failed: {e}")
            return TrainingResult(
                success=False,
                model_path=self.config.model_path,
                error=str(e)
            )
    
    def classify(self, features: Dict[str, float]) -> ClassifierOutput:
        """Classify using LightGBM or sklearn model."""
        start_time = time.time()
        
        if not self.is_ready:
            if not self.load_model():
                return ClassifierOutput(
                    success=False,
                    layer_name=self.layer_name,
                    model_used=self.config.model_type.value,
                    inference_time_ms=0,
                    device_used="cpu",
                    error="Model not loaded"
                )
        
        try:
            # Prepare features
            X = self._prepare_features(features)
            
            # Predict probabilities (handle both LightGBM and sklearn)
            if hasattr(self._model, 'predict_proba'):
                # sklearn style (GradientBoosting, etc.)
                probabilities = self._model.predict_proba(X)[0]
            else:
                # LightGBM Booster style
                probabilities = self._model.predict(X)[0]
            
            # Get predicted class
            predicted_idx = np.argmax(probabilities)
            predicted_signal = IDX_TO_SIGNAL.get(predicted_idx, "HOLD")
            confidence = float(probabilities[predicted_idx])
            
            # Signal probabilities
            signal_probs = {}
            for i in range(min(len(SIGNAL_LABELS), len(probabilities))):
                signal_probs[SIGNAL_LABELS[i]] = float(probabilities[i])
            
            # Get feature importance (handle both model types)
            feature_importance = {}
            try:
                if hasattr(self._model, 'feature_importance'):
                    # LightGBM Booster
                    importance = self._model.feature_importance(importance_type='gain')
                elif hasattr(self._model, 'feature_importances_'):
                    # sklearn style
                    importance = self._model.feature_importances_
                else:
                    importance = None
                
                if importance is not None and len(importance) > 0:
                    total_importance = importance.sum()
                    if total_importance > 0:
                        feature_importance = {
                            name: float(imp / total_importance)
                            for name, imp in zip(self._feature_names, importance)
                        }
            except Exception:
                feature_importance = {}
            
            inference_time = (time.time() - start_time) * 1000
            
            return ClassifierOutput(
                success=True,
                layer_name=self.layer_name,
                model_used=self.config.model_type.value,
                inference_time_ms=inference_time,
                device_used="cpu",
                signal=predicted_signal,
                signal_probabilities=signal_probs,
                confidence=float(confidence),
                feature_importance=feature_importance,
            )
            
        except Exception as e:
            self._logger.error(f"Classification failed: {e}")
            return ClassifierOutput(
                success=False,
                layer_name=self.layer_name,
                model_used=self.config.model_type.value,
                inference_time_ms=(time.time() - start_time) * 1000,
                device_used="cpu",
                error=str(e)
            )


class CatBoostClassifier(SignalClassifier):
    """
    CatBoost-based signal classifier.
    
    Advantages:
    - Excellent with categorical features
    - Less prone to overfitting
    - Handles missing values
    - Built-in GPU support
    """
    
    def load_model(self) -> bool:
        """Load trained model from disk."""
        if self._model is not None:
            return True
        
        model_path = self.config.model_path / "model.pkl"
        
        if not model_path.exists():
            self._logger.warning(f"No model found at {model_path}")
            self._status = ModelStatus.NOT_DOWNLOADED
            return False
        
        try:
            self._status = ModelStatus.LOADING
            
            with open(model_path, 'rb') as f:
                self._model = pickle.load(f)
            
            # Load metadata
            metadata_path = self.config.model_path / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    self._feature_names = metadata.get("feature_names", [])
            
            self._status = ModelStatus.READY
            self._logger.info("CatBoost model loaded successfully")
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to load model: {e}")
            self._status = ModelStatus.ERROR
            return False
    
    def unload_model(self) -> None:
        """Unload model from memory."""
        self._model = None
        self._status = ModelStatus.DOWNLOADED
    
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: Optional[Tuple[pd.DataFrame, pd.Series]] = None
    ) -> TrainingResult:
        """Train CatBoost classifier."""
        start_time = time.time()
        
        try:
            from catboost import CatBoostClassifier as CatBoost, Pool
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score, f1_score
        except ImportError:
            return TrainingResult(
                success=False,
                model_path=self.config.model_path,
                error="catboost not installed. Run: pip install catboost"
            )
        
        try:
            # Store feature names
            self._feature_names = list(X.columns)
            
            # Encode labels
            y_encoded = y.map(SIGNAL_TO_IDX)
            
            # Split if no eval set provided
            if eval_set is None:
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y_encoded,
                    test_size=self.training_config.validation_fraction,
                    stratify=y_encoded,
                    random_state=self.training_config.random_state
                )
            else:
                X_train, y_train = X, y_encoded
                X_val, y_val = eval_set[0], eval_set[1].map(SIGNAL_TO_IDX)
            
            # Calculate class weights
            class_counts = y_train.value_counts()
            class_weights = [1.0 / class_counts.get(i, 1) for i in range(len(SIGNAL_LABELS))]
            
            # Create model
            self._model = CatBoost(
                iterations=self.training_config.n_estimators,
                learning_rate=self.training_config.learning_rate,
                depth=self.training_config.max_depth,
                loss_function='MultiClass',
                classes_count=len(SIGNAL_LABELS),
                class_weights=class_weights,
                random_seed=self.training_config.random_state,
                verbose=100,
                early_stopping_rounds=self.training_config.early_stopping_rounds,
                task_type='CPU',  # Use 'GPU' if available
            )
            
            # Train
            self._model.fit(
                X_train, y_train,
                eval_set=(X_val, y_val),
                use_best_model=True,
            )
            
            # Evaluate
            y_pred = self._model.predict(X_val).flatten().astype(int)
            accuracy = accuracy_score(y_val, y_pred)
            f1 = f1_score(y_val, y_pred, average='weighted')
            
            # Feature importance
            importance = self._model.get_feature_importance()
            feature_importance = {
                name: float(imp) for name, imp in 
                zip(self._feature_names, importance)
            }
            
            # Save model
            self.save_model()
            
            self._status = ModelStatus.READY
            
            training_time = time.time() - start_time
            
            return TrainingResult(
                success=True,
                model_path=self.config.model_path,
                metrics={
                    "accuracy": accuracy,
                    "f1_weighted": f1,
                    "training_time_seconds": training_time,
                    "best_iteration": self._model.get_best_iteration(),
                },
                feature_importance=feature_importance,
                training_samples=len(X_train),
                validation_samples=len(X_val),
            )
            
        except Exception as e:
            self._logger.error(f"Training failed: {e}")
            return TrainingResult(
                success=False,
                model_path=self.config.model_path,
                error=str(e)
            )
    
    def classify(self, features: Dict[str, float]) -> ClassifierOutput:
        """Classify using CatBoost model."""
        start_time = time.time()
        
        if not self.is_ready:
            if not self.load_model():
                return ClassifierOutput(
                    success=False,
                    layer_name=self.layer_name,
                    model_used=self.config.model_type.value,
                    inference_time_ms=0,
                    device_used="cpu",
                    error="Model not loaded"
                )
        
        try:
            # Prepare features
            X = self._prepare_features(features)
            
            # Predict probabilities
            probabilities = self._model.predict_proba(X)[0]
            
            # Get predicted class
            predicted_idx = np.argmax(probabilities)
            predicted_signal = IDX_TO_SIGNAL[predicted_idx]
            confidence = probabilities[predicted_idx]
            
            # Signal probabilities
            signal_probs = {
                SIGNAL_LABELS[i]: float(probabilities[i])
                for i in range(len(SIGNAL_LABELS))
            }
            
            # Feature importance
            importance = self._model.get_feature_importance()
            total_importance = importance.sum()
            if total_importance > 0:
                feature_importance = {
                    name: float(imp / total_importance)
                    for name, imp in zip(self._feature_names, importance)
                }
            else:
                feature_importance = {}
            
            inference_time = (time.time() - start_time) * 1000
            
            return ClassifierOutput(
                success=True,
                layer_name=self.layer_name,
                model_used=self.config.model_type.value,
                inference_time_ms=inference_time,
                device_used="cpu",
                signal=predicted_signal,
                signal_probabilities=signal_probs,
                confidence=float(confidence),
                feature_importance=feature_importance,
            )
            
        except Exception as e:
            self._logger.error(f"Classification failed: {e}")
            return ClassifierOutput(
                success=False,
                layer_name=self.layer_name,
                model_used=self.config.model_type.value,
                inference_time_ms=(time.time() - start_time) * 1000,
                device_used="cpu",
                error=str(e)
            )


def create_classifier(
    model_type: ModelType,
    model_path: Path,
    training_config: Optional[TrainingConfig] = None
) -> SignalClassifier:
    """
    Factory function to create the appropriate classifier.
    
    Args:
        model_type: LIGHTGBM, CATBOOST, or XGBOOST
        model_path: Path to save/load model files
        training_config: Optional training configuration
    """
    config = ModelConfig(
        model_type=model_type,
        model_path=model_path,
        device="cpu",  # Gradient boosting runs on CPU
    )
    
    if model_type == ModelType.LIGHTGBM:
        return LightGBMClassifier(config, training_config)
    elif model_type == ModelType.CATBOOST:
        return CatBoostClassifier(config, training_config)
    elif model_type == ModelType.XGBOOST:
        # XGBoost implementation similar to LightGBM
        # For now, fall back to LightGBM
        logger.warning("XGBoost not fully implemented, using LightGBM")
        return LightGBMClassifier(config, training_config)
    else:
        raise ValueError(f"Unknown classifier model type: {model_type}")

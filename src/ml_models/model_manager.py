"""
Model Manager for downloading, loading, and managing ML models.

Provides a unified interface for handling all model operations across layers.
"""

import json
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .base import ModelConfig, ModelRegistry, ModelStatus, ModelType

logger = logging.getLogger(__name__)


@dataclass
class DownloadProgress:
    """Track download progress."""
    model_type: ModelType
    total_size_gb: float
    downloaded_gb: float = 0.0
    status: str = "pending"
    error: Optional[str] = None
    
    @property
    def progress_percent(self) -> float:
        if self.total_size_gb == 0:
            return 0.0
        return min(100.0, (self.downloaded_gb / self.total_size_gb) * 100)


class ModelManager:
    """
    Manages ML model lifecycle: download, load, unload, status checking.
    
    Usage:
        manager = ModelManager(models_dir="models/")
        
        # Check what's available
        available = manager.get_available_models()
        
        # Download a model
        manager.download_model(ModelType.CHRONOS_T5_BASE)
        
        # Get model config for use
        config = manager.get_model_config(ModelType.CHRONOS_T5_BASE)
    """
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for each layer
        (self.models_dir / "forecaster").mkdir(exist_ok=True)
        (self.models_dir / "classifier").mkdir(exist_ok=True)
        (self.models_dir / "explainer").mkdir(exist_ok=True)
        
        # Status file to track model states
        self.status_file = self.models_dir / "model_status.json"
        self._status_cache: Dict[str, Dict] = self._load_status()
    
    def _load_status(self) -> Dict[str, Dict]:
        """Load model status from file."""
        if self.status_file.exists():
            try:
                with open(self.status_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load model status: {e}")
        return {}
    
    def _save_status(self) -> None:
        """Save model status to file."""
        try:
            with open(self.status_file, 'w') as f:
                json.dump(self._status_cache, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save model status: {e}")
    
    def get_model_path(self, model_type: ModelType) -> Path:
        """Get the path where a model should be stored."""
        info = ModelRegistry.get_model_info(model_type)
        layer = info.get("layer", "unknown")
        return self.models_dir / layer / model_type.value
    
    def is_model_downloaded(self, model_type: ModelType) -> bool:
        """Check if a model has been downloaded."""
        model_path = self.get_model_path(model_type)
        
        if not model_path.exists():
            return False
        
        info = ModelRegistry.get_model_info(model_type)
        
        # For classifiers (trained locally), check for .pkl or .joblib
        if info.get("layer") == "classifier":
            patterns = ['*.pkl', '*.joblib', '*.bin']
        # For LLM explainers, check for GGUF files
        elif info.get("layer") == "explainer":
            patterns = ['*.gguf', '*.bin', '*.safetensors']
        # For forecasters
        else:
            patterns = ['*.bin', '*.safetensors', 'config.json', '*.pt']
        
        for pattern in patterns:
            if list(model_path.glob(pattern)):
                return True
        
        # Check subdirectories too
        for pattern in patterns:
            if list(model_path.rglob(pattern)):
                return True
                
        return False
    
    def get_model_status(self, model_type: ModelType) -> ModelStatus:
        """Get the current status of a model."""
        if self.is_model_downloaded(model_type):
            return ModelStatus.DOWNLOADED
        return ModelStatus.NOT_DOWNLOADED
    
    def get_available_models(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get all available models grouped by layer.
        
        Returns:
            {
                "forecaster": [{"type": ModelType.CHRONOS_T5_BASE, "downloaded": True, ...}, ...],
                "classifier": [...],
                "explainer": [...]
            }
        """
        result = {"forecaster": [], "classifier": [], "explainer": []}
        
        for model_type in ModelType:
            info = ModelRegistry.get_model_info(model_type)
            if not info:
                continue
                
            layer = info.get("layer")
            if layer not in result:
                continue
            
            result[layer].append({
                "type": model_type,
                "name": info.get("name"),
                "description": info.get("description"),
                "size_gb": info.get("size_gb"),
                "min_memory_gb": info.get("min_memory_gb"),
                "downloaded": self.is_model_downloaded(model_type),
                "hf_repo": info.get("hf_repo"),
            })
        
        return result
    
    def get_download_command(self, model_type: ModelType) -> str:
        """
        Get the command to download a specific model.
        
        Returns a shell command string that users can run.
        """
        info = ModelRegistry.get_model_info(model_type)
        model_path = self.get_model_path(model_type)
        layer = info.get("layer")
        
        if layer == "classifier":
            # Classifiers are trained locally, no download needed
            return f"# {model_type.value} is trained locally - no download needed"
        
        if layer == "explainer":
            # Use GGUF for efficiency
            gguf_repo = info.get("gguf_repo", info.get("hf_repo"))
            
            if "qwen" in model_type.value.lower():
                # Qwen specific GGUF download
                return f"""# Download {info.get('name')} (GGUF format for efficiency)
mkdir -p {model_path}
# Option 1: Using huggingface-cli
huggingface-cli download {gguf_repo} --local-dir {model_path} --include "*.gguf"

# Option 2: Direct download (pick one GGUF file based on your RAM)
# For 8GB RAM: wget https://huggingface.co/{gguf_repo}/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf -O {model_path}/model.gguf
# For 16GB RAM: wget https://huggingface.co/{gguf_repo}/resolve/main/qwen2.5-3b-instruct-q8_0.gguf -O {model_path}/model.gguf"""
            else:
                return f"""# Download {info.get('name')}
mkdir -p {model_path}
huggingface-cli download {gguf_repo} --local-dir {model_path} --include "*.gguf" """
        
        if layer == "forecaster":
            hf_repo = info.get("hf_repo")
            return f"""# Download {info.get('name')}
mkdir -p {model_path}
huggingface-cli download {hf_repo} --local-dir {model_path}

# Or using Python:
# python -c "from huggingface_hub import snapshot_download; snapshot_download('{hf_repo}', local_dir='{model_path}')" """
        
        return f"# Unknown model type: {model_type.value}"
    
    def download_model(
        self, 
        model_type: ModelType,
        progress_callback: Optional[callable] = None
    ) -> Tuple[bool, str]:
        """
        Download a model from HuggingFace.
        
        Args:
            model_type: The model to download
            progress_callback: Optional callback(progress: DownloadProgress)
            
        Returns:
            (success: bool, message: str)
        """
        info = ModelRegistry.get_model_info(model_type)
        model_path = self.get_model_path(model_type)
        layer = info.get("layer")
        
        # Classifiers are trained locally
        if layer == "classifier":
            return True, f"{model_type.value} is trained locally, no download needed"
        
        hf_repo = info.get("hf_repo")
        if not hf_repo:
            return False, f"No HuggingFace repo configured for {model_type.value}"
        
        try:
            # Create directory
            model_path.mkdir(parents=True, exist_ok=True)
            
            # Update status
            self._status_cache[model_type.value] = {"status": "downloading"}
            self._save_status()
            
            if progress_callback:
                progress_callback(DownloadProgress(
                    model_type=model_type,
                    total_size_gb=info.get("size_gb", 1.0),
                    status="downloading"
                ))
            
            # Use huggingface_hub for download
            from huggingface_hub import snapshot_download
            
            # For explainers, prefer GGUF
            if layer == "explainer":
                gguf_repo = info.get("gguf_repo", hf_repo)
                snapshot_download(
                    gguf_repo,
                    local_dir=str(model_path),
                    allow_patterns=["*.gguf", "*.json"],
                )
            else:
                snapshot_download(
                    hf_repo,
                    local_dir=str(model_path),
                )
            
            # Update status
            self._status_cache[model_type.value] = {"status": "downloaded"}
            self._save_status()
            
            if progress_callback:
                progress_callback(DownloadProgress(
                    model_type=model_type,
                    total_size_gb=info.get("size_gb", 1.0),
                    downloaded_gb=info.get("size_gb", 1.0),
                    status="completed"
                ))
            
            return True, f"Successfully downloaded {info.get('name')}"
            
        except ImportError:
            return False, "huggingface_hub not installed. Run: pip install huggingface_hub"
        except Exception as e:
            self._status_cache[model_type.value] = {"status": "error", "error": str(e)}
            self._save_status()
            return False, f"Download failed: {str(e)}"
    
    def get_model_config(
        self,
        model_type: ModelType,
        device: str = "auto",
        quantization: Optional[str] = None
    ) -> ModelConfig:
        """
        Get a ModelConfig for a downloaded model.
        
        Args:
            model_type: The model type
            device: "auto", "cpu", "cuda", or "mps"
            quantization: "int4", "int8", or None
        """
        model_path = self.get_model_path(model_type)
        info = ModelRegistry.get_model_info(model_type)
        
        # Auto-detect device if needed
        if device == "auto":
            device = self._detect_device()
        
        return ModelConfig(
            model_type=model_type,
            model_path=model_path,
            device=device,
            quantization=quantization,
            max_memory_gb=info.get("min_memory_gb", 4.0),
            extra_config={
                "hf_repo": info.get("hf_repo"),
                "layer": info.get("layer"),
            }
        )
    
    def _detect_device(self) -> str:
        """Auto-detect the best available device."""
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"
    
    def get_recommended_config(self) -> Dict[str, Optional[ModelType]]:
        """
        Get recommended models for each layer based on system capabilities.
        
        Returns:
            {"forecaster": ModelType, "classifier": ModelType, "explainer": ModelType}
        """
        available_memory = self._get_available_memory()
        
        recommendations = {}
        for layer in ["forecaster", "classifier", "explainer"]:
            recommended = ModelRegistry.get_recommended_model(layer, available_memory)
            recommendations[layer] = recommended
        
        return recommendations
    
    def _get_available_memory(self) -> float:
        """Get available system memory in GB."""
        try:
            import psutil
            return psutil.virtual_memory().available / (1024**3)
        except ImportError:
            return 8.0  # Conservative default
    
    def delete_model(self, model_type: ModelType) -> Tuple[bool, str]:
        """Delete a downloaded model to free disk space."""
        model_path = self.get_model_path(model_type)
        
        if not model_path.exists():
            return False, f"Model {model_type.value} not found"
        
        try:
            shutil.rmtree(model_path)
            if model_type.value in self._status_cache:
                del self._status_cache[model_type.value]
            self._save_status()
            return True, f"Deleted {model_type.value}"
        except Exception as e:
            return False, f"Failed to delete: {str(e)}"
    
    def print_status(self) -> str:
        """Get a formatted string showing all model statuses."""
        lines = ["=" * 60, "ML Models Status", "=" * 60]
        
        available = self.get_available_models()
        
        for layer in ["forecaster", "classifier", "explainer"]:
            lines.append(f"\n{layer.upper()}")
            lines.append("-" * 40)
            
            for model in available.get(layer, []):
                status = "✓ Downloaded" if model["downloaded"] else "✗ Not Downloaded"
                lines.append(
                    f"  {model['name']:<25} [{model['size_gb']:.1f}GB] {status}"
                )
        
        lines.append("\n" + "=" * 60)
        return "\n".join(lines)


def create_download_script(output_path: str = "download_models.sh") -> str:
    """Generate a shell script with all download commands."""
    manager = ModelManager()
    
    lines = [
        "#!/bin/bash",
        "# Stocron by RTR - Model Download Script",
        "# Generated automatically - run the commands you need",
        "",
        "set -e",
        "",
        "MODELS_DIR=\"models\"",
        "mkdir -p $MODELS_DIR/{forecaster,classifier,explainer}",
        "",
        "# Ensure huggingface-cli is available",
        "pip install -q huggingface_hub",
        "",
    ]
    
    for layer in ["forecaster", "classifier", "explainer"]:
        lines.append(f"\n# ===== Layer: {layer.upper()} =====\n")
        
        for model_type in ModelRegistry.get_models_for_layer(layer):
            info = ModelRegistry.get_model_info(model_type)
            lines.append(f"# {info.get('name')} - {info.get('description')}")
            lines.append(f"# Size: {info.get('size_gb')}GB | Min Memory: {info.get('min_memory_gb')}GB")
            lines.append(manager.get_download_command(model_type))
            lines.append("")
    
    script = "\n".join(lines)
    
    with open(output_path, 'w') as f:
        f.write(script)
    
    return script

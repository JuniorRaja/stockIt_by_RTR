# ML Models Setup Guide

This guide explains how to download and configure the ML models for Stocron by RTR (Indian Equity Intelligence).

## Easiest Way: Use the Download Script

```bash
python 3-download_models.py
```

This interactive script will:
1. Detect your system specs (RAM, GPU)
2. Let you choose models based on your hardware
3. Download everything automatically
4. Update your configuration

**That's it!** After running the script, just `streamlit run app.py`.

---

## Overview

The system uses a **3-layer ML architecture**:

| Layer | Purpose | Models Available | Recommended |
|-------|---------|-----------------|-------------|
| **Layer 1: Forecaster** | Price prediction | Chronos-T5, Lag-Llama | Chronos-T5-Base |
| **Layer 2: Classifier** | Signal generation | LightGBM, CatBoost | LightGBM |
| **Layer 3: Explainer** | Natural language analysis | Qwen2.5-3B, Qwen2.5-7B, Phi-3-mini | Qwen2.5-3B |

All models run **100% locally** on your machine.

---

## Hardware Requirements

### Minimum Requirements
- **RAM:** 8GB
- **Storage:** 5GB free space
- **CPU:** Any modern CPU (Intel/AMD/Apple Silicon)
- **GPU:** Not required (CPU inference supported)

### Recommended Requirements
- **RAM:** 16GB
- **Storage:** 10GB free space
- **GPU:** 6GB+ VRAM (NVIDIA/AMD/Apple Silicon)

### Model Memory Usage

| Model | Size on Disk | RAM Required | Best For |
|-------|--------------|--------------|----------|
| Chronos-T5-Tiny | ~100MB | 1GB | Quick testing |
| Chronos-T5-Small | ~200MB | 2GB | Low-memory systems |
| **Chronos-T5-Base** | ~500MB | 4GB | **Best accuracy** |
| Lag-Llama | ~1GB | 4GB | Uncertainty estimates |
| LightGBM | ~50MB | 0.5GB | Fast classification |
| CatBoost | ~100MB | 1GB | Categorical features |
| **Qwen2.5-3B (Q4)** | ~2GB | 4GB | **Best balance** |
| Qwen2.5-7B (Q4) | ~4.5GB | 8GB | Better quality |
| Phi-3-mini | ~2.3GB | 4GB | Alternative |

---

## Quick Start

### Step 1: Install Dependencies

```bash
# Install all dependencies
pip install -r requirements.txt

# For GPU acceleration (optional):

# NVIDIA GPU (CUDA)
CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir

# Apple Silicon (Metal)
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python --force-reinstall --no-cache-dir

# AMD GPU (ROCm)
CMAKE_ARGS="-DLLAMA_HIPBLAS=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

### Step 2: Create Models Directory

```bash
mkdir -p models/{forecaster,classifier,explainer}
```

### Step 3: Download Models

Choose the models you want based on your hardware:

---

## Layer 1: Time-Series Forecaster

### Option A: Chronos-T5-Base (Recommended)

Amazon's time-series foundation model. Best accuracy for price forecasting.

```bash
# Using huggingface-cli
huggingface-cli download amazon/chronos-t5-base --local-dir models/forecaster/chronos-t5-base

# Or using Python
python -c "from huggingface_hub import snapshot_download; snapshot_download('amazon/chronos-t5-base', local_dir='models/forecaster/chronos-t5-base')"
```

### Option B: Chronos-T5-Small (Lower Memory)

For systems with less RAM.

```bash
huggingface-cli download amazon/chronos-t5-small --local-dir models/forecaster/chronos-t5-small
```

### Option C: Lag-Llama (Probabilistic)

Provides uncertainty quantification in predictions.

```bash
# Install additional dependencies
pip install lag-llama gluonts

# Download model
huggingface-cli download time-series-foundation-models/Lag-Llama --local-dir models/forecaster/lag-llama
```

---

## Layer 2: Signal Classifier

**No download needed!** These models are trained locally on your data.

The classifier will be automatically trained when you:
1. Run analysis on multiple stocks
2. Use the training utility (coming soon)

### Configure in `config/settings.yaml`:

```yaml
ml_config:
  classifier:
    model: "lightgbm"  # or "catboost"
```

---

## Layer 3: LLM Explainer

### Option A: Qwen2.5-3B-Instruct (Recommended)

Best balance of quality and resource usage.

```bash
# Create directory
mkdir -p models/explainer/qwen2.5-3b

# Download GGUF format (quantized for efficiency)

# Option 1: Q4_K_M (2GB, recommended for most systems)
wget https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf \
  -O models/explainer/qwen2.5-3b/model.gguf

# Option 2: Q8_0 (3GB, better quality if you have the RAM)
wget https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q8_0.gguf \
  -O models/explainer/qwen2.5-3b/model.gguf

# Option 3: Download all variants using huggingface-cli
huggingface-cli download Qwen/Qwen2.5-3B-Instruct-GGUF \
  --local-dir models/explainer/qwen2.5-3b \
  --include "*.gguf"
```

### Option B: Qwen2.5-7B-Instruct (Higher Quality)

Better explanations, but needs more memory.

```bash
mkdir -p models/explainer/qwen2.5-7b

# Q4_K_M (4.5GB)
wget https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q4_k_m.gguf \
  -O models/explainer/qwen2.5-7b/model.gguf
```

### Option C: Phi-3-mini (Alternative)

Microsoft's efficient model.

```bash
mkdir -p models/explainer/phi-3-mini

huggingface-cli download microsoft/Phi-3-mini-4k-instruct-gguf \
  --local-dir models/explainer/phi-3-mini \
  --include "*.gguf"
```

---

## Configuration

Edit `config/settings.yaml` to configure your models:

```yaml
ml_config:
  # Enable/disable ML
  enabled: true
  
  # Device: "auto", "cpu", "cuda", "mps"
  device: "auto"
  
  # Layer 1: Forecaster
  forecaster:
    model: "chronos-t5-base"  # or "chronos-t5-small", "lag-llama", null
    horizon_days: 30
    num_samples: 100
  
  # Layer 2: Classifier
  classifier:
    model: "lightgbm"  # or "catboost", null
  
  # Layer 3: Explainer
  explainer:
    model: "qwen2.5-3b"  # or "qwen2.5-7b", "phi-3-mini", null
```

### Disable Specific Layers

Set any layer's model to `null` to disable it:

```yaml
ml_config:
  forecaster:
    model: null  # Disable price prediction
  classifier:
    model: "lightgbm"
  explainer:
    model: null  # Use rule-based explanations
```

---

## Verify Installation

Run this script to check your setup:

```python
from src.ml_models.model_manager import ModelManager

manager = ModelManager("models")
print(manager.print_status())
```

Expected output:

```
============================================================
ML Models Status
============================================================

FORECASTER
----------------------------------------
  Chronos-T5-Tiny             [0.1GB] ✗ Not Downloaded
  Chronos-T5-Small            [0.2GB] ✗ Not Downloaded
  Chronos-T5-Base             [0.5GB] ✓ Downloaded
  Lag-Llama                   [1.0GB] ✗ Not Downloaded

CLASSIFIER
----------------------------------------
  LightGBM                    [0.1GB] ✓ Downloaded
  CatBoost                    [0.1GB] ✗ Not Downloaded
  XGBoost                     [0.1GB] ✗ Not Downloaded

EXPLAINER
----------------------------------------
  Qwen2.5-3B-Instruct         [2.0GB] ✓ Downloaded
  Qwen2.5-7B-Instruct         [4.5GB] ✗ Not Downloaded
  Phi-3-Mini                  [2.3GB] ✗ Not Downloaded

============================================================
```

---

## Troubleshooting

### "chronos-forecasting not installed"

```bash
pip install chronos-forecasting
```

### "llama-cpp-python not installed"

```bash
pip install llama-cpp-python

# For GPU support, see Step 1 above
```

### "huggingface_hub not installed"

```bash
pip install huggingface_hub
```

### "CUDA out of memory"

Switch to CPU or use a smaller model:

```yaml
ml_config:
  device: "cpu"
  forecaster:
    model: "chronos-t5-small"  # Smaller model
```

### "Model loading is slow"

First load takes longer to initialize. Subsequent runs are faster.

For faster startup, keep models in RAM:
1. Initialize ML once in the app
2. Don't restart the app frequently

### "Apple Silicon / Metal issues"

```bash
# Reinstall llama-cpp-python with Metal support
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

---

## Model Selection Guide

### For 8GB RAM Systems

```yaml
ml_config:
  forecaster:
    model: "chronos-t5-small"
  classifier:
    model: "lightgbm"
  explainer:
    model: "qwen2.5-3b"  # Q4 quantization
```

### For 16GB RAM Systems

```yaml
ml_config:
  forecaster:
    model: "chronos-t5-base"
  classifier:
    model: "lightgbm"
  explainer:
    model: "qwen2.5-3b"  # Q8 quantization for better quality
```

### For Systems with GPU (8GB+ VRAM)

```yaml
ml_config:
  device: "cuda"  # or "mps" for Apple Silicon
  forecaster:
    model: "chronos-t5-base"
  classifier:
    model: "catboost"
  explainer:
    model: "qwen2.5-7b"
```

### Minimal Setup (Low Resources)

```yaml
ml_config:
  forecaster:
    model: "chronos-t5-tiny"
  classifier:
    model: "lightgbm"
  explainer:
    model: null  # Use rule-based
```

---

## Complete Download Script

Save as `download_models.sh` and run:

```bash
#!/bin/bash
set -e

echo "=== Stocron by RTR - Model Download Script ==="

# Create directories
mkdir -p models/{forecaster,classifier,explainer}

# Install dependencies
pip install -q huggingface_hub chronos-forecasting

# Download Chronos-T5-Base
echo "Downloading Chronos-T5-Base..."
huggingface-cli download amazon/chronos-t5-base --local-dir models/forecaster/chronos-t5-base

# Download Qwen2.5-3B
echo "Downloading Qwen2.5-3B..."
mkdir -p models/explainer/qwen2.5-3b
wget -q https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf \
  -O models/explainer/qwen2.5-3b/model.gguf

echo "=== Download Complete ==="
echo "Models downloaded to ./models/"
echo "Run 'streamlit run app.py' to start the application"
```

Make it executable and run:

```bash
chmod +x download_models.sh
./download_models.sh
```

---

## Support

If you encounter issues:

1. Check the Troubleshooting section above
2. Verify your `config/settings.yaml` is correct
3. Ensure models are in the correct directory structure
4. Check system resources (RAM, disk space)

The system will fall back to rule-based analysis if ML models are unavailable.

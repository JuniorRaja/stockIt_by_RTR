# Stocron by RTR | The Ultimate Market Intelligence Engine

> **30+ Years of History. Zero Noise. Pure Alpha.**


[![Docker](https://img.shields.io/badge/Docker-Enabled-blue?logo=docker&logoColor=white)](https://www.docker.com/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red?style=for-the-badge&logo=streamlit)](https://streamlit.io/)
[![Model](https://img.shields.io/badge/AI-Chronos--T5%20%2B%20LightGBM-orange?style=for-the-badge)](https://huggingface.co/amazon/chronos-t5-tiny)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

---

## 📉 The Problem
Retail investors are playing a rigged game. Most analysis tools suffer from three critical flaws:
1.  **Recency Bias:** They only look at the last 5-10 years of a Bull market.
2.  **Survivorship Bias:** They ignore delisted companies, making history look safer than it actually was.
3.  **Linear Thinking:** They use simple indicators (RSI, MA) in a complex, non-linear macro environment.

You cannot build generational wealth by just "looking at the chart." You need to understand the **Regime**, the **Fundamentals**, and the **Macro-Economic** backdrop.

## 🛡️ Why This Tool Exists
**Stocron** was built for the **RTR Unfiltered** ecosystem to answer one question:
*"If I had the same data, tools, and computing power as a hedge fund, but with 30 years of unfiltered Indian market context, how would I trade?"*

This is not just a screener. It is a **Time Machine**. It allows you to validate strategies across decades, factoring in inflation, oil prices, and corporate governance failures.

---

## ⚡ Super Powers

### 1. The "Dual-Brain" AI Core 🧠
We don't rely on a single model. Stocron uses a hybrid architecture:
* **The Forecaster (Chronos-T5):** A Transformer-based model (pretrained by Amazon) that treats stock charts like a language to predict future price sequences.
* **The Classifier (LightGBM):** A gradient-boosting decision engine that analyzes hundreds of features (Financials, Macro, Technicals) to generate a binary `BUY`/`HOLD` signal.
* **The Explainer (Qwen2.5):** A local LLM that generates human-readable explanations for AI decisions.

> **GPU Acceleration:** All models support NVIDIA CUDA, Apple Metal (MPS), and Intel OpenVINO for 10-50x faster inference.

### 2. Time Travel & Survivorship ⏳
Most backtests are fake because they test on companies that exist *today*.
* **Delisted Database:** We track companies that failed, ensuring your strategy survives the worst.
* **Time Travel Engine:** Go back to Jan 1st, 2008. The system "forgets" the future, forcing the AI to trade only on what it knew then.

### 3. Scenario Simulator 🌪️
Don't just predict; prepare.
* *What if Crude Oil hits $120?*
* *What if the Repo Rate jumps to 8%?*
The simulator stresses your portfolio against hypothetical macro-economic shocks.

### 4. Governance Guard 🕵️
The tool doesn't just chase profits; it filters out fraud.
* **Beneish M-Score:** Detects earnings manipulation.
* **Altman Z-Score:** Predicts bankruptcy risk.
* **Piotroski F-Score:** Measures fundamental strength.

---

## Architecture 🏗️

The system is containerized for stability and reproducibility.

```text
       ┌──────────────┐
       │     USER     │
       └──────┬───────┘
              │ (Browser)
              ▼
    ┌────────────────────┐
    │    STREAMLIT UI    │
    └─────────┬──────────┘
              │ (Request)
              ▼
    ┌────────────────────┐          ┌──────────────────┐
    │  ANALYSIS ENGINE   │◄─────────│   DATA LAKE DB   │
    └─────────┬──────────┘          │  (JSON / CSV)    │
              │                     └─────────▲────────┘
              │ (Inference)                   │
              │                               │ (Fetch)
    ┌─────────▼──────────┐          ┌─────────┴────────┐
    │     ML MODELS      │          │   INTERNET / NSE │
    │ ┌────────────────┐ │          └──────────────────┘
    │ │ Chronos-T5     │ │
    │ ├────────────────┤ │
    │ │ LightGBM       │ │
    │ ├────────────────┤ │
    │ │ Qwen2.5 (LLM)  │ │
    │ └────────────────┘ │
    │   ▲                │
    │   │ GPU/NPU        │
    │   │ Acceleration   │
    │ ┌─┴──────────────┐ │
    │ │ CUDA/MPS/OpenVINO│
    │ └────────────────┘ │
    └────────────────────┘
```

---

## Installation & Usage (Docker🐳)
> We strongly recommend running Stocron via Docker to avoid dependency hell.

### Prerequisites
- Docker Desktop installed and running
- **For GPU acceleration:** NVIDIA GPU with drivers + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

### Quick Start (CPU Mode)

```bash
# 1) Clone
git clone https://github.com/RTR95/stockIt_by_RTR.git
cd stockIt_by_RTR

# 2) Build + start the app container (CPU mode)
docker-compose up -d --build

# 3) Build local DB from bundled historic data (stocks + indices)
docker-compose exec stocron-by-rtr python 2-download_all_stocks.py --build-db-only

# 4) Download missing live symbols (recommended)
docker-compose exec stocron-by-rtr python 2-download_all_stocks.py

# 5) Delisted stocks (recommended for survivorship bias)
docker-compose exec stocron-by-rtr python scripts/download_delisted_stocks.py --export
docker-compose exec stocron-by-rtr python scripts/download_delisted_stocks.py --download --years 30
docker-compose exec stocron-by-rtr python 2-download_all_stocks.py --build-db-only

# 6) (Optional) Enable real crude oil data
export FRED_API_KEY="your_fred_api_key"

# 7) Download ML models (optional; required for ML forecaster/explainer)
docker-compose exec stocron-by-rtr python 3-download_models.py

# 8) Train classifier (required for ML signals)
docker-compose exec stocron-by-rtr python 4-train_classifier.py
```

Open👉 http://localhost:8501

> **Note:** Source code and models are mounted, so changes persist and reflect immediately without rebuilding.

---

## GPU Acceleration Setup 🎮

ML models (Chronos, Qwen) run **10-50x faster** on GPU. We provide a pre-configured GPU Docker image.

### Option 1: GPU Mode (NVIDIA - Recommended)

```bash
# Prerequisites: Install NVIDIA Container Toolkit first
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

# Verify your GPU is detected
nvidia-smi

# Start with GPU support (uses Dockerfile.gpu with CUDA + cuDNN)
docker-compose --profile gpu up -d --build

# The GPU container is named 'stocron-by-rtr-gpu'
docker-compose exec stocron-by-rtr-gpu python 3-download_models.py

# Run GPU diagnostics
docker-compose exec stocron-by-rtr-gpu python 3-download_models.py --diagnose

# Verify GPU is working
docker-compose exec stocron-by-rtr-gpu python -c "
import torch
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('VRAM:', round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), 'GB')
"
```

### Option 2: CPU Mode (Default)

```bash
# Standard CPU-only mode
docker-compose up -d --build

# All commands use 'stocron-by-rtr' container
docker-compose exec stocron-by-rtr python 3-download_models.py
```

### Switching Between Modes

```bash
# Stop current containers
docker-compose down

# Start CPU mode
docker-compose up -d --build

# OR start GPU mode
docker-compose --profile gpu up -d --build
```

### Other Hardware Options

<details>
<summary><b>Apple Silicon (M1/M2/M3/M4)</b></summary>

Apple Silicon uses Metal Performance Shaders (MPS). Run natively (not in Docker) for best performance:

```bash
# Install dependencies
pip install -r requirements.txt

# Install PyTorch (MPS enabled by default)
pip install torch torchvision

# Enable Metal for Qwen LLM
pip uninstall llama-cpp-python -y
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python --no-cache-dir

# Verify
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
```

</details>

<details>
<summary><b>AMD/Intel GPU (Windows - DirectML)</b></summary>

```bash
pip install torch-directml
```

> **Note:** DirectML works for PyTorch models but llama-cpp-python may still use CPU.

</details>

<details>
<summary><b>Intel NPU (Neural Processing Unit)</b></summary>

```bash
pip install openvino optimum[openvino]
```

> **Note:** NPU support is experimental.

</details>

### Troubleshooting GPU Issues

| Issue | Solution |
|-------|----------|
| `CUDA not available` in GPU container | Ensure NVIDIA Container Toolkit is installed and restart Docker |
| `GPU: Not detected` | Use `--profile gpu` flag: `docker-compose --profile gpu up -d --build` |
| `Out of memory` | Use smaller model (chronos-t5-tiny) via option [3] in download script |
| `nvidia-smi` not found | Install NVIDIA drivers from nvidia.com |
| Container starts but no GPU | Check `nvidia-smi` works on host first |

---

## ML Logic & Explainability 🧩
We believe in "Unfiltered" truth. The AI shouldn't be a black box.

SHAP Integration: We use SHAP (SHapley Additive exPlanations) to break down every signal.

Example Output: "The model is Bullish because 'ROE > 15%' (+20 impact) and 'Oil Prices Dropped' (+10 impact), despite 'RSI being Overbought' (-5 impact)."

## Disclaimer 📜
This tool is for **educational and research purposes only**. It is built for the RTR Unfiltered community to analyze market logic. It is NOT financial advice. Markets are subject to risk. *Use your own brain before making financial decisions*.

## License
MIT License — see [LICENSE](LICENSE).


_**Built with 🧠 by RTR.**_
